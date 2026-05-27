"""KB RAG 服务编排模块。

这个文件把第二阶段的核心动作串起来：文档切块、embedding、入库、向量召回、
reranker 精排、答案草稿生成和 raw Evidence 输出。业务节点和 API 都通过它使用 KB 能力。

核心流程：
    1. 文档入库（ingest）：原始文本 -> 切块 -> 生成 embedding -> 写入向量存储
    2. 知识查询（query）：用户问题 -> 生成 query embedding -> 向量召回候选 -> rerank 精排 -> 构建答案
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any
from uuid import uuid4

from app.core.config import settings
from app.kb.chunking import KBChunk, split_text_to_chunks
from app.kb.embeddings import EmbeddingProvider, HashEmbeddingProvider, OpenAICompatibleEmbeddingProvider
from app.kb.hybrid import build_search_text
from app.kb.reranker import HTTPReranker, KeywordReranker, Reranker
from app.kb.store import (
    InMemoryKnowledgeStore,
    KnowledgeStore,
    PostgreSQLPgVectorStore,
    RetrievedChunk,
    StoredChunk,
)
from app.models import EvidenceSource


@dataclass(slots=True)
class KBIngestResult:
    """文档入库操作的结果数据类。

    入库结果只暴露文档 ID 和 chunk 数量，不泄漏 embedding 向量细节，
    避免将大维度向量意外传递到上层业务逻辑。

    Attributes:
        document_id: 文档唯一标识，由调用方提供或自动生成（uuid4）
        chunk_count: 该文档被切分后产生的 chunk 总数
        store_type: 底层存储类型（"memory" 或 "pgvector"），用于日志和调试
    """

    document_id: str
    chunk_count: int
    store_type: str


@dataclass(slots=True)
class KBQueryResult:
    """知识库查询操作的结果数据类。

    KB 查询结果包含答案草稿和 raw_evidence，后续统一交给 Evidence Normalizer
    转成标准证据对象，供下游业务节点消费。

    Attributes:
        answer: 基于检索到的 chunk 构建的答案草稿（当前阶段为模板式摘要，非 LLM 生成）
        raw_evidence: 原始证据列表，每条证据是一个字典，包含 source_type、content、score 等字段
        retrieved_chunks: rerank 精排后的 chunk 列表，保留完整的检索元信息供上层使用
    """

    answer: str
    raw_evidence: list[dict[str, Any]] = field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    candidate_count: int = 0
    retrieval_sources: list[str] = field(default_factory=list)
    hybrid_retrieval_enabled: bool = False


@dataclass(slots=True)
class KBStatus:
    """知识库运行状态。

    这个状态用于 API 诊断和部署检查，只暴露 provider/store 类型、维度和 chunk 数，
    不返回 DSN、API Key 等敏感配置。
    """

    store_type: str
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    reranker_type: str
    reranker_status: dict[str, Any]
    hybrid_retrieval_enabled: bool
    lexical_top_k: int
    rrf_k: int
    chunk_count: int
    postgres_configured: bool
    pgvector_table: str | None


class KnowledgeBaseService:
    """知识库核心服务类，串起第二阶段完整 RAG 链路。

    完整链路：
        文档切块 -> embedding 向量化 -> 入库（pgvector / 内存） ->
        向量相似度召回 -> reranker 精排 -> 证据输出 & 答案草稿生成

    该类是 KB 模块对外的唯一入口，业务节点和 API 层都通过它使用 KB 能力。
    通过构造函数注入 store / embedding_provider / reranker 等依赖，方便测试时替换。
    """

    def __init__(
        self,
        *,
        store: KnowledgeStore,
        embedding_provider: EmbeddingProvider,
        reranker: Reranker,
        chunk_size: int,
        chunk_overlap: int,
        retrieve_top_k: int,
        rerank_top_k: int,
        hybrid_retrieval_enabled: bool = True,
        lexical_top_k: int = 8,
        rrf_k: int = 60,
    ) -> None:
        """初始化知识库服务。

        Args:
            store: 向量存储后端，支持 InMemoryKnowledgeStore 和 PostgreSQLPgVectorStore
            embedding_provider: embedding 提供者，负责将文本转为向量
            reranker: 重排序器，对向量召回的候选进行精排
            chunk_size: 文档切块时每个 chunk 的最大字符数
            chunk_overlap: 相邻 chunk 之间重叠的字符数，用于保留上下文连贯性
            retrieve_top_k: 向量召回阶段返回的候选 chunk 数量
            rerank_top_k: rerank 精排后保留的最终 chunk 数量
        """
        self.store = store
        self.embedding_provider = embedding_provider
        self.reranker = reranker
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retrieve_top_k = retrieve_top_k
        self.rerank_top_k = rerank_top_k
        self.hybrid_retrieval_enabled = hybrid_retrieval_enabled
        self.lexical_top_k = lexical_top_k
        self.rrf_k = rrf_k

    def status(self) -> KBStatus:
        """返回当前 KB RAG 的运行状态。

        该方法主要用于真实 pgvector/embedding 接入时的健康诊断，
        例如确认当前到底使用 hash 还是 OpenAI-compatible embedding、底层是内存还是 PostgreSQL。
        """
        provider_name = getattr(self.embedding_provider, "provider_type", self.embedding_provider.__class__.__name__)
        provider_model = getattr(self.embedding_provider, "model", provider_name)
        return KBStatus(
            store_type=self.store.store_type,
            embedding_provider=provider_name,
            embedding_model=provider_model,
            embedding_dimension=self.embedding_provider.dimension,
            reranker_type=getattr(self.reranker, "reranker_type", self.reranker.__class__.__name__),
            reranker_status=self.reranker.status(),
            hybrid_retrieval_enabled=self.hybrid_retrieval_enabled,
            lexical_top_k=self.lexical_top_k,
            rrf_k=self.rrf_k,
            chunk_count=self.store.count_chunks(),
            postgres_configured=bool(settings.postgres_dsn),
            pgvector_table=getattr(self.store, "table_name", None),
        )

    def ingest_document(
        self,
        *,
        title: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        source_id: str | None = None,
    ) -> KBIngestResult:
        """将一篇文档入库，完成切块、embedding 和存储的全流程。

        处理步骤：
            1. 确定文档 ID（优先使用 source_id，否则自动生成 uuid4）
            2. 将原始文本按 chunk_size / chunk_overlap 切分为多个 KBChunk
            3. 为每个 chunk 生成 embedding 向量，转换为 StoredChunk
            4. 批量写入向量存储（upsert 语义，重复 chunk_id 会覆盖）

        Args:
            title: 文档标题，会写入 chunk 的 metadata，用于答案生成时引用来源
            content: 文档原始全文
            metadata: 附加元数据（如 doc_type、cuisine 等），会合并到每个 chunk 的 metadata 中
            source_id: 外部来源标识，如果提供则同时作为 document_id 使用

        Returns:
            KBIngestResult: 包含文档 ID、chunk 数量和存储类型的结果对象
        """
        # 如果调用方提供了 source_id 则复用为 document_id，否则生成新的 uuid
        document_id = source_id or str(uuid4())

        # 将原始文本切分为多个 chunk，每个 chunk 携带 document_id 和 metadata
        chunks = split_text_to_chunks(
            content,
            document_id=document_id,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            metadata={**(metadata or {}), "title": title, "source_id": source_id or document_id},
        )

        # 为每个 chunk 生成 embedding 向量，转换为可存储的 StoredChunk 格式
        stored_chunks = [self._to_stored_chunk(chunk) for chunk in chunks]

        # 批量写入向量存储（upsert 语义：存在则更新，不存在则插入）
        self.store.upsert_chunks(stored_chunks)

        return KBIngestResult(
            document_id=document_id,
            chunk_count=len(stored_chunks),
            store_type=self.store.store_type,
        )

    def query(self, question: str, *, metadata_filter: dict[str, Any] | None = None) -> KBQueryResult:
        """根据用户问题检索知识库并生成答案草稿。

        完整查询链路：
            1. 将用户问题转为 embedding 向量
            2. 在向量存储中做相似度搜索，召回 top_k 个候选 chunk
            3. 使用 reranker 对候选 chunk 进行精排，保留最相关的 top_k 个
            4. 基于精排结果构建答案草稿和原始证据列表

        Args:
            question: 用户提出的自然语言问题
            metadata_filter: 可选的元数据过滤条件，用于缩小搜索范围
                            （例如只搜索特定 doc_type 或 cuisine 的文档）

        Returns:
            KBQueryResult: 包含答案草稿、原始证据和检索到的 chunk 列表
        """
        # 第一步：将用户问题转为 embedding 向量，用于后续相似度搜索
        query_embedding = self.embedding_provider.embed(question)

        # 第二步：在向量存储中进行相似度搜索，召回 retrieve_top_k 个候选 chunk
        candidates = self.store.search(
            query_embedding,
            top_k=self.retrieve_top_k,
            metadata_filter=metadata_filter,
            query_text=question,
            hybrid_enabled=self.hybrid_retrieval_enabled,
            lexical_top_k=self.lexical_top_k,
            rrf_k=self.rrf_k,
        )

        # 第三步：使用 reranker 对候选 chunk 进行精排，只保留 rerank_top_k 个最相关的
        reranked = self.reranker.rerank(question, candidates, top_k=self.rerank_top_k)

        # 如果精排后没有任何结果，返回"未检索到"的兜底回答
        if not reranked:
            return KBQueryResult(
                answer="我没有在当前知识库中检索到足够相关的资料，暂时不能基于证据回答这个问题。",
                raw_evidence=[],
                retrieved_chunks=[],
                candidate_count=0,
                retrieval_sources=[],
                hybrid_retrieval_enabled=self.hybrid_retrieval_enabled,
            )

        # 第四步：基于精排结果构建答案草稿和原始证据列表
        return KBQueryResult(
            answer=self._build_grounded_answer(question, reranked),
            raw_evidence=[self._to_raw_evidence(chunk) for chunk in reranked],
            retrieved_chunks=reranked,
            candidate_count=len(candidates),
            retrieval_sources=_retrieval_sources(candidates),
            hybrid_retrieval_enabled=self.hybrid_retrieval_enabled,
        )

    def _to_stored_chunk(self, chunk: KBChunk) -> StoredChunk:
        """将切块阶段的 KBChunk 转换为存储阶段的 StoredChunk。

        核心操作是为 chunk 的文本内容生成 embedding 向量，
        这是入库流程中计算量最大的步骤。

        Args:
            chunk: 切块阶段产生的 KBChunk 对象，包含文本内容但还没有向量

        Returns:
            StoredChunk: 包含 embedding 向量的存储就绪对象
        """
        return StoredChunk(
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            # 为 chunk 文本内容生成 embedding 向量，这是入库的核心计算步骤
            embedding=self.embedding_provider.embed(chunk.content),
            search_text=build_search_text(chunk.content, chunk.metadata),
            metadata=chunk.metadata,
        )

    def _to_raw_evidence(self, chunk: RetrievedChunk) -> dict[str, Any]:
        """将检索到的 chunk 转换为原始证据字典格式。

        原始证据后续会交给 Evidence Normalizer 统一处理，
        转成标准 Evidence 对象供下游业务节点消费。

        Args:
            chunk: rerank 精排后的 RetrievedChunk 对象

        Returns:
            包含 source_type、content、score、source_id 和 metadata 的证据字典
        """
        return {
            # 标记证据来源为知识库（区别于搜索、API 等其他来源）
            "source_type": EvidenceSource.KB,
            # chunk 的文本内容，作为证据的核心信息
            "content": chunk.content,
            # 相关性得分（由 reranker 或向量搜索给出）
            "score": chunk.score,
            # 使用 chunk_id 作为证据的唯一标识
            "source_id": chunk.chunk_id,
            # 合并 chunk 自身的 metadata 和额外的 document_id、store_type 信息
            "metadata": {
                **chunk.metadata,
                "document_id": chunk.document_id,
                "store_type": self.store.store_type,
                "embedding_provider": getattr(
                    self.embedding_provider,
                    "provider_type",
                    self.embedding_provider.__class__.__name__,
                ),
                "embedding_model": getattr(self.embedding_provider, "model", None),
            },
        }

    def _build_grounded_answer(self, question: str, chunks: list[RetrievedChunk]) -> str:
        """基于检索到的 chunk 构建答案草稿。

        第二阶段还不调用答案 LLM，因此回答采用"证据摘要式"模板。
        这样可以明确告诉用户答案来自哪些 chunk，避免在没有生成模型的情况下编造细节。
        后续阶段会替换为 LLM 生成的自然语言答案。

        Args:
            question: 用户提出的原始问题
            chunks: rerank 精排后的 chunk 列表（至少包含一个元素）

        Returns:
            基于模板生成的答案草稿字符串
        """
        # 取相关性最高的 chunk，提取其标题作为引用来源
        top_chunk = chunks[0]
        title = top_chunk.metadata.get("title", "知识库资料")

        # 取前两个 chunk 的内容拼接为摘要片段，用分号分隔
        snippets = "；".join(chunk.content for chunk in chunks[:2])

        # 答案草稿不复述原问题，避免等价问法命中缓存时带出上一种问法。
        return f"根据知识库中《{title}》等资料，相关信息是：{snippets}"


# ── 全局单例管理 ──────────────────────────────────────────────────────────────
# 使用模块级变量 + 线程锁实现线程安全的惰性单例模式。
# 避免应用导入时就连接 PostgreSQL，只有真正调用 KB 链路时才初始化服务。

_service: KnowledgeBaseService | None = None
_service_lock = Lock()


def get_kb_service() -> KnowledgeBaseService:
    """获取知识库服务的全局单例。

    使用双重检查锁定（Double-Checked Locking）实现线程安全的惰性初始化：
        - 第一次检查不加锁，避免每次调用都获取锁的开销
        - 加锁后再次检查，防止多线程同时通过第一次检查后重复创建实例

    Returns:
        KnowledgeBaseService: 已初始化的知识库服务实例
    """
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = _build_service()
    return _service


def reset_kb_service_for_tests(service: KnowledgeBaseService | None = None) -> None:
    """重置全局单例，仅供测试使用。

    在单元测试中，不同测试用例需要隔离全局单例状态，
    可以通过此函数重置为 None 或替换为 mock 对象。
    生产代码不要调用这个函数。

    Args:
        service: 要替换的服务实例，传 None 则清空单例
    """
    global _service
    with _service_lock:
        _service = service


def _build_service() -> KnowledgeBaseService:
    """根据配置构建 KnowledgeBaseService 实例。

    构建逻辑：
        1. 根据配置创建 embedding 提供者（hash 或 OpenAI-compatible）
        2. 根据是否配置了 postgres_dsn 决定使用 PostgreSQL 还是内存存储
        3. 创建 reranker、组装服务实例
        4. 如果使用内存存储，则加载种子文档用于本地开发和测试

    Returns:
        配置完备的 KnowledgeBaseService 实例
    """
    if settings.strict_external_stores and not settings.postgres_dsn:
        raise RuntimeError("生产环境必须配置 GUSTOBOT_POSTGRES_DSN，不能退回内存知识库。")

    # 创建 embedding 提供者，维度由配置决定；真实服务和 pgvector 表维度必须保持一致。
    embedding_provider = _build_embedding_provider()

    # 根据是否配置了 PostgreSQL DSN 选择存储后端
    if settings.postgres_dsn:
        # 生产环境：使用 PostgreSQL + pgvector 扩展，支持持久化和大规模向量检索
        store: KnowledgeStore = PostgreSQLPgVectorStore(
            settings.postgres_dsn,
            embedding_dimension=embedding_provider.dimension,
            table_name=settings.kb_pgvector_table,
        )
    else:
        if settings.strict_external_stores:
            raise RuntimeError("生产环境必须配置 GUSTOBOT_POSTGRES_DSN，不能退回内存知识库。")
        # 开发/测试环境：使用内存存储，无需外部依赖，但数据不持久
        store = InMemoryKnowledgeStore()

    # 组装知识库服务实例，所有参数均来自应用配置
    service = KnowledgeBaseService(
        store=store,
        embedding_provider=embedding_provider,
        reranker=_build_reranker(),
        chunk_size=settings.kb_chunk_size,
        chunk_overlap=settings.kb_chunk_overlap,
        retrieve_top_k=settings.kb_retrieve_top_k,
        rerank_top_k=settings.kb_rerank_top_k,
        hybrid_retrieval_enabled=getattr(settings, "kb_hybrid_retrieval_enabled", True),
        lexical_top_k=getattr(settings, "kb_lexical_top_k", 8),
        rrf_k=getattr(settings, "kb_rrf_k", 60),
    )

    # 如果使用内存存储，加载种子文档，让本地开发和自动测试开箱即可看到 Evidence
    if store.store_type == "memory":
        _seed_memory_knowledge_base(service)
    return service


def _build_embedding_provider() -> EmbeddingProvider:
    """根据配置构建 embedding provider。

    默认 hash provider 用于测试和本地演示；当 GUSTOBOT_KB_EMBEDDING_PROVIDER=openai-compatible 时，
    会调用真实 /v1/embeddings 服务，适配 bge、OpenAI-compatible embedding 等服务。
    """
    provider = settings.kb_embedding_provider.strip().lower()
    if provider in {"hash", "local-hash"}:
        if settings.strict_external_stores:
            raise RuntimeError("生产环境必须配置真实 KB embedding，不能使用 HashEmbeddingProvider。")
        return HashEmbeddingProvider(dimension=settings.kb_embedding_dimension)

    if provider in {"openai", "openai-compatible", "openai_compatible"}:
        if not settings.kb_embedding_base_url:
            raise RuntimeError(
                "已选择 OpenAI-compatible embedding，但缺少 GUSTOBOT_KB_EMBEDDING_BASE_URL。"
            )
        if settings.strict_external_stores and not settings.kb_embedding_api_key:
            raise RuntimeError("生产环境使用通义千问/DashScope embedding 时必须配置 GUSTOBOT_KB_EMBEDDING_API_KEY。")
        return OpenAICompatibleEmbeddingProvider(
            base_url=settings.kb_embedding_base_url,
            api_key=settings.kb_embedding_api_key,
            model=settings.kb_embedding_model,
            dimension=settings.kb_embedding_dimension,
            timeout_seconds=settings.kb_embedding_timeout_seconds,
        )

    raise RuntimeError(
        f"未知的 KB embedding provider：{settings.kb_embedding_provider}。"
        "支持值：hash、openai-compatible。"
    )


def _build_reranker() -> Reranker:
    """根据配置构建 KB reranker。

    默认关键词 reranker 适合本地测试；配置 GUSTOBOT_KB_RERANK_BASE_URL 后切换到外部服务。
    """
    if settings.kb_rerank_base_url:
        return HTTPReranker(
            base_url=settings.kb_rerank_base_url,
            endpoint=settings.kb_rerank_endpoint,
            api_key=settings.kb_rerank_api_key,
            model=settings.kb_rerank_model,
            request_format=settings.kb_rerank_format,
            timeout_seconds=settings.kb_rerank_timeout_seconds,
            max_retries=settings.kb_rerank_max_retries,
            fallback_on_failure=not settings.strict_external_stores,
        )
    if settings.strict_external_stores:
        raise RuntimeError("生产环境必须配置 GUSTOBOT_KB_RERANK_BASE_URL，不能使用 KeywordReranker。")
    return KeywordReranker()


def _retrieval_sources(chunks: list[RetrievedChunk]) -> list[str]:
    sources: list[str] = []
    for chunk in chunks:
        for source in chunk.metadata.get("retrieval_sources", []):
            if source not in sources:
                sources.append(source)
    return sources


def _seed_memory_knowledge_base(service: KnowledgeBaseService) -> None:
    """向内存知识库中加载种子文档。

    这些种子文档只服务于本地开发和自动测试，让 KB RAG 链路开箱即可看到 Evidence。
    真正的业务资料后续会通过文件入库、管理后台或批处理脚本写入 PostgreSQL + pgvector。

    当前种子文档包含两道经典川菜（宫保鸡丁、麻婆豆腐）的知识卡片，
    覆盖历史、食材、口味等常见问答角度。

    Args:
        service: 已初始化的知识库服务实例
    """
    seed_documents = [
        {
            "title": "宫保鸡丁的历史与文化",
            "content": (
                "宫保鸡丁是一道经典川菜，常见说法认为它与清代官员丁宝桢有关。"
                "这道菜通常以鸡丁、花生、干辣椒和葱段为核心材料，口味偏咸鲜、微辣、微甜。"
                "在菜谱知识问答中，宫保鸡丁适合从历史、典故、食材和口味几个角度介绍。"
            ),
            "metadata": {"doc_type": "seed", "cuisine": "川菜"},
            "source_id": "seed:gongbao-history",
        },
        {
            "title": "麻婆豆腐知识卡片",
            "content": (
                "麻婆豆腐是川菜代表菜之一，特点是麻、辣、烫、香、酥、嫩、鲜、活。"
                "常见核心食材包括豆腐、牛肉末或猪肉末、豆瓣酱、花椒和辣椒。"
                "解释类问题可以重点说明其风味特点、地方菜系和常见做法。"
            ),
            "metadata": {"doc_type": "seed", "cuisine": "川菜"},
            "source_id": "seed:mapo-card",
        },
    ]
    # 逐篇将种子文档入库
    for document in seed_documents:
        service.ingest_document(**document)
