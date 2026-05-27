"""KB RAG 单元测试模块。

这个文件验证第二阶段知识库能力本身是否可用，包括文档切块、内存入库、embedding 检索、
reranker 精排和 Evidence 输出，不依赖真实 PostgreSQL 服务。
"""

from app.kb.chunking import split_text_to_chunks
import httpx

from app.kb.embeddings import HashEmbeddingProvider, OpenAICompatibleEmbeddingProvider
from app.kb.hybrid import build_search_text, reciprocal_rank_fusion
from app.kb.reranker import HTTPReranker, KeywordReranker
from app.kb.service import KnowledgeBaseService
from app.kb.store import InMemoryKnowledgeStore


def build_test_kb_service() -> KnowledgeBaseService:
    # 测试里显式构造内存版 KB service，避免依赖全局单例和外部 PostgreSQL。
    return KnowledgeBaseService(
        store=InMemoryKnowledgeStore(),
        embedding_provider=HashEmbeddingProvider(dimension=64),
        reranker=KeywordReranker(),
        chunk_size=40,
        chunk_overlap=8,
        retrieve_top_k=5,
        rerank_top_k=2,
    )


def build_vector_only_kb_service() -> KnowledgeBaseService:
    return KnowledgeBaseService(
        store=InMemoryKnowledgeStore(),
        embedding_provider=HashEmbeddingProvider(dimension=64),
        reranker=KeywordReranker(),
        chunk_size=40,
        chunk_overlap=8,
        retrieve_top_k=5,
        rerank_top_k=2,
        hybrid_retrieval_enabled=False,
    )


def test_split_text_to_chunks() -> None:
    # 切块需要保留 document_id 和 chunk metadata，后续入库、引用和追踪都依赖这些字段。
    chunks = split_text_to_chunks(
        "宫保鸡丁是一道经典川菜。它常用鸡丁、花生和干辣椒制作。口味咸鲜微辣。",
        document_id="doc-1",
        chunk_size=25,
        chunk_overlap=5,
        metadata={"title": "宫保鸡丁"},
    )

    assert len(chunks) >= 2
    assert chunks[0].document_id == "doc-1"
    assert chunks[0].metadata["title"] == "宫保鸡丁"
    assert chunks[0].metadata["chunk_index"] == 0


def test_build_search_text_keeps_chinese_bigrams_and_metadata() -> None:
    search_text = build_search_text(
        "宫保鸡丁常用花生和干辣椒制作。",
        {"title": "川菜知识", "source_id": "doc-gongbao"},
    )

    assert "宫" in search_text
    assert "宫保" in search_text
    assert "鸡丁" in search_text
    assert "doc" in search_text
    assert "gongbao" in search_text


def test_kb_ingest_query_and_rerank() -> None:
    # 这个用例覆盖第二阶段主链路：文本入库 -> 切块 -> embedding -> 检索 -> rerank -> Evidence。
    service = build_test_kb_service()
    ingest_result = service.ingest_document(
        title="佛跳墙知识卡片",
        content="佛跳墙是福建名菜，也属于闽菜代表。它常用于宴席场景，文化含义强调食材丰富和汤香浓郁。",
        source_id="doc-fotiaoqiang",
        metadata={"cuisine": "闽菜"},
    )

    result = service.query("佛跳墙属于什么菜系，有什么文化含义")

    assert ingest_result.chunk_count >= 1
    assert result.raw_evidence
    assert result.raw_evidence[0]["source_type"].value == "kb"
    assert "佛跳墙" in result.answer
    assert result.retrieved_chunks[0].metadata["vector_score"] >= 0


def test_kb_hybrid_retrieval_marks_lexical_metadata() -> None:
    service = build_test_kb_service()
    service.ingest_document(
        title="冷门菜谱资料",
        content="雪绵豆沙是东北甜品，外层蓬松，常搭配豆沙馅。",
        source_id="doc-xuemian",
        metadata={"cuisine": "东北菜"},
    )

    result = service.query("雪绵豆沙是什么菜")

    assert result.raw_evidence
    metadata = result.raw_evidence[0]["metadata"]
    assert metadata["retrieval_mode"] in {"hybrid", "lexical"}
    assert "lexical" in metadata["retrieval_sources"]
    assert metadata["lexical_score"] > 0
    assert result.candidate_count >= 1


def test_kb_vector_only_mode_keeps_vector_retrieval_metadata() -> None:
    service = build_vector_only_kb_service()
    service.ingest_document(
        title="宫保鸡丁知识卡片",
        content="宫保鸡丁是一道经典川菜，常见于历史文化介绍。",
        source_id="doc-gongbao",
    )

    result = service.query("介绍宫保鸡丁历史")

    assert result.raw_evidence
    metadata = result.raw_evidence[0]["metadata"]
    assert metadata["retrieval_mode"] == "vector"
    assert metadata["retrieval_sources"] == ["vector"]
    assert "lexical_score" not in metadata


def test_rrf_fusion_deduplicates_vector_and_lexical_hits() -> None:
    vector = [service_chunk("doc-1", "chunk-1", "宫保鸡丁历史", 0.8)]
    lexical = [
        service_chunk("doc-1", "chunk-1", "宫保鸡丁历史", 2.0),
        service_chunk("doc-2", "chunk-2", "麻婆豆腐文化", 1.0),
    ]

    fused = reciprocal_rank_fusion(vector, lexical, top_k=5, rrf_k=60)

    assert [chunk.chunk_id for chunk in fused].count("chunk-1") == 1
    assert fused[0].chunk_id == "chunk-1"
    assert fused[0].metadata["retrieval_mode"] == "hybrid"
    assert fused[0].metadata["vector_rank"] == 1
    assert fused[0].metadata["lexical_rank"] == 1


def test_openai_compatible_embedding_provider() -> None:
    # 用 httpx MockTransport 模拟真实 /v1/embeddings 服务，避免单元测试依赖外部网络。
    def handler(request: httpx.Request) -> httpx.Response:
        payload = request.read().decode("utf-8")
        assert "bge-test" in payload
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json={"data": [{"embedding": [3.0, 4.0]}]},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleEmbeddingProvider(
        base_url="http://embedding.local/v1",
        api_key="test-key",
        model="bge-test",
        dimension=2,
        timeout_seconds=5,
        http_client=client,
    )

    assert provider.embed("宫保鸡丁") == [0.6, 0.8]


def test_dashscope_embedding_dimension_mismatch_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"embedding": [1.0, 2.0]}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleEmbeddingProvider(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="test-key",
        model="text-embedding-v4",
        dimension=1024,
        timeout_seconds=5,
        http_client=client,
    )

    try:
        provider.embed("宫保鸡丁")
    except RuntimeError as exc:
        assert "1024" in str(exc)
    else:
        raise AssertionError("text-embedding-v4 dimension mismatch should raise")


def test_http_reranker_parses_indexed_results() -> None:
    # 外部 rerank 服务常返回 index + relevance_score，适配器应按 index 写回候选分数。
    candidates = [
        service_chunk("doc-1", "chunk-1", "宫保鸡丁历史", 0.2),
        service_chunk("doc-2", "chunk-2", "麻婆豆腐做法", 0.2),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.3},
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    original_post = httpx.post

    def mocked_post(*args, **kwargs):
        with httpx.Client(transport=transport) as client:
            return client.post(*args, **kwargs)

    try:
        httpx.post = mocked_post
        reranker = HTTPReranker(base_url="http://rerank.local", model="bge-reranker")
        results = reranker.rerank("豆腐", candidates, top_k=2)
    finally:
        httpx.post = original_post

    assert results[0].chunk_id == "chunk-2"
    assert results[0].metadata["rerank_score"] == 0.9
    assert results[0].metadata["reranker_success"] is True
    assert reranker.status()["last_success"] is True


def test_http_reranker_supports_dashscope_qwen3_output() -> None:
    # qwen3-rerank 使用 DashScope compatible reranks 入口，响应结果位于 output.results。
    candidates = [
        service_chunk("doc-1", "chunk-1", "宫保鸡丁历史", 0.2),
        service_chunk("doc-2", "chunk-2", "麻婆豆腐做法", 0.2),
    ]
    captured = {}

    original_post = httpx.post

    def mocked_post(url, **kwargs):
        captured["url"] = url
        captured["payload"] = kwargs["json"]
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "output": {
                    "results": [
                        {"index": 1, "relevance_score": 0.91},
                        {"index": 0, "relevance_score": 0.22},
                    ]
                }
            },
        )

    try:
        httpx.post = mocked_post
        reranker = HTTPReranker(
            base_url="https://dashscope.aliyuncs.com/api/v1/services",
            endpoint="/rerank/text-rerank/text-rerank",
            model="qwen3-rerank",
        )
        results = reranker.rerank("豆腐", candidates, top_k=2)
    finally:
        httpx.post = original_post

    assert captured["url"].endswith("/compatible-api/v1/reranks")
    assert captured["payload"]["model"] == "qwen3-rerank"
    assert captured["payload"]["query"] == "豆腐"
    assert results[0].chunk_id == "chunk-2"
    assert results[0].metadata["reranker_success"] is True


def test_http_reranker_failure_marks_vector_fallback() -> None:
    # 外部 reranker 不可用时，主链路应继续返回向量召回结果，同时把降级状态写进 metadata 和 status。
    candidates = [
        service_chunk("doc-1", "chunk-1", "宫保鸡丁历史", 0.8),
        service_chunk("doc-2", "chunk-2", "麻婆豆腐做法", 0.4),
    ]

    original_post = httpx.post

    def failing_post(*args, **kwargs):
        raise httpx.ConnectError("reranker unavailable")

    try:
        httpx.post = failing_post
        reranker = HTTPReranker(base_url="http://rerank.local", model="bge-reranker")
        results = reranker.rerank("宫保鸡丁", candidates, top_k=2)
    finally:
        httpx.post = original_post

    assert results[0].chunk_id == "chunk-1"
    assert results[0].metadata["reranker_success"] is False
    assert results[0].metadata["reranker_fallback"] is True
    assert "reranker unavailable" in results[0].metadata["reranker_error"]
    assert reranker.status()["last_success"] is False
    assert reranker.status()["failed_calls"] == 1


def test_http_reranker_strict_mode_raises_on_failure() -> None:
    candidates = [service_chunk("doc-1", "chunk-1", "宫保鸡丁历史", 0.8)]
    original_post = httpx.post

    def failing_post(*args, **kwargs):
        raise httpx.ConnectError("reranker unavailable")

    try:
        httpx.post = failing_post
        reranker = HTTPReranker(
            base_url="http://rerank.local",
            model="qwen3-rerank",
            fallback_on_failure=False,
        )
        try:
            reranker.rerank("宫保鸡丁", candidates, top_k=1)
        except RuntimeError as exc:
            assert "reranker request failed" in str(exc)
        else:
            raise AssertionError("strict reranker should raise instead of vector fallback")
    finally:
        httpx.post = original_post


def test_kb_status_exposes_reranker_observability() -> None:
    service = build_test_kb_service()
    status = service.status()

    assert status.reranker_status["configured_type"] == "keyword"
    assert status.reranker_status["successful_calls"] == 0


def service_chunk(document_id: str, chunk_id: str, content: str, score: float):
    from app.kb.store import RetrievedChunk

    return RetrievedChunk(
        document_id=document_id,
        chunk_id=chunk_id,
        content=content,
        score=score,
        metadata={},
    )
