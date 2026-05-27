"""系统配置模块。

这个文件集中定义当前阶段需要的轻量配置，包括路由阈值、KB RAG 参数、
GraphRAG 参数、Text2SQL 安全参数、缓存、日志追踪，以及外部数据库连接信息。
"""

from dataclasses import dataclass
from os import getenv
from urllib.parse import quote_plus


def _env(*names: str, default: str | None = None) -> str | None:
    # 兼容旧项目配置：优先读取 GustoBot-v2 的新变量，缺省时回退到 GustoBot-develop 已有变量名。
    for name in names:
        value = getenv(name)
        if value not in (None, ""):
            return value
    return default


def _env_int(*names: str, default: int) -> int:
    value = _env(*names)
    return int(value) if value is not None else default


def _env_float(*names: str, default: float) -> float:
    value = _env(*names)
    return float(value) if value is not None else default


def _env_bool(*names: str, default: bool) -> bool:
    value = _env(*names)
    if value is None:
        return default
    return value.lower() == "true"


def _legacy_pgvector_configured() -> bool:
    # 旧项目 kb_ingest 使用 PGHOST/PGDATABASE/PGUSER/PGPASSWORD，也常由 KB_PG* 变量间接注入。
    return any(
        _env(name) is not None
        for name in (
            "KB_PGHOST",
            "KB_PGDATABASE",
            "KB_PGUSER",
            "PGHOST",
            "PGDATABASE",
            "PGUSER",
            "POSTGRES_HOST",
            "POSTGRES_DB",
            "POSTGRES_USER",
        )
    )


def _build_legacy_postgres_dsn() -> str | None:
    if not _legacy_pgvector_configured():
        return None
    host = _env("KB_PGHOST", "PGHOST", "POSTGRES_HOST", default="localhost")
    port = _env("KB_PGPORT", "PGPORT", "POSTGRES_PORT", default="5432")
    database = _env("KB_PGDATABASE", "PGDATABASE", "POSTGRES_DB", default="vector_db")
    user = _env("KB_PGUSER", "PGUSER", "POSTGRES_USER", default="postgres")
    password = _env("KB_PGPASSWORD", "PGPASSWORD", "POSTGRES_PASSWORD", default="")
    auth = quote_plus(user or "postgres")
    if password:
        auth = f"{auth}:{quote_plus(password)}"
    return f"postgresql://{auth}@{host}:{port}/{database}"


def _postgres_dsn() -> str | None:
    return _env("GUSTOBOT_POSTGRES_DSN", "KB_POSTGRES_DSN") or _build_legacy_postgres_dsn()


def _default_pgvector_table() -> str:
    # 旧 GustoBot-develop 的 pgvector 初始化表名是 searchable_documents。
    # 如果用户显式配置新 DSN 但没指定旧 PG 变量，则继续使用 v2 自己的 kb_chunks 表结构。
    if _legacy_pgvector_configured() and _env("GUSTOBOT_POSTGRES_DSN", "KB_POSTGRES_DSN") is None:
        return "searchable_documents"
    return "kb_chunks"


@dataclass(frozen=True)
class Settings:
    # 当前只保留主流程、检索链路、缓存和日志追踪需要的配置，避免过早引入复杂配置系统。
    app_name: str = "GustoBot-v2"
    route_confidence_threshold: float = 0.6
    environment: str = _env("GUSTOBOT_ENV", default="dev") or "dev"
    strict_external_stores: bool = _env_bool(
        "GUSTOBOT_STRICT_EXTERNAL_STORES",
        default=(_env("GUSTOBOT_ENV", default="dev") or "dev").lower() == "prod",
    )
    # Router LLM 配置：优先读取 v2 独立变量，缺省回退到通用 LLM_*，未配置时自动走规则 Router。
    router_llm_enabled: bool = _env_bool("GUSTOBOT_ROUTER_LLM_ENABLED", "ROUTER_LLM_ENABLED", default=True)
    router_llm_provider: str = _env("GUSTOBOT_ROUTER_LLM_PROVIDER", "ROUTER_LLM_PROVIDER", default="openai-compatible") or "openai-compatible"
    router_llm_base_url: str | None = _env("GUSTOBOT_ROUTER_LLM_BASE_URL", "ROUTER_LLM_BASE_URL", "LLM_BASE_URL")
    router_llm_api_key: str | None = _env("GUSTOBOT_ROUTER_LLM_API_KEY", "ROUTER_LLM_API_KEY", "LLM_API_KEY")
    router_llm_model: str | None = _env("GUSTOBOT_ROUTER_LLM_MODEL", "ROUTER_LLM_MODEL", "LLM_MODEL")
    router_llm_timeout_seconds: float = _env_float("GUSTOBOT_ROUTER_LLM_TIMEOUT_SECONDS", "ROUTER_LLM_TIMEOUT_SECONDS", "LLM_TIMEOUT_SECONDS", default=15)
    router_llm_temperature: float = _env_float("GUSTOBOT_ROUTER_LLM_TEMPERATURE", "ROUTER_LLM_TEMPERATURE", default=0.0)
    router_llm_max_retries: int = _env_int("GUSTOBOT_ROUTER_LLM_MAX_RETRIES", "ROUTER_LLM_MAX_RETRIES", default=1)
    # 选择性多路由只在明确复合意图时启用，避免普通问题无谓并行增加成本和冲突。
    multi_route_enabled: bool = _env_bool("GUSTOBOT_MULTI_ROUTE_ENABLED", default=True)
    multi_route_max_parallelism: int = _env_int("GUSTOBOT_MULTI_ROUTE_MAX_PARALLELISM", default=3)
    multi_route_subtask_timeout_seconds: float = _env_float(
        "GUSTOBOT_MULTI_ROUTE_SUBTASK_TIMEOUT_SECONDS",
        default=20,
    )
    # GraphRAG Planner LLM 只负责选择白名单模板和填充已链接实体参数，不参与实体链接或 Cypher 生成。
    graph_planner_llm_enabled: bool = _env_bool("GUSTOBOT_GRAPH_PLANNER_LLM_ENABLED", default=False)
    graph_planner_llm_base_url: str | None = _env("GUSTOBOT_GRAPH_PLANNER_LLM_BASE_URL", "GUSTOBOT_ROUTER_LLM_BASE_URL", "LLM_BASE_URL")
    graph_planner_llm_api_key: str | None = _env("GUSTOBOT_GRAPH_PLANNER_LLM_API_KEY", "GUSTOBOT_ROUTER_LLM_API_KEY", "LLM_API_KEY")
    graph_planner_llm_model: str | None = _env("GUSTOBOT_GRAPH_PLANNER_LLM_MODEL", "GUSTOBOT_ROUTER_LLM_MODEL", "LLM_MODEL")
    graph_planner_llm_timeout_seconds: float = _env_float(
        "GUSTOBOT_GRAPH_PLANNER_LLM_TIMEOUT_SECONDS",
        "GUSTOBOT_ROUTER_LLM_TIMEOUT_SECONDS",
        "LLM_TIMEOUT_SECONDS",
        default=10,
    )
    graph_planner_llm_temperature: float = _env_float("GUSTOBOT_GRAPH_PLANNER_LLM_TEMPERATURE", default=0.0)
    graph_planner_llm_max_retries: int = _env_int("GUSTOBOT_GRAPH_PLANNER_LLM_MAX_RETRIES", default=0)
    graph_planner_llm_confidence_threshold: float = _env_float("GUSTOBOT_GRAPH_PLANNER_LLM_CONFIDENCE_THRESHOLD", default=0.65)
    # KB RAG 配置：未配置 DSN 时使用内存存储，便于本地无数据库环境先跑通主链路。
    postgres_dsn: str | None = _postgres_dsn()
    taxonomy_postgres_dsn: str | None = _env("GUSTOBOT_TAXONOMY_POSTGRES_DSN")
    kb_pgvector_table: str = _env("GUSTOBOT_KB_PGVECTOR_TABLE", "KB_PGVECTOR_TABLE", default=_default_pgvector_table()) or "kb_chunks"
    kb_embedding_provider: str = _env("GUSTOBOT_KB_EMBEDDING_PROVIDER", "KB_EMBEDDING_PROVIDER", "EMBEDDING_PROVIDER", default="hash") or "hash"
    kb_embedding_base_url: str | None = _env("GUSTOBOT_KB_EMBEDDING_BASE_URL", "KB_EMBEDDING_BASE_URL", "EMBEDDING_BASE_URL")
    kb_embedding_api_key: str | None = _env("GUSTOBOT_KB_EMBEDDING_API_KEY", "KB_EMBEDDING_API_KEY", "EMBEDDING_API_KEY", "LLM_API_KEY")
    kb_embedding_model: str = _env("GUSTOBOT_KB_EMBEDDING_MODEL", "KB_EMBEDDING_MODEL", "EMBEDDING_MODEL", default="hash-embedding") or "hash-embedding"
    kb_embedding_dimension: int = _env_int("GUSTOBOT_KB_EMBEDDING_DIMENSION", "KB_EMBEDDING_DIMENSION", "EMBEDDING_DIMENSION", default=64)
    kb_embedding_timeout_seconds: float = _env_float("GUSTOBOT_KB_EMBEDDING_TIMEOUT_SECONDS", "KB_EMBEDDING_TIMEOUT_SECONDS", default=30)
    kb_chunk_size: int = _env_int("GUSTOBOT_KB_CHUNK_SIZE", "KB_CHUNK_SIZE", default=500)
    kb_chunk_overlap: int = _env_int("GUSTOBOT_KB_CHUNK_OVERLAP", "KB_CHUNK_OVERLAP", default=80)
    kb_retrieve_top_k: int = _env_int("GUSTOBOT_KB_RETRIEVE_TOP_K", "KB_TOP_K", default=8)
    kb_hybrid_retrieval_enabled: bool = _env_bool("GUSTOBOT_KB_HYBRID_RETRIEVAL_ENABLED", default=True)
    kb_lexical_top_k: int = _env_int("GUSTOBOT_KB_LEXICAL_TOP_K", default=8)
    kb_rrf_k: int = _env_int("GUSTOBOT_KB_RRF_K", default=60)
    kb_rerank_top_k: int = _env_int("GUSTOBOT_KB_RERANK_TOP_K", "KB_RERANK_TOP_N", "RERANK_TOP_N", default=3)
    kb_rerank_base_url: str | None = _env("GUSTOBOT_KB_RERANK_BASE_URL", "KB_RERANK_BASE_URL", "RERANK_BASE_URL")
    kb_rerank_endpoint: str = _env("GUSTOBOT_KB_RERANK_ENDPOINT", "KB_RERANK_ENDPOINT", "RERANK_ENDPOINT", default="/rerank") or "/rerank"
    kb_rerank_api_key: str | None = _env("GUSTOBOT_KB_RERANK_API_KEY", "KB_RERANK_API_KEY", "RERANK_API_KEY", "LLM_API_KEY")
    kb_rerank_model: str | None = _env("GUSTOBOT_KB_RERANK_MODEL", "KB_RERANK_MODEL", "RERANK_MODEL")
    kb_rerank_format: str = _env("GUSTOBOT_KB_RERANK_FORMAT", "KB_RERANK_FORMAT", "RERANK_FORMAT", default="auto") or "auto"
    kb_rerank_timeout_seconds: float = _env_float("GUSTOBOT_KB_RERANK_TIMEOUT_SECONDS", "KB_RERANK_TIMEOUT", default=30)
    kb_rerank_max_retries: int = _env_int("GUSTOBOT_KB_RERANK_MAX_RETRIES", "KB_RERANK_MAX_RETRIES", "RERANK_MAX_RETRIES", default=1)
    # 第三阶段 GraphRAG 配置：未配置 Neo4j 时使用内存种子图谱，便于本地测试关系问答。
    neo4j_uri: str | None = _env("GUSTOBOT_NEO4J_URI", "NEO4J_URI")
    neo4j_username: str = _env("GUSTOBOT_NEO4J_USERNAME", "NEO4J_USERNAME", "NEO4J_USER", default="neo4j") or "neo4j"
    neo4j_password: str = _env("GUSTOBOT_NEO4J_PASSWORD", "NEO4J_PASSWORD", default="password") or "password"
    neo4j_database: str | None = _env("GUSTOBOT_NEO4J_DATABASE", "NEO4J_DATABASE")
    graphrag_max_depth: int = _env_int("GUSTOBOT_GRAPHRAG_MAX_DEPTH", default=2)
    # 第三阶段 Text2SQL 配置：Schema Top-K 限制提示规模，max_rows 防止结构化查询返回过大结果集。
    text2sql_postgres_dsn: str | None = _env("GUSTOBOT_TEXT2SQL_POSTGRES_DSN", "TEXT2SQL_POSTGRES_DSN")
    text2sql_schema_table: str = _env("GUSTOBOT_TEXT2SQL_SCHEMA_TABLE", default="schema_catalog") or "schema_catalog"
    text2sql_schema_top_k: int = _env_int("GUSTOBOT_TEXT2SQL_SCHEMA_TOP_K", default=5)
    text2sql_max_rows: int = _env_int("GUSTOBOT_TEXT2SQL_MAX_ROWS", default=50)
    text2sql_llm_base_url: str | None = _env("GUSTOBOT_TEXT2SQL_LLM_BASE_URL", "TEXT2SQL_LLM_BASE_URL", "LLM_BASE_URL")
    text2sql_llm_api_key: str | None = _env("GUSTOBOT_TEXT2SQL_LLM_API_KEY", "TEXT2SQL_LLM_API_KEY", "LLM_API_KEY")
    text2sql_llm_model: str | None = _env("GUSTOBOT_TEXT2SQL_LLM_MODEL", "TEXT2SQL_LLM_MODEL", "LLM_MODEL")
    text2sql_llm_timeout_seconds: float = _env_float(
        "GUSTOBOT_TEXT2SQL_LLM_TIMEOUT_SECONDS",
        "TEXT2SQL_LLM_TIMEOUT_SECONDS",
        "LLM_TIMEOUT_SECONDS",
        default=20,
    )
    text2sql_llm_temperature: float = _env_float("GUSTOBOT_TEXT2SQL_LLM_TEMPERATURE", "TEXT2SQL_LLM_TEMPERATURE", default=0.0)
    text2sql_llm_max_validation_retries: int = _env_int(
        "GUSTOBOT_TEXT2SQL_LLM_MAX_VALIDATION_RETRIES",
        "TEXT2SQL_LLM_MAX_VALIDATION_RETRIES",
        default=1,
    )
    # 统一答案生成层：未配置时使用确定性 fallback，配置后调用 OpenAI-compatible chat/completions。
    answer_llm_base_url: str | None = _env("GUSTOBOT_ANSWER_LLM_BASE_URL", "ANSWER_LLM_BASE_URL", "LLM_BASE_URL")
    answer_llm_api_key: str | None = _env("GUSTOBOT_ANSWER_LLM_API_KEY", "ANSWER_LLM_API_KEY", "LLM_API_KEY")
    answer_llm_model: str | None = _env("GUSTOBOT_ANSWER_LLM_MODEL", "ANSWER_LLM_MODEL", "LLM_MODEL")
    answer_llm_timeout_seconds: float = _env_float("GUSTOBOT_ANSWER_LLM_TIMEOUT_SECONDS", "ANSWER_LLM_TIMEOUT_SECONDS", "LLM_TIMEOUT_SECONDS", default=30)
    answer_llm_max_retries: int = _env_int("GUSTOBOT_ANSWER_LLM_MAX_RETRIES", "ANSWER_LLM_MAX_RETRIES", default=1)
    answer_llm_allow_general_recipe_fallback: bool = _env_bool("GUSTOBOT_ALLOW_GENERAL_RECIPE_FALLBACK", default=True)
    # 多模态理解层只把图片转为结构化文本，后续仍回流 Router 复用业务链路。
    vision_base_url: str | None = _env("GUSTOBOT_VISION_BASE_URL", "VISION_BASE_URL", "GUSTOBOT_ANSWER_LLM_BASE_URL", "ANSWER_LLM_BASE_URL", "LLM_BASE_URL")
    vision_api_key: str | None = _env("GUSTOBOT_VISION_API_KEY", "VISION_API_KEY", "GUSTOBOT_ANSWER_LLM_API_KEY", "ANSWER_LLM_API_KEY", "LLM_API_KEY")
    vision_model: str | None = _env("GUSTOBOT_VISION_MODEL", "VISION_MODEL", "GUSTOBOT_ANSWER_LLM_MODEL", "ANSWER_LLM_MODEL", "LLM_MODEL")
    vision_timeout_seconds: float = _env_float("GUSTOBOT_VISION_TIMEOUT_SECONDS", "VISION_TIMEOUT_SECONDS", "LLM_TIMEOUT_SECONDS", default=30)
    vision_max_retries: int = _env_int("GUSTOBOT_VISION_MAX_RETRIES", "VISION_MAX_RETRIES", default=1)
    ocr_base_url: str | None = _env("GUSTOBOT_OCR_BASE_URL", "OCR_BASE_URL")
    ocr_api_key: str | None = _env("GUSTOBOT_OCR_API_KEY", "OCR_API_KEY")
    ocr_timeout_seconds: float = _env_float("GUSTOBOT_OCR_TIMEOUT_SECONDS", "OCR_TIMEOUT_SECONDS", default=20)
    ocr_max_retries: int = _env_int("GUSTOBOT_OCR_MAX_RETRIES", "OCR_MAX_RETRIES", default=1)
    # 上传目录只保存已登记文件，业务链路通过 upload://file_id 读取，避免暴露任意本地路径。
    upload_dir: str = _env("GUSTOBOT_UPLOAD_DIR", default="data/uploads") or "data/uploads"
    upload_max_mb: int = _env_int("GUSTOBOT_UPLOAD_MAX_MB", default=20)
    allowed_file_extensions: str = _env(
        "GUSTOBOT_ALLOWED_FILE_EXTENSIONS",
        default=".txt,.md,.json,.csv,.xlsx,.pdf,.docx",
    ) or ".txt,.md,.json,.csv,.xlsx,.pdf,.docx"
    allowed_image_extensions: str = _env(
        "GUSTOBOT_ALLOWED_IMAGE_EXTENSIONS",
        default=".jpg,.jpeg,.png,.webp,.bmp",
    ) or ".jpg,.jpeg,.png,.webp,.bmp"
    # 第四阶段缓存配置：未配置 Redis 时使用进程内缓存，保证本地开发无需额外服务。
    redis_url: str | None = _env("GUSTOBOT_REDIS_URL", "REDIS_URL")
    cache_enabled: bool = _env_bool("GUSTOBOT_CACHE_ENABLED", default=True)
    cache_ttl_seconds: int = _env_int("GUSTOBOT_CACHE_TTL_SECONDS", "REDIS_CACHE_EXPIRE", default=300)
    text2sql_cache_version: str = _env("GUSTOBOT_TEXT2SQL_CACHE_VERSION", default="v1") or "v1"
    kb_corpus_version: str = _env("GUSTOBOT_KB_CORPUS_VERSION", default="v1") or "v1"
    # 第四阶段日志追踪配置：默认写入 logs/traces.jsonl，便于按 trace_id 排查一次请求的完整路径。
    trace_enabled: bool = _env_bool("GUSTOBOT_TRACE_ENABLED", default=True)
    trace_log_path: str = _env("GUSTOBOT_TRACE_LOG_PATH", default="logs/traces.jsonl") or "logs/traces.jsonl"


# settings 作为轻量级单例被 Router 和工作流节点复用。
settings = Settings()
