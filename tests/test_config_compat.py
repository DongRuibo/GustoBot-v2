"""旧项目配置兼容测试模块。

这个文件验证 GustoBot-v2 能识别 GustoBot-develop 已经使用的环境变量命名，
例如 PGHOST/PGDATABASE/EMBEDDING_*，避免后续接入旧数据库和旧数据时要求用户重写配置。
"""

import importlib

import app.core.config as config_module


def test_legacy_pg_env_builds_postgres_dsn(monkeypatch) -> None:
    # 旧 kb_ingest 服务使用 PGHOST/PGDATABASE/PGUSER/PGPASSWORD 连接 pgvector。
    monkeypatch.setenv("PGHOST", "kb_postgres")
    monkeypatch.setenv("PGPORT", "5432")
    monkeypatch.setenv("PGDATABASE", "vector_db")
    monkeypatch.setenv("PGUSER", "postgres")
    monkeypatch.setenv("PGPASSWORD", "secret")

    reloaded = importlib.reload(config_module)

    assert reloaded.settings.postgres_dsn == "postgresql://postgres:secret@kb_postgres:5432/vector_db"
    assert reloaded.settings.kb_pgvector_table == "searchable_documents"
    for name in ("PGHOST", "PGPORT", "PGDATABASE", "PGUSER", "PGPASSWORD"):
        monkeypatch.delenv(name, raising=False)
    importlib.reload(config_module)


def test_strict_external_stores_defaults_to_prod_only(monkeypatch) -> None:
    monkeypatch.delenv("GUSTOBOT_STRICT_EXTERNAL_STORES", raising=False)
    monkeypatch.setenv("GUSTOBOT_ENV", "prod")
    reloaded = importlib.reload(config_module)
    assert reloaded.settings.strict_external_stores is True

    monkeypatch.setenv("GUSTOBOT_ENV", "dev")
    reloaded = importlib.reload(config_module)
    assert reloaded.settings.strict_external_stores is False

    monkeypatch.setenv("GUSTOBOT_STRICT_EXTERNAL_STORES", "true")
    reloaded = importlib.reload(config_module)
    assert reloaded.settings.strict_external_stores is True

    monkeypatch.delenv("GUSTOBOT_ENV", raising=False)
    monkeypatch.delenv("GUSTOBOT_STRICT_EXTERNAL_STORES", raising=False)
    importlib.reload(config_module)


def test_legacy_embedding_env_is_supported(monkeypatch) -> None:
    # 旧主项目使用 EMBEDDING_*，v2 应自动映射到 KB embedding 配置。
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_MODEL", "bge-m3")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://embedding.local/v1")
    monkeypatch.setenv("EMBEDDING_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "1024")

    reloaded = importlib.reload(config_module)

    assert reloaded.settings.kb_embedding_provider == "openai"
    assert reloaded.settings.kb_embedding_model == "bge-m3"
    assert reloaded.settings.kb_embedding_base_url == "http://embedding.local/v1"
    assert reloaded.settings.kb_embedding_dimension == 1024
    for name in (
        "EMBEDDING_PROVIDER",
        "EMBEDDING_MODEL",
        "EMBEDDING_BASE_URL",
        "EMBEDDING_API_KEY",
        "EMBEDDING_DIMENSION",
    ):
        monkeypatch.delenv(name, raising=False)
    importlib.reload(config_module)


def test_legacy_llm_and_rerank_env_are_supported(monkeypatch) -> None:
    # 真实模型接入时，v2 应能复用旧项目已有的 LLM_* 和 RERANK_* 变量。
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.local/v1")
    monkeypatch.setenv("LLM_API_KEY", "llm-key")
    monkeypatch.setenv("LLM_MODEL", "qwen3-max")
    monkeypatch.setenv("RERANK_BASE_URL", "http://rerank.local")
    monkeypatch.setenv("RERANK_ENDPOINT", "/reranks")
    monkeypatch.setenv("RERANK_API_KEY", "rerank-key")
    monkeypatch.setenv("RERANK_MODEL", "qwen3-rerank")

    reloaded = importlib.reload(config_module)

    assert reloaded.settings.answer_llm_base_url == "http://llm.local/v1"
    assert reloaded.settings.answer_llm_api_key == "llm-key"
    assert reloaded.settings.answer_llm_model == "qwen3-max"
    assert reloaded.settings.text2sql_llm_base_url == "http://llm.local/v1"
    assert reloaded.settings.text2sql_llm_model == "qwen3-max"
    assert reloaded.settings.router_llm_base_url == "http://llm.local/v1"
    assert reloaded.settings.router_llm_api_key == "llm-key"
    assert reloaded.settings.router_llm_model == "qwen3-max"
    assert reloaded.settings.graph_planner_llm_base_url == "http://llm.local/v1"
    assert reloaded.settings.graph_planner_llm_model == "qwen3-max"
    assert reloaded.settings.vision_base_url == "http://llm.local/v1"
    assert reloaded.settings.vision_model == "qwen3-max"
    assert reloaded.settings.kb_rerank_base_url == "http://rerank.local"
    assert reloaded.settings.kb_rerank_endpoint == "/reranks"
    assert reloaded.settings.kb_rerank_model == "qwen3-rerank"
    for name in (
        "LLM_BASE_URL",
        "LLM_API_KEY",
        "LLM_MODEL",
        "RERANK_BASE_URL",
        "RERANK_ENDPOINT",
        "RERANK_API_KEY",
        "RERANK_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    importlib.reload(config_module)


def test_router_llm_provider_defaults_and_dashscope_env(monkeypatch) -> None:
    monkeypatch.delenv("GUSTOBOT_ROUTER_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("ROUTER_LLM_PROVIDER", raising=False)

    reloaded = importlib.reload(config_module)
    assert reloaded.settings.router_llm_provider == "openai-compatible"

    monkeypatch.setenv("GUSTOBOT_ROUTER_LLM_PROVIDER", "dashscope")
    monkeypatch.setenv("GUSTOBOT_ROUTER_LLM_BASE_URL", "https://dashscope.aliyuncs.com/api/v1")
    monkeypatch.setenv("GUSTOBOT_ROUTER_LLM_MODEL", "router-ft")
    reloaded = importlib.reload(config_module)

    assert reloaded.settings.router_llm_provider == "dashscope"
    assert reloaded.settings.router_llm_base_url == "https://dashscope.aliyuncs.com/api/v1"
    assert reloaded.settings.router_llm_model == "router-ft"
    for name in (
        "GUSTOBOT_ROUTER_LLM_PROVIDER",
        "GUSTOBOT_ROUTER_LLM_BASE_URL",
        "GUSTOBOT_ROUTER_LLM_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    importlib.reload(config_module)


def test_embedding_and_rerank_can_reuse_legacy_llm_api_key(monkeypatch) -> None:
    # DashScope 通常同一个 API Key 可同时调用 chat、embedding 和 rerank。
    for name in (
        "GUSTOBOT_KB_EMBEDDING_API_KEY",
        "KB_EMBEDDING_API_KEY",
        "EMBEDDING_API_KEY",
        "GUSTOBOT_KB_RERANK_API_KEY",
        "KB_RERANK_API_KEY",
        "RERANK_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_API_KEY", "shared-dashscope-key")

    reloaded = importlib.reload(config_module)

    assert reloaded.settings.kb_embedding_api_key == "shared-dashscope-key"
    assert reloaded.settings.kb_rerank_api_key == "shared-dashscope-key"
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    importlib.reload(config_module)
