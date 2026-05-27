"""P0 strict readiness 检查。

这个模块只做脱敏配置和运行时 provider/store 类型检查，不打印 API Key 或数据库 DSN。
脚本、FastAPI 启动门禁和后续失败注入测试都复用同一套规则。
"""

from __future__ import annotations

from typing import Any, Callable


RuntimeSnapshotFunc = Callable[[], dict[str, Any]]


class P0ReadinessError(RuntimeError):
    """P0 strict readiness 未通过。"""


def should_run_startup_check(settings: Any) -> bool:
    """只在生产或 strict 外部依赖模式下执行启动门禁，避免影响本地开发。"""

    return bool(getattr(settings, "strict_external_stores", False)) or (
        str(getattr(settings, "environment", "")).lower() == "prod"
    )


def check_readiness(
    settings: Any,
    *,
    config_only: bool = False,
    runtime_snapshot_func: RuntimeSnapshotFunc | None = None,
) -> dict[str, Any]:
    issues = config_issues(settings)
    if issues:
        return {"status": "failed", "stage": "config", "issues": issues, "snapshot": config_snapshot(settings)}

    if config_only:
        return {"status": "ok", "stage": "config", "issues": [], "snapshot": config_snapshot(settings)}

    snapshot_func = runtime_snapshot_func or runtime_snapshot
    try:
        snapshot = snapshot_func()
    except Exception as exc:
        return {"status": "failed", "stage": "runtime", "error": str(exc)[:500]}

    issues = runtime_issues(snapshot)
    return {"status": "failed" if issues else "ok", "issues": issues, "snapshot": snapshot}


def assert_p0_startup_ready(
    settings: Any,
    *,
    runtime_snapshot_func: RuntimeSnapshotFunc | None = None,
) -> dict[str, Any]:
    if not should_run_startup_check(settings):
        return {
            "status": "skipped",
            "reason": "not_prod_or_strict",
            "snapshot": config_snapshot(settings),
        }

    result = check_readiness(settings, runtime_snapshot_func=runtime_snapshot_func)
    if result.get("status") != "ok":
        stage = result.get("stage", "runtime")
        details = result.get("issues") or result.get("error") or "unknown"
        raise P0ReadinessError(f"P0 strict readiness failed at {stage}: {details}")
    return result


def config_issues(settings: Any) -> list[str]:
    issues: list[str] = []
    if not settings.strict_external_stores or settings.environment.lower() != "prod":
        issues.append("P0 实验必须设置 GUSTOBOT_ENV=prod 且 GUSTOBOT_STRICT_EXTERNAL_STORES=true。")
    for name, value in {
        "GUSTOBOT_POSTGRES_DSN": settings.postgres_dsn,
        "GUSTOBOT_TEXT2SQL_POSTGRES_DSN 或 GUSTOBOT_POSTGRES_DSN": settings.text2sql_postgres_dsn
        or settings.postgres_dsn,
        "GUSTOBOT_NEO4J_URI": settings.neo4j_uri,
        "GUSTOBOT_REDIS_URL": settings.redis_url,
        "GUSTOBOT_KB_EMBEDDING_BASE_URL": settings.kb_embedding_base_url,
        "GUSTOBOT_KB_EMBEDDING_API_KEY": settings.kb_embedding_api_key,
        "GUSTOBOT_KB_RERANK_BASE_URL": settings.kb_rerank_base_url,
        "GUSTOBOT_KB_RERANK_API_KEY": settings.kb_rerank_api_key,
        "GUSTOBOT_TEXT2SQL_LLM_BASE_URL": settings.text2sql_llm_base_url,
        "GUSTOBOT_TEXT2SQL_LLM_API_KEY": settings.text2sql_llm_api_key,
        "GUSTOBOT_TEXT2SQL_LLM_MODEL": settings.text2sql_llm_model,
        "GUSTOBOT_ROUTER_LLM_BASE_URL": settings.router_llm_base_url,
        "GUSTOBOT_ROUTER_LLM_API_KEY": settings.router_llm_api_key,
        "GUSTOBOT_ROUTER_LLM_MODEL": settings.router_llm_model,
        "GUSTOBOT_ANSWER_LLM_BASE_URL": settings.answer_llm_base_url,
        "GUSTOBOT_ANSWER_LLM_API_KEY": settings.answer_llm_api_key,
        "GUSTOBOT_ANSWER_LLM_MODEL": settings.answer_llm_model,
        "GUSTOBOT_VISION_BASE_URL": settings.vision_base_url,
        "GUSTOBOT_VISION_API_KEY": settings.vision_api_key,
        "GUSTOBOT_VISION_MODEL": settings.vision_model,
    }.items():
        if not value:
            issues.append(f"缺少 {name}。")
    if settings.kb_embedding_provider.strip().lower() not in {"openai", "openai-compatible", "openai_compatible"}:
        issues.append("GUSTOBOT_KB_EMBEDDING_PROVIDER 必须是真实 OpenAI-compatible provider。")
    if settings.kb_embedding_model != "text-embedding-v4":
        issues.append("P0 DashScope 实验要求 GUSTOBOT_KB_EMBEDDING_MODEL=text-embedding-v4。")
    if settings.kb_embedding_dimension != 1024:
        issues.append("text-embedding-v4 的 GUSTOBOT_KB_EMBEDDING_DIMENSION 必须是 1024。")
    if settings.kb_rerank_model != "qwen3-rerank":
        issues.append("P0 DashScope 实验要求 GUSTOBOT_KB_RERANK_MODEL=qwen3-rerank。")
    if is_image_generation_model(settings.vision_model):
        issues.append(
            "GUSTOBOT_VISION_MODEL 当前是图片生成/编辑模型，不能用于 P0 图片理解；"
            "请使用 qwen3-vl-plus、qwen-vl-plus 或 qwen-vl-max 等支持 image_url 的视觉理解模型。"
        )
    return issues


def config_snapshot(settings: Any) -> dict[str, Any]:
    return {
        "environment": settings.environment,
        "strict_external_stores": settings.strict_external_stores,
        "postgres_dsn_configured": bool(settings.postgres_dsn),
        "neo4j_uri_configured": bool(settings.neo4j_uri),
        "redis_url_configured": bool(settings.redis_url),
        "kb_embedding_provider": settings.kb_embedding_provider,
        "kb_embedding_model": settings.kb_embedding_model,
        "kb_embedding_dimension": settings.kb_embedding_dimension,
        "kb_embedding_api_key_configured": bool(settings.kb_embedding_api_key),
        "kb_rerank_model": settings.kb_rerank_model,
        "kb_rerank_api_key_configured": bool(settings.kb_rerank_api_key),
        "text2sql_llm_model": settings.text2sql_llm_model,
        "text2sql_llm_api_key_configured": bool(settings.text2sql_llm_api_key),
        "router_llm_model": settings.router_llm_model,
        "router_llm_api_key_configured": bool(settings.router_llm_api_key),
        "answer_llm_model": settings.answer_llm_model,
        "answer_llm_api_key_configured": bool(settings.answer_llm_api_key),
        "vision_model": settings.vision_model,
        "vision_api_key_configured": bool(settings.vision_api_key),
    }


def runtime_snapshot() -> dict[str, Any]:
    from app.cache.store import get_cache_store
    from app.graphrag.service import get_graphrag_service
    from app.kb.service import get_kb_service
    from app.sessions.service import get_session_service
    from app.text2sql.service import get_text2sql_service
    from app.uploads.service import get_upload_service

    kb_status = get_kb_service().status()
    graph_service = get_graphrag_service()
    text2sql_service = get_text2sql_service()
    cache_store = get_cache_store()
    session_service = get_session_service()
    upload_service = get_upload_service()
    schema_embedding_provider = text2sql_service.schema_catalog.embedding_provider
    return {
        "kb": {
            "store_type": kb_status.store_type,
            "embedding_provider": kb_status.embedding_provider,
            "embedding_model": kb_status.embedding_model,
            "embedding_dimension": kb_status.embedding_dimension,
            "reranker_type": kb_status.reranker_type,
            "reranker_status": kb_status.reranker_status,
        },
        "graphrag": {"store_type": graph_service.store.store_type},
        "text2sql": {
            "executor_type": text2sql_service.executor.executor_type,
            "generator_type": text2sql_service.sql_generator.__class__.__name__,
            "schema_embedding_provider": getattr(
                schema_embedding_provider,
                "provider_type",
                schema_embedding_provider.__class__.__name__,
            ),
            "schema_embedding_model": getattr(schema_embedding_provider, "model", None),
        },
        "cache": {"store_type": cache_store.store_type},
        "sessions": {"store_type": session_service.store.__class__.__name__},
        "uploads": {"store_type": upload_service.store.__class__.__name__},
    }


def runtime_issues(snapshot: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    expected = {
        ("kb", "store_type"): "postgres_pgvector",
        ("kb", "embedding_provider"): "openai-compatible",
        ("kb", "reranker_type"): "http",
        ("graphrag", "store_type"): "neo4j",
        ("text2sql", "executor_type"): "postgres_readonly",
        ("text2sql", "generator_type"): "LLMSQLGenerator",
        ("text2sql", "schema_embedding_provider"): "openai-compatible",
        ("cache", "store_type"): "redis",
    }
    for path, expected_value in expected.items():
        value: Any = snapshot
        for key in path:
            value = value[key]
        if value != expected_value:
            issues.append(f"{'.'.join(path)}={value!r}，期望 {expected_value!r}。")
    if snapshot["sessions"]["store_type"].startswith("InMemory"):
        issues.append("sessions 命中了内存会话存储。")
    if snapshot["uploads"]["store_type"].startswith("InMemory"):
        issues.append("uploads 命中了内存上传记录。")
    return issues


def is_image_generation_model(model: str | None) -> bool:
    # qwen-image 系列是图片生成/编辑模型，不返回图片理解 JSON 文本。
    return bool(model and model.strip().lower().startswith("qwen-image"))
