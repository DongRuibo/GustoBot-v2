"""生产环境外部存储强校验测试。"""

from types import SimpleNamespace

import pytest

from app.graphrag import service as graphrag_service
from app.kb import service as kb_service
from app.cache import store as cache_store
from app.sessions import service as session_service
from app.text2sql import service as text2sql_service
from app.text2sql.schema import SchemaCatalog
from app.uploads import service as upload_service
from app.kb.embeddings import HashEmbeddingProvider


def test_kb_strict_mode_requires_postgres(monkeypatch) -> None:
    monkeypatch.setattr(
        kb_service,
        "settings",
        SimpleNamespace(
            postgres_dsn=None,
            strict_external_stores=True,
            kb_embedding_provider="hash",
            kb_embedding_dimension=64,
        ),
    )

    with pytest.raises(RuntimeError, match="GUSTOBOT_POSTGRES_DSN"):
        kb_service._build_service()


def test_kb_dev_mode_keeps_memory_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        kb_service,
        "settings",
        SimpleNamespace(
            postgres_dsn=None,
            strict_external_stores=False,
            kb_embedding_provider="hash",
            kb_embedding_dimension=64,
            kb_chunk_size=500,
            kb_chunk_overlap=80,
            kb_retrieve_top_k=8,
            kb_rerank_top_k=3,
            kb_rerank_base_url=None,
        ),
    )

    service = kb_service._build_service()

    assert service.store.store_type == "memory"


def test_kb_strict_mode_rejects_hash_embedding(monkeypatch) -> None:
    monkeypatch.setattr(
        kb_service,
        "settings",
        SimpleNamespace(
            postgres_dsn="postgresql://configured",
            strict_external_stores=True,
            kb_embedding_provider="hash",
            kb_embedding_dimension=64,
        ),
    )

    with pytest.raises(RuntimeError, match="HashEmbeddingProvider"):
        kb_service._build_embedding_provider()


def test_kb_strict_mode_requires_http_reranker(monkeypatch) -> None:
    monkeypatch.setattr(
        kb_service,
        "settings",
        SimpleNamespace(strict_external_stores=True, kb_rerank_base_url=None),
    )

    with pytest.raises(RuntimeError, match="GUSTOBOT_KB_RERANK_BASE_URL"):
        kb_service._build_reranker()


def test_graphrag_strict_mode_requires_neo4j(monkeypatch) -> None:
    monkeypatch.setattr(
        graphrag_service,
        "settings",
        SimpleNamespace(neo4j_uri=None, strict_external_stores=True, graphrag_max_depth=2),
    )

    with pytest.raises(RuntimeError, match="GUSTOBOT_NEO4J_URI"):
        graphrag_service._build_service()


def test_graphrag_dev_mode_keeps_memory_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        graphrag_service,
        "settings",
        SimpleNamespace(neo4j_uri=None, strict_external_stores=False, graphrag_max_depth=2),
    )

    service = graphrag_service._build_service()

    assert service.store.store_type == "memory"


def test_session_strict_mode_rejects_memory_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        session_service,
        "settings",
        SimpleNamespace(postgres_dsn=None, strict_external_stores=True),
    )

    with pytest.raises(RuntimeError, match="GUSTOBOT_POSTGRES_DSN"):
        session_service._build_store()


def test_session_strict_mode_reraises_postgres_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        session_service,
        "settings",
        SimpleNamespace(postgres_dsn="postgresql://invalid", strict_external_stores=True),
    )
    monkeypatch.setattr(
        session_service,
        "PostgreSQLSessionStore",
        lambda dsn: (_ for _ in ()).throw(RuntimeError("postgres unavailable")),
    )

    with pytest.raises(RuntimeError, match="postgres unavailable"):
        session_service._build_store()


def test_upload_strict_mode_rejects_memory_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        upload_service,
        "settings",
        SimpleNamespace(postgres_dsn=None, strict_external_stores=True),
    )

    with pytest.raises(RuntimeError, match="GUSTOBOT_POSTGRES_DSN"):
        upload_service._build_store()


def test_upload_strict_mode_reraises_postgres_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        upload_service,
        "settings",
        SimpleNamespace(postgres_dsn="postgresql://invalid", strict_external_stores=True),
    )
    monkeypatch.setattr(
        upload_service,
        "PostgreSQLUploadStore",
        lambda dsn: (_ for _ in ()).throw(RuntimeError("postgres unavailable")),
    )

    with pytest.raises(RuntimeError, match="postgres unavailable"):
        upload_service._build_store()


def test_cache_strict_mode_requires_redis(monkeypatch) -> None:
    monkeypatch.setattr(
        cache_store,
        "settings",
        SimpleNamespace(redis_url=None, strict_external_stores=True),
    )

    with pytest.raises(RuntimeError, match="GUSTOBOT_REDIS_URL"):
        cache_store._build_cache_store()


def test_cache_strict_mode_reraises_redis_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        cache_store,
        "settings",
        SimpleNamespace(redis_url="redis://invalid", strict_external_stores=True),
    )
    monkeypatch.setattr(
        cache_store,
        "RedisCacheStore",
        lambda redis_url: (_ for _ in ()).throw(RuntimeError("redis unavailable")),
    )

    with pytest.raises(RuntimeError, match="redis unavailable"):
        cache_store._build_cache_store()


def test_text2sql_strict_mode_requires_postgres(monkeypatch) -> None:
    monkeypatch.setattr(
        text2sql_service,
        "settings",
        SimpleNamespace(
            strict_external_stores=True,
            text2sql_postgres_dsn=None,
            postgres_dsn=None,
        ),
    )

    with pytest.raises(RuntimeError, match="GUSTOBOT_TEXT2SQL_POSTGRES_DSN"):
        text2sql_service._build_service()


def test_text2sql_strict_mode_requires_llm(monkeypatch) -> None:
    monkeypatch.setattr(
        text2sql_service,
        "settings",
        SimpleNamespace(
            strict_external_stores=True,
            text2sql_postgres_dsn="postgresql://configured",
            postgres_dsn=None,
            text2sql_llm_base_url=None,
            text2sql_llm_api_key=None,
            text2sql_llm_model=None,
        ),
    )

    with pytest.raises(RuntimeError, match="GUSTOBOT_TEXT2SQL_LLM_BASE_URL"):
        text2sql_service._build_service()


def test_text2sql_strict_mode_rejects_hash_schema_embedding(monkeypatch) -> None:
    monkeypatch.setattr(
        text2sql_service,
        "settings",
        SimpleNamespace(
            strict_external_stores=True,
            kb_embedding_provider="hash",
            kb_embedding_base_url="http://embedding.local/v1",
            kb_embedding_api_key="test-key",
            kb_embedding_model="text-embedding-v4",
            kb_embedding_dimension=1024,
            kb_embedding_timeout_seconds=30,
        ),
    )

    with pytest.raises(RuntimeError, match="schema catalog"):
        text2sql_service._build_schema_embedding_provider()


def test_text2sql_strict_mode_uses_real_schema_embedding_provider(monkeypatch) -> None:
    monkeypatch.setattr(
        text2sql_service,
        "settings",
        SimpleNamespace(
            strict_external_stores=True,
            kb_embedding_provider="openai-compatible",
            kb_embedding_base_url="http://embedding.local/v1",
            kb_embedding_api_key="test-key",
            kb_embedding_model="text-embedding-v4",
            kb_embedding_dimension=1024,
            kb_embedding_timeout_seconds=30,
        ),
    )

    provider = text2sql_service._build_schema_embedding_provider()

    assert provider.provider_type == "openai-compatible"
    assert provider.model == "text-embedding-v4"
    assert provider.dimension == 1024


def test_text2sql_strict_mode_rejects_empty_schema_catalog(monkeypatch) -> None:
    monkeypatch.setattr(
        text2sql_service,
        "settings",
        SimpleNamespace(
            strict_external_stores=True,
            text2sql_postgres_dsn="postgresql://configured",
            postgres_dsn=None,
            text2sql_schema_table="schema_catalog",
            text2sql_llm_base_url="http://llm.local/v1",
            text2sql_llm_api_key="test-key",
            text2sql_llm_model="qwen3-max",
            kb_embedding_provider="openai-compatible",
            kb_embedding_base_url="http://embedding.local/v1",
            kb_embedding_api_key="test-key",
            kb_embedding_model="text-embedding-v4",
            kb_embedding_dimension=1024,
            kb_embedding_timeout_seconds=30,
        ),
    )
    monkeypatch.setattr(
        text2sql_service,
        "load_schema_catalog_from_postgres",
        lambda *args, **kwargs: SchemaCatalog([], HashEmbeddingProvider(dimension=64)),
    )

    with pytest.raises(RuntimeError, match="schema_catalog"):
        text2sql_service._build_service()
