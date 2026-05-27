"""P0 readiness 脚本配置加载测试。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

from scripts import check_p0_dashscope_readiness as readiness
from app.core import p0_readiness

PROJECT_ROOT = Path(__file__).resolve().parents[1]


P0_ENV_NAMES = (
    "GUSTOBOT_ENV",
    "GUSTOBOT_STRICT_EXTERNAL_STORES",
    "GUSTOBOT_POSTGRES_DSN",
    "GUSTOBOT_TEXT2SQL_POSTGRES_DSN",
    "GUSTOBOT_NEO4J_URI",
    "GUSTOBOT_REDIS_URL",
    "GUSTOBOT_KB_EMBEDDING_PROVIDER",
    "GUSTOBOT_KB_EMBEDDING_BASE_URL",
    "GUSTOBOT_KB_EMBEDDING_API_KEY",
    "GUSTOBOT_KB_EMBEDDING_MODEL",
    "GUSTOBOT_KB_EMBEDDING_DIMENSION",
    "GUSTOBOT_KB_RERANK_BASE_URL",
    "GUSTOBOT_KB_RERANK_ENDPOINT",
    "GUSTOBOT_KB_RERANK_API_KEY",
    "GUSTOBOT_KB_RERANK_MODEL",
    "GUSTOBOT_TEXT2SQL_LLM_BASE_URL",
    "GUSTOBOT_TEXT2SQL_LLM_API_KEY",
    "GUSTOBOT_TEXT2SQL_LLM_MODEL",
    "GUSTOBOT_ROUTER_LLM_BASE_URL",
    "GUSTOBOT_ROUTER_LLM_API_KEY",
    "GUSTOBOT_ROUTER_LLM_MODEL",
    "GUSTOBOT_ANSWER_LLM_BASE_URL",
    "GUSTOBOT_ANSWER_LLM_API_KEY",
    "GUSTOBOT_ANSWER_LLM_MODEL",
    "GUSTOBOT_VISION_BASE_URL",
    "GUSTOBOT_VISION_API_KEY",
    "GUSTOBOT_VISION_MODEL",
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_MODEL",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_BASE_URL",
    "EMBEDDING_API_KEY",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSION",
    "RERANK_BASE_URL",
    "RERANK_ENDPOINT",
    "RERANK_API_KEY",
    "RERANK_MODEL",
    "VISION_BASE_URL",
    "VISION_API_KEY",
    "VISION_MODEL",
)


def test_readiness_config_only_loads_legacy_env_file(monkeypatch, capsys) -> None:
    previous_env = _snapshot_env()
    env_file = _test_env_file()
    try:
        _clear_env()
        env_file.write_text(
            "\n".join(
                [
                    "GUSTOBOT_ENV=prod",
                    "GUSTOBOT_STRICT_EXTERNAL_STORES=true",
                    "GUSTOBOT_POSTGRES_DSN=postgresql://gustobot:gustobot@postgres:5432/gustobot",
                    "GUSTOBOT_TEXT2SQL_POSTGRES_DSN=postgresql://gustobot:gustobot@postgres:5432/gustobot",
                    "GUSTOBOT_NEO4J_URI=bolt://neo4j:7687",
                    "GUSTOBOT_REDIS_URL=redis://redis:6379/0",
                    "LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "LLM_API_KEY=test-llm-key",
                    "LLM_MODEL=qwen3-max",
                    "VISION_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "VISION_API_KEY=test-vision-key",
                    "VISION_MODEL=qwen3-vl-plus",
                    "EMBEDDING_PROVIDER=openai",
                    "EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "EMBEDDING_API_KEY=test-embedding-key",
                    "EMBEDDING_MODEL=text-embedding-v4",
                    "EMBEDDING_DIMENSION=1024",
                    "RERANK_BASE_URL=https://dashscope.aliyuncs.com/api/v1/services",
                    "RERANK_ENDPOINT=/rerank/text-rerank/text-rerank",
                    "RERANK_API_KEY=test-rerank-key",
                    "RERANK_MODEL=qwen3-rerank",
                ]
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(readiness, "_runtime_snapshot", lambda: (_ for _ in ()).throw(AssertionError("runtime called")))

        exit_code = readiness.main(["--env-file", str(env_file), "--config-only"])

        output = capsys.readouterr().out
        payload = json.loads(output)
        assert exit_code == 0
        assert payload["status"] == "ok"
        assert payload["stage"] == "config"
        assert payload["snapshot"]["kb_embedding_model"] == "text-embedding-v4"
        assert payload["snapshot"]["kb_embedding_api_key_configured"] is True
        assert "test-embedding-key" not in output
    finally:
        _restore_env(previous_env)
        env_file.unlink(missing_ok=True)


def test_readiness_config_only_rejects_image_generation_model_for_vision(monkeypatch, capsys) -> None:
    previous_env = _snapshot_env()
    env_file = _test_env_file()
    try:
        _clear_env()
        env_file.write_text(
            "\n".join(
                [
                    "GUSTOBOT_ENV=prod",
                    "GUSTOBOT_STRICT_EXTERNAL_STORES=true",
                    "GUSTOBOT_POSTGRES_DSN=postgresql://gustobot:gustobot@postgres:5432/gustobot",
                    "GUSTOBOT_TEXT2SQL_POSTGRES_DSN=postgresql://gustobot:gustobot@postgres:5432/gustobot",
                    "GUSTOBOT_NEO4J_URI=bolt://neo4j:7687",
                    "GUSTOBOT_REDIS_URL=redis://redis:6379/0",
                    "LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "LLM_API_KEY=test-llm-key",
                    "LLM_MODEL=qwen3-max",
                    "VISION_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "VISION_API_KEY=test-vision-key",
                    "VISION_MODEL=qwen-image-2.0-pro",
                    "EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "EMBEDDING_API_KEY=test-embedding-key",
                    "EMBEDDING_MODEL=text-embedding-v4",
                    "EMBEDDING_DIMENSION=1024",
                    "RERANK_BASE_URL=https://dashscope.aliyuncs.com/api/v1/services",
                    "RERANK_API_KEY=test-rerank-key",
                    "RERANK_MODEL=qwen3-rerank",
                ]
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(readiness, "_runtime_snapshot", lambda: (_ for _ in ()).throw(AssertionError("runtime called")))

        exit_code = readiness.main(["--env-file", str(env_file), "--config-only"])

        payload = json.loads(capsys.readouterr().out)
        assert exit_code == 1
        assert payload["status"] == "failed"
        assert any("图片生成/编辑模型" in issue for issue in payload["issues"])
    finally:
        _restore_env(previous_env)
        env_file.unlink(missing_ok=True)


def test_env_loader_does_not_override_explicit_environment() -> None:
    previous_env = _snapshot_env()
    env_file = _test_env_file()
    try:
        _clear_env()
        env_file.write_text("LLM_MODEL=from-file\n", encoding="utf-8")
        os.environ["LLM_MODEL"] = "explicit"

        readiness._load_env_file(env_file)

        assert readiness.os.environ["LLM_MODEL"] == "explicit"
    finally:
        _restore_env(previous_env)
        env_file.unlink(missing_ok=True)


def test_runtime_issues_reject_hash_schema_embedding_provider() -> None:
    snapshot = {
        "kb": {"store_type": "postgres_pgvector", "embedding_provider": "openai-compatible", "reranker_type": "http"},
        "graphrag": {"store_type": "neo4j"},
        "text2sql": {
            "executor_type": "postgres_readonly",
            "generator_type": "LLMSQLGenerator",
            "schema_embedding_provider": "hash",
        },
        "cache": {"store_type": "redis"},
        "sessions": {"store_type": "PostgreSQLSessionStore"},
        "uploads": {"store_type": "PostgreSQLUploadStore"},
    }

    issues = p0_readiness.runtime_issues(snapshot)

    assert any("schema_embedding_provider" in issue for issue in issues)


def _test_env_file() -> Path:
    tmp_dir = PROJECT_ROOT / "tmp" / "pytest_p0_readiness"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir / f"{uuid4().hex}.env"


def _snapshot_env() -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in P0_ENV_NAMES}


def _clear_env() -> None:
    for name in P0_ENV_NAMES:
        os.environ.pop(name, None)


def _restore_env(snapshot: dict[str, str | None]) -> None:
    _clear_env()
    for name, value in snapshot.items():
        if value is not None:
            os.environ[name] = value
