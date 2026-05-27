"""P0 Docker compose override 配置测试。"""

from __future__ import annotations

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_p0_compose_override_uses_strict_external_stores() -> None:
    payload = yaml.safe_load((PROJECT_ROOT / "docker-compose.p0.yml").read_text(encoding="utf-8"))
    env = payload["services"]["api"]["environment"]

    assert env["GUSTOBOT_ENV"] == "prod"
    assert env["GUSTOBOT_STRICT_EXTERNAL_STORES"] == "true"
    assert env["GUSTOBOT_POSTGRES_DSN"] == "postgresql://gustobot:gustobot@postgres:5432/gustobot"
    assert env["GUSTOBOT_NEO4J_URI"] == "bolt://neo4j:7687"
    assert env["GUSTOBOT_REDIS_URL"] == "redis://redis:6379/0"
    assert env["GUSTOBOT_KB_EMBEDDING_PROVIDER"] == "openai-compatible"
    assert env["GUSTOBOT_KB_EMBEDDING_MODEL"] == "text-embedding-v4"
    assert env["GUSTOBOT_KB_EMBEDDING_DIMENSION"] == "1024"
    assert env["GUSTOBOT_KB_RERANK_MODEL"] == "qwen3-rerank"


def test_p0_compose_override_maps_legacy_model_env_without_real_keys() -> None:
    text = (PROJECT_ROOT / "docker-compose.p0.yml").read_text(encoding="utf-8")
    payload = yaml.safe_load(text)
    env = payload["services"]["api"]["environment"]

    assert env["GUSTOBOT_ROUTER_LLM_API_KEY"] == "${LLM_API_KEY}"
    assert env["GUSTOBOT_TEXT2SQL_LLM_API_KEY"] == "${LLM_API_KEY}"
    assert env["GUSTOBOT_ANSWER_LLM_API_KEY"] == "${LLM_API_KEY}"
    assert env["GUSTOBOT_VISION_API_KEY"] == "${VISION_API_KEY:-}"
    assert env["GUSTOBOT_VISION_MODEL"] == "${VISION_UNDERSTANDING_MODEL:-qwen3-vl-plus}"
    assert env["GUSTOBOT_KB_EMBEDDING_API_KEY"] == "${EMBEDDING_API_KEY:-}"
    assert env["GUSTOBOT_KB_RERANK_API_KEY"] == "${RERANK_API_KEY:-}"

    # override 文件只能引用本地环境变量，不能把真实 API Key 写进仓库。
    for name, value in env.items():
        if name.endswith("API_KEY"):
            assert isinstance(value, str)
            assert value.startswith("${") and value.endswith("}")
    assert "your-dashscope-api-key" not in text
    assert "sk-" not in text.lower()
