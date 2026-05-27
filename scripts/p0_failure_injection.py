"""P0 strict 失败注入检查。

该脚本通过子进程临时覆盖环境变量来验证 readiness 会 fail-fast。
它不会修改 `.env`，也不会停止或改动 Docker 服务；建议在 P0 API 容器内执行。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
READINESS_SCRIPT = PROJECT_ROOT / "scripts" / "check_p0_dashscope_readiness.py"


@dataclass(frozen=True)
class FailureCase:
    name: str
    description: str
    overrides: dict[str, str]
    config_only: bool = True
    expected_stage: str | None = "config"


API_KEY_ENV_NAMES = (
    "LLM_API_KEY",
    "EMBEDDING_API_KEY",
    "RERANK_API_KEY",
    "VISION_API_KEY",
    "GUSTOBOT_ROUTER_LLM_API_KEY",
    "GUSTOBOT_TEXT2SQL_LLM_API_KEY",
    "GUSTOBOT_ANSWER_LLM_API_KEY",
    "GUSTOBOT_KB_EMBEDDING_API_KEY",
    "GUSTOBOT_KB_RERANK_API_KEY",
    "GUSTOBOT_VISION_API_KEY",
)


FAILURE_CASES = (
    FailureCase(
        name="missing_model_keys",
        description="移除所有模型 API key 后，配置检查必须失败。",
        overrides={name: "" for name in API_KEY_ENV_NAMES},
    ),
    FailureCase(
        name="hash_embedding_provider",
        description="KB embedding provider 被改成 hash 时，配置检查必须失败。",
        overrides={"GUSTOBOT_KB_EMBEDDING_PROVIDER": "hash", "KB_EMBEDDING_PROVIDER": "", "EMBEDDING_PROVIDER": ""},
    ),
    FailureCase(
        name="wrong_embedding_dimension",
        description="text-embedding-v4 维度不是 1024 时，配置检查必须失败。",
        overrides={"GUSTOBOT_KB_EMBEDDING_DIMENSION": "64", "EMBEDDING_DIMENSION": ""},
    ),
    FailureCase(
        name="missing_rerank_base_url",
        description="移除 rerank base_url 后，配置检查必须失败。",
        overrides={"GUSTOBOT_KB_RERANK_BASE_URL": "", "KB_RERANK_BASE_URL": "", "RERANK_BASE_URL": ""},
    ),
    FailureCase(
        name="image_generation_model_for_vision",
        description="Vision 使用 qwen-image 系列生成模型时，配置检查必须失败。",
        overrides={"GUSTOBOT_VISION_MODEL": "qwen-image-2.0-pro", "VISION_MODEL": ""},
    ),
    FailureCase(
        name="bad_postgres_runtime",
        description="PostgreSQL 地址不可达时，运行时 readiness 必须失败。",
        overrides={
            "GUSTOBOT_POSTGRES_DSN": "postgresql://gustobot:gustobot@127.0.0.1:1/gustobot",
            "GUSTOBOT_TEXT2SQL_POSTGRES_DSN": "postgresql://gustobot:gustobot@127.0.0.1:1/gustobot",
        },
        config_only=False,
        expected_stage="runtime",
    ),
    FailureCase(
        name="bad_neo4j_runtime",
        description="Neo4j 地址不可达时，运行时 readiness 必须失败。",
        overrides={"GUSTOBOT_NEO4J_URI": "bolt://127.0.0.1:1"},
        config_only=False,
        expected_stage="runtime",
    ),
    FailureCase(
        name="bad_redis_runtime",
        description="Redis 地址不可达时，运行时 readiness 必须失败。",
        overrides={"GUSTOBOT_REDIS_URL": "redis://127.0.0.1:1/0"},
        config_only=False,
        expected_stage="runtime",
    ),
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    selected = _select_cases(args.cases)
    results = [_run_case(case) for case in selected]
    payload = {
        "status": "ok" if all(item["passed"] for item in results) else "failed",
        "cases": results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "ok" else 1


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run P0 strict failure-injection readiness checks.")
    parser.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="可选：只运行指定 case 名称；默认运行全部。",
    )
    return parser.parse_args(argv)


def _select_cases(names: list[str] | None) -> list[FailureCase]:
    if not names:
        return list(FAILURE_CASES)
    by_name = {case.name: case for case in FAILURE_CASES}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise SystemExit(f"未知失败注入 case: {', '.join(missing)}")
    return [by_name[name] for name in names]


def _run_case(case: FailureCase) -> dict[str, Any]:
    command = [sys.executable, str(READINESS_SCRIPT), "--env-file", ""]
    if case.config_only:
        command.append("--config-only")
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=_case_env(case),
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    parsed = _parse_readiness_output(completed.stdout)
    passed = _case_passed(case, completed.returncode, parsed)
    return {
        "name": case.name,
        "description": case.description,
        "passed": passed,
        "exit_code": completed.returncode,
        "expected_stage": case.expected_stage,
        "actual_status": parsed.get("status"),
        "actual_stage": parsed.get("stage"),
        "issues": parsed.get("issues", []),
        "error": parsed.get("error"),
        "stderr": completed.stderr[-500:] if completed.stderr else "",
    }


def _case_env(case: FailureCase) -> dict[str, str]:
    env = os.environ.copy()
    env.update(case.overrides)
    # 避免子进程重新读取本地 .env 后掩盖注入值。
    return env


def _parse_readiness_output(stdout: str) -> dict[str, Any]:
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return {"status": "unparseable", "raw_stdout": stdout[-500:]}


def _case_passed(case: FailureCase, exit_code: int, parsed: dict[str, Any]) -> bool:
    if exit_code == 0:
        return False
    if parsed.get("status") != "failed":
        return False
    if case.expected_stage and parsed.get("stage") != case.expected_stage:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
