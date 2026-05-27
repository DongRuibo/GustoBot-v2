"""检查 P0 DashScope 替换实验是否仍然命中占位实现。"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _load_env_file(_resolve_env_file(args.env_file))
    _ensure_project_on_path()
    settings = _load_settings()
    from app.core import p0_readiness

    result = p0_readiness.check_readiness(
        settings,
        config_only=args.config_only,
        runtime_snapshot_func=_runtime_snapshot,
    )
    _print_result(result)
    return 0 if result.get("status") == "ok" else 1


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check P0 DashScope strict readiness for GustoBot-v2.")
    parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_FILE),
        help="可选 .env 文件；默认读取项目根目录 .env，且不会覆盖已显式设置的环境变量。",
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="只检查配置和旧变量映射，不初始化 PostgreSQL、Neo4j、Redis 或模型服务。",
    )
    return parser.parse_args(argv)


def _resolve_env_file(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve() if path.exists() else None


def _load_env_file(path: Path | None) -> None:
    """读取 .env，但保留调用方已经显式设置的环境变量。"""

    if path is None:
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _ensure_project_on_path() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_settings() -> Any:
    # readiness 可能先被测试进程导入，这里强制重载配置，确保 --env-file 刚加载的旧变量生效。
    config_module = importlib.import_module("app.core.config")
    return importlib.reload(config_module).settings


def _runtime_snapshot() -> dict[str, Any]:
    from app.core import p0_readiness

    return p0_readiness.runtime_snapshot()


def _print_result(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
