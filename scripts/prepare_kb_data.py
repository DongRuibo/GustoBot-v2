"""KB 本地数据准备与批量入库脚本。

这个脚本把“数据读取过程”放回 GustoBot-v2 自己的项目目录中完成：
    1. 从 data/raw/kb 扫描本项目内的原始资料；
    2. 将 txt/md/json/jsonl/csv/xlsx 解析为 PreparedDocument；
    3. 可先 dry-run 查看准备结果；
    4. 正式执行时复用 KBService 完成切块、embedding 和向量存储写入。

注意：脚本默认拒绝读取项目目录外的 input-dir/env-file，避免新系统运行时依赖旧项目路径。
如果要复用旧项目资料，请先把资料复制到 GustoBot-v2/data/raw/kb。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "kb"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    input_dir = _resolve_project_path(args.input_dir, label="input-dir")
    env_file = _resolve_optional_project_path(args.env_file, label="env-file")

    _load_env_file(env_file)
    _ensure_project_on_path()

    from app.files.local_loader import LocalKnowledgeFileLoader

    loader = LocalKnowledgeFileLoader(input_dir)
    documents = loader.load()
    if args.dry_run:
        _print_json(
            {
                "dry_run": True,
                "input_dir": str(input_dir),
                "documents_found": len(documents),
                "documents": [_document_preview(document) for document in documents],
            }
        )
        return 0

    from app.kb.service import get_kb_service

    if args.reset:
        _reset_kb_storage()

    kb_service = get_kb_service()
    ingested = []
    total_chunks = 0
    for document in documents:
        # 每个 PreparedDocument 进入 KBService 后，会继续完成切块、embedding 和存储。
        result = kb_service.ingest_document(
            title=document.title,
            content=document.content,
            metadata=document.metadata,
            source_id=document.source_id,
        )
        total_chunks += result.chunk_count
        ingested.append(
            {
                "source_id": result.document_id,
                "title": document.title,
                "chunk_count": result.chunk_count,
            }
        )

    status = kb_service.status()
    _print_json(
        {
            "dry_run": False,
            "input_dir": str(input_dir),
            "documents_found": len(documents),
            "ingested_documents": len(ingested),
            "chunk_count": total_chunks,
            "store_type": status.store_type,
            "pgvector_table": status.pgvector_table,
            "embedding_provider": status.embedding_provider,
            "embedding_model": status.embedding_model,
            "embedding_dimension": status.embedding_dimension,
            "documents": ingested,
        }
    )
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and ingest local KB data for GustoBot-v2.")
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="本项目内的原始知识资料目录，默认 data/raw/kb。",
    )
    parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_FILE),
        help="可选 .env 文件，默认读取项目根目录 .env；文件不存在时自动跳过。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只读取和展示数据准备结果，不执行切块、embedding 或入库。",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="入库前清空当前 KB pgvector 数据，用于重建真实 embedding。",
    )
    return parser.parse_args(argv)


def _reset_kb_storage() -> None:
    """清空当前 KB 存储，便于用真实 embedding 重新构建向量。"""

    from app.core.config import settings

    if not settings.postgres_dsn:
        return

    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("已配置 PostgreSQL，但当前环境缺少 psycopg，无法清空 KB 数据。") from exc

    if settings.kb_pgvector_table == "kb_chunks":
        with psycopg.connect(settings.postgres_dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM kb_documents")
        return

    if settings.kb_pgvector_table == "searchable_documents":
        with psycopg.connect(settings.postgres_dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM searchable_documents")
        return

    raise RuntimeError(f"不支持清空的 KB pgvector 表：{settings.kb_pgvector_table}")


def _resolve_project_path(path_value: str, *, label: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    resolved = path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise SystemExit(
            f"{label} 必须位于 GustoBot-v2 项目目录内：{resolved}。"
            "请先把旧项目资料复制到本项目，再从本项目读取。"
        ) from exc
    return resolved


def _resolve_optional_project_path(path_value: str, *, label: str) -> Path | None:
    resolved = _resolve_project_path(path_value, label=label)
    return resolved if resolved.exists() else None


def _load_env_file(path: Path | None) -> None:
    """读取项目内 .env，且不覆盖调用方已经显式设置的环境变量。"""

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


def _document_preview(document) -> dict[str, object]:
    # dry-run 只展示短预览，避免终端输出超长正文或大表格 metadata。
    return {
        "source_id": document.source_id,
        "title": document.title,
        "content_length": len(document.content),
        "content_preview": document.content[:120],
        "metadata": {
            key: value
            for key, value in document.metadata.items()
            if key not in {"row_data"}
        },
    }


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
