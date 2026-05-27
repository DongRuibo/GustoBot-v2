"""Neo4j 图谱初始化脚本。

这个脚本把当前开发用的种子图谱写入 Neo4j：
    1. 读取项目内 .env；
    2. 创建节点 node_id 唯一约束；
    3. 写入菜谱、食材、步骤、菜系、口味节点；
    4. 写入关系边。

脚本不读取旧项目路径，Neo4j 连接只来自 v2 自己的环境变量。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _load_env_file(_resolve_optional_project_path(args.env_file, label="env-file"))
    _ensure_project_on_path()

    from app.core.config import settings
    from app.graphrag.service import _build_seed_memory_graph

    neo4j_uri = args.uri or settings.neo4j_uri
    neo4j_username = args.username or settings.neo4j_username
    neo4j_password = args.password or settings.neo4j_password
    neo4j_database = args.database if args.database is not None else settings.neo4j_database

    if not neo4j_uri:
        raise SystemExit("缺少 GUSTOBOT_NEO4J_URI，无法初始化 Neo4j。")

    neo4j = _ensure_driver()
    graph = _build_seed_memory_graph()
    with neo4j.GraphDatabase.driver(
        neo4j_uri,
        auth=(neo4j_username, neo4j_password),
    ) as driver:
        with driver.session(database=neo4j_database) as session:
            session.execute_write(_create_constraints, sorted({node.label for node in graph.nodes.values()}))
            for node in graph.nodes.values():
                session.execute_write(_upsert_node, node)
            for edge in graph.edges:
                session.execute_write(_upsert_edge, edge)

    print(
        json.dumps(
            {
                "neo4j_uri": neo4j_uri,
                "node_count": len(graph.nodes),
                "edge_count": len(graph.edges),
                "status": "ok",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap GustoBot-v2 seed graph into Neo4j.")
    parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_FILE),
        help="项目内 .env 文件路径，默认读取项目根目录 .env。",
    )
    parser.add_argument("--uri", default=None, help="Neo4j Bolt URI；不传则读取 GUSTOBOT_NEO4J_URI。")
    parser.add_argument("--username", default=None, help="Neo4j 用户名；不传则读取环境变量。")
    parser.add_argument("--password", default=None, help="Neo4j 密码；不传则读取环境变量。")
    parser.add_argument("--database", default=None, help="Neo4j database；不传则读取环境变量。")
    return parser.parse_args(argv)


def _create_constraints(tx: Any, labels: list[str]) -> None:
    for label in labels:
        safe_label = _safe_symbol(label)
        tx.run(
            f"CREATE CONSTRAINT gustobot_{safe_label.lower()}_node_id IF NOT EXISTS "
            f"FOR (n:{safe_label}) REQUIRE n.node_id IS UNIQUE"
        )


def _upsert_node(tx: Any, node: Any) -> None:
    safe_label = _safe_symbol(node.label)
    tx.run(
        f"""
        MERGE (n:{safe_label} {{node_id: $node_id}})
        SET n.name = $name,
            n.aliases = $aliases,
            n += $properties
        """,
        node_id=node.node_id,
        name=node.name,
        aliases=list(node.aliases),
        properties=node.properties,
    )


def _upsert_edge(tx: Any, edge: Any) -> None:
    safe_relation = _safe_symbol(edge.relation)
    tx.run(
        f"""
        MATCH (source {{node_id: $source_id}})
        MATCH (target {{node_id: $target_id}})
        MERGE (source)-[r:{safe_relation} {{edge_id: $edge_id}}]->(target)
        SET r += $properties
        """,
        source_id=edge.source_id,
        target_id=edge.target_id,
        edge_id=edge.edge_id,
        properties=edge.properties,
    )


def _safe_symbol(value: str) -> str:
    # 标签和关系类型不能参数化，必须先限制为普通符号。
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"非法 Neo4j 标识符：{value}")
    return value


def _resolve_optional_project_path(path_value: str, *, label: str) -> Path | None:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    resolved = path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise SystemExit(f"{label} 必须位于 GustoBot-v2 项目目录内：{resolved}") from exc
    return resolved if resolved.exists() else None


def _load_env_file(path: Path | None) -> None:
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


def _ensure_driver():
    try:
        import neo4j
    except ImportError as exc:
        raise RuntimeError("当前环境缺少 neo4j 包，请先安装 requirements.txt。") from exc
    return neo4j


if __name__ == "__main__":
    raise SystemExit(main())
