"""把食品图谱 CSV 导入 Neo4j。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.food_dataset import PROCESSED_DIR, resolve_project_path  # noqa: E402


DEFAULT_NODES = PROCESSED_DIR / "graph_nodes.csv"
DEFAULT_EDGES = PROCESSED_DIR / "graph_edges.csv"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _load_env_file(resolve_project_path(args.env_file, label="env-file") if args.env_file else None)

    from app.core.config import settings

    uri = args.neo4j_uri or settings.neo4j_uri
    username = args.neo4j_username or settings.neo4j_username
    password = args.neo4j_password or settings.neo4j_password
    database = args.neo4j_database if args.neo4j_database is not None else settings.neo4j_database
    if not uri:
        raise SystemExit("缺少 Neo4j URI，无法导入食品图谱。")

    nodes_path = resolve_project_path(args.nodes, label="nodes")
    edges_path = resolve_project_path(args.edges, label="edges")
    nodes = _read_nodes(nodes_path)
    edges = _read_edges(edges_path)

    neo4j = _ensure_neo4j_driver()
    with neo4j.GraphDatabase.driver(uri, auth=(username, password)) as driver:
        with driver.session(database=database) as session:
            if args.reset:
                session.execute_write(lambda tx: tx.run("MATCH (n) DETACH DELETE n"))
            session.execute_write(_create_constraints, sorted({node["label"] for node in nodes}))
            _upsert_nodes(session, nodes, batch_size=args.batch_size)
            _upsert_edges(session, edges, batch_size=args.batch_size)

    print(
        json.dumps(
            {
                "neo4j_uri": uri,
                "node_count": len(nodes),
                "edge_count": len(edges),
                "status": "ok",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import food graph CSV into Neo4j.")
    parser.add_argument("--nodes", default=str(DEFAULT_NODES))
    parser.add_argument("--edges", default=str(DEFAULT_EDGES))
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--neo4j-uri", default=None)
    parser.add_argument("--neo4j-username", default=None)
    parser.add_argument("--neo4j-password", default=None)
    parser.add_argument("--neo4j-database", default=None)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1000)
    return parser.parse_args(argv)


def _read_nodes(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return [
            {
                "node_id": row["node_id"],
                "label": row["label"],
                "name": row["name"],
                "aliases": json.loads(row["aliases"] or "[]"),
                "properties": json.loads(row["properties"] or "{}"),
            }
            for row in csv.DictReader(file)
        ]


def _read_edges(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return [
            {
                "edge_id": row["edge_id"],
                "source_id": row["source_id"],
                "target_id": row["target_id"],
                "relation": row["relation"],
                "properties": json.loads(row["properties"] or "{}"),
            }
            for row in csv.DictReader(file)
        ]


def _create_constraints(tx: Any, labels: list[str]) -> None:
    tx.run(
        "CREATE CONSTRAINT gustobot_foodnode_node_id IF NOT EXISTS "
        "FOR (n:FoodNode) REQUIRE n.node_id IS UNIQUE"
    )
    for label in labels:
        safe_label = _safe_symbol(label)
        tx.run(
            f"CREATE CONSTRAINT gustobot_food_{safe_label.lower()}_node_id IF NOT EXISTS "
            f"FOR (n:{safe_label}) REQUIRE n.node_id IS UNIQUE"
        )


def _upsert_nodes(session: Any, nodes: list[dict[str, Any]], *, batch_size: int) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        grouped[node["label"]].append(node)
    for label, label_nodes in grouped.items():
        safe_label = _safe_symbol(label)
        for batch in _batches(label_nodes, batch_size):
            session.execute_write(_upsert_node_batch, safe_label, batch)


def _upsert_node_batch(tx: Any, safe_label: str, nodes: list[dict[str, Any]]) -> None:
    tx.run(
        f"""
        UNWIND $nodes AS node
        MERGE (n:FoodNode:{safe_label} {{node_id: node.node_id}})
        SET n.name = node.name,
            n.aliases = node.aliases,
            n += node.properties
        """,
        nodes=nodes,
    )


def _upsert_edges(session: Any, edges: list[dict[str, Any]], *, batch_size: int) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        grouped[edge["relation"]].append(edge)
    for relation, relation_edges in grouped.items():
        safe_relation = _safe_symbol(relation)
        for batch in _batches(relation_edges, batch_size):
            session.execute_write(_upsert_edge_batch, safe_relation, batch)


def _upsert_edge_batch(tx: Any, safe_relation: str, edges: list[dict[str, Any]]) -> None:
    tx.run(
        f"""
        UNWIND $edges AS edge
        MATCH (source:FoodNode {{node_id: edge.source_id}})
        MATCH (target:FoodNode {{node_id: edge.target_id}})
        MERGE (source)-[r:{safe_relation} {{edge_id: edge.edge_id}}]->(target)
        SET r += edge.properties
        """,
        edges=edges,
    )


def _safe_symbol(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"非法 Neo4j 标识符：{value}")
    return value


def _batches(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    size = max(1, batch_size)
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def _load_env_file(path: Path | None) -> None:
    if path is None or not path.exists():
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


def _ensure_neo4j_driver():
    try:
        import neo4j
    except ImportError as exc:
        raise RuntimeError("当前环境缺少 neo4j 包，无法写入 Neo4j。") from exc
    return neo4j


if __name__ == "__main__":
    raise SystemExit(main())
