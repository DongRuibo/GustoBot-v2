"""从 PostgreSQL 真实菜谱数据构建 Neo4j 图谱。"""

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

    postgres_dsn = args.postgres_dsn or settings.text2sql_postgres_dsn or settings.postgres_dsn
    neo4j_uri = args.neo4j_uri or settings.neo4j_uri
    neo4j_username = args.neo4j_username or settings.neo4j_username
    neo4j_password = args.neo4j_password or settings.neo4j_password
    neo4j_database = args.neo4j_database if args.neo4j_database is not None else settings.neo4j_database

    if not postgres_dsn:
        raise SystemExit("缺少 PostgreSQL DSN，无法读取真实菜谱数据。")
    if not neo4j_uri:
        raise SystemExit("缺少 Neo4j URI，无法构建真实图谱。")

    graph = _load_graph_payload(postgres_dsn, limit=args.limit)
    neo4j = _ensure_neo4j_driver()
    with neo4j.GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password)) as driver:
        with driver.session(database=neo4j_database) as session:
            if args.reset:
                session.execute_write(lambda tx: tx.run("MATCH (n) DETACH DELETE n"))
            session.execute_write(_create_constraints, ["Recipe", "Ingredient", "IngredientCategory", "Step", "Cuisine", "Tool"])
            for node in graph["nodes"]:
                session.execute_write(_upsert_node, node)
            for edge in graph["edges"]:
                session.execute_write(_upsert_edge, edge)

    _print_json(
        {
            "postgres_dsn_configured": True,
            "neo4j_uri": neo4j_uri,
            "node_count": len(graph["nodes"]),
            "edge_count": len(graph["edges"]),
            **_taxonomy_stats(graph, report_uncategorized=args.report_uncategorized),
            "status": "ok",
        }
    )
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap Neo4j graph from real PostgreSQL recipe data.")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--postgres-dsn", default=None)
    parser.add_argument("--neo4j-uri", default=None)
    parser.add_argument("--neo4j-username", default=None)
    parser.add_argument("--neo4j-password", default=None)
    parser.add_argument("--neo4j-database", default=None)
    parser.add_argument("--limit", type=int, default=0, help="限制导入菜谱数量，0 表示不限制。")
    parser.add_argument("--reset", action="store_true", help="导入前清空当前 Neo4j 图谱。")
    parser.add_argument("--report-uncategorized", action="store_true", help="在输出 JSON 中列出未归类食材名称。")
    return parser.parse_args(argv)


def _load_graph_payload(dsn: str, *, limit: int) -> dict[str, list[dict[str, Any]]]:
    psycopg, dict_row = _ensure_postgres_driver()
    from app.graphrag.ingredient_taxonomy import (
        BELONGS_TO_CATEGORY,
        aliases_for_ingredient,
        categories_for_ingredient,
        category_hierarchy_edges,
        category_node_data,
        load_taxonomy,
    )

    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}
    taxonomy_data = load_taxonomy(dsn, use_cache=False)
    for node in category_node_data(taxonomy_data=taxonomy_data):
        nodes[str(node["node_id"])] = node
    for edge in category_hierarchy_edges(taxonomy_data=taxonomy_data):
        edges[str(edge["edge_id"])] = edge
    with psycopg.connect(dsn, row_factory=dict_row) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT r.id, r.name, r.description, r.total_time, r.servings, r.difficulty,
                       c.id AS cuisine_id, c.name AS cuisine_name
                FROM recipe_records r
                LEFT JOIN recipe_cuisines c ON c.id = r.cuisine_id
                ORDER BY r.id
                """
            )
            recipes = cursor.fetchall()
            if limit > 0:
                recipes = recipes[:limit]
            recipe_ids = [recipe["id"] for recipe in recipes]
            for recipe in recipes:
                recipe_node_id = f"recipe:{recipe['id']}"
                nodes[recipe_node_id] = _node(
                    recipe_node_id,
                    "Recipe",
                    recipe["name"],
                    recipe_id=recipe["id"],
                    description=recipe.get("description"),
                    total_time=recipe.get("total_time"),
                    servings=recipe.get("servings"),
                    difficulty=recipe.get("difficulty"),
                )
                if recipe.get("cuisine_id"):
                    cuisine_node_id = f"cuisine:{recipe['cuisine_id']}"
                    nodes[cuisine_node_id] = _node(
                        cuisine_node_id,
                        "Cuisine",
                        recipe["cuisine_name"],
                        cuisine_id=recipe["cuisine_id"],
                    )
                    edges[f"edge:{recipe_node_id}:cuisine"] = _edge(
                        f"edge:{recipe_node_id}:cuisine",
                        recipe_node_id,
                        cuisine_node_id,
                        "BELONGS_TO_CUISINE",
                    )

            if not recipe_ids:
                return {"nodes": list(nodes.values()), "edges": list(edges.values())}

            cursor.execute(
                """
                SELECT ri.recipe_id, ri.quantity, ri.unit, ri.prep_method, ri.ingredient_type, ri.is_main,
                       i.id AS ingredient_id, i.name, i.category
                FROM recipe_ingredients ri
                JOIN recipe_ingredients_master i ON i.id = ri.ingredient_id
                WHERE ri.recipe_id = ANY(%s)
                """,
                (recipe_ids,),
            )
            for item in cursor.fetchall():
                source_id = f"recipe:{item['recipe_id']}"
                target_id = f"ingredient:{item['ingredient_id']}"
                nodes[target_id] = _node(
                    target_id,
                    "Ingredient",
                    item["name"],
                    aliases=aliases_for_ingredient(item["name"]),
                    ingredient_id=item["ingredient_id"],
                    category=item.get("category"),
                )
                edges[f"edge:{source_id}:ingredient:{item['ingredient_id']}"] = _edge(
                    f"edge:{source_id}:ingredient:{item['ingredient_id']}",
                    source_id,
                    target_id,
                    "USES_INGREDIENT",
                    quantity=item.get("quantity"),
                    unit=item.get("unit"),
                    prep_method=item.get("prep_method"),
                    ingredient_type=item.get("ingredient_type"),
                    is_main=item.get("is_main"),
                )
                for category in categories_for_ingredient(
                    item["name"],
                    item.get("category"),
                    taxonomy_data=taxonomy_data,
                ):
                    edges[f"edge:{target_id}:category:{category.slug}"] = _edge(
                        f"edge:{target_id}:category:{category.slug}",
                        target_id,
                        category.node_id,
                        BELONGS_TO_CATEGORY,
                        source="taxonomy",
                    )

            cursor.execute(
                """
                SELECT id, recipe_id, step_number, action, instruction, duration, temperature
                FROM recipe_steps
                WHERE recipe_id = ANY(%s)
                ORDER BY recipe_id, step_number
                """,
                (recipe_ids,),
            )
            step_ids = []
            for item in cursor.fetchall():
                step_ids.append(item["id"])
                source_id = f"recipe:{item['recipe_id']}"
                target_id = f"step:{item['id']}"
                nodes[target_id] = _node(
                    target_id,
                    "Step",
                    f"第{item['step_number']}步 {item['action']}",
                    step_id=item["id"],
                    order=item["step_number"],
                    action=item["action"],
                    instruction=item["instruction"],
                    duration=item.get("duration"),
                    temperature=item.get("temperature"),
                )
                edges[f"edge:{source_id}:step:{item['id']}"] = _edge(
                    f"edge:{source_id}:step:{item['id']}",
                    source_id,
                    target_id,
                    "HAS_STEP",
                    order=item["step_number"],
                )

            if step_ids:
                cursor.execute(
                    """
                    SELECT st.step_id, st.usage_text, t.id AS tool_id, t.name, t.type, t.material, t.capacity
                    FROM recipe_step_tools st
                    JOIN recipe_tools t ON t.id = st.tool_id
                    WHERE st.step_id = ANY(%s)
                    """,
                    (step_ids,),
                )
                for item in cursor.fetchall():
                    source_id = f"step:{item['step_id']}"
                    target_id = f"tool:{item['tool_id']}"
                    nodes[target_id] = _node(
                        target_id,
                        "Tool",
                        item["name"],
                        tool_id=item["tool_id"],
                        type=item.get("type"),
                        material=item.get("material"),
                        capacity=item.get("capacity"),
                    )
                    edges[f"edge:{source_id}:tool:{item['tool_id']}"] = _edge(
                        f"edge:{source_id}:tool:{item['tool_id']}",
                        source_id,
                        target_id,
                        "USES_TOOL",
                        usage=item.get("usage_text"),
                    )

    return {"nodes": list(nodes.values()), "edges": list(edges.values())}


def _node(node_id: str, label: str, name: str, aliases: list[str] | None = None, **properties: Any) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "label": label,
        "name": name,
        "aliases": aliases or [name],
        "properties": {key: value for key, value in properties.items() if value is not None},
    }


def _edge(edge_id: str, source_id: str, target_id: str, relation: str, **properties: Any) -> dict[str, Any]:
    return {
        "edge_id": edge_id,
        "source_id": source_id,
        "target_id": target_id,
        "relation": relation,
        "properties": {key: value for key, value in properties.items() if value is not None},
    }


def _taxonomy_stats(graph: dict[str, list[dict[str, Any]]], *, report_uncategorized: bool) -> dict[str, Any]:
    ingredient_nodes = {
        node["node_id"]: node
        for node in graph["nodes"]
        if node.get("label") == "Ingredient"
    }
    category_nodes = {
        str(node["node_id"]): str(node["name"])
        for node in graph["nodes"]
        if node.get("label") == "IngredientCategory"
    }
    category_edges = [
        edge
        for edge in graph["edges"]
        if edge.get("relation") == "BELONGS_TO_CATEGORY" and edge.get("source_id") in ingredient_nodes
    ]
    categorized_ids = {str(edge["source_id"]) for edge in category_edges}
    category_ingredients: dict[str, set[str]] = {}
    for edge in category_edges:
        category_id = str(edge["target_id"])
        category_ingredients.setdefault(category_id, set()).add(str(edge["source_id"]))
    category_coverage = {
        category_nodes.get(category_id, category_id): len(ingredient_ids)
        for category_id, ingredient_ids in category_ingredients.items()
    }
    uncategorized_names = [
        str(node["name"])
        for node_id, node in ingredient_nodes.items()
        if node_id not in categorized_ids
    ]
    stats: dict[str, Any] = {
        "ingredient_count": len(ingredient_nodes),
        "categorized_ingredient_count": len(categorized_ids),
        "uncategorized_ingredient_count": len(uncategorized_names),
        "ingredient_category_edge_count": len(category_edges),
        "category_coverage": dict(sorted(category_coverage.items())),
    }
    if report_uncategorized:
        stats["uncategorized_ingredients"] = uncategorized_names
        stats["uncategorized_ingredients_top"] = uncategorized_names[:20]
    return stats


def _create_constraints(tx: Any, labels: list[str]) -> None:
    for label in labels:
        safe_label = _safe_symbol(label)
        tx.run(
            f"CREATE CONSTRAINT gustobot_real_{safe_label.lower()}_node_id IF NOT EXISTS "
            f"FOR (n:{safe_label}) REQUIRE n.node_id IS UNIQUE"
        )


def _upsert_node(tx: Any, node: dict[str, Any]) -> None:
    safe_label = _safe_symbol(node["label"])
    tx.run(
        f"""
        MERGE (n:{safe_label} {{node_id: $node_id}})
        SET n.name = $name,
            n.aliases = $aliases,
            n += $properties
        """,
        node_id=node["node_id"],
        name=node["name"],
        aliases=node["aliases"],
        properties=node["properties"],
    )


def _upsert_edge(tx: Any, edge: dict[str, Any]) -> None:
    safe_relation = _safe_symbol(edge["relation"])
    tx.run(
        f"""
        MATCH (source {{node_id: $source_id}})
        MATCH (target {{node_id: $target_id}})
        MERGE (source)-[r:{safe_relation} {{edge_id: $edge_id}}]->(target)
        SET r += $properties
        """,
        source_id=edge["source_id"],
        target_id=edge["target_id"],
        edge_id=edge["edge_id"],
        properties=edge["properties"],
    )


def _safe_symbol(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"非法 Neo4j 标识符：{value}")
    return value


def _resolve_project_path(path_value: str, *, label: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    resolved = path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise SystemExit(f"{label} 必须位于 GustoBot-v2 项目目录内：{resolved}") from exc
    return resolved


def _resolve_optional_project_path(path_value: str, *, label: str) -> Path | None:
    resolved = _resolve_project_path(path_value, label=label)
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


def _ensure_postgres_driver():
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise RuntimeError("当前环境缺少 psycopg，无法读取 PostgreSQL。") from exc
    return psycopg, dict_row


def _ensure_neo4j_driver():
    try:
        import neo4j
    except ImportError as exc:
        raise RuntimeError("当前环境缺少 neo4j 包，无法写入 Neo4j。") from exc
    return neo4j


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
