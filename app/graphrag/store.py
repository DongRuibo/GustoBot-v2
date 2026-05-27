"""GraphRAG 图谱存储模块。

这个文件定义 GraphStore 抽象，并提供内存图谱和 Neo4j 图谱两个实现。
上层 GraphRAGService 只依赖实体链接和子图抽取接口，不直接关心底层是本地内存还是 Neo4j。
"""

from __future__ import annotations

import re
from collections import deque
from difflib import SequenceMatcher
from typing import Protocol

from app.graphrag.ingredient_taxonomy import BELONGS_TO_CATEGORY, IS_A_CATEGORY
from app.graphrag.models import GraphEdge, GraphNode, GraphQueryPlan, GraphSubgraph, LinkedEntity
from app.graphrag.templates import (
    ALLERGEN_TO_PRODUCTS_TEMPLATE,
    COMMON_INGREDIENTS_TEMPLATE,
    CUISINE_TO_RECIPES_TEMPLATE,
    DIRECT_NEIGHBORS_TEMPLATE,
    INGREDIENT_TO_RECIPES_TEMPLATE,
    PRODUCT_ALLERGENS_TEMPLATE,
    PRODUCT_CATEGORY_TEMPLATE,
    PRODUCT_DETAIL_TEMPLATE,
    PRODUCT_INGREDIENTS_TEMPLATE,
    RECIPE_DETAIL_TEMPLATE,
    RECIPE_INGREDIENT_AMOUNT_TEMPLATE,
    RECIPE_INGREDIENTS_TEMPLATE,
    RECIPE_STEPS_TEMPLATE,
    RECIPE_TOOLS_TEMPLATE,
)


class GraphStore(Protocol):
    # GraphStore 是图谱查询的最小协议：先做实体链接，再围绕实体抽取有限跳数子图。
    store_type: str

    def find_entities(self, text: str, *, limit: int = 5) -> list[LinkedEntity]:
        ...

    def extract_subgraph(self, start_node_ids: list[str], *, max_depth: int = 2) -> GraphSubgraph:
        ...

    def execute_plan(self, plan: GraphQueryPlan, *, max_depth: int = 2) -> GraphSubgraph:
        ...


class InMemoryGraphStore:
    # 内存图谱用于本地开发和测试，数据规模很小，但保留了实体链接、关系查询和子图抽取的完整动作。
    store_type = "memory"

    def __init__(self) -> None:
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []

    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, edge: GraphEdge) -> None:
        self.edges.append(edge)

    def find_entities(self, text: str, *, limit: int = 5) -> list[LinkedEntity]:
        matches: list[LinkedEntity] = []
        for node in self.nodes.values():
            candidates = (node.name, *node.aliases)
            matched = False
            for candidate in candidates:
                if candidate and candidate in text:
                    # 名称越长越可信，完全命中名称比命中别名更可信。
                    base_score = 1.0 if candidate == node.name else 0.85
                    length_bonus = min(len(candidate) / 10, 0.1)
                    matches.append(
                        LinkedEntity(
                            node=node,
                            matched_text=candidate,
                            score=min(1.0, base_score + length_bonus),
                        )
                    )
                    matched = True
                    break
            if matched:
                continue

            fuzzy_match = _fuzzy_match_entity(text, candidates)
            if fuzzy_match:
                matched_text, score = fuzzy_match
                matches.append(LinkedEntity(node=node, matched_text=matched_text, score=score))

        deduped: dict[str, LinkedEntity] = {}
        for match in matches:
            current = deduped.get(match.node.node_id)
            if current is None or match.score > current.score:
                deduped[match.node.node_id] = match
        return sorted(deduped.values(), key=lambda item: item.score, reverse=True)[:limit]

    def extract_subgraph(self, start_node_ids: list[str], *, max_depth: int = 2) -> GraphSubgraph:
        # 使用无向 BFS 抽取有限跳数子图，因为用户问题可能从菜谱问食材，也可能从食材反查菜谱。
        visited_nodes = set(start_node_ids)
        visited_edges: dict[str, GraphEdge] = {}
        queue: deque[tuple[str, int]] = deque((node_id, 0) for node_id in start_node_ids)

        while queue:
            node_id, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for edge in self._incident_edges(node_id):
                visited_edges[edge.edge_id] = edge
                neighbor_id = edge.target_id if edge.source_id == node_id else edge.source_id
                if neighbor_id not in visited_nodes:
                    visited_nodes.add(neighbor_id)
                    queue.append((neighbor_id, depth + 1))

        nodes = [self.nodes[node_id] for node_id in visited_nodes if node_id in self.nodes]
        edges = list(visited_edges.values())
        return GraphSubgraph(nodes=nodes, edges=edges)

    def execute_plan(self, plan: GraphQueryPlan, *, max_depth: int = 2) -> GraphSubgraph:
        if plan.template_id == RECIPE_DETAIL_TEMPLATE:
            return self._recipe_detail_subgraph(plan.params["recipe_node_id"])
        if plan.template_id == RECIPE_INGREDIENTS_TEMPLATE:
            return self._relation_subgraph([plan.params["recipe_node_id"]], {"USES_INGREDIENT"})
        if plan.template_id == RECIPE_INGREDIENT_AMOUNT_TEMPLATE:
            return self._relation_subgraph(
                [plan.params["recipe_node_id"]],
                {"USES_INGREDIENT"},
                target_ids={plan.params["ingredient_node_id"]},
            )
        if plan.template_id == RECIPE_STEPS_TEMPLATE:
            return self._recipe_steps_subgraph(plan.params["recipe_node_id"])
        if plan.template_id == RECIPE_TOOLS_TEMPLATE:
            return self._recipe_tools_subgraph(plan.params["recipe_node_id"])
        if plan.template_id == INGREDIENT_TO_RECIPES_TEMPLATE:
            ingredient_node_ids = list(plan.params.get("ingredient_node_ids") or [])
            if not ingredient_node_ids and plan.params.get("ingredient_node_id"):
                ingredient_node_ids = [str(plan.params["ingredient_node_id"])]
            category_node_ids = list(plan.params.get("category_node_ids") or [])
            return self._ingredient_or_category_to_recipes_subgraph(list(ingredient_node_ids), category_node_ids)
        if plan.template_id == CUISINE_TO_RECIPES_TEMPLATE:
            return self._incoming_relation_subgraph([plan.params["cuisine_node_id"]], {"BELONGS_TO_CUISINE"})
        if plan.template_id == COMMON_INGREDIENTS_TEMPLATE:
            return self._common_ingredients_subgraph(
                plan.params["recipe_a_id"],
                plan.params["recipe_b_id"],
            )
        if plan.template_id == PRODUCT_DETAIL_TEMPLATE:
            return self._relation_subgraph(
                [plan.params["product_node_id"]],
                {"HAS_INGREDIENT", "HAS_ALLERGEN", "BELONGS_TO", "HAS_NUTRIENT"},
            )
        if plan.template_id == PRODUCT_INGREDIENTS_TEMPLATE:
            return self._relation_subgraph([plan.params["product_node_id"]], {"HAS_INGREDIENT"})
        if plan.template_id == PRODUCT_ALLERGENS_TEMPLATE:
            return self._relation_subgraph([plan.params["product_node_id"]], {"HAS_ALLERGEN"})
        if plan.template_id == PRODUCT_CATEGORY_TEMPLATE:
            return self._relation_subgraph([plan.params["product_node_id"]], {"BELONGS_TO"})
        if plan.template_id == ALLERGEN_TO_PRODUCTS_TEMPLATE:
            return self._incoming_relation_subgraph([plan.params["allergen_node_id"]], {"HAS_ALLERGEN"})
        return self.extract_subgraph(plan.start_node_ids, max_depth=max_depth)

    def _incident_edges(self, node_id: str) -> list[GraphEdge]:
        return [
            edge
            for edge in self.edges
            if edge.source_id == node_id or edge.target_id == node_id
        ]

    def _recipe_detail_subgraph(self, recipe_id: str) -> GraphSubgraph:
        base = self._relation_subgraph(
            [recipe_id],
            {"USES_INGREDIENT", "HAS_STEP", "BELONGS_TO_CUISINE", "HAS_FLAVOR"},
        )
        step_ids = [node.node_id for node in base.nodes if node.label == "Step"]
        if not step_ids:
            return base
        step_tools = self._relation_subgraph(step_ids, {"USES_TOOL"})
        return _merge_subgraphs(base, step_tools)

    def _recipe_steps_subgraph(self, recipe_id: str) -> GraphSubgraph:
        base = self._relation_subgraph([recipe_id], {"HAS_STEP"})
        step_ids = [node.node_id for node in base.nodes if node.label == "Step"]
        if not step_ids:
            return base
        return _merge_subgraphs(base, self._relation_subgraph(step_ids, {"USES_TOOL"}))

    def _recipe_tools_subgraph(self, recipe_id: str) -> GraphSubgraph:
        return self._recipe_steps_subgraph(recipe_id)

    def _relation_subgraph(
        self,
        source_ids: list[str],
        relations: set[str],
        *,
        target_ids: set[str] | None = None,
    ) -> GraphSubgraph:
        node_ids = set(source_ids)
        selected_edges: list[GraphEdge] = []
        for edge in self.edges:
            if (
                edge.source_id in source_ids
                and edge.relation in relations
                and (target_ids is None or edge.target_id in target_ids)
            ):
                selected_edges.append(edge)
                node_ids.add(edge.target_id)
        return GraphSubgraph(
            nodes=[self.nodes[node_id] for node_id in node_ids if node_id in self.nodes],
            edges=selected_edges,
        )

    def _incoming_relation_subgraph(self, target_ids: list[str], relations: set[str]) -> GraphSubgraph:
        target_id_set = set(target_ids)
        node_ids = set(target_id_set)
        selected_edges: list[GraphEdge] = []
        for edge in self.edges:
            if edge.target_id in target_id_set and edge.relation in relations:
                selected_edges.append(edge)
                node_ids.add(edge.source_id)
        return GraphSubgraph(
            nodes=[self.nodes[node_id] for node_id in node_ids if node_id in self.nodes],
            edges=selected_edges,
        )

    def _ingredient_or_category_to_recipes_subgraph(
        self,
        ingredient_ids: list[str],
        category_ids: list[str],
    ) -> GraphSubgraph:
        target_ingredient_ids = set(ingredient_ids)
        target_category_ids = set(category_ids)
        selected_edges: dict[str, GraphEdge] = {}
        node_ids = set(target_ingredient_ids) | set(target_category_ids)

        if target_category_ids:
            for edge in self.edges:
                if edge.relation != BELONGS_TO_CATEGORY or not self._category_reaches_target(
                    edge.target_id,
                    target_category_ids,
                ):
                    continue
                selected_edges[edge.edge_id] = edge
                target_ingredient_ids.add(edge.source_id)
                node_ids.add(edge.source_id)
                node_ids.add(edge.target_id)
                for path_edge in self._category_path_edges(edge.target_id, target_category_ids):
                    selected_edges[path_edge.edge_id] = path_edge
                    node_ids.add(path_edge.source_id)
                    node_ids.add(path_edge.target_id)

        for edge in self.edges:
            if edge.target_id in target_ingredient_ids and edge.relation == "USES_INGREDIENT":
                selected_edges[edge.edge_id] = edge
                node_ids.add(edge.source_id)
                node_ids.add(edge.target_id)

        return GraphSubgraph(
            nodes=[self.nodes[node_id] for node_id in node_ids if node_id in self.nodes],
            edges=list(selected_edges.values()),
        )

    def _category_reaches_target(self, category_id: str, target_ids: set[str]) -> bool:
        if category_id in target_ids:
            return True
        visited = {category_id}
        queue: deque[str] = deque([category_id])
        while queue:
            current_id = queue.popleft()
            for edge in self.edges:
                if edge.source_id != current_id or edge.relation != IS_A_CATEGORY:
                    continue
                if edge.target_id in target_ids:
                    return True
                if edge.target_id not in visited:
                    visited.add(edge.target_id)
                    queue.append(edge.target_id)
        return False

    def _category_path_edges(self, category_id: str, target_ids: set[str]) -> list[GraphEdge]:
        if category_id in target_ids:
            return []
        selected: list[GraphEdge] = []
        visited = {category_id}
        queue: deque[str] = deque([category_id])
        while queue:
            current_id = queue.popleft()
            for edge in self.edges:
                if edge.source_id != current_id or edge.relation != IS_A_CATEGORY:
                    continue
                selected.append(edge)
                if edge.target_id in target_ids:
                    return selected
                if edge.target_id not in visited:
                    visited.add(edge.target_id)
                    queue.append(edge.target_id)
        return []

    def _common_ingredients_subgraph(self, recipe_a_id: str, recipe_b_id: str) -> GraphSubgraph:
        recipe_a_edges = [
            edge
            for edge in self.edges
            if edge.source_id == recipe_a_id and edge.relation == "USES_INGREDIENT"
        ]
        recipe_b_edges = [
            edge
            for edge in self.edges
            if edge.source_id == recipe_b_id and edge.relation == "USES_INGREDIENT"
        ]
        recipe_b_targets = {edge.target_id for edge in recipe_b_edges}
        common_ids = {edge.target_id for edge in recipe_a_edges if edge.target_id in recipe_b_targets}
        selected_edges = [
            edge
            for edge in recipe_a_edges + recipe_b_edges
            if edge.target_id in common_ids
        ]
        node_ids = {recipe_a_id, recipe_b_id, *common_ids}
        return GraphSubgraph(
            nodes=[self.nodes[node_id] for node_id in node_ids if node_id in self.nodes],
            edges=selected_edges,
        )


class Neo4jGraphStore:
    # Neo4j 适配器提供真实图数据库入口；只有配置 GUSTOBOT_NEO4J_URI 时才会初始化。
    # 当前实现只允许 MATCH 读查询，不暴露任何写 Cypher，避免 LLM 或上层代码直接修改图谱。
    store_type = "neo4j"

    def __init__(self, uri: str, username: str, password: str, database: str | None = None) -> None:
        self.uri = uri
        self.database = database
        neo4j = self._ensure_driver()
        self.driver = neo4j.GraphDatabase.driver(uri, auth=(username, password))

    def find_entities(self, text: str, *, limit: int = 5) -> list[LinkedEntity]:
        keywords = _entity_keywords(text)
        query = """
        MATCH (n)
        WITH n,
             CASE
               WHEN coalesce(n.name, '') <> '' AND $text CONTAINS n.name THEN 1.0
               WHEN any(alias IN coalesce(n.aliases, []) WHERE alias <> '' AND $text CONTAINS alias) THEN 0.92
               WHEN any(keyword IN $keywords WHERE keyword <> '' AND coalesce(n.name, '') CONTAINS keyword) THEN 0.62
               WHEN any(alias IN coalesce(n.aliases, []) WHERE any(keyword IN $keywords WHERE keyword <> '' AND alias CONTAINS keyword)) THEN 0.58
               ELSE 0.0
             END AS score,
             CASE
               WHEN coalesce(n.name, '') <> '' AND $text CONTAINS n.name THEN n.name
               WHEN any(alias IN coalesce(n.aliases, []) WHERE alias <> '' AND $text CONTAINS alias)
                    THEN head([alias IN coalesce(n.aliases, []) WHERE alias <> '' AND $text CONTAINS alias])
               WHEN any(keyword IN $keywords WHERE keyword <> '' AND coalesce(n.name, '') CONTAINS keyword)
                    THEN head([keyword IN $keywords WHERE keyword <> '' AND coalesce(n.name, '') CONTAINS keyword])
               WHEN any(alias IN coalesce(n.aliases, []) WHERE any(keyword IN $keywords WHERE keyword <> '' AND alias CONTAINS keyword))
                    THEN head([alias IN coalesce(n.aliases, []) WHERE any(keyword IN $keywords WHERE keyword <> '' AND alias CONTAINS keyword)])
               ELSE coalesce(n.name, '')
             END AS matched_text,
             CASE labels(n)[0]
               WHEN 'Recipe' THEN 0
               WHEN 'Ingredient' THEN 1
               WHEN 'IngredientCategory' THEN 2
               WHEN 'Cuisine' THEN 3
               WHEN 'Flavor' THEN 4
               WHEN 'Step' THEN 5
               WHEN 'Product' THEN 6
               WHEN 'Allergen' THEN 7
               WHEN 'FoodCategory' THEN 8
               WHEN 'Nutrient' THEN 9
               ELSE 9
             END AS label_priority
        WHERE score > 0
        RETURN coalesce(n.node_id, elementId(n)) AS node_id,
               labels(n)[0] AS label,
               coalesce(n.name, '') AS name,
               coalesce(n.aliases, []) AS aliases,
               properties(n) AS properties,
               score,
               matched_text
        ORDER BY score DESC, label_priority ASC, size(coalesce(n.name, '')) DESC
        LIMIT $limit
        """
        with self.driver.session(database=self.database) as session:
            rows = session.execute_read(
                lambda tx: list(
                    tx.run(
                        query,
                        text=text,
                        keywords=keywords,
                        limit=limit,
                    )
                )
            )

        linked: list[LinkedEntity] = []
        for row in rows:
            node = GraphNode(
                node_id=row["node_id"],
                label=row["label"],
                name=row["name"],
                aliases=tuple(row["aliases"]),
                properties=dict(row["properties"]),
            )
            linked.append(LinkedEntity(node=node, matched_text=row["matched_text"] or node.name, score=float(row["score"])))
        return linked

    def extract_subgraph(self, start_node_ids: list[str], *, max_depth: int = 2) -> GraphSubgraph:
        query = """
        MATCH (n)-[r]-(m)
        WHERE coalesce(n.node_id, elementId(n)) IN $start_node_ids
        RETURN n AS start_node, r AS relationship, m AS neighbor
        LIMIT 120
        """
        with self.driver.session(database=self.database) as session:
            rows = session.execute_read(
                lambda tx: list(tx.run(query, start_node_ids=start_node_ids))
            )

        nodes: dict[str, GraphNode] = {}
        edges: dict[str, GraphEdge] = {}
        for row in rows:
            for raw_node in (row["start_node"], row["neighbor"]):
                node = self._to_graph_node(raw_node)
                nodes[node.node_id] = node
            edge = self._to_graph_edge(row["relationship"])
            edges[edge.edge_id] = edge
        return GraphSubgraph(nodes=list(nodes.values()), edges=list(edges.values()))

    def execute_plan(self, plan: GraphQueryPlan, *, max_depth: int = 2) -> GraphSubgraph:
        query, params = self._query_for_plan(plan)
        if query is None:
            return self.extract_subgraph(plan.start_node_ids, max_depth=max_depth)
        with self.driver.session(database=self.database) as session:
            rows = session.execute_read(lambda tx: list(tx.run(query, **params)))
        return self._rows_to_subgraph(rows)

    def _query_for_plan(self, plan: GraphQueryPlan) -> tuple[str | None, dict[str, object]]:
        if plan.template_id == RECIPE_INGREDIENT_AMOUNT_TEMPLATE:
            return (
                """
                MATCH (r:Recipe {node_id: $recipe_node_id})-[rel:USES_INGREDIENT]->(i:Ingredient {node_id: $ingredient_node_id})
                RETURN r AS start_node, rel AS relationship, i AS neighbor
                LIMIT 20
                """,
                {
                    "recipe_node_id": str(plan.params["recipe_node_id"]),
                    "ingredient_node_id": str(plan.params["ingredient_node_id"]),
                },
            )

        if plan.template_id == RECIPE_TOOLS_TEMPLATE:
            return (
                """
                MATCH (r:Recipe {node_id: $recipe_node_id})-[step_rel:HAS_STEP]->(s:Step)
                RETURN r AS start_node, step_rel AS relationship, s AS neighbor
                UNION
                MATCH (r:Recipe {node_id: $recipe_node_id})-[:HAS_STEP]->(s:Step)-[rel:USES_TOOL]->(n:Tool)
                RETURN s AS start_node, rel AS relationship, n AS neighbor
                LIMIT 80
                """,
                {"recipe_node_id": str(plan.params["recipe_node_id"])},
            )

        if plan.template_id == CUISINE_TO_RECIPES_TEMPLATE:
            return (
                """
                MATCH (r:Recipe)-[rel:BELONGS_TO_CUISINE]->(c:Cuisine {node_id: $cuisine_node_id})
                RETURN r AS start_node, rel AS relationship, c AS neighbor
                ORDER BY r.name
                LIMIT 80
                """,
                {"cuisine_node_id": str(plan.params["cuisine_node_id"])},
            )

        if plan.template_id in {RECIPE_DETAIL_TEMPLATE, RECIPE_INGREDIENTS_TEMPLATE, RECIPE_STEPS_TEMPLATE}:
            recipe_node_id = str(plan.params["recipe_node_id"])
            if plan.template_id == RECIPE_INGREDIENTS_TEMPLATE:
                return (
                    """
                    MATCH (r:Recipe {node_id: $recipe_node_id})-[rel:USES_INGREDIENT]->(n:Ingredient)
                    RETURN r AS start_node, rel AS relationship, n AS neighbor
                    LIMIT 80
                    """,
                    {"recipe_node_id": recipe_node_id},
                )
            if plan.template_id == RECIPE_STEPS_TEMPLATE:
                return (
                    """
                    MATCH (r:Recipe {node_id: $recipe_node_id})-[rel:HAS_STEP]->(n:Step)
                    RETURN r AS start_node, rel AS relationship, n AS neighbor
                    UNION
                    MATCH (r:Recipe {node_id: $recipe_node_id})-[:HAS_STEP]->(s:Step)-[rel:USES_TOOL]->(n:Tool)
                    RETURN s AS start_node, rel AS relationship, n AS neighbor
                    LIMIT 80
                    """,
                    {"recipe_node_id": recipe_node_id},
                )
            return (
                """
                MATCH (r:Recipe {node_id: $recipe_node_id})-[rel]->(n)
                WHERE type(rel) IN ['USES_INGREDIENT', 'HAS_STEP', 'BELONGS_TO_CUISINE', 'HAS_FLAVOR']
                RETURN r AS start_node, rel AS relationship, n AS neighbor
                UNION
                MATCH (r:Recipe {node_id: $recipe_node_id})-[:HAS_STEP]->(s:Step)-[rel:USES_TOOL]->(n:Tool)
                RETURN s AS start_node, rel AS relationship, n AS neighbor
                LIMIT 120
                """,
                {"recipe_node_id": recipe_node_id},
            )

        if plan.template_id == INGREDIENT_TO_RECIPES_TEMPLATE:
            ingredient_node_ids = [
                str(node_id)
                for node_id in (plan.params.get("ingredient_node_ids") or [])
            ]
            if not ingredient_node_ids and plan.params.get("ingredient_node_id"):
                ingredient_node_ids = [str(plan.params["ingredient_node_id"])]
            category_node_ids = [str(node_id) for node_id in (plan.params.get("category_node_ids") or [])]
            return (
                """
                MATCH (r:Recipe)-[rel:USES_INGREDIENT]->(i:Ingredient)
                WHERE i.node_id IN $ingredient_node_ids
                RETURN r AS start_node, rel AS relationship, i AS neighbor
                UNION
                MATCH (r:Recipe)-[rel:USES_INGREDIENT]->(i:Ingredient)-[cat_rel:BELONGS_TO_CATEGORY]->(c:IngredientCategory)
                WHERE c.node_id IN $category_node_ids
                   OR EXISTS {
                       MATCH (c)-[:IS_A*1..2]->(parent:IngredientCategory)
                       WHERE parent.node_id IN $category_node_ids
                   }
                RETURN r AS start_node, rel AS relationship, i AS neighbor
                UNION
                MATCH (i:Ingredient)-[rel:BELONGS_TO_CATEGORY]->(c:IngredientCategory)
                WHERE c.node_id IN $category_node_ids
                   OR EXISTS {
                       MATCH (c)-[:IS_A*1..2]->(parent:IngredientCategory)
                       WHERE parent.node_id IN $category_node_ids
                   }
                RETURN i AS start_node, rel AS relationship, c AS neighbor
                UNION
                MATCH (i:Ingredient)-[:BELONGS_TO_CATEGORY]->(c:IngredientCategory)
                MATCH path = (c)-[:IS_A*1..2]->(parent:IngredientCategory)
                WHERE parent.node_id IN $category_node_ids
                UNWIND relationships(path) AS rel
                WITH DISTINCT startNode(rel) AS start_node, rel AS relationship, endNode(rel) AS neighbor
                RETURN start_node, relationship, neighbor
                LIMIT 80
                """,
                {"ingredient_node_ids": ingredient_node_ids, "category_node_ids": category_node_ids},
            )

        if plan.template_id == COMMON_INGREDIENTS_TEMPLATE:
            return (
                """
                MATCH (r1:Recipe {node_id: $recipe_a_id})-[rel1:USES_INGREDIENT]->(i:Ingredient)
                MATCH (r2:Recipe {node_id: $recipe_b_id})-[rel2:USES_INGREDIENT]->(i)
                RETURN [r1, r2, i] AS nodes, [rel1, rel2] AS relationships
                LIMIT 80
                """,
                {
                    "recipe_a_id": str(plan.params["recipe_a_id"]),
                    "recipe_b_id": str(plan.params["recipe_b_id"]),
                },
            )

        if plan.template_id == PRODUCT_DETAIL_TEMPLATE:
            return (
                """
                MATCH (p:Product {node_id: $product_node_id})-[rel]->(n)
                WHERE type(rel) IN ['HAS_INGREDIENT', 'HAS_ALLERGEN', 'BELONGS_TO', 'HAS_NUTRIENT']
                RETURN p AS start_node, rel AS relationship, n AS neighbor
                LIMIT 120
                """,
                {"product_node_id": str(plan.params["product_node_id"])},
            )

        if plan.template_id in {PRODUCT_INGREDIENTS_TEMPLATE, PRODUCT_ALLERGENS_TEMPLATE, PRODUCT_CATEGORY_TEMPLATE}:
            relation = {
                PRODUCT_INGREDIENTS_TEMPLATE: "HAS_INGREDIENT",
                PRODUCT_ALLERGENS_TEMPLATE: "HAS_ALLERGEN",
                PRODUCT_CATEGORY_TEMPLATE: "BELONGS_TO",
            }[plan.template_id]
            target_label = {
                PRODUCT_INGREDIENTS_TEMPLATE: "Ingredient",
                PRODUCT_ALLERGENS_TEMPLATE: "Allergen",
                PRODUCT_CATEGORY_TEMPLATE: "FoodCategory",
            }[plan.template_id]
            return (
                f"""
                MATCH (p:Product {{node_id: $product_node_id}})-[rel:{relation}]->(n:{target_label})
                RETURN p AS start_node, rel AS relationship, n AS neighbor
                LIMIT 80
                """,
                {"product_node_id": str(plan.params["product_node_id"])},
            )

        if plan.template_id == ALLERGEN_TO_PRODUCTS_TEMPLATE:
            return (
                """
                MATCH (p:Product)-[rel:HAS_ALLERGEN]->(a:Allergen {node_id: $allergen_node_id})
                RETURN p AS start_node, rel AS relationship, a AS neighbor
                ORDER BY p.name
                LIMIT 80
                """,
                {"allergen_node_id": str(plan.params["allergen_node_id"])},
            )

        if plan.template_id == DIRECT_NEIGHBORS_TEMPLATE:
            return None, {}
        return None, {}

    def _rows_to_subgraph(self, rows) -> GraphSubgraph:
        nodes: dict[str, GraphNode] = {}
        edges: dict[str, GraphEdge] = {}
        for row in rows:
            if "nodes" in row and "relationships" in row:
                for raw_node in row["nodes"]:
                    node = self._to_graph_node(raw_node)
                    nodes[node.node_id] = node
                for raw_edge in row["relationships"]:
                    edge = self._to_graph_edge(raw_edge)
                    edges[edge.edge_id] = edge
                continue
            for raw_node in (row["start_node"], row["neighbor"]):
                node = self._to_graph_node(raw_node)
                nodes[node.node_id] = node
            edge = self._to_graph_edge(row["relationship"])
            edges[edge.edge_id] = edge
        return GraphSubgraph(nodes=list(nodes.values()), edges=list(edges.values()))

    def _to_graph_node(self, raw_node) -> GraphNode:
        node_id = raw_node.get("node_id", raw_node.element_id)
        return GraphNode(
            node_id=node_id,
            label=list(raw_node.labels)[0] if raw_node.labels else "Entity",
            name=raw_node.get("name", node_id),
            aliases=tuple(raw_node.get("aliases", [])),
            properties=dict(raw_node),
        )

    def _to_graph_edge(self, raw_edge) -> GraphEdge:
        source_id = raw_edge.start_node.get("node_id", raw_edge.start_node.element_id)
        target_id = raw_edge.end_node.get("node_id", raw_edge.end_node.element_id)
        edge_id = raw_edge.get("edge_id", raw_edge.element_id)
        return GraphEdge(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            relation=raw_edge.type,
            properties=dict(raw_edge),
        )

    @staticmethod
    def _ensure_driver():
        try:
            import neo4j
        except ImportError as exc:
            raise RuntimeError(
                "已配置 GUSTOBOT_NEO4J_URI，但当前环境缺少 neo4j 驱动。"
                "请先安装 neo4j，或取消该环境变量改用内存图谱。"
            ) from exc
        return neo4j


def _entity_keywords(text: str) -> list[str]:
    """从自然问题中提取少量实体候选词，用于 Neo4j 弱匹配。"""
    cleaned = text.strip()
    for marker in (
        "怎么做",
        "如何做",
        "做法",
        "制作方法",
        "烹饪方法",
        "需要哪些食材",
        "能做什么菜",
        "哪些菜",
        "食材",
        "步骤",
        "调料",
        "介绍",
        "历史",
        "文化",
        "哪些产品",
        "产品",
        "商品",
        "食品",
        "配料",
        "成分",
        "过敏原",
        "营养标签",
        "类别",
        "分类",
        "吗",
        "呢",
        "请",
        "一下",
    ):
        cleaned = cleaned.replace(marker, " ")
    tokens = [
        token
        for token in re.split(r"[\s,，。！？?；;：:、]+", cleaned)
        if len(token) >= 2
    ]
    keywords: list[str] = []
    for token in tokens:
        keywords.append(token)
    return list(dict.fromkeys(keywords))[:8]


def _fuzzy_match_entity(text: str, candidates: tuple[str, ...], *, threshold: float = 0.74) -> tuple[str, float] | None:
    cleaned_text = re.sub(r"\s+", "", text)
    best_candidate = ""
    best_score = 0.0
    for candidate in candidates:
        candidate = re.sub(r"\s+", "", candidate)
        if len(candidate) < 2:
            continue
        for window in _fuzzy_windows(cleaned_text, len(candidate)):
            score = SequenceMatcher(None, candidate, window).ratio()
            if score > best_score:
                best_candidate = candidate
                best_score = score
    if best_score < threshold or not best_candidate:
        return None
    return best_candidate, min(0.72, 0.45 + best_score * 0.3)


def _fuzzy_windows(text: str, candidate_length: int) -> list[str]:
    if not text:
        return []
    sizes = {candidate_length - 1, candidate_length, candidate_length + 1}
    windows: list[str] = []
    for size in sorted(size for size in sizes if size > 0):
        if len(text) <= size:
            windows.append(text)
            continue
        for index in range(0, len(text) - size + 1):
            windows.append(text[index : index + size])
    return windows


def _merge_subgraphs(*subgraphs: GraphSubgraph) -> GraphSubgraph:
    nodes: dict[str, GraphNode] = {}
    edges: dict[str, GraphEdge] = {}
    for subgraph in subgraphs:
        for node in subgraph.nodes:
            nodes[node.node_id] = node
        for edge in subgraph.edges:
            edges[edge.edge_id] = edge
    return GraphSubgraph(nodes=list(nodes.values()), edges=list(edges.values()))
