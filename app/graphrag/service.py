"""GraphRAG 服务编排模块。

这个文件把图谱问答链路串起来：实体链接、子图抽取、关系解释、Evidence 输出。
当前答案采用模板化证据摘要，后续接入 LLM 后可以把子图文本作为上下文生成更自然的回答。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Any

from app.core.config import settings
from app.graphrag.ingredient_taxonomy import (
    BELONGS_TO_CATEGORY,
    IS_A_CATEGORY,
    categories_for_ingredient,
    category_hierarchy_edges,
    category_node_data,
)
from app.graphrag.models import GraphEdge, GraphIntent, GraphNode, GraphQueryPlan, GraphRAGQueryResult, GraphSubgraph, LinkedEntity
from app.graphrag.planner import GraphIntentPlanner, LLMGraphIntentPlanner
from app.graphrag.store import GraphStore, InMemoryGraphStore, Neo4jGraphStore
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
from app.models import EvidenceSource


RELATION_LABELS = {
    "USES_INGREDIENT": "使用食材",
    "HAS_INGREDIENT": "包含配料",
    "HAS_ALLERGEN": "标注过敏原",
    "BELONGS_TO": "属于分类",
    "HAS_NUTRIENT": "包含营养素",
    "BELONGS_TO_CUISINE": "属于菜系",
    "BELONGS_TO_CATEGORY": "属于食材类别",
    "IS_A": "上位类别",
    "HAS_STEP": "包含步骤",
    "HAS_FLAVOR": "具有口味",
    "USES_TOOL": "使用工具",
}

HOWTO_KEYWORDS = ("步骤", "怎么做", "如何做", "做法", "制作方法", "烹饪方法", "怎么烧", "怎么煮", "怎么炒", "怎么炖", "怎么蒸")


@dataclass(slots=True)
class GraphAnswerContext:
    # GraphAnswerContext 保存模板回答需要的中间信息，避免把关系筛选逻辑散落在多个函数里。
    linked_entities: list[LinkedEntity]
    subgraph: GraphSubgraph
    query_plan: GraphQueryPlan


@dataclass(slots=True)
class GraphPlanPreview:
    # 预规划只做实体链接和模板选择，不执行图查询，供语义缓存提前判断。
    linked_entities: list[LinkedEntity]
    query_plan: GraphQueryPlan | None = None
    semantic_cache_allowed: bool = False
    semantic_cache_disabled_reason: str | None = None


class GraphRAGService:
    # GraphRAGService 是图谱链路入口，上层节点只调用 query，不直接操作图谱存储。
    def __init__(self, store: GraphStore, *, max_depth: int, planner: GraphIntentPlanner | LLMGraphIntentPlanner | None = None) -> None:
        self.store = store
        self.max_depth = max_depth
        self.planner = planner or LLMGraphIntentPlanner()

    def plan_query(self, question: str) -> GraphPlanPreview:
        linked_entities = self.store.find_entities(question)
        linked_entities = _filter_ambiguous_howto_category_matches(question, linked_entities)
        if not linked_entities:
            return GraphPlanPreview(
                linked_entities=[],
                semantic_cache_allowed=False,
                semantic_cache_disabled_reason="no_linked_entities",
            )

        query_plan = self.planner.plan(question, linked_entities)
        disabled_reason = _semantic_cache_disabled_reason(query_plan)
        return GraphPlanPreview(
            linked_entities=linked_entities,
            query_plan=query_plan,
            semantic_cache_allowed=disabled_reason is None,
            semantic_cache_disabled_reason=disabled_reason,
        )

    def query(self, question: str, *, plan_preview: GraphPlanPreview | None = None) -> GraphRAGQueryResult:
        preview = plan_preview or self.plan_query(question)
        if not preview.linked_entities:
            return GraphRAGQueryResult(
                answer="我没有在图谱中链接到明确的菜谱或食材实体，暂时不能基于关系图回答这个问题。",
                raw_evidence=[],
                linked_entities=[],
                subgraph=None,
            )

        if preview.query_plan is None:
            preview = self.plan_query(question)
        if preview.query_plan is None:
            return GraphRAGQueryResult(
                answer="我没有在图谱中链接到明确的菜谱或食材实体，暂时不能基于关系图回答这个问题。",
                raw_evidence=[],
                linked_entities=preview.linked_entities,
                subgraph=None,
            )

        linked_entities = preview.linked_entities
        query_plan = preview.query_plan
        subgraph = self.store.execute_plan(query_plan, max_depth=self.max_depth)
        context = GraphAnswerContext(
            linked_entities=linked_entities,
            subgraph=subgraph,
            query_plan=query_plan,
        )
        answer = self._build_answer(question, context)
        return GraphRAGQueryResult(
            answer=answer,
            raw_evidence=[self._to_raw_evidence(context)],
            linked_entities=linked_entities,
            subgraph=subgraph,
        )

    def _build_answer(self, question: str, context: GraphAnswerContext) -> str:
        # 当前阶段不用 LLM，而是根据问题意图从子图中提取确定性答案，保证每句话都有图谱依据。
        primary = context.linked_entities[0].node
        if context.query_plan.graph_intent == GraphIntent.RECIPE_COMPARE:
            common = self._common_ingredients(context.subgraph, context.query_plan.start_node_ids)
            if common:
                recipe_names = [
                    node.name
                    for node in context.subgraph.nodes
                    if node.node_id in context.query_plan.start_node_ids
                ]
                return f"{' 和 '.join(recipe_names)} 的共同食材有：{_join_names(common)}。"
            return "当前图谱没有找到这两道菜的共同食材关系。"

        if context.query_plan.graph_intent == GraphIntent.ALLERGEN_TO_PRODUCTS:
            allergen_node_id = str(context.query_plan.params["allergen_node_id"])
            products = self._incoming_neighbors(
                context.subgraph,
                allergen_node_id,
                "HAS_ALLERGEN",
                "Product",
            )
            allergen_names = self._node_names(context.subgraph, [allergen_node_id], "Allergen")
            allergen_name = allergen_names[0] if allergen_names else primary.name
            if products:
                return f"当前图谱中标注 {allergen_name} 过敏原的产品有：{_join_names(products)}。"
            return f"当前图谱没有找到标注 {allergen_name} 过敏原的产品。"

        if context.query_plan.graph_intent == GraphIntent.PRODUCT_INGREDIENTS:
            product_node_id = str(context.query_plan.params["product_node_id"])
            ingredients = self._neighbors(context.subgraph, product_node_id, "HAS_INGREDIENT", "Ingredient")
            product_name = self._product_name(context.subgraph, product_node_id, primary.name)
            if ingredients:
                return f"{product_name} 在当前图谱中的配料有：{_join_names(ingredients)}。"
            return f"当前图谱没有找到 {product_name} 的配料关系。"

        if context.query_plan.graph_intent == GraphIntent.PRODUCT_ALLERGENS:
            product_node_id = str(context.query_plan.params["product_node_id"])
            allergens = self._neighbors(context.subgraph, product_node_id, "HAS_ALLERGEN", "Allergen")
            product_name = self._product_name(context.subgraph, product_node_id, primary.name)
            if allergens:
                return f"{product_name} 在当前图谱中标注的过敏原有：{_join_names(allergens)}。"
            return f"当前图谱没有找到 {product_name} 的过敏原关系。"

        if context.query_plan.graph_intent == GraphIntent.PRODUCT_CATEGORY:
            product_node_id = str(context.query_plan.params["product_node_id"])
            categories = self._neighbors(context.subgraph, product_node_id, "BELONGS_TO", "FoodCategory")
            product_name = self._product_name(context.subgraph, product_node_id, primary.name)
            if categories:
                return f"{product_name} 在当前图谱中属于：{_join_names(categories)}。"
            return f"当前图谱没有找到 {product_name} 的分类关系。"

        if context.query_plan.graph_intent == GraphIntent.PRODUCT_DETAIL:
            product_node_id = str(context.query_plan.params["product_node_id"])
            return self._format_product_detail(context.subgraph, product_node_id, primary.name)

        if context.query_plan.graph_intent == GraphIntent.RECIPE_INGREDIENT_AMOUNT:
            details = self._ingredient_amount_details(
                context.subgraph,
                str(context.query_plan.params["recipe_node_id"]),
                str(context.query_plan.params["ingredient_node_id"]),
            )
            if details:
                recipe_names = self._node_names(
                    context.subgraph,
                    [str(context.query_plan.params["recipe_node_id"])],
                    "Recipe",
                )
                recipe_name = recipe_names[0] if recipe_names else primary.name
                return f"{recipe_name} 中这个食材的图谱用量是：{'；'.join(details)}。"
            return "当前图谱没有找到这道菜和该食材之间的用量关系。"

        if context.query_plan.graph_intent == GraphIntent.RECIPE_TOOLS:
            tools = self._recipe_tools(context.subgraph, str(context.query_plan.params["recipe_node_id"]))
            if tools:
                return f"{primary.name} 在当前图谱中涉及的工具有：{_join_names(tools)}。"
            return "当前图谱没有找到这道菜关联的工具。"

        if context.query_plan.graph_intent == GraphIntent.CUISINE_TO_RECIPES:
            cuisine_node_id = str(context.query_plan.params["cuisine_node_id"])
            recipes = self._incoming_neighbors(
                context.subgraph,
                cuisine_node_id,
                "BELONGS_TO_CUISINE",
                "Recipe",
            )
            cuisine_names = self._node_names(context.subgraph, [cuisine_node_id], "Cuisine")
            cuisine_name = cuisine_names[0] if cuisine_names else primary.name
            if recipes:
                return f"{cuisine_name} 在当前图谱中关联的菜谱有：{_join_names(recipes)}。"
            return f"当前图谱没有找到属于 {cuisine_name} 的菜谱。"

        if primary.label == "Recipe" and _has_howto_intent(question):
            return self._format_recipe_howto(primary, context.subgraph)

        if "需要哪些食材" in question or "食材" in question:
            ingredient_details = self._ingredient_details(context.subgraph, primary.node_id)
            if ingredient_details:
                return f"{primary.name} 在当前图谱中关联的主要食材有：{'；'.join(ingredient_details)}。"

        if context.query_plan.graph_intent == GraphIntent.INGREDIENT_TO_RECIPES:
            ingredient_node_ids = _as_optional_text_list(context.query_plan.params.get("ingredient_node_ids"))
            if not ingredient_node_ids and context.query_plan.params.get("ingredient_node_id"):
                ingredient_node_ids = [str(context.query_plan.params["ingredient_node_id"])]
            category_node_ids = _as_optional_text_list(context.query_plan.params.get("category_node_ids"))
            category_ingredient_ids = self._category_ingredient_ids(context.subgraph, category_node_ids)
            target_ingredient_ids = list(dict.fromkeys([*ingredient_node_ids, *category_ingredient_ids]))
            recipes = self._incoming_neighbors_for_targets(
                context.subgraph,
                target_ingredient_ids,
                "USES_INGREDIENT",
                "Recipe",
            )
            if recipes:
                ingredient_names = self._node_names(context.subgraph, target_ingredient_ids, "Ingredient")
                if category_node_ids and ingredient_names:
                    category_names = self._node_names(context.subgraph, category_node_ids, "IngredientCategory")
                    category_name = category_names[0] if category_names else primary.name
                    if primary.label == "IngredientCategory" and context.linked_entities[0].matched_text:
                        category_name = context.linked_entities[0].matched_text
                    return (
                        f"当前图谱将“{category_name}”泛化到了这些具体食材："
                        f"{'、'.join(ingredient_names)}；可以关联到这些菜谱：{_join_names(recipes)}。"
                    )
                if len(ingredient_names) > 1:
                    return (
                        "当前图谱把这个食材问题泛化到了这些具体食材："
                        f"{'、'.join(ingredient_names)}；可以关联到这些菜谱：{_join_names(recipes)}。"
                    )
                if category_node_ids:
                    category_names = self._node_names(context.subgraph, category_node_ids, "IngredientCategory")
                    category_name = category_names[0] if category_names else primary.name
                    query_term = context.linked_entities[0].matched_text if primary.label == "IngredientCategory" else primary.name
                    return f"当前图谱将“{query_term}”识别为“{category_name}”类别；可以关联到这些菜谱：{_join_names(recipes)}。"
                return f"{primary.name} 在当前图谱中可以关联到这些菜谱：{_join_names(recipes)}。"
            if category_node_ids:
                category_names = self._node_names(context.subgraph, category_node_ids, "IngredientCategory")
                category_name = category_names[0] if category_names else primary.name
                return f"当前图谱已识别“{category_name}”为食材类别，但没有找到该类别下具体食材关联的菜谱。"

        if "能做什么菜" in question or "哪些菜" in question:
            recipes = self._incoming_neighbors(
                context.subgraph,
                primary.node_id,
                "USES_INGREDIENT",
                "Recipe",
            )
            if recipes:
                return f"{primary.name} 在当前图谱中可以关联到这些菜谱：{_join_names(recipes)}。"

        if "步骤" in question or "怎么做" in question:
            step_details = self._step_details(context.subgraph, primary.node_id)
            if step_details:
                return f"{primary.name} 的图谱步骤包括：{'；'.join(step_details)}。"

        relation_summary = self._summarize_relations(context.subgraph, primary.node_id)
        if relation_summary:
            return f"围绕 {primary.name} 抽取到的图谱关系包括：{relation_summary}"
        return f"已链接到实体 {primary.name}，但当前子图中没有足够关系支撑更具体的回答。"

    def _to_raw_evidence(self, context: GraphAnswerContext) -> dict[str, Any]:
        linked = [
            {
                "node_id": entity.node.node_id,
                "name": entity.node.name,
                "label": entity.node.label,
                "matched_text": entity.matched_text,
                "score": entity.score,
            }
            for entity in context.linked_entities
        ]
        return {
            "source_type": EvidenceSource.GRAPH,
            "content": self._subgraph_to_text(context.subgraph),
            "score": context.linked_entities[0].score if context.linked_entities else 0.0,
            "source_id": "graphrag_subgraph",
            "metadata": {
                "store_type": self.store.store_type,
                "graph_intent": context.query_plan.graph_intent.value,
                "template_id": context.query_plan.template_id,
                "plan_reason": context.query_plan.reason,
                "planner_provider": context.query_plan.planner_provider,
                "planner_model": context.query_plan.planner_model,
                "planner_confidence": context.query_plan.confidence,
                "planner_fallback_reason": context.query_plan.fallback_reason,
                "linked_entities": linked,
                "node_count": len(context.subgraph.nodes),
                "edge_count": len(context.subgraph.edges),
            },
        }

    def _neighbors(
        self,
        subgraph: GraphSubgraph,
        source_id: str,
        relation: str,
        target_label: str,
    ) -> list[GraphNode]:
        node_map = {node.node_id: node for node in subgraph.nodes}
        return [
            node_map[edge.target_id]
            for edge in subgraph.edges
            if edge.source_id == source_id
            and edge.relation == relation
            and edge.target_id in node_map
            and node_map[edge.target_id].label == target_label
        ]

    def _incoming_neighbors(
        self,
        subgraph: GraphSubgraph,
        target_id: str,
        relation: str,
        source_label: str,
    ) -> list[GraphNode]:
        node_map = {node.node_id: node for node in subgraph.nodes}
        return [
            node_map[edge.source_id]
            for edge in subgraph.edges
            if edge.target_id == target_id
            and edge.relation == relation
            and edge.source_id in node_map
            and node_map[edge.source_id].label == source_label
        ]

    def _incoming_neighbors_for_targets(
        self,
        subgraph: GraphSubgraph,
        target_ids: list[str],
        relation: str,
        source_label: str,
    ) -> list[GraphNode]:
        node_map = {node.node_id: node for node in subgraph.nodes}
        target_id_set = set(target_ids)
        neighbors: dict[str, GraphNode] = {}
        for edge in subgraph.edges:
            if (
                edge.target_id in target_id_set
                and edge.relation == relation
                and edge.source_id in node_map
                and node_map[edge.source_id].label == source_label
            ):
                neighbors[edge.source_id] = node_map[edge.source_id]
        return list(neighbors.values())

    def _category_ingredient_ids(self, subgraph: GraphSubgraph, category_ids: list[str]) -> list[str]:
        if not category_ids:
            return []
        node_map = {node.node_id: node for node in subgraph.nodes}
        target_id_set = set(category_ids)
        ingredient_ids: list[str] = []
        for edge in subgraph.edges:
            if edge.relation != BELONGS_TO_CATEGORY:
                continue
            ingredient = node_map.get(edge.source_id)
            if not ingredient or ingredient.label != "Ingredient":
                continue
            if self._category_reaches_targets_in_subgraph(subgraph, edge.target_id, target_id_set):
                ingredient_ids.append(edge.source_id)
        return list(dict.fromkeys(ingredient_ids))

    def _category_reaches_targets_in_subgraph(
        self,
        subgraph: GraphSubgraph,
        category_id: str,
        target_ids: set[str],
    ) -> bool:
        if category_id in target_ids:
            return True
        children: dict[str, list[str]] = {}
        for edge in subgraph.edges:
            if edge.relation == IS_A_CATEGORY:
                children.setdefault(edge.source_id, []).append(edge.target_id)
        visited = {category_id}
        queue: deque[str] = deque([category_id])
        while queue:
            current_id = queue.popleft()
            for parent_id in children.get(current_id, []):
                if parent_id in target_ids:
                    return True
                if parent_id not in visited:
                    visited.add(parent_id)
                    queue.append(parent_id)
        return False

    def _node_names(self, subgraph: GraphSubgraph, node_ids: list[str], label: str) -> list[str]:
        node_map = {node.node_id: node for node in subgraph.nodes}
        names: list[str] = []
        for node_id in node_ids:
            node = node_map.get(node_id)
            if node and node.label == label and node.name not in names:
                names.append(node.name)
        return names

    def _summarize_relations(self, subgraph: GraphSubgraph, primary_id: str) -> str:
        node_map = {node.node_id: node for node in subgraph.nodes}
        parts: list[str] = []
        for edge in subgraph.edges:
            if edge.source_id == primary_id and edge.target_id in node_map:
                relation_label = RELATION_LABELS.get(edge.relation, edge.relation)
                parts.append(f"{relation_label}：{node_map[edge.target_id].name}")
            elif edge.target_id == primary_id and edge.source_id in node_map:
                relation_label = RELATION_LABELS.get(edge.relation, edge.relation)
                parts.append(f"{node_map[edge.source_id].name} -> {relation_label}")
        return "；".join(parts[:8])

    def _subgraph_to_text(self, subgraph: GraphSubgraph) -> str:
        # 子图文本化是 GraphRAG 的关键步骤，后续接 LLM 时可以直接把这段文本作为可引用上下文。
        node_map = {node.node_id: node for node in subgraph.nodes}
        lines: list[str] = []
        for node in subgraph.nodes:
            if node.label == "Recipe":
                lines.extend(self._recipe_evidence_lines(node, subgraph))
            if node.label == "Product":
                lines.extend(self._product_evidence_lines(node, subgraph))
        for edge in subgraph.edges:
            source = node_map.get(edge.source_id)
            target = node_map.get(edge.target_id)
            if not source or not target:
                continue
            relation_label = RELATION_LABELS.get(edge.relation, edge.relation)
            detail = self._edge_detail(edge, target)
            suffix = f"（{detail}）" if detail else ""
            lines.append(f"{source.name} -[{relation_label}]-> {target.name}{suffix}")
        return "\n".join(dict.fromkeys(lines))

    def _format_recipe_howto(self, recipe: GraphNode, subgraph: GraphSubgraph) -> str:
        summary_parts = self._recipe_summary_parts(recipe, subgraph)
        ingredients = self._ingredient_details(subgraph, recipe.node_id)
        steps = self._step_details(subgraph, recipe.node_id)
        answer_parts = [f"{recipe.name} 的图谱做法如下。"]
        if summary_parts:
            answer_parts.append("基础信息：" + "；".join(summary_parts) + "。")
        if ingredients:
            answer_parts.append("食材：" + "；".join(ingredients) + "。")
        if steps:
            answer_parts.append("步骤：" + "；".join(steps) + "。")
        if len(answer_parts) == 1:
            answer_parts.append("当前图谱只命中了菜谱实体，还没有步骤或食材关系。")
        return "\n".join(answer_parts)

    def _recipe_evidence_lines(self, recipe: GraphNode, subgraph: GraphSubgraph) -> list[str]:
        lines = [f"菜谱：{recipe.name}"]
        summary_parts = self._recipe_summary_parts(recipe, subgraph)
        if summary_parts:
            lines.append("基础信息：" + "；".join(summary_parts))
        ingredient_details = self._ingredient_details(subgraph, recipe.node_id)
        if ingredient_details:
            lines.append("食材清单：" + "；".join(ingredient_details))
        step_details = self._step_details(subgraph, recipe.node_id)
        if step_details:
            lines.append("制作步骤：" + "；".join(step_details))
        return lines

    def _recipe_summary_parts(self, recipe: GraphNode, subgraph: GraphSubgraph) -> list[str]:
        parts: list[str] = []
        description = _text_property(recipe.properties.get("description"))
        if description:
            parts.append(f"简介：{description}")
        cuisine_names = _join_names(self._neighbors(subgraph, recipe.node_id, "BELONGS_TO_CUISINE", "Cuisine"))
        if cuisine_names:
            parts.append(f"菜系：{cuisine_names}")
        total_time = _text_property(recipe.properties.get("total_time"))
        if total_time and total_time != "0":
            parts.append(f"预计耗时：{total_time}分钟")
        servings = _text_property(recipe.properties.get("servings"))
        if servings and servings != "0":
            parts.append(f"份量：{servings}人份")
        difficulty = _text_property(recipe.properties.get("difficulty"))
        if difficulty:
            parts.append(f"难度：{difficulty}")
        return parts

    def _format_product_detail(self, subgraph: GraphSubgraph, product_id: str, fallback_name: str) -> str:
        product_name = self._product_name(subgraph, product_id, fallback_name)
        parts: list[str] = []
        categories = self._neighbors(subgraph, product_id, "BELONGS_TO", "FoodCategory")
        if categories:
            parts.append(f"分类：{_join_names(categories)}")
        ingredients = self._neighbors(subgraph, product_id, "HAS_INGREDIENT", "Ingredient")
        if ingredients:
            parts.append(f"配料：{_join_names(ingredients[:8])}")
        allergens = self._neighbors(subgraph, product_id, "HAS_ALLERGEN", "Allergen")
        if allergens:
            parts.append(f"过敏原：{_join_names(allergens)}")
        nutrients = self._nutrient_details(subgraph, product_id)
        if nutrients:
            parts.append(f"营养素：{'；'.join(nutrients[:8])}")
        if parts:
            return f"{product_name} 的图谱信息包括：" + "；".join(parts) + "。"
        return f"已链接到食品商品 {product_name}，但当前子图中没有足够关系支撑更具体的回答。"

    def _product_evidence_lines(self, product: GraphNode, subgraph: GraphSubgraph) -> list[str]:
        lines = [f"食品商品：{product.name}"]
        summary_parts = self._product_summary_parts(product, subgraph)
        if summary_parts:
            lines.append("商品信息：" + "；".join(summary_parts))
        ingredients = self._neighbors(subgraph, product.node_id, "HAS_INGREDIENT", "Ingredient")
        if ingredients:
            lines.append("配料：" + _join_names(ingredients))
        allergens = self._neighbors(subgraph, product.node_id, "HAS_ALLERGEN", "Allergen")
        if allergens:
            lines.append("过敏原：" + _join_names(allergens))
        nutrients = self._nutrient_details(subgraph, product.node_id)
        if nutrients:
            lines.append("营养标签：" + "；".join(nutrients))
        return lines

    def _product_summary_parts(self, product: GraphNode, subgraph: GraphSubgraph) -> list[str]:
        parts: list[str] = []
        brand = _text_property(product.properties.get("brand"))
        if brand:
            parts.append(f"品牌：{brand}")
        country = _text_property(product.properties.get("country"))
        if country:
            parts.append(f"国家/地区：{country}")
        categories = self._neighbors(subgraph, product.node_id, "BELONGS_TO", "FoodCategory")
        if categories:
            parts.append(f"分类：{_join_names(categories)}")
        return parts

    def _product_name(self, subgraph: GraphSubgraph, product_id: str, fallback_name: str) -> str:
        node_map = {node.node_id: node for node in subgraph.nodes}
        product = node_map.get(product_id)
        return product.name if product and product.label == "Product" else fallback_name

    def _nutrient_details(self, subgraph: GraphSubgraph, product_id: str) -> list[str]:
        node_map = {node.node_id: node for node in subgraph.nodes}
        details: list[str] = []
        for edge in subgraph.edges:
            nutrient = node_map.get(edge.target_id)
            if edge.source_id != product_id or edge.relation != "HAS_NUTRIENT" or nutrient is None:
                continue
            value = _text_property(edge.properties.get("value"))
            unit = _text_property(edge.properties.get("unit"))
            if value:
                details.append(f"{nutrient.name}={value}{unit or ''}")
            else:
                details.append(nutrient.name)
        return details

    def _ingredient_details(self, subgraph: GraphSubgraph, recipe_id: str) -> list[str]:
        node_map = {node.node_id: node for node in subgraph.nodes}
        details: list[str] = []
        for edge in subgraph.edges:
            ingredient = node_map.get(edge.target_id)
            if (
                edge.source_id != recipe_id
                or edge.relation != "USES_INGREDIENT"
                or ingredient is None
                or ingredient.label != "Ingredient"
            ):
                continue
            quantity = _text_property(edge.properties.get("quantity"))
            unit = _text_property(edge.properties.get("unit"))
            prep_method = _text_property(edge.properties.get("prep_method"))
            ingredient_type = _text_property(edge.properties.get("ingredient_type"))
            parts = [ingredient.name]
            if quantity:
                parts.append(f"{quantity}{unit or ''}")
            if prep_method:
                parts.append(prep_method)
            if edge.properties.get("is_main"):
                parts.append("主料")
            elif ingredient_type:
                parts.append(ingredient_type)
            details.append(" ".join(parts))
        return details

    def _ingredient_amount_details(self, subgraph: GraphSubgraph, recipe_id: str, ingredient_id: str) -> list[str]:
        node_map = {node.node_id: node for node in subgraph.nodes}
        details: list[str] = []
        for edge in subgraph.edges:
            ingredient = node_map.get(edge.target_id)
            if (
                edge.source_id != recipe_id
                or edge.target_id != ingredient_id
                or edge.relation != "USES_INGREDIENT"
                or ingredient is None
                or ingredient.label != "Ingredient"
            ):
                continue
            quantity = _text_property(edge.properties.get("quantity"))
            unit = _text_property(edge.properties.get("unit"))
            prep_method = _text_property(edge.properties.get("prep_method"))
            parts = [ingredient.name]
            if quantity:
                parts.append(f"{quantity}{unit or ''}")
            if prep_method:
                parts.append(prep_method)
            if edge.properties.get("is_main"):
                parts.append("主料")
            details.append(" ".join(parts))
        return details

    def _recipe_tools(self, subgraph: GraphSubgraph, recipe_id: str) -> list[GraphNode]:
        node_map = {node.node_id: node for node in subgraph.nodes}
        step_ids = {
            edge.target_id
            for edge in subgraph.edges
            if edge.source_id == recipe_id and edge.relation == "HAS_STEP"
        }
        tools: dict[str, GraphNode] = {}
        for edge in subgraph.edges:
            target = node_map.get(edge.target_id)
            if edge.source_id in step_ids and edge.relation == "USES_TOOL" and target and target.label == "Tool":
                tools[target.node_id] = target
        return list(tools.values())

    def _step_details(self, subgraph: GraphSubgraph, recipe_id: str) -> list[str]:
        node_map = {node.node_id: node for node in subgraph.nodes}
        step_edges = [
            edge
            for edge in subgraph.edges
            if edge.source_id == recipe_id and edge.relation == "HAS_STEP" and edge.target_id in node_map
        ]
        ordered_edges = sorted(
            step_edges,
            key=lambda edge: _numeric_property(
                node_map[edge.target_id].properties.get("order"),
                edge.properties.get("order"),
            ),
        )
        return [self._format_step(node_map[edge.target_id], subgraph) for edge in ordered_edges]

    def _format_step(self, step: GraphNode, subgraph: GraphSubgraph) -> str:
        order = _numeric_property(step.properties.get("order"), default=0)
        action = _text_property(step.properties.get("action"))
        instruction = _text_property(step.properties.get("instruction")) or step.name
        duration = _text_property(step.properties.get("duration"))
        temperature = _text_property(step.properties.get("temperature"))
        tools = self._neighbors(subgraph, step.node_id, "USES_TOOL", "Tool")
        prefix = f"{order}. " if order else ""
        title = f"{action}：" if action else ""
        suffix_parts: list[str] = []
        if duration and duration != "0":
            suffix_parts.append(f"约{duration}分钟")
        if temperature:
            suffix_parts.append(temperature)
        if tools:
            suffix_parts.append(f"工具：{_join_names(tools)}")
        suffix = f"（{'，'.join(suffix_parts)}）" if suffix_parts else ""
        return f"{prefix}{title}{instruction}{suffix}"

    def _edge_detail(self, edge: GraphEdge, target: GraphNode) -> str:
        if edge.relation == "USES_INGREDIENT":
            quantity = _text_property(edge.properties.get("quantity"))
            unit = _text_property(edge.properties.get("unit"))
            prep_method = _text_property(edge.properties.get("prep_method"))
            parts = []
            if quantity:
                parts.append(f"用量：{quantity}{unit or ''}")
            if prep_method:
                parts.append(f"处理：{prep_method}")
            if edge.properties.get("is_main"):
                parts.append("主料")
            return "，".join(parts)
        if edge.relation == "HAS_STEP":
            instruction = _text_property(target.properties.get("instruction"))
            duration = _text_property(target.properties.get("duration"))
            temperature = _text_property(target.properties.get("temperature"))
            parts = []
            if instruction:
                parts.append(instruction)
            if duration and duration != "0":
                parts.append(f"约{duration}分钟")
            if temperature:
                parts.append(temperature)
            return "，".join(parts)
        if edge.relation == "USES_TOOL":
            usage = _text_property(edge.properties.get("usage"))
            return f"用途：{usage}" if usage else ""
        if edge.relation == "HAS_NUTRIENT":
            value = _text_property(edge.properties.get("value"))
            unit = _text_property(edge.properties.get("unit"))
            return f"数值：{value}{unit or ''}" if value else ""
        return ""

    def _common_ingredients(self, subgraph: GraphSubgraph, recipe_ids: list[str]) -> list[GraphNode]:
        if len(recipe_ids) < 2:
            return []
        node_map = {node.node_id: node for node in subgraph.nodes}
        ingredient_sets: list[set[str]] = []
        for recipe_id in recipe_ids[:2]:
            ingredient_sets.append(
                {
                    edge.target_id
                    for edge in subgraph.edges
                    if edge.source_id == recipe_id
                    and edge.relation == "USES_INGREDIENT"
                    and edge.target_id in node_map
                    and node_map[edge.target_id].label == "Ingredient"
                }
            )
        common_ids = set.intersection(*ingredient_sets) if ingredient_sets else set()
        return [node_map[node_id] for node_id in common_ids if node_id in node_map]


_service: GraphRAGService | None = None
_service_lock = Lock()


def get_graphrag_service() -> GraphRAGService:
    # 惰性初始化避免应用导入时就连接 Neo4j；只有真正进入 GraphRAG 节点时才建立服务。
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = _build_service()
    return _service


def reset_graphrag_service_for_tests(service: GraphRAGService | None = None) -> None:
    # 测试使用，用来隔离全局单例，避免不同用例之间共享图谱状态。
    global _service
    with _service_lock:
        _service = service


def _build_service() -> GraphRAGService:
    if settings.neo4j_uri:
        store: GraphStore = Neo4jGraphStore(
            settings.neo4j_uri,
            settings.neo4j_username,
            settings.neo4j_password,
            settings.neo4j_database,
        )
    else:
        if settings.strict_external_stores:
            raise RuntimeError("生产环境必须配置 GUSTOBOT_NEO4J_URI，不能退回内存图谱。")
        store = _build_seed_memory_graph()
    return GraphRAGService(store=store, max_depth=settings.graphrag_max_depth)


def _build_seed_memory_graph() -> InMemoryGraphStore:
    # 种子图谱只用于本地开发和测试，覆盖菜谱、食材、步骤、菜系和口味几类典型节点。
    store = InMemoryGraphStore()
    nodes = [
        GraphNode("recipe:gongbao", "Recipe", "宫保鸡丁", ("宫保鸡丁", "宫爆鸡丁")),
        GraphNode("recipe:mapo", "Recipe", "麻婆豆腐", ("麻婆豆腐",)),
        GraphNode("recipe:cabbage_tofu", "Recipe", "白菜炖豆腐", ("白菜炖豆腐",)),
        GraphNode("ingredient:chicken", "Ingredient", "鸡肉", ("鸡丁", "鸡胸肉")),
        GraphNode("ingredient:peanut", "Ingredient", "花生", ("花生米",)),
        GraphNode("ingredient:tofu", "Ingredient", "豆腐", ("嫩豆腐",)),
        GraphNode("ingredient:cabbage", "Ingredient", "白菜", ("大白菜",)),
        GraphNode("ingredient:pepper", "Ingredient", "辣椒", ("干辣椒",)),
        GraphNode("cuisine:sichuan", "Cuisine", "川菜", ("四川菜",)),
        GraphNode("cuisine:home", "Cuisine", "家常菜", ("家常",)),
        GraphNode("flavor:spicy", "Flavor", "麻辣", ("辣", "微辣")),
        GraphNode("step:gongbao:1", "Step", "鸡肉切丁并腌制", properties={"order": 1}),
        GraphNode("step:gongbao:2", "Step", "炒香干辣椒后下鸡丁", properties={"order": 2}),
        GraphNode("step:gongbao:3", "Step", "加入花生和调味汁收汁", properties={"order": 3}),
        GraphNode("tool:wok", "Tool", "炒锅", ("铁锅", "锅")),
        GraphNode("tool:bowl", "Tool", "调味碗", ("碗",)),
    ]
    for node in nodes:
        store.add_node(node)
    for category in category_node_data():
        store.add_node(
            GraphNode(
                str(category["node_id"]),
                str(category["label"]),
                str(category["name"]),
                tuple(str(alias) for alias in category["aliases"]),
                dict(category["properties"]),
            )
        )

    edges = [
        GraphEdge(
            "edge:gongbao:chicken",
            "recipe:gongbao",
            "ingredient:chicken",
            "USES_INGREDIENT",
            {"quantity": "250", "unit": "克", "prep_method": "切丁", "is_main": True},
        ),
        GraphEdge(
            "edge:gongbao:peanut",
            "recipe:gongbao",
            "ingredient:peanut",
            "USES_INGREDIENT",
            {"quantity": "50", "unit": "克"},
        ),
        GraphEdge(
            "edge:gongbao:pepper",
            "recipe:gongbao",
            "ingredient:pepper",
            "USES_INGREDIENT",
            {"quantity": "8", "unit": "个"},
        ),
        GraphEdge("edge:gongbao:cuisine", "recipe:gongbao", "cuisine:sichuan", "BELONGS_TO_CUISINE"),
        GraphEdge("edge:gongbao:flavor", "recipe:gongbao", "flavor:spicy", "HAS_FLAVOR"),
        GraphEdge("edge:gongbao:step1", "recipe:gongbao", "step:gongbao:1", "HAS_STEP"),
        GraphEdge("edge:gongbao:step2", "recipe:gongbao", "step:gongbao:2", "HAS_STEP"),
        GraphEdge("edge:gongbao:step3", "recipe:gongbao", "step:gongbao:3", "HAS_STEP"),
        GraphEdge("edge:gongbao:step1:bowl", "step:gongbao:1", "tool:bowl", "USES_TOOL", {"usage": "腌制鸡丁"}),
        GraphEdge("edge:gongbao:step2:wok", "step:gongbao:2", "tool:wok", "USES_TOOL", {"usage": "爆香和翻炒"}),
        GraphEdge("edge:mapo:tofu", "recipe:mapo", "ingredient:tofu", "USES_INGREDIENT"),
        GraphEdge("edge:mapo:pepper", "recipe:mapo", "ingredient:pepper", "USES_INGREDIENT"),
        GraphEdge("edge:mapo:cuisine", "recipe:mapo", "cuisine:sichuan", "BELONGS_TO_CUISINE"),
        GraphEdge("edge:mapo:flavor", "recipe:mapo", "flavor:spicy", "HAS_FLAVOR"),
        GraphEdge("edge:cabbage:tofu", "recipe:cabbage_tofu", "ingredient:tofu", "USES_INGREDIENT"),
        GraphEdge("edge:cabbage:cabbage", "recipe:cabbage_tofu", "ingredient:cabbage", "USES_INGREDIENT"),
        GraphEdge("edge:cabbage:cuisine", "recipe:cabbage_tofu", "cuisine:home", "BELONGS_TO_CUISINE"),
    ]
    for edge in edges:
        store.add_edge(edge)
    for edge in category_hierarchy_edges():
        store.add_edge(
            GraphEdge(
                str(edge["edge_id"]),
                str(edge["source_id"]),
                str(edge["target_id"]),
                str(edge["relation"]),
                dict(edge["properties"]),
            )
        )
    for node in list(store.nodes.values()):
        if node.label != "Ingredient":
            continue
        for category in categories_for_ingredient(node.name, _text_property(node.properties.get("category"))):
            store.add_edge(
                GraphEdge(
                    f"edge:{node.node_id}:category:{category.slug}",
                    node.node_id,
                    category.node_id,
                    BELONGS_TO_CATEGORY,
                    {"source": "taxonomy"},
                )
            )
    return store


def _join_names(nodes: list[GraphNode]) -> str:
    return "、".join(node.name for node in nodes)


def _has_howto_intent(question: str) -> bool:
    return any(keyword in question for keyword in HOWTO_KEYWORDS)


def _as_optional_text_list(value: Any) -> list[str]:
    if isinstance(value, list | tuple):
        return [str(item) for item in value if item]
    if value:
        return [str(value)]
    return []


SEMANTIC_CACHEABLE_TEMPLATES = {
    RECIPE_DETAIL_TEMPLATE,
    RECIPE_INGREDIENTS_TEMPLATE,
    RECIPE_INGREDIENT_AMOUNT_TEMPLATE,
    RECIPE_STEPS_TEMPLATE,
    RECIPE_TOOLS_TEMPLATE,
    INGREDIENT_TO_RECIPES_TEMPLATE,
    CUISINE_TO_RECIPES_TEMPLATE,
    COMMON_INGREDIENTS_TEMPLATE,
    PRODUCT_DETAIL_TEMPLATE,
    PRODUCT_INGREDIENTS_TEMPLATE,
    PRODUCT_ALLERGENS_TEMPLATE,
    PRODUCT_CATEGORY_TEMPLATE,
    ALLERGEN_TO_PRODUCTS_TEMPLATE,
}
MIN_SEMANTIC_CACHE_CONFIDENCE = 0.8


def _semantic_cache_disabled_reason(plan: GraphQueryPlan) -> str | None:
    if plan.template_id == DIRECT_NEIGHBORS_TEMPLATE or plan.graph_intent == GraphIntent.UNKNOWN:
        return "unstable_template"
    if plan.template_id not in SEMANTIC_CACHEABLE_TEMPLATES:
        return "template_not_cacheable"
    if plan.confidence < MIN_SEMANTIC_CACHE_CONFIDENCE:
        return "low_planner_confidence"
    if plan.fallback_reason or plan.planner_provider == "rule_fallback":
        return "planner_fallback"
    if not plan.start_node_ids or not plan.params:
        return "incomplete_plan_params"
    return None


def _filter_ambiguous_howto_category_matches(question: str, linked_entities: list[LinkedEntity]) -> list[LinkedEntity]:
    if not linked_entities or not _has_howto_intent(question):
        return linked_entities
    if any(entity.node.label == "Recipe" for entity in linked_entities):
        return linked_entities
    return [
        entity
        for entity in linked_entities
        if not (entity.node.label == "IngredientCategory" and len(entity.matched_text.strip()) <= 1)
    ]


def _text_property(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _numeric_property(*values: Any, default: int = 0) -> int:
    for value in values:
        if value in (None, ""):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return default
