"""GraphRAG 数据模型模块。

这个文件定义图谱节点、关系边、实体链接结果和 GraphRAG 查询结果。
这些结构把底层 Neo4j/内存图谱和 LangGraph 节点隔离开，方便后续替换图数据库实现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GraphIntent(str, Enum):
    # GraphIntent 是 GraphRAG 内部的细分意图，不等同于全局 route_type。
    RECIPE_DETAIL = "recipe_detail"
    RECIPE_INGREDIENTS = "recipe_ingredients"
    RECIPE_INGREDIENT_AMOUNT = "recipe_ingredient_amount"
    RECIPE_STEPS = "recipe_steps"
    RECIPE_TOOLS = "recipe_tools"
    INGREDIENT_TO_RECIPES = "ingredient_to_recipes"
    CUISINE_TO_RECIPES = "cuisine_to_recipes"
    RECIPE_COMPARE = "recipe_compare"
    PRODUCT_DETAIL = "product_detail"
    PRODUCT_INGREDIENTS = "product_ingredients"
    PRODUCT_ALLERGENS = "product_allergens"
    PRODUCT_CATEGORY = "product_category"
    ALLERGEN_TO_PRODUCTS = "allergen_to_products"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class GraphNode:
    # GraphNode 是图谱中的实体节点，label 用来区分菜谱、食材、步骤、菜系等类型。
    node_id: str
    label: str
    name: str
    aliases: tuple[str, ...] = ()
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GraphEdge:
    # GraphEdge 表示实体之间的关系，relation 使用稳定英文枚举，便于代码判断和后续 Cypher 映射。
    edge_id: str
    source_id: str
    target_id: str
    relation: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LinkedEntity:
    # LinkedEntity 是实体链接结果，score 表示当前文本与图谱节点的匹配置信度。
    node: GraphNode
    matched_text: str
    score: float


@dataclass(slots=True)
class GraphQueryPlan:
    # GraphQueryPlan 只允许选择白名单模板，避免恢复无约束多跳扩散。
    graph_intent: GraphIntent
    template_id: str
    start_node_ids: list[str]
    params: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    max_nodes: int = 40
    confidence: float = 1.0
    planner_provider: str = "rule"
    planner_model: str | None = None
    fallback_reason: str | None = None


@dataclass(slots=True)
class GraphSubgraph:
    # GraphSubgraph 是围绕命中实体抽取出来的小子图，后续会被文本化并作为 Evidence。
    nodes: list[GraphNode]
    edges: list[GraphEdge]


@dataclass(slots=True)
class GraphRAGQueryResult:
    # GraphRAGQueryResult 是图谱问答链路的内部结果，节点层会把 raw_evidence 交给统一归一化。
    answer: str
    raw_evidence: list[dict[str, Any]]
    linked_entities: list[LinkedEntity] = field(default_factory=list)
    subgraph: GraphSubgraph | None = None
