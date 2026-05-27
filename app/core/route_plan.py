"""多路由计划模块。

这个模块只识别明确的复合意图，把问题拆成少量可并行执行的证据型链路。
默认不处理文件入库、图片理解等副作用或前置理解链路，避免并行执行改变现有语义。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.config import settings
from app.models import RouteDecision, RoutePlan, RouteSubtask, RouteType


SUPPORTED_MULTI_ROUTES = {RouteType.KB, RouteType.GRAPHRAG, RouteType.TEXT2SQL}
COMPOSITE_MARKERS = ("并", "同时", "另外", "以及", "且", "，", ",", "；", ";")
FILE_SIDE_EFFECT_KEYWORDS = ("入库", "上传", "导入文件", "保存文件", "解析文件")
KB_INTENT_KEYWORDS = (
    "历史",
    "来历",
    "典故",
    "文化",
    "解释",
    "介绍",
    "为什么",
    "知识",
    "作用",
    "是什么意思",
    "是什么",
    "区别",
    "怎么看",
    "含义",
    "字段",
)
GRAPH_INTENT_KEYWORDS = (
    "需要哪些食材",
    "有哪些食材",
    "能做什么菜",
    "可以做什么菜",
    "可以做哪些菜",
    "哪些菜",
    "食材",
    "步骤",
    "做法",
    "怎么做",
    "如何做",
    "调料",
    "用量",
    "放多少",
    "几克",
    "几勺",
    "哪些产品",
    "哪些商品",
    "配料",
    "成分",
    "过敏原",
    "类别",
    "分类",
    "属于",
)
TEXT2SQL_INTENT_KEYWORDS = (
    "统计",
    "菜谱数量",
    "道菜谱",
    "每个菜系",
    "计数",
    "排名",
    "top",
    "趋势",
    "平均",
    "占比",
    "最大",
    "最小",
    "最高",
    "最低",
    "产品数量",
    "品牌",
    "糖分",
)
NON_SQL_AMOUNT_KEYWORDS = ("用量", "放多少", "要放多少", "需要多少", "几克", "几两", "几勺", "多少克")


@dataclass(slots=True)
class _IntentHit:
    route_type: RouteType
    position: int
    reason: str


def plan_parallel_routes(
    normalized_input: str,
    input_features: dict[str, Any],
    decision: RouteDecision,
) -> RoutePlan:
    """根据 Router 结果生成选择性多路由计划。"""
    if not settings.multi_route_enabled:
        return _single_plan("multi_route_disabled")
    if input_features.get("has_image") or input_features.get("has_file"):
        return _single_plan("attachment_requires_single_route")
    if decision.need_clarification or decision.confidence < settings.route_confidence_threshold:
        return _single_plan("route_not_stable")
    if decision.route_type not in SUPPORTED_MULTI_ROUTES:
        return _single_plan("route_not_parallelizable")

    text = normalized_input.strip()
    if not text or any(keyword in text for keyword in FILE_SIDE_EFFECT_KEYWORDS):
        return _single_plan("side_effect_intent")
    if not any(marker in text for marker in COMPOSITE_MARKERS):
        return _single_plan("no_composite_marker")

    hits = _detect_intents(text)
    if len(hits) < 2:
        return _single_plan("single_intent")

    subtasks = [
        RouteSubtask(
            subtask_id=f"multi-{hit.route_type.value}-{index}",
            route_type=hit.route_type,
            question=text,
            reason=hit.reason,
        )
        for index, hit in enumerate(hits[: max(1, settings.multi_route_max_parallelism)], start=1)
    ]
    return RoutePlan(
        is_multi=True,
        reason="问题包含多个独立证据意图，启用选择性并行多路由。",
        subtasks=subtasks,
    )


def _detect_intents(text: str) -> list[_IntentHit]:
    hits: list[_IntentHit] = []
    kb_position = _first_keyword_position(text, KB_INTENT_KEYWORDS)
    if kb_position is not None:
        hits.append(_IntentHit(RouteType.KB, kb_position, "命中知识解释/历史文化意图。"))

    graph_position = _first_keyword_position(text, GRAPH_INTENT_KEYWORDS)
    if graph_position is not None and not _looks_like_sql_category_stat(text):
        hits.append(_IntentHit(RouteType.GRAPHRAG, graph_position, "命中菜谱、食材、步骤或关系意图。"))

    sql_position = _first_keyword_position(text.lower(), TEXT2SQL_INTENT_KEYWORDS)
    if sql_position is not None and not _looks_like_ingredient_amount_question(text):
        hits.append(_IntentHit(RouteType.TEXT2SQL, sql_position, "命中统计、聚合、排名或趋势意图。"))

    deduped: dict[RouteType, _IntentHit] = {}
    for hit in hits:
        current = deduped.get(hit.route_type)
        if current is None or hit.position < current.position:
            deduped[hit.route_type] = hit
    return sorted(deduped.values(), key=lambda item: item.position)


def _first_keyword_position(text: str, keywords: tuple[str, ...]) -> int | None:
    positions = [text.find(keyword) for keyword in keywords if keyword in text]
    positions = [position for position in positions if position >= 0]
    return min(positions) if positions else None


def _looks_like_ingredient_amount_question(text: str) -> bool:
    return any(keyword in text for keyword in NON_SQL_AMOUNT_KEYWORDS) and ("里" in text or "中" in text)


def _looks_like_sql_category_stat(text: str) -> bool:
    if not any(keyword in text for keyword in ("统计", "数量", "计数", "排名", "排序", "多少")):
        return False
    return any(keyword in text for keyword in ("按分类", "分类统计", "分类下", "类别统计", "类别下", "产品数量", "品牌"))


def _single_plan(reason: str) -> RoutePlan:
    return RoutePlan(is_multi=False, reason=reason, subtasks=[])
