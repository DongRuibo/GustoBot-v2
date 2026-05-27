"""LangGraph 状态定义模块。

这个文件定义 WorkflowState，描述一次问答请求在各个节点之间流转时会携带哪些字段。
显式维护这些字段，可以让后续日志追踪、调试和评估更容易落地。
"""

from __future__ import annotations

from typing import Any, TypedDict

from app.graphrag.service import GraphPlanPreview
from app.models import Evidence, GuardrailResult, RouteDecision, RoutePlan


class WorkflowState(TypedDict, total=False):
    # LangGraph 节点之间通过同一个 state 传递上下文，字段保持显式便于调试和追踪。
    trace_id: str
    session_id: str | None
    user_input: str
    conversation_history: list[dict[str, Any]]
    attachments: list[dict[str, Any]]
    # 输入预处理结果。
    normalized_input: str
    input_features: dict[str, Any]
    image_understanding: dict[str, Any]
    # Guardrails 和 Router 的结构化决策。
    guardrail_result: GuardrailResult
    answer_guardrail_result: GuardrailResult
    route_decision: RouteDecision
    route_plan: RoutePlan
    # 节点先产出 raw_*，再由 Evidence Normalizer 统一规范化。
    raw_answer: str
    context_evidence: list[dict[str, Any]]
    raw_evidence: list[dict[str, Any]]
    evidences: list[Evidence]
    answer: str
    blocked: bool
    # 语义缓存只服务 GraphRAG，保存 Router 后的轻量预规划和缓存命中状态。
    graphrag_plan_preview: GraphPlanPreview
    text2sql_prepared_query: Any
    semantic_cache_key: str
    semantic_cache_route: str
    semantic_cache_template_id: str
    semantic_cache_hit: bool
    semantic_cache_disabled_reason: str
