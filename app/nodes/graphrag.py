"""GraphRAG 工作流节点模块。

这个文件把 LangGraph 主流程和 GraphRAG 服务连接起来。
当 Router 判断问题属于食材、菜谱、步骤或关系推理时，该节点会链接实体、抽取子图并返回图谱 Evidence。
"""

from __future__ import annotations

from typing import Any

from app.graph.state import WorkflowState
from app.graphrag.service import get_graphrag_service
from app.graphrag.templates import DIRECT_NEIGHBORS_TEMPLATE
from app.kb.service import get_kb_service
from app.models import RouteDecision, RouteType
from app.observability.tracing import record_trace_event


def graphrag_node(state: WorkflowState) -> dict[str, Any]:
    # GraphRAG 节点只处理关系型问题，不承担所有知识问答；历史、文化解释仍由 KB RAG 负责。
    question = state.get("normalized_input") or state["user_input"]
    result = get_graphrag_service().query(question, plan_preview=state.get("graphrag_plan_preview"))
    raw_evidence = result.raw_evidence
    _attach_cache_metadata(raw_evidence, state)
    fallback_reason = _kb_fallback_reason(raw_evidence)
    if fallback_reason:
        kb_fallback = _fallback_to_kb(state, question, graph_answer=result.answer, fallback_reason=fallback_reason)
        if kb_fallback is not None:
            return kb_fallback
    return {
        "raw_answer": result.answer,
        "raw_evidence": raw_evidence,
    }


def _fallback_to_kb(
    state: WorkflowState,
    question: str,
    *,
    graph_answer: str,
    fallback_reason: str,
) -> dict[str, Any] | None:
    # 上传文件只进入 KB，不会同步构造成 Neo4j 图谱；图谱无命中时回查 KB，保证文件入库后可继续追问。
    kb_result = get_kb_service().query(question)
    if not kb_result.raw_evidence:
        return None

    for item in kb_result.raw_evidence:
        metadata = item.setdefault("metadata", {})
        metadata["fallback_from"] = "graphrag"
        metadata["fallback_reason"] = fallback_reason
        metadata["graph_answer_preview"] = graph_answer[:200]

    record_trace_event(
        state["trace_id"],
        "graphrag_kb_fallback",
        {
            "reason": fallback_reason,
            "kb_evidence_count": len(kb_result.raw_evidence),
        },
    )
    return {
        "route_decision": _kb_fallback_decision(state.get("route_decision"), fallback_reason),
        "raw_answer": kb_result.answer,
        "raw_evidence": kb_result.raw_evidence,
    }


def _kb_fallback_reason(raw_evidence: list[dict[str, Any]]) -> str | None:
    if not raw_evidence:
        return "graph_empty_retrieval"
    metadata = raw_evidence[0].get("metadata", {})
    if metadata.get("graph_intent") == "unknown" or metadata.get("template_id") == DIRECT_NEIGHBORS_TEMPLATE:
        return "graph_low_confidence_retrieval"
    planner_confidence = metadata.get("planner_confidence")
    if isinstance(planner_confidence, (int, float)) and planner_confidence < 0.5:
        return "graph_low_confidence_retrieval"
    return None


def _kb_fallback_decision(decision: RouteDecision | None, fallback_reason: str) -> RouteDecision | None:
    if decision is None:
        return None
    return decision.model_copy(
        update={
            "route_type": RouteType.KB,
            "confidence": min(decision.confidence, 0.74),
            "reason": f"{decision.reason}；图谱未命中实体或关系，已回退到 KB 检索上传/知识库资料。",
            "slots": {
                **decision.slots,
                "fallback_from": "graphrag",
                "fallback_reason": fallback_reason,
            },
        }
    )


def _attach_cache_metadata(raw_evidence: list[dict[str, Any]], state: WorkflowState) -> None:
    if not raw_evidence:
        return
    for item in raw_evidence:
        metadata = item.setdefault("metadata", {})
        if state.get("semantic_cache_key"):
            metadata.update(
                {
                    "cache_hit": False,
                    "cache_key_type": "semantic",
                    "semantic_cache_key_version": "v1",
                    "semantic_cache_route": "graphrag",
                    "semantic_cache_template_id": state.get("semantic_cache_template_id"),
                }
            )
        if state.get("semantic_cache_disabled_reason"):
            metadata["semantic_cache_disabled_reason"] = state["semantic_cache_disabled_reason"]
