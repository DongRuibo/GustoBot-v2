"""LangGraph 主工作流模块。

这个文件是当前系统的主干，负责编译并运行完整问答流程：
Global Guardrails、Input Preprocess、Router、General/KB/GraphRAG/Text2SQL/Image/File/Clarify 节点、
Evidence Normalizer、Answer Generator、Answer Guardrails、缓存和日志追踪。
"""

from __future__ import annotations

import copy
import hashlib
import re
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from app.answer.service import AnswerGenerationInput, get_answer_generation_service
from app.cache.semantic import (
    SEMANTIC_CACHE_KEY_VERSION,
    scrub_cached_response_payload,
    semantic_cache_key,
)
from app.cache.store import get_cache_store
from app.core.config import settings
from app.core.guardrails import run_answer_guardrails, run_global_guardrails
from app.core.preprocess import preprocess_input
from app.core.route_plan import plan_parallel_routes
from app.core.router import route_question
from app.graph.state import WorkflowState
from app.graphrag.models import GraphQueryPlan
from app.graphrag.service import get_graphrag_service
from app.kb.service import get_kb_service
from app.models import ChatRequest, ChatResponse, Evidence, EvidenceSource, RouteDecision, RouteSubtask, RouteType
from app.nodes.clarify import clarify_node
from app.nodes.file import file_ingest_node
from app.nodes.general import general_node
from app.nodes.graphrag import graphrag_node
from app.nodes.image import image_understanding_node
from app.nodes.kb import kb_rag_node
from app.nodes.text2sql import text2sql_node
from app.observability.tracing import record_trace_event
from app.text2sql.service import Text2SQLPreparedQuery, Text2SQLService, get_text2sql_service


def _model_to_dict(value: Any) -> dict[str, Any]:
    # 兼容 Pydantic v1/v2，避免后续依赖版本变化影响附件序列化。
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value.dict()


def global_guardrails_node(state: WorkflowState) -> dict[str, Any]:
    # Guardrails 失败时也生成证据，便于前端展示和后续审计。
    result = run_global_guardrails(state["user_input"])
    updates: dict[str, Any] = {"guardrail_result": result, "blocked": not result.allowed}
    if not result.allowed:
        updates["raw_evidence"] = [
            {
                "source_type": EvidenceSource.GUARDRAIL,
                "content": result.reason,
                "score": 1.0,
                "source_id": "global_guardrails",
                "metadata": {"categories": result.categories},
            }
        ]
    return updates


def input_preprocess_node(state: WorkflowState) -> dict[str, Any]:
    # 预处理节点只负责输入规整，不参与业务路由判断。
    return preprocess_input(state["user_input"], state.get("attachments", []))


def router_node(state: WorkflowState) -> dict[str, Any]:
    # Router 只做决策，不直接调用数据库、图谱或 LLM。
    decision = route_question(state["normalized_input"], state.get("input_features", {}))
    record_trace_event(
        state["trace_id"],
        "router_decided",
        {
            "route_type": decision.route_type.value,
            "confidence": decision.confidence,
            "need_clarification": decision.need_clarification,
            "router_provider": decision.slots.get("router_provider"),
            "router_model": decision.slots.get("router_model"),
            "fallback_used": decision.slots.get("fallback_used"),
            "fallback_reason": decision.slots.get("fallback_reason"),
        },
    )
    return {"route_decision": decision}


def route_planner_node(state: WorkflowState) -> dict[str, Any]:
    # 多路由 Planner 只在明确复合意图时改写为 multi，普通单意图继续沿用原 Router 决策。
    decision = state["route_decision"]
    route_plan = plan_parallel_routes(
        state.get("normalized_input") or state["user_input"],
        state.get("input_features", {}),
        decision,
    )
    updates: dict[str, Any] = {"route_plan": route_plan}
    if not route_plan.is_multi:
        return updates

    planned_routes = [subtask.route_type.value for subtask in route_plan.subtasks]
    record_trace_event(
        state["trace_id"],
        "route_plan_created",
        {
            "is_multi": True,
            "reason": route_plan.reason,
            "planned_routes": planned_routes,
            "subtask_count": len(route_plan.subtasks),
        },
    )
    updates["route_decision"] = RouteDecision(
        route_type=RouteType.MULTI,
        confidence=min(0.9, max(decision.confidence, 0.72)),
        reason=route_plan.reason,
        slots={
            **decision.slots,
            "primary_route": decision.route_type.value,
            "planned_routes": planned_routes,
            "route_plan_reason": route_plan.reason,
        },
        need_clarification=False,
    )
    return updates


def semantic_cache_lookup_node(state: WorkflowState) -> dict[str, Any]:
    # 这里只处理能在业务执行前得到稳定结构化签名的链路。
    decision = state.get("route_decision")
    if decision and decision.route_type == RouteType.GRAPHRAG:
        return _graphrag_semantic_cache_lookup(state)
    if decision and decision.route_type == RouteType.TEXT2SQL:
        return _text2sql_semantic_cache_lookup(state)
    return {"semantic_cache_disabled_reason": "route_not_supported"}


def _graphrag_semantic_cache_lookup(state: WorkflowState) -> dict[str, Any]:
    disabled_reason = _semantic_cache_disabled_reason_for_state(state, RouteType.GRAPHRAG)
    if disabled_reason:
        record_trace_event(
            state["trace_id"],
            "semantic_cache_disabled",
            {"reason": disabled_reason},
        )
        return {"semantic_cache_disabled_reason": disabled_reason}

    question = state.get("normalized_input") or state["user_input"]
    plan_preview = get_graphrag_service().plan_query(question)
    updates: dict[str, Any] = {"graphrag_plan_preview": plan_preview}
    if not plan_preview.semantic_cache_allowed or plan_preview.query_plan is None:
        disabled_reason = plan_preview.semantic_cache_disabled_reason or "plan_not_cacheable"
        record_trace_event(
            state["trace_id"],
            "semantic_cache_disabled",
            {"reason": disabled_reason},
        )
        updates["semantic_cache_disabled_reason"] = disabled_reason
        return updates

    semantic_cache_key = _graphrag_semantic_cache_key(plan_preview.query_plan)
    updates.update(
        {
            "semantic_cache_key": semantic_cache_key,
            "semantic_cache_route": RouteType.GRAPHRAG.value,
            "semantic_cache_template_id": plan_preview.query_plan.template_id,
        }
    )
    cached_response = _load_cached_response_by_key(
        semantic_cache_key,
        state["trace_id"],
        cache_key_type="semantic",
        semantic_route=RouteType.GRAPHRAG.value,
        semantic_template_id=plan_preview.query_plan.template_id,
    )
    if cached_response is None:
        record_trace_event(
            state["trace_id"],
            "semantic_cache_miss",
            {
                "route_type": RouteType.GRAPHRAG.value,
                "template_id": plan_preview.query_plan.template_id,
            },
        )
        return updates

    record_trace_event(
        state["trace_id"],
        "semantic_cache_hit",
        {
            "route_type": cached_response.route_decision.route_type.value,
            "template_id": plan_preview.query_plan.template_id,
        },
    )
    updates.update(
        {
            "semantic_cache_hit": True,
            "answer": cached_response.answer,
            "route_decision": cached_response.route_decision,
            "evidences": cached_response.evidences,
        }
    )
    return updates


def _text2sql_semantic_cache_lookup(state: WorkflowState) -> dict[str, Any]:
    disabled_reason = _semantic_cache_disabled_reason_for_state(state, RouteType.TEXT2SQL)
    if disabled_reason:
        record_trace_event(
            state["trace_id"],
            "semantic_cache_disabled",
            {"route_type": RouteType.TEXT2SQL.value, "reason": disabled_reason},
        )
        return {"semantic_cache_disabled_reason": disabled_reason}

    question = state.get("normalized_input") or state["user_input"]
    service = get_text2sql_service()
    prepared = service.prepare_query(question)
    updates: dict[str, Any] = {"text2sql_prepared_query": prepared}
    if not prepared.semantic_cache_allowed:
        disabled_reason = prepared.semantic_cache_disabled_reason or "text2sql_not_cacheable"
        record_trace_event(
            state["trace_id"],
            "semantic_cache_disabled",
            {"route_type": RouteType.TEXT2SQL.value, "reason": disabled_reason},
        )
        updates["semantic_cache_disabled_reason"] = disabled_reason
        return updates

    text2sql_cache_key = _text2sql_semantic_cache_key(service, prepared)
    updates.update(
        {
            "semantic_cache_key": text2sql_cache_key,
            "semantic_cache_route": RouteType.TEXT2SQL.value,
        }
    )
    cached_response = _load_cached_response_by_key(
        text2sql_cache_key,
        state["trace_id"],
        cache_key_type="semantic",
        semantic_route=RouteType.TEXT2SQL.value,
    )
    if cached_response is None:
        record_trace_event(
            state["trace_id"],
            "semantic_cache_miss",
            {
                "route_type": RouteType.TEXT2SQL.value,
                "validated_sql": prepared.validation.sql,
            },
        )
        return updates

    record_trace_event(
        state["trace_id"],
        "semantic_cache_hit",
        {
            "route_type": cached_response.route_decision.route_type.value,
            "semantic_route": RouteType.TEXT2SQL.value,
        },
    )
    updates.update(
        {
            "semantic_cache_hit": True,
            "answer": cached_response.answer,
            "route_decision": cached_response.route_decision,
            "evidences": cached_response.evidences,
        }
    )
    return updates


def evidence_normalizer_node(state: WorkflowState) -> dict[str, Any]:
    # 不同链路都先转成统一 Evidence，答案生成层只消费这一种证据结构。
    normalized: list[Evidence] = []
    # context_evidence 用来保留图片理解等前置节点的证据，raw_evidence 则来自最终业务链路。
    evidence_items = state.get("context_evidence", []) + state.get("raw_evidence", [])
    for index, item in enumerate(evidence_items, start=1):
        payload = dict(item)
        payload.setdefault("score", 0.0)
        payload.setdefault("source_id", f"evidence-{index}")
        payload.setdefault("metadata", {})
        payload["trace_id"] = state["trace_id"]
        normalized.append(Evidence(**payload))
    return {"evidences": normalized}


def parallel_route_executor_node(state: WorkflowState) -> dict[str, Any]:
    # 并行节点只复用无副作用证据链路，避免文件入库、图片理解等链路被重复执行。
    route_plan = state.get("route_plan")
    subtasks = list(route_plan.subtasks if route_plan and route_plan.is_multi else [])
    if not subtasks:
        return {
            "raw_answer": "当前没有可执行的多路由子任务。",
            "raw_evidence": [],
        }

    max_workers = max(1, min(settings.multi_route_max_parallelism, len(subtasks)))
    timeout_seconds = max(0.1, settings.multi_route_subtask_timeout_seconds)
    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures: dict[Future[dict[str, Any]], RouteSubtask] = {}
    try:
        for subtask in subtasks:
            futures[executor.submit(_run_parallel_subtask, state, subtask)] = subtask

        results: list[dict[str, Any]] = []
        for future, subtask in futures.items():
            try:
                results.append(future.result(timeout=timeout_seconds))
            except FutureTimeoutError:
                future.cancel()
                record_trace_event(
                    state["trace_id"],
                    "parallel_subtask_failed",
                    {
                        "subtask_id": subtask.subtask_id,
                        "route_type": subtask.route_type.value,
                        "reason": "timeout",
                    },
                )
                results.append(
                    {
                        "subtask_id": subtask.subtask_id,
                        "route_type": subtask.route_type,
                        "ok": False,
                        "error": "timeout",
                        "raw_answer": "",
                        "raw_evidence": [],
                    }
                )
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    successful = [result for result in results if result.get("ok")]
    raw_evidence: list[dict[str, Any]] = []
    raw_answer_parts: list[str] = []
    executed_routes: list[str] = []
    failed_routes: list[str] = []

    for result in results:
        route_type = result["route_type"]
        route_value = route_type.value if isinstance(route_type, RouteType) else str(route_type)
        if result.get("ok"):
            executed_routes.append(route_value)
            raw_answer = str(result.get("raw_answer") or "").strip()
            if raw_answer:
                raw_answer_parts.append(f"{_multi_route_section_title(route_value)}：{raw_answer}")
            for item in result.get("raw_evidence", []):
                payload = dict(item)
                metadata = payload.setdefault("metadata", {})
                metadata["subtask_id"] = result["subtask_id"]
                metadata["subtask_route"] = route_value
                raw_evidence.append(payload)
        else:
            failed_routes.append(route_value)

    record_trace_event(
        state["trace_id"],
        "parallel_evidence_merged",
        {
            "executed_routes": executed_routes,
            "failed_routes": failed_routes,
            "evidence_count": len(raw_evidence),
        },
    )
    decision = state["route_decision"]
    return {
        "route_decision": decision.model_copy(
            update={
                "slots": {
                    **decision.slots,
                    "executed_routes": executed_routes,
                    "failed_routes": failed_routes,
                    "successful_subtask_count": len(successful),
                }
            }
        ),
        "raw_answer": "\n".join(raw_answer_parts)
        or "多路由子任务均未返回可用答案。",
        "raw_evidence": raw_evidence,
    }


def _run_parallel_subtask(state: WorkflowState, subtask: RouteSubtask) -> dict[str, Any]:
    record_trace_event(
        state["trace_id"],
        "parallel_subtask_started",
        {
            "subtask_id": subtask.subtask_id,
            "route_type": subtask.route_type.value,
            "reason": subtask.reason,
        },
    )
    sub_state = _subtask_state(state, subtask)
    try:
        if subtask.route_type == RouteType.KB:
            result = kb_rag_node(sub_state)
        elif subtask.route_type == RouteType.GRAPHRAG:
            result = graphrag_node(sub_state)
        elif subtask.route_type == RouteType.TEXT2SQL:
            result = text2sql_node(sub_state)
        else:
            raise RuntimeError(f"unsupported_parallel_route:{subtask.route_type.value}")
        raw_evidence = result.get("raw_evidence", [])
        record_trace_event(
            state["trace_id"],
            "parallel_subtask_finished",
            {
                "subtask_id": subtask.subtask_id,
                "route_type": subtask.route_type.value,
                "evidence_count": len(raw_evidence),
            },
        )
        return {
            "subtask_id": subtask.subtask_id,
            "route_type": subtask.route_type,
            "ok": True,
            "raw_answer": result.get("raw_answer", ""),
            "raw_evidence": raw_evidence,
        }
    except Exception as exc:
        record_trace_event(
            state["trace_id"],
            "parallel_subtask_failed",
            {
                "subtask_id": subtask.subtask_id,
                "route_type": subtask.route_type.value,
                "reason": str(exc)[:300],
            },
        )
        return {
            "subtask_id": subtask.subtask_id,
            "route_type": subtask.route_type,
            "ok": False,
            "error": str(exc)[:300],
            "raw_answer": "",
            "raw_evidence": [],
        }


def _subtask_state(state: WorkflowState, subtask: RouteSubtask) -> WorkflowState:
    # 子任务显式清理语义缓存字段，避免复合问题复用单路由缓存签名。
    sub_state = copy.deepcopy(state)
    sub_state["normalized_input"] = subtask.question
    sub_state["route_decision"] = RouteDecision(
        route_type=subtask.route_type,
        confidence=0.82,
        reason=subtask.reason,
        slots={"subtask_id": subtask.subtask_id, "parent_route": RouteType.MULTI.value},
        need_clarification=False,
    )
    for key in (
        "graphrag_plan_preview",
        "text2sql_prepared_query",
        "semantic_cache_key",
        "semantic_cache_route",
        "semantic_cache_template_id",
        "semantic_cache_hit",
        "semantic_cache_disabled_reason",
        "raw_answer",
        "raw_evidence",
        "evidences",
        "answer",
    ):
        sub_state.pop(key, None)
    return sub_state


def _multi_route_section_title(route_value: str) -> str:
    return {
        RouteType.KB.value: "知识库信息",
        RouteType.GRAPHRAG.value: "图谱信息",
        RouteType.TEXT2SQL.value: "统计结果",
    }.get(route_value, route_value)


def semantic_answer_cache_lookup_node(state: WorkflowState) -> dict[str, Any]:
    # KB 的安全签名依赖实际检索证据，因此放在证据归一化之后，只跳过最终答案生成。
    decision = state.get("route_decision")
    if not decision or decision.route_type != RouteType.KB:
        return {}

    disabled_reason = _semantic_cache_disabled_reason_for_state(state, RouteType.KB)
    if disabled_reason:
        record_trace_event(
            state["trace_id"],
            "semantic_cache_disabled",
            {"route_type": RouteType.KB.value, "reason": disabled_reason},
        )
        return {"semantic_cache_disabled_reason": disabled_reason}

    evidences = state.get("evidences", [])
    kb_cache_key = _kb_semantic_cache_key(evidences)
    if kb_cache_key is None:
        record_trace_event(
            state["trace_id"],
            "semantic_cache_disabled",
            {"route_type": RouteType.KB.value, "reason": "kb_empty_retrieval"},
        )
        return {"semantic_cache_disabled_reason": "kb_empty_retrieval"}

    cached_response = _load_cached_response_by_key(
        kb_cache_key,
        state["trace_id"],
        cache_key_type="semantic",
        semantic_route=RouteType.KB.value,
    )
    if cached_response is not None:
        record_trace_event(
            state["trace_id"],
            "semantic_cache_hit",
            {
                "route_type": cached_response.route_decision.route_type.value,
                "semantic_route": RouteType.KB.value,
            },
        )
        return {
            "semantic_cache_key": kb_cache_key,
            "semantic_cache_route": RouteType.KB.value,
            "semantic_cache_hit": True,
            "answer": cached_response.answer,
            "route_decision": cached_response.route_decision,
            "evidences": cached_response.evidences,
        }

    record_trace_event(
        state["trace_id"],
        "semantic_cache_miss",
        {
            "route_type": RouteType.KB.value,
            "evidence_count": len(evidences),
        },
    )
    return {
        "semantic_cache_key": kb_cache_key,
        "semantic_cache_route": RouteType.KB.value,
        "evidences": _mark_semantic_cache_miss(evidences, RouteType.KB.value),
    }


def answer_generator_node(state: WorkflowState) -> dict[str, Any]:
    # 统一答案层只消费 Evidence 和上游草稿，不直接访问 KB/Neo4j/SQL。
    decision = state.get("route_decision")
    answer = get_answer_generation_service().generate(
        AnswerGenerationInput(
            question=state.get("normalized_input") or state["user_input"],
            route_type=decision.route_type if decision else None,
            raw_answer=state.get("raw_answer", ""),
            evidences=state.get("evidences", []),
            conversation_history=state.get("conversation_history", []),
            blocked=bool(state.get("blocked")),
            trace_id=state["trace_id"],
        )
    )
    return {"answer": answer}


def answer_guardrails_node(state: WorkflowState) -> dict[str, Any]:
    # 答案生成后再做一次输出检查，防止空答案或不可信答案直接返回。
    result = run_answer_guardrails(state.get("answer", ""))
    if result.allowed:
        return {"answer_guardrail_result": result}
    return {
        "answer_guardrail_result": result,
        "answer": "抱歉，当前无法生成可靠答案。请补充更明确的问题后再试。",
    }


def _after_guardrails(state: WorkflowState) -> str:
    # 被拦截的请求跳过 Router，直接归一化 Guardrails 证据并生成拒答。
    if state.get("blocked"):
        return "evidence_normalizer"
    return "input_preprocess"


def _after_router(state: WorkflowState) -> str:
    # 第四阶段已接入 image/file；图片节点会改写输入后回到 Router，文件节点会直接执行入库。
    decision = state["route_decision"]
    if decision.need_clarification or decision.confidence < settings.route_confidence_threshold:
        return "clarify_node"
    if decision.route_type == RouteType.MULTI:
        return "parallel_route_executor"
    if decision.route_type == RouteType.IMAGE:
        return "image_understanding_node"
    if decision.route_type == RouteType.FILE:
        return "file_ingest_node"
    if decision.route_type == RouteType.GENERAL:
        return "general_node"
    if decision.route_type == RouteType.KB:
        return "kb_rag_node"
    if decision.route_type == RouteType.GRAPHRAG:
        return "semantic_cache_lookup"
    if decision.route_type == RouteType.TEXT2SQL:
        return "semantic_cache_lookup"
    return "clarify_node"


def _after_semantic_cache_lookup(state: WorkflowState) -> str:
    if state.get("semantic_cache_hit"):
        return "cached_response"
    decision = state.get("route_decision")
    if decision and decision.route_type == RouteType.TEXT2SQL:
        return "text2sql_node"
    return "graphrag_node"


def _after_semantic_answer_cache_lookup(state: WorkflowState) -> str:
    if state.get("semantic_cache_hit"):
        return "cached_response"
    return "answer_generator"


def build_workflow():
    # 这里定义主骨架；第四阶段已把图片理解、文件入库、缓存和日志追踪补到主流程周边。
    builder = StateGraph(WorkflowState)
    builder.add_node("global_guardrails", global_guardrails_node)
    builder.add_node("input_preprocess", input_preprocess_node)
    builder.add_node("router", router_node)
    builder.add_node("route_planner", route_planner_node)
    builder.add_node("semantic_cache_lookup", semantic_cache_lookup_node)
    builder.add_node("parallel_route_executor", parallel_route_executor_node)
    builder.add_node("general_node", general_node)
    builder.add_node("kb_rag_node", kb_rag_node)
    builder.add_node("graphrag_node", graphrag_node)
    builder.add_node("text2sql_node", text2sql_node)
    builder.add_node("image_understanding_node", image_understanding_node)
    builder.add_node("file_ingest_node", file_ingest_node)
    builder.add_node("clarify_node", clarify_node)
    builder.add_node("evidence_normalizer", evidence_normalizer_node)
    builder.add_node("answer_generator", answer_generator_node)
    builder.add_node("semantic_answer_cache_lookup", semantic_answer_cache_lookup_node)
    builder.add_node("answer_guardrails", answer_guardrails_node)

    builder.add_edge(START, "global_guardrails")
    # Global Guardrails 位于 Router 前面，是所有请求的第一道安全门。
    builder.add_conditional_edges(
        "global_guardrails",
        _after_guardrails,
        {
            "input_preprocess": "input_preprocess",
            "evidence_normalizer": "evidence_normalizer",
        },
    )
    builder.add_edge("input_preprocess", "router")
    builder.add_edge("router", "route_planner")
    # Router 的 route_type/confidence 决定进入普通回答还是反问/占位节点。
    builder.add_conditional_edges(
        "route_planner",
        _after_router,
        {
            "general_node": "general_node",
            "kb_rag_node": "kb_rag_node",
            "semantic_cache_lookup": "semantic_cache_lookup",
            "parallel_route_executor": "parallel_route_executor",
            "text2sql_node": "text2sql_node",
            "image_understanding_node": "image_understanding_node",
            "file_ingest_node": "file_ingest_node",
            "clarify_node": "clarify_node",
        },
    )
    builder.add_conditional_edges(
        "semantic_cache_lookup",
        _after_semantic_cache_lookup,
        {
            "cached_response": END,
            "graphrag_node": "graphrag_node",
            "text2sql_node": "text2sql_node",
        },
    )
    # 图片理解完成后重新进入 Router，这是“图片转结构化文本再路由”的关键。
    builder.add_edge("image_understanding_node", "router")
    builder.add_edge("general_node", "evidence_normalizer")
    builder.add_edge("kb_rag_node", "evidence_normalizer")
    builder.add_edge("graphrag_node", "evidence_normalizer")
    builder.add_edge("text2sql_node", "evidence_normalizer")
    builder.add_edge("parallel_route_executor", "evidence_normalizer")
    builder.add_edge("file_ingest_node", "evidence_normalizer")
    builder.add_edge("clarify_node", "evidence_normalizer")
    builder.add_edge("evidence_normalizer", "semantic_answer_cache_lookup")
    builder.add_conditional_edges(
        "semantic_answer_cache_lookup",
        _after_semantic_answer_cache_lookup,
        {
            "cached_response": END,
            "answer_generator": "answer_generator",
        },
    )
    builder.add_edge("answer_generator", "answer_guardrails")
    builder.add_edge("answer_guardrails", END)

    return builder.compile()


# 编译后的 workflow_app 可被 API、测试和脚本复用。
workflow_app = build_workflow()


def run_chat(request: ChatRequest) -> ChatResponse:
    # trace_id 在入口生成，贯穿 Evidence 和最终响应，后续日志追踪会继续复用。
    trace_id = str(uuid4())
    record_trace_event(
        trace_id,
        "request_started",
        {
            "session_id": request.session_id,
            "message": request.message,
            "attachment_count": len(request.attachments),
            "history_count": len(request.conversation_history),
        },
    )

    cached_response = _load_cached_response(request, trace_id)
    if cached_response is not None:
        record_trace_event(
            trace_id,
            "cache_hit",
            {
                "route_type": cached_response.route_decision.route_type.value,
                "cache_key_type": "exact",
            },
        )
        return cached_response

    initial_state: WorkflowState = {
        "trace_id": trace_id,
        "session_id": request.session_id,
        "user_input": request.message,
        "conversation_history": [_model_to_dict(item) for item in request.conversation_history],
        "attachments": [_model_to_dict(item) for item in request.attachments],
    }
    final_state = workflow_app.invoke(initial_state)
    # 若请求在 Router 前被拦截，需要补一个默认路由决策，保证响应结构稳定。
    route_decision = final_state.get(
        "route_decision",
        RouteDecision(
            route_type=RouteType.CLARIFY,
            confidence=0.0,
            reason="请求未进入 Router，可能已被 Guardrails 拦截。",
            slots={},
            need_clarification=True,
        ),
    )
    response = ChatResponse(
        trace_id=trace_id,
        answer=final_state["answer"],
        route_decision=route_decision,
        evidences=final_state.get("evidences", []),
        need_clarification=route_decision.need_clarification,
    )
    _store_cached_response(request, response, semantic_cache_key=final_state.get("semantic_cache_key"))
    record_trace_event(
        trace_id,
        "request_finished",
        {
            "route_type": response.route_decision.route_type.value,
            "need_clarification": response.need_clarification,
            "evidence_count": len(response.evidences),
        },
    )
    return response


def _load_cached_response(request: ChatRequest, trace_id: str) -> ChatResponse | None:
    # 只缓存无附件、无会话历史请求：历史相关问题必须每次结合当前 session 生成。
    if not settings.cache_enabled or request.attachments or request.conversation_history:
        return None

    return _load_cached_response_by_key(
        _chat_cache_key(request.message),
        trace_id,
        cache_key_type="exact",
    )


def _load_cached_response_by_key(
    cache_key: str,
    trace_id: str,
    *,
    cache_key_type: str,
    semantic_route: str | None = None,
    semantic_template_id: str | None = None,
) -> ChatResponse | None:
    payload = get_cache_store().get_json(cache_key)
    if payload is None:
        return None

    payload = copy.deepcopy(payload)
    payload["trace_id"] = trace_id
    for evidence in payload.get("evidences", []):
        evidence["trace_id"] = trace_id
        metadata = evidence.setdefault("metadata", {})
        metadata["cache_hit"] = True
        metadata["cache_key_type"] = cache_key_type
        if cache_key_type == "semantic":
            metadata["semantic_cache_key_version"] = SEMANTIC_CACHE_KEY_VERSION
            metadata["semantic_cache_route"] = semantic_route
            if semantic_template_id or metadata.get("template_id"):
                metadata["semantic_cache_template_id"] = semantic_template_id or metadata.get("template_id")
    return ChatResponse(**payload)


def _store_cached_response(
    request: ChatRequest,
    response: ChatResponse,
    *,
    semantic_cache_key: str | None = None,
) -> None:
    # 只缓存无需反问、无附件、非文件/图片的稳定回答，避免缓存副作用型请求或低置信度结果。
    if (
        not settings.cache_enabled
        or request.attachments
        or request.conversation_history
        or response.need_clarification
        or response.route_decision.route_type in {RouteType.IMAGE, RouteType.FILE}
    ):
        return

    payload = _cache_payload_from_response(response)
    cache_store = get_cache_store()
    cache_store.set_json(
        _chat_cache_key(request.message),
        payload,
        ttl_seconds=settings.cache_ttl_seconds,
    )
    if semantic_cache_key and response.route_decision.route_type in {RouteType.GRAPHRAG, RouteType.TEXT2SQL, RouteType.KB}:
        cache_store.set_json(
            semantic_cache_key,
            payload,
            ttl_seconds=settings.cache_ttl_seconds,
        )


def _chat_cache_key(message: str) -> str:
    # 当前是精确文本缓存；后续可在这里换成 embedding 相似度缓存或 Redis 语义缓存索引。
    digest = hashlib.sha256(message.strip().lower().encode("utf-8")).hexdigest()
    return f"chat:v3:{digest}"


def _cache_payload_from_response(response: ChatResponse) -> dict[str, Any]:
    return scrub_cached_response_payload(response.model_dump(mode="json"))


def _semantic_cache_disabled_reason_for_state(
    state: WorkflowState,
    expected_route_type: RouteType | None = None,
) -> str | None:
    if not settings.cache_enabled:
        return "cache_disabled"
    if state.get("attachments"):
        return "attachments_present"
    if state.get("conversation_history"):
        return "conversation_history_present"
    decision = state.get("route_decision")
    if decision is None:
        return "route_missing"
    if expected_route_type is not None and decision.route_type != expected_route_type:
        return f"route_not_{expected_route_type.value}"
    if decision.need_clarification or decision.confidence < settings.route_confidence_threshold:
        return "route_not_stable"
    return None


UNORDERED_SEMANTIC_PARAM_KEYS = {"ingredient_node_ids", "category_node_ids"}


def _graphrag_semantic_cache_key(plan: GraphQueryPlan) -> str:
    canonical_payload = {
        "route_type": RouteType.GRAPHRAG.value,
        "template_id": plan.template_id,
        "params": _canonical_graph_params(plan.params),
    }
    return semantic_cache_key(RouteType.GRAPHRAG.value, canonical_payload, discriminator=plan.template_id)


def _canonical_graph_params(params: dict[str, Any]) -> dict[str, Any]:
    canonical: dict[str, Any] = {}
    for key in sorted(params):
        value = params[key]
        if key in UNORDERED_SEMANTIC_PARAM_KEYS:
            values = value if isinstance(value, list | tuple | set) else [value]
            canonical[key] = sorted({str(item) for item in values if item})
        elif isinstance(value, list | tuple):
            canonical[key] = [str(item) for item in value if item]
        elif isinstance(value, dict):
            canonical[key] = _canonical_graph_params(value)
        elif value is not None:
            canonical[key] = str(value)
    return canonical


def _text2sql_semantic_cache_key(service: Text2SQLService, prepared: Text2SQLPreparedQuery) -> str:
    canonical_payload = {
        "route_type": RouteType.TEXT2SQL.value,
        "validated_sql": _canonical_sql(prepared.validation.sql),
        "schema_fingerprint": service.schema_fingerprint(),
        "max_rows": service.sql_validator.max_rows,
        "executor_type": service.executor.executor_type,
        "text2sql_cache_version": settings.text2sql_cache_version,
    }
    return semantic_cache_key(RouteType.TEXT2SQL.value, canonical_payload)


def _canonical_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).strip()


def _kb_semantic_cache_key(evidences: list[Evidence]) -> str | None:
    kb_evidences = [evidence for evidence in evidences if evidence.source_type == EvidenceSource.KB]
    if not kb_evidences:
        return None

    service = get_kb_service()
    provider_name = getattr(service.embedding_provider, "provider_type", service.embedding_provider.__class__.__name__)
    reranker_model = getattr(service.reranker, "model", None)
    # KB fallback 答案只使用前两个 chunk 构造正文；混合召回的尾部候选可能因词法分数轻微波动，
    # 这里用主要证据生成语义缓存签名，避免等价问法因为第三条候选不同而错过缓存。
    primary_evidences = kb_evidences[:2]
    canonical_payload = {
        "route_type": RouteType.KB.value,
        "metadata_filter": {},
        "retrieved_chunk_ids": [evidence.source_id for evidence in primary_evidences],
        "document_ids": [
            str(evidence.metadata.get("document_id"))
            for evidence in primary_evidences
            if evidence.metadata.get("document_id")
        ],
        "retrieve_top_k": service.retrieve_top_k,
        "rerank_top_k": service.rerank_top_k,
        "hybrid_retrieval_enabled": service.hybrid_retrieval_enabled,
        "lexical_top_k": service.lexical_top_k,
        "rrf_k": service.rrf_k,
        "embedding_provider": provider_name,
        "embedding_model": getattr(service.embedding_provider, "model", None),
        "embedding_dimension": service.embedding_provider.dimension,
        "reranker_type": getattr(service.reranker, "reranker_type", service.reranker.__class__.__name__),
        "reranker_model": reranker_model,
        "kb_corpus_version": settings.kb_corpus_version,
    }
    return semantic_cache_key(RouteType.KB.value, canonical_payload)


def _mark_semantic_cache_miss(evidences: list[Evidence], route: str) -> list[Evidence]:
    updated: list[Evidence] = []
    for evidence in evidences:
        payload = evidence.model_dump(mode="json")
        metadata = payload.setdefault("metadata", {})
        metadata.update(
            {
                "cache_hit": False,
                "cache_key_type": "semantic",
                "semantic_cache_key_version": SEMANTIC_CACHE_KEY_VERSION,
                "semantic_cache_route": route,
            }
        )
        updated.append(Evidence(**payload))
    return updated
