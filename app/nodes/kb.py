"""KB RAG 工作流节点模块。

这个文件把 LangGraph 工作流和 KB RAG 服务连接起来。
当 Router 判断问题属于知识库问答时，该节点会执行检索、rerank，并返回答案草稿和 KB Evidence。
"""

from __future__ import annotations

from typing import Any

from app.graph.state import WorkflowState
from app.kb.service import get_kb_service
from app.observability.tracing import record_trace_event


def kb_rag_node(state: WorkflowState) -> dict[str, Any]:
    # KB RAG 节点只处理普通知识库问题：先检索证据，再把证据交给统一答案生成层。
    # 当前阶段不调用大模型生成最终表述，避免在证据不足时出现“看似流畅但无来源”的答案。
    question = state.get("normalized_input") or state["user_input"]
    result = get_kb_service().query(question)
    record_trace_event(
        state["trace_id"],
        "kb_retrieval_finished",
        {
            "hybrid_enabled": result.hybrid_retrieval_enabled,
            "candidate_count": result.candidate_count,
            "retrieval_sources": result.retrieval_sources,
            "evidence_count": len(result.raw_evidence),
        },
    )
    return {
        "raw_answer": result.answer,
        "raw_evidence": result.raw_evidence,
    }
