"""Text2SQL 工作流节点模块。

这个文件把 LangGraph 主流程和 Text2SQL 服务连接起来。
当 Router 判断问题属于统计、排名、聚合或趋势分析时，该节点会检索 Schema Catalog、生成并校验 SQL，
再使用只读执行器查询结构化数据并返回 SQL Evidence。
"""

from __future__ import annotations

from typing import Any

from app.graph.state import WorkflowState
from app.text2sql.service import get_text2sql_service


def text2sql_node(state: WorkflowState) -> dict[str, Any]:
    # Text2SQL 节点严禁直接执行未经校验的 SQL；真正执行逻辑全部封装在 Text2SQLService 内部。
    question = state.get("normalized_input") or state["user_input"]
    service = get_text2sql_service()
    prepared = state.get("text2sql_prepared_query") or service.prepare_query(question)
    result = service.execute_prepared(prepared)
    raw_evidence = result.raw_evidence
    _attach_cache_metadata(raw_evidence, state)
    return {
        "raw_answer": result.answer,
        "raw_evidence": raw_evidence,
    }


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
                    "semantic_cache_route": "text2sql",
                }
            )
        if state.get("semantic_cache_disabled_reason"):
            metadata["semantic_cache_disabled_reason"] = state["semantic_cache_disabled_reason"]
