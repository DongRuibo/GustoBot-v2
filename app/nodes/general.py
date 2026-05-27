"""General 普通回答节点模块。

这个文件处理问候、闲聊和系统能力咨询等无需查库的问题。
它会生成普通回答和对应 Evidence，保持所有链路输出结构一致。
"""

from __future__ import annotations

from typing import Any

from app.graph.state import WorkflowState
from app.models import EvidenceSource


def general_node(state: WorkflowState) -> dict[str, Any]:
    # General 节点处理问候、闲聊和系统能力咨询，不访问知识库或数据库。
    answer = (
        "你好，我是 GustoBot-v2。当前第一阶段已接通 FastAPI、LangGraph 主流程、"
        "Global Guardrails、Router、General/Clarify 节点和统一 Evidence 输出。"
    )
    return {
        # 即使是普通回答，也生成一条 Evidence，保持响应结构统一。
        "raw_answer": answer,
        "raw_evidence": [
            {
                "source_type": EvidenceSource.GENERAL,
                "content": "通用对话节点基于系统当前能力生成回复。",
                "score": 1.0,
                "source_id": "general_node",
                "metadata": {"node": "general_node"},
            }
        ],
    }
