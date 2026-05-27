"""Clarify 反问节点模块。

这个文件实现低置信度问题的反问逻辑，也负责对尚未接入的未来链路给出安全占位提示。
这样系统不会在信息不足或链路未完成时伪造检索、图谱或数据库结果。
"""

from __future__ import annotations

from typing import Any

from app.graph.state import WorkflowState
from app.models import EvidenceSource, RouteType


# 当前主要保留这个映射扩展点；第四阶段已接入 image/file，因此暂时没有未接入的高置信度路由。
FUTURE_ROUTE_LABELS: dict[RouteType, str] = {}


def clarify_node(state: WorkflowState) -> dict[str, Any]:
    # Clarify 同时承担两种职责：低置信度反问，以及未来链路的安全占位提示。
    decision = state.get("route_decision")
    if decision and decision.route_type in FUTURE_ROUTE_LABELS and not decision.need_clarification:
        route_label = FUTURE_ROUTE_LABELS[decision.route_type]
        answer = (
            f"我已判断这个问题更适合进入 {route_label} 链路，但当前阶段还没有接入该执行链路。"
            "当前不会伪造检索、图谱或数据库结果。"
        )
        content = f"已识别未来链路：{route_label}；当前阶段需要等待对应模块接入。"
    else:
        # 问题信息不足时，反问会尽量围绕菜谱知识、关系查询和统计口径三类入口收敛。
        answer = "我需要再确认一下：你想问菜谱知识、食材关系，还是做统计分析？请补充菜名、食材或统计口径。"
        content = "问题信息不足，需要用户补充路由所需关键信息。"

    return {
        # Clarify 也输出 Evidence，保证答案生成层和日志链路始终有依据。
        "raw_answer": answer,
        "raw_evidence": [
            {
                "source_type": EvidenceSource.CLARIFY,
                "content": content,
                "score": 1.0,
                "source_id": "clarify_node",
                "metadata": {
                    "route_type": decision.route_type.value if decision else "unknown",
                    "reason": decision.reason if decision else "no_route_decision",
                },
            }
        ],
    }
