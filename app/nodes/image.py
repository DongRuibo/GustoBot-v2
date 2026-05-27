"""图片理解工作流节点模块。

这个文件实现图片链路的第一步：把图片附件转换成结构化文本，并把 normalized_input 改写成可路由文本。
节点执行后会回到 Router，从而复用 KB RAG、GraphRAG、Text2SQL 等已有业务链路。
"""

from __future__ import annotations

from typing import Any

from app.graph.state import WorkflowState
from app.models import EvidenceSource
from app.multimodal.service import get_image_understanding_service


def image_understanding_node(state: WorkflowState) -> dict[str, Any]:
    # 图片节点不直接回答最终问题，而是把多模态输入变成结构化文本后重新分流。
    image_attachments = [
        attachment for attachment in state.get("attachments", []) if attachment.get("type") == "image"
    ]
    result = get_image_understanding_service().understand(state["user_input"], image_attachments)
    return {
        "normalized_input": result.reroute_text,
        "input_features": {
            "has_image": False,
            "has_file": False,
            "attachment_count": len(state.get("attachments", [])),
            "rerouted_from_image": True,
        },
        "image_understanding": {
            "dish_name": result.dish_name,
            "possible_ingredients": result.possible_ingredients,
            "cooking_state": result.cooking_state,
            "user_intent": result.user_intent,
            "structured_text": result.structured_text,
            "metadata": result.metadata,
        },
        "context_evidence": [
            {
                "source_type": EvidenceSource.IMAGE,
                "content": result.structured_text,
                "score": 0.8 if result.dish_name else 0.4,
                "source_id": "image_understanding",
                "metadata": result.metadata,
            }
        ],
    }
