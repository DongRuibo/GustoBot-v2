"""输入预处理模块。

这个文件负责把用户输入规整成后续 Router 更容易消费的结构，
当前主要做文本标准化和附件模态识别，图片理解和文件入库会由后续工作流节点继续处理。
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any


def normalize_text(text: str) -> str:
    # NFKC 可以统一全角/半角等形式，降低 Router 规则匹配的不稳定性。
    normalized = unicodedata.normalize("NFKC", text)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def preprocess_input(text: str, attachments: list[dict[str, Any]]) -> dict[str, Any]:
    # 预处理层只做轻量标准化，避免在 Router 前执行耗时的图片理解或文件入库。
    normalized = normalize_text(text)
    # 附件这里只提取模态特征，真正的图片理解和文件入库由对应 LangGraph 节点负责。
    attachment_types = {item.get("type") for item in attachments}
    return {
        "normalized_input": normalized,
        "input_features": {
            "has_image": "image" in attachment_types,
            "has_file": "file" in attachment_types,
            "attachment_count": len(attachments),
        },
    }
