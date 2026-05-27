"""语义缓存通用工具。

这里的“语义”不是直接用自然语言相似度猜测等价问题，而是基于路由后的结构化结果
生成稳定缓存键，例如 GraphRAG 模板参数、Text2SQL 校验 SQL、KB 检索证据签名。
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any


SEMANTIC_CACHE_KEY_VERSION = "v1"

TRANSIENT_CACHE_METADATA_KEYS = (
    "cache_hit",
    "cache_key_type",
    "semantic_cache_key_version",
    "semantic_cache_route",
    "semantic_cache_template_id",
    "semantic_cache_disabled_reason",
)


def stable_json_hash(payload: dict[str, Any]) -> str:
    canonical_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def semantic_cache_key(route: str, payload: dict[str, Any], *, discriminator: str | None = None) -> str:
    suffix = f":{discriminator}" if discriminator else ""
    return f"chat:semantic:{SEMANTIC_CACHE_KEY_VERSION}:{route}{suffix}:{stable_json_hash(payload)}"


def scrub_cached_response_payload(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned = copy.deepcopy(payload)
    for evidence in cleaned.get("evidences", []):
        metadata = evidence.get("metadata")
        if not isinstance(metadata, dict):
            continue
        for key in TRANSIENT_CACHE_METADATA_KEYS:
            metadata.pop(key, None)
    return cleaned
