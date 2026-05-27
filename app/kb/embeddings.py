"""知识库 embedding 模块。

这个文件定义 embedding 抽象接口，并提供本地 hash embedding 和 OpenAI-compatible embedding 两种实现。
hash 实现用于本地测试，OpenAI-compatible 实现用于接入 bge、text-embedding 或其他兼容 /v1/embeddings 的真实向量服务。
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Protocol

import httpx


class EmbeddingProvider(Protocol):
    # EmbeddingProvider 抽象出向量模型，后续可以无缝替换为 bge、OpenAI-compatible embedding 服务等。
    dimension: int

    def embed(self, text: str) -> list[float]:
        ...


class HashEmbeddingProvider:
    # 当前默认实现是确定性的 hash embedding，不依赖外部模型，适合本地测试和流程联调。
    # 它不是生产级语义向量，只用于让“切块 -> 向量入库 -> 向量检索”链路先完整跑起来。
    provider_type = "hash"
    model = "hash-embedding"

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = tokenize_for_retrieval(text)
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 else -1.0
            vector[index] += sign

        return _l2_normalize(vector)


class OpenAICompatibleEmbeddingProvider:
    # 真实 embedding provider：兼容 OpenAI /v1/embeddings 协议，也适配很多本地 bge embedding 服务。
    # 这里把调用逻辑封装在 KB 层，避免业务节点直接关心 HTTP 细节、鉴权头和响应解析。
    provider_type = "openai-compatible"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        model: str,
        dimension: int,
        timeout_seconds: float,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dimension = dimension
        self.timeout_seconds = timeout_seconds
        self._http_client = http_client

    def embed(self, text: str) -> list[float]:
        response_payload = self._post_embeddings(text)
        embedding = _extract_embedding(response_payload)
        if len(embedding) != self.dimension:
            raise RuntimeError(
                f"embedding 维度不匹配：配置为 {self.dimension}，实际返回 {len(embedding)}。"
                "请检查 GUSTOBOT_KB_EMBEDDING_DIMENSION 与模型输出维度是否一致。"
            )
        # 对向量做 L2 归一化，保证内存检索和 pgvector cosine 检索的分数语义更一致。
        return _l2_normalize(embedding)

    def _post_embeddings(self, text: str) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {"model": self.model, "input": text}
        url = f"{self.base_url}/embeddings"
        if self._http_client is not None:
            response = self._http_client.post(url, headers=headers, json=payload)
        else:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()


def tokenize_for_retrieval(text: str) -> list[str]:
    # 中文没有天然空格，这里同时保留汉字单字、连续英文/数字、以及相邻汉字 bigram。
    # 单字能提高召回，bigram 能稍微增强“宫保”“鸡丁”“麻婆”这类短语的匹配度。
    lowered = text.lower()
    basic_tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", lowered)
    chinese_chars = [item for item in basic_tokens if "\u4e00" <= item <= "\u9fff"]
    bigrams = [f"{left}{right}" for left, right in zip(chinese_chars, chinese_chars[1:])]
    return basic_tokens + bigrams


def cosine_similarity(left: list[float], right: list[float]) -> float:
    # hash embedding 已做归一化，因此点积就是余弦相似度；这里仍保留防御式长度检查。
    if len(left) != len(right) or not left:
        return 0.0
    return sum(left_item * right_item for left_item, right_item in zip(left, right))


def _extract_embedding(response_payload: dict[str, Any]) -> list[float]:
    # OpenAI-compatible 响应通常是 {"data": [{"embedding": [...]}]}，这里集中校验，报错信息更容易定位。
    data = response_payload.get("data")
    if not isinstance(data, list) or not data:
        raise RuntimeError("embedding 服务响应缺少 data[0].embedding。")
    embedding = data[0].get("embedding") if isinstance(data[0], dict) else None
    if not isinstance(embedding, list) or not embedding:
        raise RuntimeError("embedding 服务响应中的 embedding 为空或格式不正确。")
    return [float(item) for item in embedding]


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(item * item for item in vector))
    if norm == 0:
        return vector
    return [item / norm for item in vector]
