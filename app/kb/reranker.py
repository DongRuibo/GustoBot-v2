"""知识库 reranker 模块。

这个文件负责对向量召回后的候选 chunk 做二次排序。
当前使用关键词重叠作为轻量占位实现，后续可以替换成 bge-reranker 或独立 rerank 服务。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from app.kb.embeddings import tokenize_for_retrieval
from app.kb.store import RetrievedChunk


class Reranker(Protocol):
    # Reranker 协议让 KB 服务可以在关键词重排和外部 rerank 服务之间切换。
    reranker_type: str

    def rerank(self, query: str, candidates: list[RetrievedChunk], *, top_k: int) -> list[RetrievedChunk]:
        ...

    def status(self) -> dict[str, Any]:
        ...


class KeywordReranker:
    # 第二阶段先用轻量关键词重排模拟 reranker 的位置，后续可替换为 bge-reranker 或自建 rerank 服务。
    # 注意：reranker 只处理 KB/GraphRAG 的候选证据，不参与 Text2SQL，避免把结构化查询问题复杂化。
    reranker_type = "keyword"

    def __init__(self) -> None:
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.last_success: bool | None = None
        self.last_error: str | None = None
        self.last_fallback = False

    def rerank(self, query: str, candidates: list[RetrievedChunk], *, top_k: int) -> list[RetrievedChunk]:
        self._mark_success()
        query_tokens = set(tokenize_for_retrieval(query))
        if not query_tokens:
            return [
                _annotate_chunk(
                    candidate.with_score(candidate.score, rerank_score=0.0),
                    self._evidence_metadata(),
                )
                for candidate in candidates[:top_k]
            ]

        reranked: list[RetrievedChunk] = []
        for candidate in candidates:
            candidate_tokens = set(tokenize_for_retrieval(candidate.content))
            overlap = len(query_tokens & candidate_tokens) / max(len(query_tokens), 1)
            # 向量分负责语义召回，关键词重叠负责把明显同词的菜谱知识排到更前面。
            final_score = min(1.0, max(0.0, candidate.score * 0.7 + overlap * 0.3))
            reranked.append(
                _annotate_chunk(
                    candidate.with_score(final_score, rerank_score=overlap),
                    self._evidence_metadata(),
                )
            )

        return sorted(reranked, key=lambda item: item.score, reverse=True)[:top_k]

    def status(self) -> dict[str, Any]:
        return {
            "configured_type": self.reranker_type,
            "effective_type": self.reranker_type,
            "last_success": self.last_success,
            "last_error": self.last_error,
            "last_fallback": self.last_fallback,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
        }

    def _mark_success(self) -> None:
        self.total_calls += 1
        self.successful_calls += 1
        self.last_success = True
        self.last_error = None
        self.last_fallback = False

    def _evidence_metadata(self) -> dict[str, Any]:
        return {
            "reranker_type": self.reranker_type,
            "reranker_success": True,
            "reranker_fallback": False,
        }


@dataclass(slots=True)
class HTTPReranker:
    # 外部 rerank 服务适配器，兼容常见 bge-reranker 风格接口。
    base_url: str
    endpoint: str = "/rerank"
    api_key: str | None = None
    model: str | None = None
    request_format: str = "auto"
    timeout_seconds: float = 30
    max_retries: int = 1
    fallback_on_failure: bool = True
    total_calls: int = field(default=0, init=False)
    successful_calls: int = field(default=0, init=False)
    failed_calls: int = field(default=0, init=False)
    last_success: bool | None = field(default=None, init=False)
    last_error: str | None = field(default=None, init=False)
    last_fallback: bool = field(default=False, init=False)

    reranker_type = "http"

    def rerank(self, query: str, candidates: list[RetrievedChunk], *, top_k: int) -> list[RetrievedChunk]:
        if not candidates:
            return []
        self.total_calls += 1
        try:
            scores = self._request_scores(query, candidates, top_k)
        except Exception as exc:
            self.failed_calls += 1
            self.last_success = False
            self.last_error = str(exc)[:300]
            self.last_fallback = True
            if not self.fallback_on_failure:
                raise RuntimeError(f"reranker request failed: {self.last_error}") from exc
            # 外部 reranker 不可用时降级为原始向量分，避免开发/测试主流程直接失败。
            return [
                _annotate_chunk(
                    candidate.with_score(candidate.score, rerank_score=0.0),
                    self._evidence_metadata(success=False, error=self.last_error, fallback=True),
                )
                for candidate in candidates[:top_k]
            ]

        self.successful_calls += 1
        self.last_success = True
        self.last_error = None
        self.last_fallback = False
        reranked = [
            _annotate_chunk(
                candidate.with_score(
                    min(1.0, max(0.0, score)),
                    rerank_score=score,
                ),
                self._evidence_metadata(success=True),
            )
            for candidate, score in zip(candidates, scores)
        ]
        return sorted(reranked, key=lambda item: item.score, reverse=True)[:top_k]

    def status(self) -> dict[str, Any]:
        return {
            "configured_type": self.reranker_type,
            "effective_type": "not_called"
            if self.last_success is None
            else ("http" if self.last_success else "vector_fallback"),
            "last_success": self.last_success,
            "last_error": self.last_error,
            "last_fallback": self.last_fallback,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "model": self.model,
            "endpoint": self.endpoint,
            "request_format": self._effective_request_format(),
            "max_retries": self.max_retries,
            "fallback_on_failure": self.fallback_on_failure,
        }

    def _evidence_metadata(
        self,
        *,
        success: bool,
        error: str | None = None,
        fallback: bool = False,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "reranker_type": self.reranker_type,
            "reranker_success": success,
            "reranker_fallback": fallback,
        }
        if self.model:
            metadata["reranker_model"] = self.model
        if error:
            metadata["reranker_error"] = error
        return metadata

    def _request_scores(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int,
    ) -> list[float]:
        request_format = self._effective_request_format()
        url = self._request_url(request_format)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = self._request_payload(
            query,
            [candidate.content for candidate in candidates],
            top_k=top_k,
            request_format=request_format,
        )

        response = self._post_with_retries(url, headers=headers, payload=payload)
        return _parse_rerank_scores(response.json(), candidate_count=len(candidates))

    def _post_with_retries(self, url: str, *, headers: dict[str, str], payload: dict[str, Any]) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(max(0, self.max_retries) + 1):
            try:
                response = httpx.post(url, headers=headers, json=payload, timeout=self.timeout_seconds)
                if response.status_code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                    continue
                response.raise_for_status()
                return response
            except Exception as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    raise
        raise last_exc or RuntimeError("reranker request failed")

    def _effective_request_format(self) -> str:
        configured = self.request_format.strip().lower()
        if configured not in {"", "auto"}:
            return configured
        model = (self.model or "").lower()
        if model.startswith("qwen3-rerank"):
            return "dashscope-qwen3"
        return "generic"

    def _request_url(self, request_format: str) -> str:
        base_url = self.base_url.rstrip("/")
        if request_format == "dashscope-qwen3":
            # qwen3-rerank 使用 DashScope OpenAI-compatible reranks 入口。
            if "/api/v1/services" in base_url:
                return f"{base_url.split('/api/v1/services', 1)[0]}/compatible-api/v1/reranks"
            if "/compatible-mode/v1" in base_url:
                return f"{base_url.split('/compatible-mode/v1', 1)[0]}/compatible-api/v1/reranks"
            if base_url.endswith("/compatible-api/v1"):
                return f"{base_url}/reranks"
            if base_url.endswith("/compatible-api/v1/reranks"):
                return base_url
        return f"{base_url}/{self.endpoint.lstrip('/')}"

    def _request_payload(
        self,
        query: str,
        documents: list[str],
        *,
        top_k: int,
        request_format: str,
    ) -> dict[str, Any]:
        if request_format == "dashscope-legacy":
            payload: dict[str, Any] = {
                "input": {
                    "query": query,
                    "documents": documents,
                },
                "parameters": {
                    "top_n": top_k,
                },
            }
            if self.model:
                payload["model"] = self.model
            return payload

        payload = {
            "query": query,
            "documents": documents,
            "top_n": top_k,
        }
        if self.model:
            payload["model"] = self.model
        return payload


def _parse_rerank_scores(payload: Any, *, candidate_count: int) -> list[float]:
    # 支持 {"results":[{"index":0,"relevance_score":0.9}]} 或 {"scores":[...]} 两类常见响应。
    scores = [0.0] * candidate_count
    if isinstance(payload, dict) and isinstance(payload.get("scores"), list):
        raw_scores = payload["scores"][:candidate_count]
        return [float(item) for item in raw_scores] + [0.0] * (candidate_count - len(raw_scores))

    results = None
    if isinstance(payload, dict):
        output = payload.get("output")
        if isinstance(output, dict) and isinstance(output.get("results"), list):
            results = output["results"]
        else:
            results = payload.get("results")
    else:
        results = payload
    if isinstance(results, list):
        for index, item in enumerate(results):
            if isinstance(item, dict):
                candidate_index = int(item.get("index", index))
                score = item.get("relevance_score", item.get("score", 0.0))
            else:
                candidate_index = index
                score = item
            if 0 <= candidate_index < candidate_count:
                scores[candidate_index] = float(score)
    return scores


def _annotate_chunk(chunk: RetrievedChunk, metadata: dict[str, Any]) -> RetrievedChunk:
    return RetrievedChunk(
        document_id=chunk.document_id,
        chunk_id=chunk.chunk_id,
        content=chunk.content,
        score=chunk.score,
        metadata={**chunk.metadata, **metadata},
    )
