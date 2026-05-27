"""KB 混合召回工具模块。

这里放置与具体存储无关的词法检索和 RRF 融合逻辑，让内存 store 与
PostgreSQL/pgvector store 能复用同一套候选合并规则。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from app.kb.embeddings import tokenize_for_retrieval


_SAFE_TSQUERY_TOKEN = re.compile(r"^[a-z0-9\u4e00-\u9fff]+$")


def build_search_text(content: str, metadata: dict[str, Any] | None = None) -> str:
    """把正文和关键 metadata 转成 PostgreSQL FTS 友好的空格分隔 token 文本。"""
    metadata = metadata or {}
    parts = [
        str(metadata.get("title") or ""),
        str(metadata.get("source_id") or ""),
        str(metadata.get("filename") or ""),
        content,
    ]
    tokens = tokenize_for_retrieval(" ".join(part for part in parts if part))
    return " ".join(tokens)


def lexical_query_tokens(query_text: str, *, max_tokens: int = 32) -> list[str]:
    """生成词法召回 query token，并限制长度避免构造过大的 tsquery。"""
    tokens: list[str] = []
    for token in tokenize_for_retrieval(query_text):
        if token in tokens or not _SAFE_TSQUERY_TOKEN.fullmatch(token):
            continue
        tokens.append(token)
        if len(tokens) >= max_tokens:
            break
    return tokens


def postgres_tsquery(query_text: str) -> str | None:
    """构造 simple 配置下的 OR tsquery；没有可用 token 时返回 None。"""
    tokens = lexical_query_tokens(query_text)
    if not tokens:
        return None
    return " | ".join(tokens)


def bm25_scores(query_text: str, documents: dict[str, str]) -> dict[str, float]:
    """对内存 store 的 token 化文本计算轻量 BM25 分数。"""
    query_tokens = lexical_query_tokens(query_text)
    if not query_tokens or not documents:
        return {}

    tokenized_docs = {
        doc_id: search_text.split()
        for doc_id, search_text in documents.items()
        if search_text.strip()
    }
    if not tokenized_docs:
        return {}

    doc_count = len(tokenized_docs)
    avg_doc_len = sum(len(tokens) for tokens in tokenized_docs.values()) / max(doc_count, 1)
    doc_freq: Counter[str] = Counter()
    for tokens in tokenized_docs.values():
        doc_freq.update(set(tokens))

    query_terms = set(query_tokens)
    scores: dict[str, float] = {}
    k1 = 1.5
    b = 0.75
    for doc_id, tokens in tokenized_docs.items():
        frequencies = Counter(tokens)
        doc_len = len(tokens)
        score = 0.0
        for term in query_terms:
            tf = frequencies.get(term, 0)
            if tf == 0:
                continue
            idf = math.log(1 + (doc_count - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            denominator = tf + k1 * (1 - b + b * doc_len / max(avg_doc_len, 1.0))
            score += idf * (tf * (k1 + 1)) / denominator
        if score > 0:
            scores[doc_id] = score
    return scores


def reciprocal_rank_fusion(
    vector_results: list[Any],
    lexical_results: list[Any],
    *,
    top_k: int,
    rrf_k: int,
) -> list[Any]:
    """按 chunk_id 对向量和词法结果去重，并用 RRF 生成 reranker 前候选分。"""
    merged: dict[str, dict[str, Any]] = {}

    for rank, chunk in enumerate(vector_results, start=1):
        entry = merged.setdefault(chunk.chunk_id, {"chunk": chunk, "metadata": dict(chunk.metadata), "rrf": 0.0})
        entry["rrf"] += 1 / (rrf_k + rank)
        entry["metadata"].update(
            {
                "vector_score": chunk.score,
                "vector_rank": rank,
            }
        )

    for rank, chunk in enumerate(lexical_results, start=1):
        entry = merged.setdefault(chunk.chunk_id, {"chunk": chunk, "metadata": dict(chunk.metadata), "rrf": 0.0})
        entry["rrf"] += 1 / (rrf_k + rank)
        entry["metadata"].update(
            {
                "lexical_score": chunk.score,
                "lexical_rank": rank,
            }
        )

    if not merged:
        return []

    max_rrf = 2 / (rrf_k + 1)
    fused = []
    for entry in merged.values():
        chunk = entry["chunk"]
        metadata = entry["metadata"]
        sources = []
        if "vector_rank" in metadata:
            sources.append("vector")
        if "lexical_rank" in metadata:
            sources.append("lexical")
        metadata["retrieval_sources"] = sources
        metadata["retrieval_mode"] = "hybrid" if len(sources) > 1 else sources[0]
        metadata["rrf_score"] = entry["rrf"]
        fused_score = min(1.0, entry["rrf"] / max_rrf)
        fused.append(
            chunk.__class__(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                score=fused_score,
                metadata=metadata,
            )
        )

    return sorted(fused, key=lambda item: item.score, reverse=True)[:top_k]


def annotate_vector_results(results: list[Any]) -> list[Any]:
    """为向量单召回结果补充与 hybrid 一致的调试 metadata。"""
    annotated = []
    for rank, chunk in enumerate(results, start=1):
        metadata = dict(chunk.metadata)
        metadata.update(
            {
                "retrieval_mode": "vector",
                "retrieval_sources": ["vector"],
                "vector_score": chunk.score,
                "vector_rank": rank,
            }
        )
        annotated.append(
            chunk.__class__(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                score=chunk.score,
                metadata=metadata,
            )
        )
    return annotated
