"""知识库文档切块模块。

这个文件负责把一整段文档文本拆成适合 embedding 和向量检索的 KBChunk。
切块时会保留 document_id、chunk_id 和 metadata，方便后续来源引用、权限过滤和日志追踪。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from app.core.preprocess import normalize_text


@dataclass(slots=True)
class KBChunk:
    # KBChunk 是入库和检索的最小文本单元；后续无论来自 PDF、Word 还是网页，都会先转成这种结构。
    document_id: str
    chunk_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


def split_text_to_chunks(
    text: str,
    *,
    document_id: str | None = None,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    metadata: dict[str, Any] | None = None,
) -> list[KBChunk]:
    # 这里采用确定性的字符窗口切块，原因是中文菜谱资料常常没有稳定空格分词。
    # 每个窗口保留少量 overlap，可以减少答案所需信息被切到两个 chunk 中间的概率。
    # 第二阶段先保证稳定可运行；后续文件入库时可替换为按标题、段落、表格结构切块。
    normalized = normalize_text(text)
    if not normalized:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size，避免切块窗口无法向前推进。")

    base_metadata = dict(metadata or {})
    doc_id = document_id or str(uuid4())
    chunks: list[KBChunk] = []
    start = 0
    index = 0

    while start < len(normalized):
        raw_end = min(start + chunk_size, len(normalized))
        end = _prefer_sentence_boundary(normalized, start, raw_end)
        content = normalized[start:end].strip()
        if content:
            chunk_metadata = {
                **base_metadata,
                "chunk_index": index,
                "chunk_start": start,
                "chunk_end": end,
            }
            chunks.append(
                KBChunk(
                    document_id=doc_id,
                    chunk_id=f"{doc_id}:{index}",
                    content=content,
                    metadata=chunk_metadata,
                )
            )
            index += 1

        if raw_end >= len(normalized):
            break

        # 下一段从当前 end 往前回退 overlap，保证连续 chunk 之间有上下文重叠。
        start = max(end - chunk_overlap, 0)
        if start >= end:
            start = end

    return chunks


def _prefer_sentence_boundary(text: str, start: int, raw_end: int) -> int:
    # 在窗口尾部附近优先寻找中文/英文标点，让 chunk 更像自然段落，而不是生硬截断。
    # 只在窗口后 30% 范围内回退，避免为了找标点导致 chunk 过短、召回信息不足。
    if raw_end >= len(text):
        return raw_end

    min_boundary = start + max(1, int((raw_end - start) * 0.7))
    punctuation = "。！？；.!?;"
    for index in range(raw_end - 1, min_boundary - 1, -1):
        if text[index] in punctuation:
            return index + 1
    return raw_end
