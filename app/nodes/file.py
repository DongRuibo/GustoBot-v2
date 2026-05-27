"""文件入库工作流节点模块。

这个文件处理 file 路由，把附件文本解析成 ParsedFile，再复用 KB RAG 的入库服务完成切块和向量写入。
文件节点返回入库结果 Evidence，后续用户可以继续通过 KB RAG 查询这些资料。
"""

from __future__ import annotations

from typing import Any

from app.files.parser import UnsupportedFileTypeError, get_file_parser
from app.graph.state import WorkflowState
from app.kb.service import get_kb_service
from app.models import EvidenceSource


def file_ingest_node(state: WorkflowState) -> dict[str, Any]:
    # 文件节点只负责入库，不直接基于文件内容回答复杂问题；入库后的问答仍走 KB RAG。
    parser = get_file_parser()
    kb_service = get_kb_service()
    parsed_files = []
    ingest_results = []
    unsupported_errors = []

    for attachment in state.get("attachments", []):
        if attachment.get("type") != "file":
            continue
        try:
            parsed = parser.parse(attachment)
        except UnsupportedFileTypeError as exc:
            unsupported_errors.append(str(exc))
            continue
        if parsed is None:
            continue
        parsed_files.append(parsed)
        result = kb_service.ingest_document(
            title=parsed.title,
            content=parsed.content,
            metadata=parsed.metadata,
            source_id=f"file:{parsed.filename}",
        )
        ingest_results.append(result)

    if not ingest_results:
        if unsupported_errors:
            answer = "文件格式不支持：" + "；".join(unsupported_errors)
            return {
                "raw_answer": answer,
                "raw_evidence": [
                    {
                        "source_type": EvidenceSource.FILE,
                        "content": answer,
                        "score": 0.0,
                        "source_id": "file_ingest",
                        "metadata": {
                            "file_count": len(state.get("attachments", [])),
                            "unsupported_errors": unsupported_errors,
                        },
                    }
                ],
            }
        return {
            "raw_answer": "没有解析到可入库的文件文本，请上传带文本内容的文件或先完成文件解析。",
            "raw_evidence": [
                {
                    "source_type": EvidenceSource.FILE,
                    "content": "文件附件缺少可解析文本，未执行 KB 入库。",
                    "score": 0.0,
                    "source_id": "file_ingest",
                    "metadata": {"file_count": len(state.get("attachments", []))},
                }
            ],
        }

    total_chunks = sum(result.chunk_count for result in ingest_results)
    filenames = [parsed.filename for parsed in parsed_files]
    answer = f"已完成文件入库：{', '.join(filenames)}，共生成 {total_chunks} 个知识库 chunk。"
    return {
        "raw_answer": answer,
        "raw_evidence": [
            {
                "source_type": EvidenceSource.FILE,
                "content": answer,
                "score": 1.0,
                "source_id": "file_ingest",
                "metadata": {
                    "files": filenames,
                    "chunk_count": total_chunks,
                    "store_type": kb_service.store.store_type,
                },
            }
        ],
    }
