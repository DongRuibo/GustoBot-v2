"""HTTP 路由入口。

这个文件定义当前系统对外暴露的 FastAPI 接口，包括健康检查、聊天入口、KB 文档入库和文件入库。
它的职责是把 HTTP 请求转换为内部模型，然后委托给工作流或 KB 服务处理。
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.sessions import router as sessions_router
from app.api.uploads import router as uploads_router
from app.graph.workflow import run_chat
from app.kb.service import get_kb_service
from app.files.parser import UnsupportedFileTypeError, get_file_parser
from app.models import (
    ChatRequest,
    ChatResponse,
    ConversationMessage,
    FileIngestRequest,
    FileIngestResponse,
    KBDocumentIngestRequest,
    KBDocumentIngestResponse,
    KBStatusResponse,
)
from app.sessions.service import get_session_service

# 当前 API 暴露健康检查、聊天入口、KB 文本入库和文件入库；评估等工程接口放在脚本层。
router = APIRouter(prefix="/api/v1")
router.include_router(sessions_router)
router.include_router(uploads_router)

_WORKFLOW_HISTORY_LIMIT = 12


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "GustoBot-v2"}


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    # HTTP 层只补会话外壳，核心问答仍统一委托给 LangGraph 主流程。
    return _run_chat_with_session(request)


@router.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    # 流式接口保持与 /chat 相同的会话与工作流能力，只把最终答案拆成增量事件返回给前端。
    return StreamingResponse(
        _stream_chat_with_session(request),
        media_type="application/x-ndjson; charset=utf-8",
    )


def _run_chat_with_session(request: ChatRequest) -> ChatResponse:
    session_service = get_session_service()
    session = session_service.get_or_create_session(
        session_id=request.session_id,
        user_id=request.user_id,
        title_seed=request.message,
    )
    user_message = session_service.save_user_message(session_id=session.session_id, content=request.message)
    conversation_history = _conversation_history_for_workflow(
        session_service,
        session.session_id,
        exclude_message_id=user_message.message_id,
    )
    workflow_request = request.model_copy(
        update={"session_id": session.session_id, "conversation_history": conversation_history}
    )
    try:
        response = run_chat(workflow_request)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)[:500]) from exc
    assistant_message = session_service.save_assistant_response(
        session_id=session.session_id,
        response=response,
    )
    response.session_id = session.session_id
    response.message_id = assistant_message.message_id
    return response


def _conversation_history_for_workflow(
    session_service,
    session_id: str,
    *,
    exclude_message_id: str,
) -> list[ConversationMessage]:
    # 只把当前问题之前的最近几轮带进工作流，避免当前用户消息被重复注入 prompt。
    messages = session_service.list_messages(session_id, skip=0, limit=200)
    prior_messages = [
        message
        for message in messages
        if message.message_id != exclude_message_id and message.role in {"user", "assistant"}
    ]
    return [
        ConversationMessage(role=message.role, content=message.content)
        for message in prior_messages[-_WORKFLOW_HISTORY_LIMIT:]
        if message.content.strip()
    ]


def _stream_chat_with_session(request: ChatRequest) -> Iterator[str]:
    try:
        response = _run_chat_with_session(request)
        yield _stream_event(
            "assistant_start",
            {
                "session_id": response.session_id,
                "message_id": response.message_id,
                "trace_id": response.trace_id,
                "route_decision": response.route_decision.model_dump(mode="json"),
                "evidences": [evidence.model_dump(mode="json") for evidence in response.evidences],
                "need_clarification": response.need_clarification,
            },
        )
        for chunk in _answer_chunks(response.answer):
            yield _stream_event("answer_delta", {"delta": chunk})
            # 轻微节流让浏览器有机会逐块渲染，不改变后端业务结果。
            time.sleep(0.015)
        yield _stream_event("done", {"response": response.model_dump(mode="json")})
    except Exception as exc:
        yield _stream_event("error", {"message": str(exc)[:500]})


def _stream_event(event: str, payload: dict) -> str:
    return json.dumps({"event": event, **payload}, ensure_ascii=False) + "\n"


def _answer_chunks(answer: str, chunk_size: int = 24) -> Iterator[str]:
    for index in range(0, len(answer), chunk_size):
        yield answer[index : index + chunk_size]


@router.post("/kb/documents", response_model=KBDocumentIngestResponse)
def ingest_kb_document(request: KBDocumentIngestRequest) -> KBDocumentIngestResponse:
    # 第二阶段提供最小入库接口：文本资料进入后会完成切块、embedding 和向量存储。
    # 后续文件上传链路会复用同一套 KB service，而不是在 API 层重复实现入库逻辑。
    result = get_kb_service().ingest_document(
        title=request.title,
        content=request.content,
        metadata=request.metadata,
        source_id=request.source_id,
    )
    return KBDocumentIngestResponse(
        document_id=result.document_id,
        chunk_count=result.chunk_count,
        store_type=result.store_type,
    )


@router.get("/kb/status", response_model=KBStatusResponse)
def kb_status() -> KBStatusResponse:
    # 状态接口用于真实 embedding/pgvector 接入时做部署自检，不返回 API Key 或数据库 DSN。
    status = get_kb_service().status()
    return KBStatusResponse(
        store_type=status.store_type,
        embedding_provider=status.embedding_provider,
        embedding_model=status.embedding_model,
        embedding_dimension=status.embedding_dimension,
        reranker_type=status.reranker_type,
        reranker_status=status.reranker_status,
        hybrid_retrieval_enabled=status.hybrid_retrieval_enabled,
        lexical_top_k=status.lexical_top_k,
        rrf_k=status.rrf_k,
        chunk_count=status.chunk_count,
        postgres_configured=status.postgres_configured,
        pgvector_table=status.pgvector_table,
    )


@router.post("/files/ingest", response_model=FileIngestResponse)
def ingest_files(request: FileIngestRequest) -> FileIngestResponse:
    # 文件入库接口用于显式把文件解析文本写入 KB；聊天接口中的 file 附件也会复用同一套能力。
    parser = get_file_parser()
    kb_service = get_kb_service()
    ingested_files: list[str] = []
    total_chunks = 0
    for attachment in request.files:
        try:
            parsed = parser.parse(_model_to_dict(attachment))
        except UnsupportedFileTypeError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if parsed is None:
            continue
        result = kb_service.ingest_document(
            title=parsed.title,
            content=parsed.content,
            metadata=parsed.metadata,
            source_id=f"file:{parsed.filename}",
        )
        ingested_files.append(parsed.filename)
        total_chunks += result.chunk_count

    message = (
        f"已入库 {len(ingested_files)} 个文件，共生成 {total_chunks} 个 chunk。"
        if ingested_files
        else "没有解析到可入库的文件文本。"
    )
    return FileIngestResponse(
        ingested_files=ingested_files,
        chunk_count=total_chunks,
        store_type=kb_service.store.store_type,
        message=message,
    )


def _model_to_dict(value) -> dict:
    # 兼容 Pydantic v1/v2，避免接口层因为版本差异无法序列化附件。
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value.dict()
