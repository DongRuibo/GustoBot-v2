"""系统数据模型模块。

这个文件集中定义请求、响应、路由决策、Guardrails 结果、Evidence、KB 入库和文件入库模型。
这些模型是 API 层、LangGraph 工作流、业务节点和测试之间共享的数据协议。
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RouteType(str, Enum):
    # 路由类型是 Router、LangGraph 条件边和评估样本之间的统一协议。
    GENERAL = "general"
    KB = "kb"
    GRAPHRAG = "graphrag"
    TEXT2SQL = "text2sql"
    IMAGE = "image"
    FILE = "file"
    CLARIFY = "clarify"
    MULTI = "multi"


class EvidenceSource(str, Enum):
    # Evidence 来源用于答案引用、日志追踪和后续评估，不等同于 route_type。
    GENERAL = "general"
    CLARIFY = "clarify"
    GUARDRAIL = "guardrail"
    KB = "kb"
    GRAPH = "graph"
    SQL = "sql"
    IMAGE = "image"
    FILE = "file"


class Attachment(BaseModel):
    # 第一阶段只保留附件元信息；二进制内容、OCR 和文件解析后续由专门链路处理。
    type: str = Field(..., description="附件类型，例如 image 或 file。")
    filename: str | None = None
    content_type: str | None = None
    uri: str | None = None
    text: str | None = None
    content_base64: str | None = None


class ConversationMessage(BaseModel):
    # 传给工作流的轻量会话历史，只保留回答生成需要的角色和文本。
    role: str
    content: str


class ChatRequest(BaseModel):
    # chat 接口的统一请求模型，后续会继续承载多模态和会话上下文。
    message: str = Field(..., min_length=1)
    session_id: str | None = None
    user_id: str | None = "anonymous"
    attachments: list[Attachment] = Field(default_factory=list)
    conversation_history: list[ConversationMessage] = Field(default_factory=list)


class GuardrailResult(BaseModel):
    # Guardrails 返回结构化结果，方便节点决定是否中断后续链路。
    allowed: bool
    reason: str
    categories: list[str] = Field(default_factory=list)


class RouteDecision(BaseModel):
    # Router 不只返回类别，还要给出置信度、原因、槽位和是否需要反问。
    route_type: RouteType
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str
    slots: dict[str, Any] = Field(default_factory=dict)
    need_clarification: bool


class RouteSubtask(BaseModel):
    # 多路由计划中的单个子任务，只允许证据型无副作用链路参与并行。
    subtask_id: str
    route_type: RouteType
    question: str
    reason: str


class RoutePlan(BaseModel):
    # 多路由计划用于描述一次复合问题需要并行执行哪些证据链路。
    is_multi: bool
    reason: str
    subtasks: list[RouteSubtask] = Field(default_factory=list)


class Evidence(BaseModel):
    # 所有业务链路最终都归一化成 Evidence，答案生成层不关心证据来自哪里。
    source_type: EvidenceSource
    content: str
    score: float = Field(..., ge=0.0, le=1.0)
    source_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    trace_id: str


class ChatResponse(BaseModel):
    # 前端和调用方只依赖这一层响应结构，不直接感知 LangGraph 内部 state。
    trace_id: str
    answer: str
    route_decision: RouteDecision
    evidences: list[Evidence] = Field(default_factory=list)
    need_clarification: bool
    session_id: str | None = None
    message_id: str | None = None


class SessionCreate(BaseModel):
    user_id: str | None = "anonymous"
    title: str | None = None


class SessionUpdate(BaseModel):
    title: str | None = None
    is_active: bool | None = None


class SessionSummary(BaseModel):
    session_id: str
    user_id: str | None = None
    title: str
    is_active: bool
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class MessageItem(BaseModel):
    message_id: str
    session_id: str
    role: str
    content: str
    route_type: str | None = None
    trace_id: str | None = None
    evidences: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    order_index: int


class SessionSnapshotItem(BaseModel):
    snapshot_id: str
    session_id: str
    message_id: str
    trace_id: str | None = None
    route_type: str | None = None
    answer: str
    evidences: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class UploadResponse(BaseModel):
    file_id: str
    kind: str
    original_name: str
    size_bytes: int
    content_type: str | None = None
    uri: str
    attachment: Attachment


class KBDocumentIngestRequest(BaseModel):
    # 文档入库请求用于第二阶段 KB RAG，把外部文本资料切块、向量化并写入知识库存储。
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    source_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class KBDocumentIngestResponse(BaseModel):
    # 入库响应只返回可追踪的文档 id、chunk 数和当前使用的存储类型。
    document_id: str
    chunk_count: int
    store_type: str


class KBStatusResponse(BaseModel):
    # KB 状态响应用于检查当前 embedding provider、向量存储和 chunk 数，不暴露敏感密钥或 DSN。
    store_type: str
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    reranker_type: str
    reranker_status: dict[str, Any] = Field(default_factory=dict)
    hybrid_retrieval_enabled: bool = True
    lexical_top_k: int = 8
    rrf_k: int = 60
    chunk_count: int
    postgres_configured: bool
    pgvector_table: str | None = None


class FileIngestRequest(BaseModel):
    # 文件入库接口复用 Attachment 模型，当前要求附件 text 字段中已经有解析后的文件文本。
    files: list[Attachment] = Field(default_factory=list)


class FileIngestResponse(BaseModel):
    # 文件入库响应聚合多个文件的入库结果，便于前端展示“哪些文件已进入知识库”。
    ingested_files: list[str] = Field(default_factory=list)
    chunk_count: int
    store_type: str
    message: str
