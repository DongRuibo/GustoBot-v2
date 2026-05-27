"""API 层测试模块。

这个文件验证 FastAPI 暴露的接口是否正常工作，包括健康检查、聊天入口和 KB 文档入库接口。
它主要从 HTTP 调用视角确认外部调用方能访问当前系统能力。
"""

import json

from fastapi.testclient import TestClient

from app.main import app
from app.models import ChatRequest, ChatResponse, Evidence, EvidenceSource, RouteDecision, RouteType


client = TestClient(app)


def test_health() -> None:
    # 健康检查用于确认 FastAPI 应用和路由挂载正常。
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_api() -> None:
    # 从 HTTP 层验证 chat 接口能走完整 LangGraph 主流程。
    response = client.post("/api/v1/chat", json={"message": "你好"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["route_decision"]["route_type"] == "general"
    assert payload["evidences"]
    assert payload["session_id"]
    assert payload["message_id"]


def test_chat_creates_session_and_appends_history() -> None:
    # chat 接口现在负责保存用户消息和助手消息，同一 session_id 会继续追加历史。
    first = client.post("/api/v1/chat", json={"message": "你好", "user_id": "api-user"})
    assert first.status_code == 200
    session_id = first.json()["session_id"]

    second = client.post(
        "/api/v1/chat",
        json={"message": "再介绍一下你自己", "user_id": "api-user", "session_id": session_id},
    )
    assert second.status_code == 200
    second_payload = second.json()
    assert second_payload["session_id"] == session_id

    sessions = client.get("/api/v1/sessions", params={"user_id": "api-user"})
    assert sessions.status_code == 200
    session_payload = sessions.json()
    assert len(session_payload) == 1
    assert session_payload[0]["session_id"] == session_id
    assert session_payload[0]["message_count"] == 4

    messages = client.get(f"/api/v1/sessions/{session_id}/messages")
    assert messages.status_code == 200
    message_payload = messages.json()
    assert [item["role"] for item in message_payload] == ["user", "assistant", "user", "assistant"]
    assert message_payload[-1]["trace_id"]

    snapshots = client.get(f"/api/v1/sessions/{session_id}/snapshots")
    assert snapshots.status_code == 200
    snapshot_payload = snapshots.json()
    assert len(snapshot_payload) == 2
    assert snapshot_payload[0]["session_id"] == session_id
    assert snapshot_payload[0]["message_id"] == message_payload[-1]["message_id"]
    assert snapshot_payload[0]["trace_id"]
    assert snapshot_payload[0]["route_type"] == second_payload["route_decision"]["route_type"]
    assert snapshot_payload[0]["evidences"]

    snapshot_detail = client.get(f"/api/v1/sessions/{session_id}/snapshots/{snapshot_payload[0]['snapshot_id']}")
    assert snapshot_detail.status_code == 200
    assert snapshot_detail.json()["answer"] == snapshot_payload[0]["answer"]


def test_chat_passes_session_history_to_workflow(monkeypatch) -> None:
    captured_requests: list[ChatRequest] = []

    def fake_run_chat(request: ChatRequest) -> ChatResponse:
        captured_requests.append(request)
        trace_id = f"trace-{len(captured_requests)}"
        return ChatResponse(
            trace_id=trace_id,
            answer=f"收到：{request.message}",
            route_decision=RouteDecision(
                route_type=RouteType.GENERAL,
                confidence=0.95,
                reason="test",
                slots={},
                need_clarification=False,
            ),
            evidences=[
                Evidence(
                    source_type=EvidenceSource.GENERAL,
                    content="test",
                    score=1.0,
                    source_id="general_node",
                    trace_id=trace_id,
                )
            ],
            need_clarification=False,
        )

    monkeypatch.setattr("app.api.routes.run_chat", fake_run_chat)

    first = client.post("/api/v1/chat", json={"message": "你好，我叫冻睿博。", "user_id": "memory-user"})
    assert first.status_code == 200
    session_id = first.json()["session_id"]

    second = client.post(
        "/api/v1/chat",
        json={"message": "你还记得我叫什么吗？", "user_id": "memory-user", "session_id": session_id},
    )
    assert second.status_code == 200

    history = captured_requests[1].conversation_history
    assert [(item.role, item.content) for item in history] == [
        ("user", "你好，我叫冻睿博。"),
        ("assistant", "收到：你好，我叫冻睿博。"),
    ]


def test_chat_stream_api_appends_history() -> None:
    # 流式接口应复用同一套会话保存逻辑，并以 NDJSON 事件逐块返回答案。
    with client.stream(
        "POST",
        "/api/v1/chat/stream",
        json={"message": "你好", "user_id": "stream-user"},
    ) as response:
        assert response.status_code == 200
        raw_lines = [line for line in response.iter_lines() if line]

    events = [json.loads(line.decode("utf-8") if isinstance(line, bytes) else line) for line in raw_lines]
    assert events[0]["event"] == "assistant_start"
    assert any(event["event"] == "answer_delta" for event in events)
    assert events[-1]["event"] == "done"
    done_response = events[-1]["response"]
    assert done_response["session_id"]
    assert done_response["message_id"]
    assert done_response["route_decision"]["route_type"] == "general"

    messages = client.get(f"/api/v1/sessions/{done_response['session_id']}/messages")
    assert messages.status_code == 200
    assert [item["role"] for item in messages.json()] == ["user", "assistant"]


def test_session_soft_delete_hides_from_active_list() -> None:
    # 删除会话只做软删除，默认 active_only 列表不再返回。
    created = client.post("/api/v1/sessions", json={"user_id": "api-user", "title": "测试会话"})
    assert created.status_code == 201
    session_id = created.json()["session_id"]

    deleted = client.delete(f"/api/v1/sessions/{session_id}")
    assert deleted.status_code == 204

    active_sessions = client.get("/api/v1/sessions", params={"user_id": "api-user"})
    assert active_sessions.status_code == 200
    assert active_sessions.json() == []

    all_sessions = client.get("/api/v1/sessions", params={"user_id": "api-user", "active_only": False})
    assert all_sessions.status_code == 200
    assert all_sessions.json()[0]["is_active"] is False


def test_kb_document_ingest_api() -> None:
    # 入库接口验证第二阶段的 API 入口，确保文本能被切块并写入当前 KB 存储。
    response = client.post(
        "/api/v1/kb/documents",
        json={
            "title": "测试菜谱知识",
            "content": "佛跳墙是闽菜代表菜，常见于宴席文化介绍。",
            "source_id": "api-test-fotiaoqiang",
            "metadata": {"doc_type": "test"},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["document_id"] == "api-test-fotiaoqiang"
    assert payload["chunk_count"] >= 1


def test_kb_status_api() -> None:
    # KB 状态接口用于确认当前 embedding provider、向量存储和 chunk 数，不能泄漏敏感连接信息。
    response = client.get("/api/v1/kb/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["store_type"] in {"memory", "postgres_pgvector"}
    assert payload["embedding_provider"] in {"hash", "openai-compatible"}
    assert payload["reranker_type"] in {"keyword", "http"}
    assert "reranker_status" in payload
    assert "hybrid_retrieval_enabled" in payload
    assert payload["lexical_top_k"] >= 1
    assert payload["rrf_k"] >= 1
    assert "postgres_configured" in payload


def test_file_ingest_api() -> None:
    # 文件入库接口验证第四阶段的文件链路，当前使用附件 text 字段作为已解析文本。
    response = client.post(
        "/api/v1/files/ingest",
        json={
            "files": [
                {
                    "type": "file",
                    "filename": "闽菜资料.txt",
                    "text": "佛跳墙属于闽菜，常用于宴席场景。",
                }
            ]
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ingested_files"] == ["闽菜资料.txt"]
    assert payload["chunk_count"] >= 1


def test_upload_file_can_be_ingested_with_upload_uri() -> None:
    # multipart 上传返回 upload:// 附件，文件入库接口只能通过已登记 URI 读取服务端文件。
    uploaded = client.post(
        "/api/v1/upload/file",
        files={"file": ("recipe.txt", "佛跳墙是闽菜代表菜。".encode("utf-8"), "text/plain")},
    )
    assert uploaded.status_code == 200
    attachment = uploaded.json()["attachment"]
    assert attachment["uri"].startswith("upload://")

    ingested = client.post("/api/v1/files/ingest", json={"files": [attachment]})
    assert ingested.status_code == 200
    payload = ingested.json()
    assert payload["ingested_files"] == ["recipe.txt"]
    assert payload["chunk_count"] >= 1


def test_upload_image_attachment_can_enter_chat_flow() -> None:
    # 图片上传后不直接回答，而是以 upload:// 附件进入现有图片理解与重路由流程。
    uploaded = client.post(
        "/api/v1/upload/image",
        files={"image": ("gongbao.png", b"fake-image", "image/png")},
    )
    assert uploaded.status_code == 200
    attachment = uploaded.json()["attachment"]

    response = client.post(
        "/api/v1/chat",
        json={
            "message": "这道菜需要哪些食材？",
            "attachments": [attachment],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["route_decision"]["route_type"] == "graphrag"
    assert payload["evidences"][0]["source_type"] == "image"


def test_upload_rejects_unsupported_extension() -> None:
    # 上传入口按类型限制扩展名，避免任意文件进入解析链路。
    response = client.post(
        "/api/v1/upload/file",
        files={"file": ("payload.exe", b"bad", "application/octet-stream")},
    )
    assert response.status_code == 400
