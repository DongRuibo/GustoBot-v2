"""Docker 端到端 smoke 验证脚本。

该脚本假设 FastAPI、PostgreSQL/pgvector、Neo4j、Redis 已由 docker compose 启动。
它通过 HTTP 调用验证业务链路，并可直连数据库检查初始化表、图谱数据和 Redis 连接。
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SmokeFailure(RuntimeError):
    """Smoke 检查失败。"""


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _ensure_project_on_path()

    run_id = uuid4().hex[:8]
    base_url = args.base_url.rstrip("/")
    summary: dict[str, Any] = {"run_id": run_id, "checks": []}

    with httpx.Client(base_url=base_url, timeout=args.timeout) as client:
        _wait_for_api(client, args.wait_seconds)
        _check_health(client, summary)
        _check_session_flow(client, run_id, summary)
        _check_upload_flow(client, run_id, summary)
        _check_kb_status(client, summary, args.expect_embedding_provider)
        kb_response = _check_kb_flow(client, run_id, summary, args.expect_reranker_success)
        graph_response = _check_graphrag_flow(client, run_id, summary)
        _check_natural_router_flow(client, run_id, summary)
        sql_response = _check_text2sql_flow(client, run_id, summary)
        _check_file_ingest_flow(client, run_id, summary)

    if not args.skip_db_checks:
        _check_postgres(args.postgres_dsn, summary, expect_real_data=args.expect_real_data)
        _check_neo4j(
            args.neo4j_uri,
            args.neo4j_username,
            args.neo4j_password,
            args.neo4j_database,
            summary,
            expect_real_data=args.expect_real_data,
        )
        _check_redis(args.redis_url, summary)

    _check_trace_log(
        args.trace_log_path,
        [kb_response["trace_id"], graph_response["trace_id"], sql_response["trace_id"]],
        summary,
        expected_answer_mode=args.expect_answer_mode,
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GustoBot-v2 Docker E2E smoke checks.")
    parser.add_argument("--base-url", default=os.getenv("GUSTOBOT_SMOKE_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument(
        "--postgres-dsn",
        default=os.getenv("GUSTOBOT_TEXT2SQL_POSTGRES_DSN")
        or os.getenv("GUSTOBOT_POSTGRES_DSN")
        or "postgresql://gustobot:gustobot@127.0.0.1:5432/gustobot",
    )
    parser.add_argument("--neo4j-uri", default=os.getenv("GUSTOBOT_NEO4J_URI", "bolt://127.0.0.1:17687"))
    parser.add_argument("--neo4j-username", default=os.getenv("GUSTOBOT_NEO4J_USERNAME", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("GUSTOBOT_NEO4J_PASSWORD", "gustobotneo4j"))
    parser.add_argument("--neo4j-database", default=os.getenv("GUSTOBOT_NEO4J_DATABASE"))
    parser.add_argument("--redis-url", default=os.getenv("GUSTOBOT_REDIS_URL", "redis://127.0.0.1:6379/0"))
    parser.add_argument(
        "--trace-log-path",
        type=Path,
        default=Path(os.getenv("GUSTOBOT_TRACE_LOG_PATH", PROJECT_ROOT / "logs" / "traces.jsonl")),
    )
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--wait-seconds", type=float, default=60)
    parser.add_argument("--skip-db-checks", action="store_true")
    parser.add_argument(
        "--expect-embedding-provider",
        default=os.getenv("GUSTOBOT_SMOKE_EXPECT_EMBEDDING_PROVIDER"),
        help="可选：断言 KB status 中的 embedding_provider，例如 openai-compatible。",
    )
    parser.add_argument(
        "--expect-reranker-success",
        action="store_true",
        default=_env_bool("GUSTOBOT_SMOKE_EXPECT_RERANKER_SUCCESS"),
        help="可选：断言 KB Evidence 中 reranker_success=true。",
    )
    parser.add_argument(
        "--expect-answer-mode",
        choices=("llm", "template"),
        default=os.getenv("GUSTOBOT_SMOKE_EXPECT_ANSWER_MODE"),
        help="可选：断言 trace 中 answer_generated 的 mode。",
    )
    parser.add_argument(
        "--expect-real-data",
        action="store_true",
        default=_env_bool("GUSTOBOT_SMOKE_EXPECT_REAL_DATA"),
        help="断言 PostgreSQL/Neo4j 已导入真实菜谱数据。",
    )
    return parser.parse_args(argv)


def _wait_for_api(client: httpx.Client, wait_seconds: float) -> None:
    deadline = time.time() + wait_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = client.get("/api/v1/health")
            if response.status_code == 200:
                return
        except Exception as exc:
            last_error = exc
        time.sleep(1)
    raise SmokeFailure(f"API health check did not become ready within {wait_seconds}s: {last_error}")


def _check_health(client: httpx.Client, summary: dict[str, Any]) -> None:
    response = client.get("/api/v1/health")
    response.raise_for_status()
    payload = response.json()
    _assert(payload.get("status") == "ok", "health status should be ok")
    _record(summary, "health", payload)


def _check_session_flow(client: httpx.Client, run_id: str, summary: dict[str, Any]) -> None:
    user_id = f"smoke-user-{run_id}"
    first = client.post("/api/v1/chat", json={"message": "你好", "user_id": user_id})
    first.raise_for_status()
    first_payload = first.json()
    session_id = first_payload.get("session_id")
    _assert(bool(session_id), "chat response should include session_id")
    _assert(bool(first_payload.get("message_id")), "chat response should include assistant message_id")

    second = client.post(
        "/api/v1/chat",
        json={"message": "再介绍一下你自己", "user_id": user_id, "session_id": session_id},
    )
    second.raise_for_status()
    _assert(second.json().get("session_id") == session_id, "chat should reuse provided session_id")

    sessions = client.get("/api/v1/sessions", params={"user_id": user_id})
    sessions.raise_for_status()
    session_payload = sessions.json()
    _assert(len(session_payload) == 1, "session list should contain current smoke session")
    _assert(session_payload[0]["message_count"] >= 4, "session should contain user and assistant messages")

    messages = client.get(f"/api/v1/sessions/{session_id}/messages")
    messages.raise_for_status()
    message_payload = messages.json()
    _assert(len(message_payload) >= 4, "message history should include both chat rounds")

    deleted = client.delete(f"/api/v1/sessions/{session_id}")
    deleted.raise_for_status()
    active_sessions = client.get("/api/v1/sessions", params={"user_id": user_id})
    active_sessions.raise_for_status()
    _assert(active_sessions.json() == [], "soft-deleted session should not appear in active list")
    _record(
        summary,
        "session_flow",
        {"session_id": session_id, "message_count": len(message_payload), "user_id": user_id},
    )


def _check_upload_flow(client: httpx.Client, run_id: str, summary: dict[str, Any]) -> None:
    upload_response = client.post(
        "/api/v1/upload/file",
        files={"file": (f"smoke-{run_id}.txt", f"佛跳墙是闽菜代表菜。smoke_run={run_id}".encode("utf-8"), "text/plain")},
    )
    upload_response.raise_for_status()
    upload_payload = upload_response.json()
    attachment = upload_payload["attachment"]
    _assert(attachment["uri"].startswith("upload://"), "file upload should return upload:// attachment")

    ingest_response = client.post("/api/v1/files/ingest", json={"files": [attachment]})
    ingest_response.raise_for_status()
    ingest_payload = ingest_response.json()
    _assert(ingest_payload["chunk_count"] >= 1, "uploaded file should be ingestible")

    image_response = client.post(
        "/api/v1/upload/image",
        files={"image": (f"gongbao-{run_id}.png", _sample_dish_png_bytes(), "image/png")},
    )
    image_response.raise_for_status()
    image_attachment = image_response.json()["attachment"]
    image_attachment["text"] = "图片线索：宫保鸡丁，鸡肉、花生、辣椒。"
    chat_payload = client.post(
        "/api/v1/chat",
        json={"message": "这道菜需要哪些食材？", "attachments": [image_attachment]},
    )
    try:
        chat_payload.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise SmokeFailure(f"uploaded image chat failed: {chat_payload.status_code} {chat_payload.text[:500]}") from exc
    chat_json = chat_payload.json()
    _assert(chat_json["route_decision"]["route_type"] == "graphrag", "uploaded image should reroute to graphrag")
    _assert(_has_evidence(chat_json, "image"), "uploaded image chat should include image evidence")

    _record(
        summary,
        "upload_flow",
        {
            "file_id": upload_payload["file_id"],
            "file_ingest_chunks": ingest_payload["chunk_count"],
            "image_route": chat_json["route_decision"]["route_type"],
        },
    )


def _sample_dish_png_bytes() -> bytes:
    """生成一张有效 PNG，避免 strict Vision smoke 使用无效 fake bytes。"""

    width = 96
    height = 72
    rows: list[bytes] = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            dx = (x - width / 2) / (width / 2)
            dy = (y - height / 2) / (height / 2)
            distance = dx * dx + dy * dy
            if distance > 0.82:
                row.extend((245, 245, 238))
            elif distance > 0.64:
                row.extend((235, 232, 220))
            elif (x + y) % 17 < 4:
                row.extend((180, 40, 30))
            elif (x * 3 + y) % 23 < 5:
                row.extend((236, 184, 70))
            else:
                row.extend((143, 83, 42))
        rows.append(b"\x00" + bytes(row))
    raw = b"".join(rows)
    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            _png_chunk(b"IDAT", zlib.compress(raw)),
            _png_chunk(b"IEND", b""),
        )
    )


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", checksum)


def _check_kb_status(
    client: httpx.Client,
    summary: dict[str, Any],
    expected_embedding_provider: str | None,
) -> None:
    response = client.get("/api/v1/kb/status")
    response.raise_for_status()
    payload = response.json()
    _assert(payload["store_type"] == "postgres_pgvector", "KB should use PostgreSQL pgvector in Docker smoke")
    _assert(payload["postgres_configured"] is True, "KB postgres_configured should be true")
    _assert(payload["hybrid_retrieval_enabled"] is True, "KB hybrid retrieval should be enabled by default")
    _assert(payload["lexical_top_k"] >= 1, "KB lexical_top_k should be positive")
    if expected_embedding_provider:
        _assert(
            payload["embedding_provider"] == expected_embedding_provider,
            f"KB embedding_provider should be {expected_embedding_provider}",
        )
    _record(summary, "kb_status", payload)


def _check_kb_flow(
    client: httpx.Client,
    run_id: str,
    summary: dict[str, Any],
    expect_reranker_success: bool,
) -> dict[str, Any]:
    source_id = f"smoke:kb:{run_id}"
    ingest_response = client.post(
        "/api/v1/kb/documents",
        json={
            "title": f"Smoke 佛跳墙资料 {run_id}",
            "content": f"佛跳墙是闽菜代表菜，常见于宴席文化介绍。smoke_run={run_id}",
            "source_id": source_id,
            "metadata": {"doc_type": "smoke", "run_id": run_id},
        },
    )
    ingest_response.raise_for_status()
    ingest_payload = ingest_response.json()
    _assert(ingest_payload["store_type"] == "postgres_pgvector", "KB ingest should write pgvector")
    _assert(ingest_payload["chunk_count"] >= 1, "KB ingest should create chunks")

    chat_payload = _chat(client, f"介绍一下佛跳墙的历史和文化 smoke {run_id}")
    _assert(chat_payload["route_decision"]["route_type"] == "kb", "KB question should route to kb")
    _assert(_has_evidence(chat_payload, "kb"), "KB response should include kb evidence")
    _assert(
        any(run_id in evidence.get("content", "") for evidence in chat_payload.get("evidences", [])),
        "KB response should retrieve the current smoke chunk",
    )
    _assert(
        any(
            evidence.get("metadata", {}).get("retrieval_mode") in {"hybrid", "vector", "lexical"}
            for evidence in chat_payload.get("evidences", [])
            if evidence.get("source_type") == "kb"
        ),
        "KB Evidence should include retrieval metadata",
    )
    if expect_reranker_success:
        _assert(
            any(
                evidence.get("metadata", {}).get("reranker_success") is True
                for evidence in chat_payload.get("evidences", [])
                if evidence.get("source_type") == "kb"
            ),
            "KB Evidence should show reranker_success=true",
        )
    _record(summary, "kb_flow", {"ingest": ingest_payload, "chat": _compact_chat(chat_payload)})
    return chat_payload


def _check_graphrag_flow(client: httpx.Client, run_id: str, summary: dict[str, Any]) -> dict[str, Any]:
    payload = _chat(client, f"宫保鸡丁需要哪些食材 smoke {run_id}")
    _assert(payload["route_decision"]["route_type"] == "graphrag", "relation question should route to graphrag")
    graph_evidence = _first_evidence(payload, "graph")
    _assert(graph_evidence is not None, "GraphRAG response should include graph evidence")
    _assert(
        graph_evidence["metadata"].get("store_type") == "neo4j",
        "GraphRAG should use Neo4j store in Docker smoke",
    )
    _record(summary, "graphrag_flow", _compact_chat(payload))
    return payload


def _check_natural_router_flow(client: httpx.Client, run_id: str, summary: dict[str, Any]) -> None:
    payload = _chat(client, f"红烧排骨怎么做 smoke {run_id}")
    _assert(
        payload["route_decision"]["route_type"] == "graphrag",
        "natural recipe howto question should route to graphrag instead of clarify",
    )
    _assert(
        payload["route_decision"].get("need_clarification") is False,
        "natural recipe howto question should not need clarification",
    )
    graph_evidence = _first_evidence(payload, "graph")
    _assert(graph_evidence is not None, "natural recipe howto question should include graph evidence")
    _assert("制作步骤" in graph_evidence["content"], "recipe howto evidence should include step details")
    _record(summary, "natural_router_flow", _compact_chat(payload))


def _check_text2sql_flow(client: httpx.Client, run_id: str, summary: dict[str, Any]) -> dict[str, Any]:
    payload = _chat(client, f"统计一下每个菜系的菜谱数量 smoke {run_id}")
    _assert(payload["route_decision"]["route_type"] == "text2sql", "statistics question should route to text2sql")
    sql_evidence = _first_evidence(payload, "sql")
    _assert(sql_evidence is not None, "Text2SQL response should include sql evidence")
    _assert(
        sql_evidence["metadata"].get("executor_type") == "postgres_readonly",
        "Text2SQL should use PostgreSQL readonly executor",
    )
    _record(summary, "text2sql_flow", _compact_chat(payload))
    return payload


def _check_file_ingest_flow(client: httpx.Client, run_id: str, summary: dict[str, Any]) -> None:
    response = client.post(
        "/api/v1/files/ingest",
        json={
            "files": [
                {
                    "type": "file",
                    "filename": f"smoke-闽菜资料-{run_id}.txt",
                    "text": f"佛跳墙属于闽菜，常用于宴席场景。smoke_run={run_id}",
                }
            ]
        },
    )
    response.raise_for_status()
    payload = response.json()
    _assert(payload["store_type"] == "postgres_pgvector", "file ingest should write pgvector")
    _assert(payload["chunk_count"] >= 1, "file ingest should create chunks")
    _record(summary, "file_ingest_flow", payload)


def _check_postgres(dsn: str, summary: dict[str, Any], *, expect_real_data: bool) -> None:
    try:
        import psycopg
    except ImportError as exc:
        raise SmokeFailure("缺少 psycopg，无法执行 PostgreSQL smoke 检查。") from exc

    expected_tables = [
        "kb_documents",
        "kb_chunks",
        "recipes",
        "schema_catalog",
        "evaluation_logs",
        "trace_events",
        "recipe_records",
        "recipe_cuisines",
        "recipe_ingredients_master",
        "recipe_steps",
        "recipe_ingredients",
        "recipe_tools",
        "recipe_step_tools",
    ]
    with psycopg.connect(dsn) as connection:
        with connection.cursor() as cursor:
            table_status: dict[str, bool] = {}
            for table in expected_tables:
                cursor.execute("SELECT to_regclass(%s)", (table,))
                table_status[table] = cursor.fetchone()[0] is not None
            _assert(all(table_status.values()), f"PostgreSQL tables missing: {table_status}")
            cursor.execute("SELECT COUNT(*) FROM recipes")
            recipe_count = int(cursor.fetchone()[0])
            cursor.execute("SELECT COUNT(*) FROM schema_catalog")
            schema_count = int(cursor.fetchone()[0])
            cursor.execute("SELECT COUNT(*) FROM kb_chunks")
            chunk_count = int(cursor.fetchone()[0])
            cursor.execute("SELECT COUNT(*) FROM kb_chunks WHERE search_text <> ''")
            searchable_chunk_count = int(cursor.fetchone()[0])
            cursor.execute("SELECT COUNT(*) FROM recipe_records")
            real_recipe_count = int(cursor.fetchone()[0])
            cursor.execute("SELECT COUNT(*) FROM recipe_ingredients")
            real_relation_count = int(cursor.fetchone()[0])

    _assert(recipe_count > 0, "recipes table should contain seed data")
    _assert(schema_count > 0, "schema_catalog should contain seed schema")
    _assert(chunk_count > 0, "kb_chunks should contain smoke-ingested chunks")
    _assert(searchable_chunk_count > 0, "kb_chunks should contain tokenized search_text for hybrid retrieval")
    if expect_real_data:
        _assert(real_recipe_count > 0, "recipe_records should contain real imported recipes")
        _assert(real_relation_count > 0, "recipe_ingredients should contain real imported relations")
    _record(
        summary,
        "postgres_state",
        {
            "tables": table_status,
            "recipes": recipe_count,
            "schema_catalog": schema_count,
            "kb_chunks": chunk_count,
            "kb_chunks_with_search_text": searchable_chunk_count,
            "recipe_records": real_recipe_count,
            "recipe_ingredients": real_relation_count,
        },
    )


def _check_neo4j(
    uri: str,
    username: str,
    password: str,
    database: str | None,
    summary: dict[str, Any],
    *,
    expect_real_data: bool,
) -> None:
    try:
        import neo4j
    except ImportError as exc:
        raise SmokeFailure("缺少 neo4j，无法执行 Neo4j smoke 检查。") from exc

    with neo4j.GraphDatabase.driver(uri, auth=(username, password)) as driver:
        with driver.session(database=database) as session:
            node_count = session.execute_read(lambda tx: tx.run("MATCH (n) RETURN count(n) AS count").single()["count"])
            edge_count = session.execute_read(lambda tx: tx.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"])
            real_recipe_count = session.execute_read(
                lambda tx: tx.run("MATCH (n:Recipe) RETURN count(n) AS count").single()["count"]
            )

    _assert(node_count > 0, "Neo4j should contain nodes; run scripts/bootstrap_neo4j.py first")
    _assert(edge_count > 0, "Neo4j should contain relationships; run scripts/bootstrap_neo4j.py first")
    if expect_real_data:
        _assert(real_recipe_count > 0, "Neo4j should contain real Recipe nodes")
    _record(
        summary,
        "neo4j_state",
        {"nodes": node_count, "relationships": edge_count, "recipe_nodes": real_recipe_count},
    )


def _check_redis(redis_url: str, summary: dict[str, Any]) -> None:
    try:
        import redis
    except ImportError as exc:
        raise SmokeFailure("缺少 redis 包，无法执行 Redis smoke 检查。") from exc

    client = redis.Redis.from_url(redis_url, decode_responses=True)
    _assert(client.ping() is True, "Redis ping should return true")
    key = f"smoke:{uuid4().hex}"
    client.setex(key, 30, "ok")
    _assert(client.get(key) == "ok", "Redis set/get should work")
    client.delete(key)
    _record(summary, "redis_state", {"ping": True})


def _check_trace_log(
    path: Path,
    trace_ids: list[str],
    summary: dict[str, Any],
    *,
    expected_answer_mode: str | None,
) -> None:
    # trace 是旁路能力：文件存在时必须能找到本次 trace_id；不存在时给出明确错误。
    deadline = time.time() + 5
    found: set[str] = set()
    matching_answer_modes: dict[str, str] = {}
    semantic_cache_hits: set[str] = set()
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    while time.time() < deadline:
        if resolved.exists():
            content = resolved.read_text(encoding="utf-8", errors="ignore")
            found = {trace_id for trace_id in trace_ids if trace_id in content}
            matching_answer_modes = _read_answer_modes(resolved, trace_ids)
            semantic_cache_hits = _read_semantic_cache_hits(resolved, trace_ids)
            if len(found) == len(trace_ids):
                break
        time.sleep(0.2)

    _assert(len(found) == len(trace_ids), f"trace log missing trace ids: {set(trace_ids) - found}")
    if expected_answer_mode:
        wrong_modes = {
            trace_id: matching_answer_modes.get(trace_id)
            for trace_id in trace_ids
            if trace_id not in semantic_cache_hits and matching_answer_modes.get(trace_id) != expected_answer_mode
        }
        _assert(not wrong_modes, f"answer_generated mode mismatch: {wrong_modes}")
        _assert(
            any(mode == expected_answer_mode for mode in matching_answer_modes.values()),
            f"trace log should include at least one fresh answer_generated mode={expected_answer_mode}",
        )
    _record(
        summary,
        "trace_log",
        {
            "path": str(resolved),
            "trace_ids": trace_ids,
            "answer_modes": matching_answer_modes,
            "semantic_cache_hits": sorted(semantic_cache_hits),
        },
    )


def _chat(client: httpx.Client, message: str) -> dict[str, Any]:
    response = client.post("/api/v1/chat", json={"message": message})
    response.raise_for_status()
    return response.json()


def _first_evidence(payload: dict[str, Any], source_type: str) -> dict[str, Any] | None:
    for evidence in payload.get("evidences", []):
        if evidence.get("source_type") == source_type:
            return evidence
    return None


def _has_evidence(payload: dict[str, Any], source_type: str) -> bool:
    return _first_evidence(payload, source_type) is not None


def _compact_chat(payload: dict[str, Any]) -> dict[str, Any]:
    evidence_metadata = [
        {
            "source_type": item["source_type"],
            "source_id": item["source_id"],
            "metadata": item.get("metadata", {}),
        }
        for item in payload.get("evidences", [])[:3]
    ]
    return {
        "trace_id": payload["trace_id"],
        "route_type": payload["route_decision"]["route_type"],
        "route_slots": payload["route_decision"].get("slots", {}),
        "evidence_types": [item["source_type"] for item in payload.get("evidences", [])],
        "evidence_metadata": evidence_metadata,
        "answer_preview": payload.get("answer", "")[:120],
    }


def _record(summary: dict[str, Any], name: str, payload: dict[str, Any]) -> None:
    summary["checks"].append({"name": name, "status": "ok", "payload": payload})


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def _ensure_project_on_path() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _read_answer_modes(path: Path, trace_ids: list[str]) -> dict[str, str]:
    modes: dict[str, str] = {}
    wanted = set(trace_ids)
    if not path.exists():
        return modes
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        trace_id = event.get("trace_id")
        if trace_id not in wanted or event.get("event_type") != "answer_generated":
            continue
        mode = event.get("payload", {}).get("mode")
        if isinstance(mode, str):
            modes[trace_id] = mode
    return modes


def _read_semantic_cache_hits(path: Path, trace_ids: list[str]) -> set[str]:
    hits: set[str] = set()
    wanted = set(trace_ids)
    if not path.exists():
        return hits
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        trace_id = event.get("trace_id")
        if trace_id in wanted and event.get("event_type") == "semantic_cache_hit":
            hits.add(trace_id)
    return hits


def _env_bool(name: str) -> bool:
    value = os.getenv(name)
    return value is not None and value.lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    raise SystemExit(main())
