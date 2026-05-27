"""会话与消息服务。

该模块只负责应用外壳层的会话记录，不参与 RAG、GraphRAG 或 Text2SQL 决策。
正式环境使用 PostgreSQL；未配置 DSN 时使用内存实现，保证本地测试不依赖数据库。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Protocol
from uuid import uuid4

from app.core.config import settings
from app.models import ChatResponse, MessageItem, SessionSnapshotItem, SessionSummary


class SessionStore(Protocol):
    def create_session(self, *, user_id: str | None, title: str, session_id: str | None = None) -> SessionSummary:
        ...

    def get_session(self, session_id: str) -> SessionSummary | None:
        ...

    def list_sessions(
        self,
        *,
        user_id: str | None,
        skip: int,
        limit: int,
        active_only: bool,
    ) -> list[SessionSummary]:
        ...

    def update_session(self, session_id: str, *, title: str | None, is_active: bool | None) -> SessionSummary | None:
        ...

    def append_message(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        route_type: str | None = None,
        trace_id: str | None = None,
        evidences: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MessageItem:
        ...

    def list_messages(self, session_id: str, *, skip: int, limit: int) -> list[MessageItem]:
        ...

    def create_snapshot(
        self,
        *,
        session_id: str,
        message_id: str,
        trace_id: str | None,
        route_type: str | None,
        answer: str,
        evidences: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> SessionSnapshotItem:
        ...

    def list_snapshots(self, session_id: str, *, skip: int, limit: int) -> list[SessionSnapshotItem]:
        ...

    def get_snapshot(self, session_id: str, snapshot_id: str) -> SessionSnapshotItem | None:
        ...


class InMemorySessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}
        self._messages: dict[str, list[dict[str, Any]]] = {}
        self._snapshots: dict[str, list[dict[str, Any]]] = {}
        self._lock = RLock()

    def create_session(self, *, user_id: str | None, title: str, session_id: str | None = None) -> SessionSummary:
        now = _now()
        sid = session_id or str(uuid4())
        with self._lock:
            self._sessions[sid] = {
                "session_id": sid,
                "user_id": user_id or "anonymous",
                "title": title,
                "is_active": True,
                "created_at": now,
                "updated_at": now,
            }
            self._messages.setdefault(sid, [])
            self._snapshots.setdefault(sid, [])
            return self._to_summary(sid)

    def get_session(self, session_id: str) -> SessionSummary | None:
        with self._lock:
            if session_id not in self._sessions:
                return None
            return self._to_summary(session_id)

    def list_sessions(
        self,
        *,
        user_id: str | None,
        skip: int,
        limit: int,
        active_only: bool,
    ) -> list[SessionSummary]:
        with self._lock:
            records = list(self._sessions.values())
            if user_id:
                records = [record for record in records if record.get("user_id") == user_id]
            if active_only:
                records = [record for record in records if record.get("is_active")]
            records.sort(key=lambda record: record["updated_at"], reverse=True)
            return [self._to_summary(record["session_id"]) for record in records[skip : skip + limit]]

    def update_session(self, session_id: str, *, title: str | None, is_active: bool | None) -> SessionSummary | None:
        with self._lock:
            record = self._sessions.get(session_id)
            if not record:
                return None
            if title is not None:
                record["title"] = title
            if is_active is not None:
                record["is_active"] = is_active
            record["updated_at"] = _now()
            return self._to_summary(session_id)

    def append_message(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        route_type: str | None = None,
        trace_id: str | None = None,
        evidences: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MessageItem:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(session_id)
            messages = self._messages.setdefault(session_id, [])
            record = {
                "message_id": str(uuid4()),
                "session_id": session_id,
                "role": role,
                "content": content,
                "route_type": route_type,
                "trace_id": trace_id,
                "evidences": evidences or [],
                "metadata": metadata or {},
                "created_at": _now(),
                "order_index": len(messages) + 1,
            }
            messages.append(record)
            self._sessions[session_id]["updated_at"] = record["created_at"]
            return MessageItem(**record)

    def list_messages(self, session_id: str, *, skip: int, limit: int) -> list[MessageItem]:
        with self._lock:
            return [MessageItem(**record) for record in self._messages.get(session_id, [])[skip : skip + limit]]

    def create_snapshot(
        self,
        *,
        session_id: str,
        message_id: str,
        trace_id: str | None,
        route_type: str | None,
        answer: str,
        evidences: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> SessionSnapshotItem:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(session_id)
            record = {
                "snapshot_id": str(uuid4()),
                "session_id": session_id,
                "message_id": message_id,
                "trace_id": trace_id,
                "route_type": route_type,
                "answer": answer,
                "evidences": evidences,
                "metadata": metadata,
                "created_at": _now(),
            }
            self._snapshots.setdefault(session_id, []).append(record)
            return SessionSnapshotItem(**record)

    def list_snapshots(self, session_id: str, *, skip: int, limit: int) -> list[SessionSnapshotItem]:
        with self._lock:
            records = list(self._snapshots.get(session_id, []))
            records.sort(key=lambda record: record["created_at"], reverse=True)
            return [SessionSnapshotItem(**record) for record in records[skip : skip + limit]]

    def get_snapshot(self, session_id: str, snapshot_id: str) -> SessionSnapshotItem | None:
        with self._lock:
            for record in self._snapshots.get(session_id, []):
                if record["snapshot_id"] == snapshot_id:
                    return SessionSnapshotItem(**record)
            return None

    def _to_summary(self, session_id: str) -> SessionSummary:
        record = self._sessions[session_id]
        return SessionSummary(
            **record,
            message_count=len(self._messages.get(session_id, [])),
        )


class PostgreSQLSessionStore:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._psycopg, self._dict_row = _ensure_postgres_driver()
        self.ensure_schema()

    def ensure_schema(self) -> None:
        with self._psycopg.connect(self.dsn, row_factory=self._dict_row) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        session_id text PRIMARY KEY,
                        user_id text,
                        title text NOT NULL,
                        is_active boolean NOT NULL DEFAULT true,
                        created_at timestamptz NOT NULL DEFAULT now(),
                        updated_at timestamptz NOT NULL DEFAULT now()
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        message_id text PRIMARY KEY,
                        session_id text NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                        role text NOT NULL,
                        content text NOT NULL,
                        route_type text,
                        trace_id text,
                        evidences jsonb NOT NULL DEFAULT '[]'::jsonb,
                        metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
                        order_index integer NOT NULL,
                        created_at timestamptz NOT NULL DEFAULT now()
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_session_snapshots (
                        snapshot_id text PRIMARY KEY,
                        session_id text NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                        message_id text NOT NULL REFERENCES chat_messages(message_id) ON DELETE CASCADE,
                        trace_id text,
                        route_type text,
                        answer text NOT NULL,
                        evidences jsonb NOT NULL DEFAULT '[]'::jsonb,
                        metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
                        created_at timestamptz NOT NULL DEFAULT now()
                    )
                    """
                )
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_sessions_user ON chat_sessions(user_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id, order_index)")
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chat_session_snapshots_session "
                    "ON chat_session_snapshots(session_id, created_at DESC)"
                )

    def create_session(self, *, user_id: str | None, title: str, session_id: str | None = None) -> SessionSummary:
        sid = session_id or str(uuid4())
        with self._psycopg.connect(self.dsn, row_factory=self._dict_row) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO chat_sessions (session_id, user_id, title)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (session_id) DO NOTHING
                    """,
                    (sid, user_id or "anonymous", title),
                )
        session = self.get_session(sid)
        if session is None:
            raise RuntimeError("session creation failed")
        return session

    def get_session(self, session_id: str) -> SessionSummary | None:
        with self._psycopg.connect(self.dsn, row_factory=self._dict_row) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT s.session_id, s.user_id, s.title, s.is_active, s.created_at, s.updated_at,
                           COUNT(m.message_id)::int AS message_count
                    FROM chat_sessions s
                    LEFT JOIN chat_messages m ON m.session_id = s.session_id
                    WHERE s.session_id = %s
                    GROUP BY s.session_id
                    """,
                    (session_id,),
                )
                row = cursor.fetchone()
        return _summary_from_row(row) if row else None

    def list_sessions(
        self,
        *,
        user_id: str | None,
        skip: int,
        limit: int,
        active_only: bool,
    ) -> list[SessionSummary]:
        where: list[str] = []
        params: list[Any] = []
        if user_id:
            where.append("s.user_id = %s")
            params.append(user_id)
        if active_only:
            where.append("s.is_active = true")
        where_sql = "WHERE " + " AND ".join(where) if where else ""
        params.extend([limit, skip])
        with self._psycopg.connect(self.dsn, row_factory=self._dict_row) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT s.session_id, s.user_id, s.title, s.is_active, s.created_at, s.updated_at,
                           COUNT(m.message_id)::int AS message_count
                    FROM chat_sessions s
                    LEFT JOIN chat_messages m ON m.session_id = s.session_id
                    {where_sql}
                    GROUP BY s.session_id
                    ORDER BY s.updated_at DESC
                    LIMIT %s OFFSET %s
                    """,
                    params,
                )
                rows = cursor.fetchall()
        return [_summary_from_row(row) for row in rows]

    def update_session(self, session_id: str, *, title: str | None, is_active: bool | None) -> SessionSummary | None:
        current = self.get_session(session_id)
        if current is None:
            return None
        with self._psycopg.connect(self.dsn, row_factory=self._dict_row) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE chat_sessions
                    SET title = COALESCE(%s, title),
                        is_active = COALESCE(%s, is_active),
                        updated_at = now()
                    WHERE session_id = %s
                    """,
                    (title, is_active, session_id),
                )
        return self.get_session(session_id)

    def append_message(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        route_type: str | None = None,
        trace_id: str | None = None,
        evidences: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MessageItem:
        message_id = str(uuid4())
        with self._psycopg.connect(self.dsn, row_factory=self._dict_row) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1 FROM chat_sessions WHERE session_id = %s", (session_id,))
                if cursor.fetchone() is None:
                    raise KeyError(session_id)
                cursor.execute(
                    "SELECT COALESCE(MAX(order_index), 0) + 1 AS next_order FROM chat_messages WHERE session_id = %s",
                    (session_id,),
                )
                order_index = int(cursor.fetchone()["next_order"])
                cursor.execute(
                    """
                    INSERT INTO chat_messages
                        (message_id, session_id, role, content, route_type, trace_id, evidences, metadata, order_index)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s)
                    RETURNING message_id, session_id, role, content, route_type, trace_id,
                              evidences, metadata, created_at, order_index
                    """,
                    (
                        message_id,
                        session_id,
                        role,
                        content,
                        route_type,
                        trace_id,
                        json.dumps(evidences or [], ensure_ascii=False),
                        json.dumps(metadata or {}, ensure_ascii=False),
                        order_index,
                    ),
                )
                row = cursor.fetchone()
                cursor.execute("UPDATE chat_sessions SET updated_at = now() WHERE session_id = %s", (session_id,))
        return _message_from_row(row)

    def list_messages(self, session_id: str, *, skip: int, limit: int) -> list[MessageItem]:
        with self._psycopg.connect(self.dsn, row_factory=self._dict_row) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT message_id, session_id, role, content, route_type, trace_id,
                           evidences, metadata, created_at, order_index
                    FROM chat_messages
                    WHERE session_id = %s
                    ORDER BY order_index
                    LIMIT %s OFFSET %s
                    """,
                    (session_id, limit, skip),
                )
                rows = cursor.fetchall()
        return [_message_from_row(row) for row in rows]

    def create_snapshot(
        self,
        *,
        session_id: str,
        message_id: str,
        trace_id: str | None,
        route_type: str | None,
        answer: str,
        evidences: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> SessionSnapshotItem:
        snapshot_id = str(uuid4())
        with self._psycopg.connect(self.dsn, row_factory=self._dict_row) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1 FROM chat_sessions WHERE session_id = %s", (session_id,))
                if cursor.fetchone() is None:
                    raise KeyError(session_id)
                cursor.execute(
                    """
                    INSERT INTO chat_session_snapshots
                        (snapshot_id, session_id, message_id, trace_id, route_type, answer, evidences, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                    RETURNING snapshot_id, session_id, message_id, trace_id, route_type,
                              answer, evidences, metadata, created_at
                    """,
                    (
                        snapshot_id,
                        session_id,
                        message_id,
                        trace_id,
                        route_type,
                        answer,
                        json.dumps(evidences or [], ensure_ascii=False),
                        json.dumps(metadata or {}, ensure_ascii=False),
                    ),
                )
                row = cursor.fetchone()
        return _snapshot_from_row(row)

    def list_snapshots(self, session_id: str, *, skip: int, limit: int) -> list[SessionSnapshotItem]:
        with self._psycopg.connect(self.dsn, row_factory=self._dict_row) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT snapshot_id, session_id, message_id, trace_id, route_type,
                           answer, evidences, metadata, created_at
                    FROM chat_session_snapshots
                    WHERE session_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                    """,
                    (session_id, limit, skip),
                )
                rows = cursor.fetchall()
        return [_snapshot_from_row(row) for row in rows]

    def get_snapshot(self, session_id: str, snapshot_id: str) -> SessionSnapshotItem | None:
        with self._psycopg.connect(self.dsn, row_factory=self._dict_row) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT snapshot_id, session_id, message_id, trace_id, route_type,
                           answer, evidences, metadata, created_at
                    FROM chat_session_snapshots
                    WHERE session_id = %s AND snapshot_id = %s
                    """,
                    (session_id, snapshot_id),
                )
                row = cursor.fetchone()
        return _snapshot_from_row(row) if row else None


class SessionService:
    def __init__(self, store: SessionStore) -> None:
        self.store = store

    def get_or_create_session(self, *, session_id: str | None, user_id: str | None, title_seed: str) -> SessionSummary:
        if session_id:
            existing = self.store.get_session(session_id)
            if existing:
                return existing
        return self.store.create_session(user_id=user_id or "anonymous", title=_session_title(title_seed), session_id=session_id)

    def list_sessions(self, *, user_id: str | None, skip: int, limit: int, active_only: bool) -> list[SessionSummary]:
        return self.store.list_sessions(user_id=user_id, skip=skip, limit=limit, active_only=active_only)

    def create_session(self, *, user_id: str | None, title: str | None) -> SessionSummary:
        return self.store.create_session(user_id=user_id or "anonymous", title=title or "新会话")

    def get_session(self, session_id: str) -> SessionSummary | None:
        return self.store.get_session(session_id)

    def update_session(self, session_id: str, *, title: str | None, is_active: bool | None) -> SessionSummary | None:
        return self.store.update_session(session_id, title=title, is_active=is_active)

    def soft_delete_session(self, session_id: str) -> bool:
        return self.update_session(session_id, title=None, is_active=False) is not None

    def list_messages(self, session_id: str, *, skip: int, limit: int) -> list[MessageItem]:
        return self.store.list_messages(session_id, skip=skip, limit=limit)

    def list_snapshots(self, session_id: str, *, skip: int, limit: int) -> list[SessionSnapshotItem]:
        return self.store.list_snapshots(session_id, skip=skip, limit=limit)

    def get_snapshot(self, session_id: str, snapshot_id: str) -> SessionSnapshotItem | None:
        return self.store.get_snapshot(session_id, snapshot_id)

    def save_user_message(self, *, session_id: str, content: str) -> MessageItem:
        return self.store.append_message(session_id=session_id, role="user", content=content)

    def save_assistant_response(self, *, session_id: str, response: ChatResponse) -> MessageItem:
        evidence_payload = [evidence.model_dump(mode="json") for evidence in response.evidences[:10]]
        metadata = {
            "route": response.route_decision.model_dump(mode="json"),
            "need_clarification": response.need_clarification,
            "evidence_count": len(response.evidences),
        }
        message = self.store.append_message(
            session_id=session_id,
            role="assistant",
            content=response.answer,
            route_type=response.route_decision.route_type.value,
            trace_id=response.trace_id,
            evidences=evidence_payload,
            metadata=metadata,
        )
        self.store.create_snapshot(
            session_id=session_id,
            message_id=message.message_id,
            trace_id=response.trace_id,
            route_type=response.route_decision.route_type.value,
            answer=response.answer,
            evidences=evidence_payload,
            metadata=metadata,
        )
        return message


_session_service: SessionService | None = None
_service_lock = RLock()


def get_session_service() -> SessionService:
    global _session_service
    if _session_service is None:
        with _service_lock:
            if _session_service is None:
                _session_service = SessionService(_build_store())
    return _session_service


def reset_session_service_for_tests(service: SessionService | None = None) -> None:
    global _session_service
    with _service_lock:
        _session_service = service


def _build_store() -> SessionStore:
    if settings.postgres_dsn:
        try:
            return PostgreSQLSessionStore(settings.postgres_dsn)
        except Exception:
            if settings.strict_external_stores:
                raise
            return InMemorySessionStore()
    if settings.strict_external_stores:
        raise RuntimeError("生产环境必须配置 GUSTOBOT_POSTGRES_DSN，不能退回内存会话存储。")
    return InMemorySessionStore()


def _summary_from_row(row: dict[str, Any]) -> SessionSummary:
    return SessionSummary(
        session_id=row["session_id"],
        user_id=row.get("user_id"),
        title=row["title"],
        is_active=bool(row["is_active"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        message_count=int(row.get("message_count") or 0),
    )


def _message_from_row(row: dict[str, Any]) -> MessageItem:
    return MessageItem(
        message_id=row["message_id"],
        session_id=row["session_id"],
        role=row["role"],
        content=row["content"],
        route_type=row.get("route_type"),
        trace_id=row.get("trace_id"),
        evidences=list(row.get("evidences") or []),
        metadata=dict(row.get("metadata") or {}),
        created_at=row["created_at"],
        order_index=int(row["order_index"]),
    )


def _snapshot_from_row(row: dict[str, Any]) -> SessionSnapshotItem:
    return SessionSnapshotItem(
        snapshot_id=row["snapshot_id"],
        session_id=row["session_id"],
        message_id=row["message_id"],
        trace_id=row.get("trace_id"),
        route_type=row.get("route_type"),
        answer=row["answer"],
        evidences=list(row.get("evidences") or []),
        metadata=dict(row.get("metadata") or {}),
        created_at=row["created_at"],
    )


def _session_title(text: str) -> str:
    normalized = " ".join(text.strip().split())
    return (normalized[:40] or "新会话")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_postgres_driver() -> tuple[Any, Any]:
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ModuleNotFoundError as exc:
        raise RuntimeError("psycopg is required for PostgreSQL session storage") from exc
    return psycopg, dict_row
