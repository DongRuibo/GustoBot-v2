"""上传文件服务。

上传接口只登记服务端保存过的文件，并通过 upload://file_id 暴露给业务链路。
文件解析和图片理解不能读取任意本地路径，只能通过这里解析已登记文件。
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Protocol
from uuid import uuid4

from fastapi import UploadFile

from app.core.config import settings
from app.models import Attachment, UploadResponse


@dataclass(slots=True)
class UploadRecord:
    file_id: str
    kind: str
    original_name: str
    stored_name: str
    relative_path: str
    content_type: str | None
    size_bytes: int
    metadata: dict[str, Any]
    created_at: datetime
    deleted_at: datetime | None = None

    @property
    def uri(self) -> str:
        return f"upload://{self.file_id}"


class UploadStore(Protocol):
    def save_record(self, record: UploadRecord) -> None:
        ...

    def get_record(self, file_id: str) -> UploadRecord | None:
        ...

    def mark_deleted(self, file_id: str) -> bool:
        ...


class InMemoryUploadStore:
    def __init__(self) -> None:
        self._records: dict[str, UploadRecord] = {}
        self._lock = RLock()

    def save_record(self, record: UploadRecord) -> None:
        with self._lock:
            self._records[record.file_id] = record

    def get_record(self, file_id: str) -> UploadRecord | None:
        with self._lock:
            record = self._records.get(file_id)
            if record is None or record.deleted_at is not None:
                return None
            return record

    def mark_deleted(self, file_id: str) -> bool:
        with self._lock:
            record = self._records.get(file_id)
            if record is None:
                return False
            record.deleted_at = _now()
            return True


class PostgreSQLUploadStore:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._psycopg, self._dict_row = _ensure_postgres_driver()
        self.ensure_schema()

    def ensure_schema(self) -> None:
        with self._psycopg.connect(self.dsn, row_factory=self._dict_row) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS uploaded_files (
                        file_id text PRIMARY KEY,
                        kind text NOT NULL,
                        original_name text NOT NULL,
                        stored_name text NOT NULL,
                        relative_path text NOT NULL,
                        content_type text,
                        size_bytes bigint NOT NULL,
                        metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
                        created_at timestamptz NOT NULL DEFAULT now(),
                        deleted_at timestamptz
                    )
                    """
                )
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_uploaded_files_kind ON uploaded_files(kind)")

    def save_record(self, record: UploadRecord) -> None:
        with self._psycopg.connect(self.dsn, row_factory=self._dict_row) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO uploaded_files
                        (file_id, kind, original_name, stored_name, relative_path,
                         content_type, size_bytes, metadata, created_at, deleted_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                    ON CONFLICT (file_id) DO UPDATE
                    SET kind = EXCLUDED.kind,
                        original_name = EXCLUDED.original_name,
                        stored_name = EXCLUDED.stored_name,
                        relative_path = EXCLUDED.relative_path,
                        content_type = EXCLUDED.content_type,
                        size_bytes = EXCLUDED.size_bytes,
                        metadata = EXCLUDED.metadata,
                        deleted_at = EXCLUDED.deleted_at
                    """,
                    (
                        record.file_id,
                        record.kind,
                        record.original_name,
                        record.stored_name,
                        record.relative_path,
                        record.content_type,
                        record.size_bytes,
                        json.dumps(record.metadata, ensure_ascii=False),
                        record.created_at,
                        record.deleted_at,
                    ),
                )

    def get_record(self, file_id: str) -> UploadRecord | None:
        with self._psycopg.connect(self.dsn, row_factory=self._dict_row) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT file_id, kind, original_name, stored_name, relative_path,
                           content_type, size_bytes, metadata, created_at, deleted_at
                    FROM uploaded_files
                    WHERE file_id = %s AND deleted_at IS NULL
                    """,
                    (file_id,),
                )
                row = cursor.fetchone()
        return _record_from_row(row) if row else None

    def mark_deleted(self, file_id: str) -> bool:
        with self._psycopg.connect(self.dsn, row_factory=self._dict_row) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "UPDATE uploaded_files SET deleted_at = now() WHERE file_id = %s AND deleted_at IS NULL",
                    (file_id,),
                )
                return cursor.rowcount > 0


class UploadService:
    def __init__(self, store: UploadStore, *, upload_dir: Path) -> None:
        self.store = store
        self.upload_dir = upload_dir
        self.files_dir = upload_dir / "files"
        self.images_dir = upload_dir / "images"
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

    async def save_upload(self, upload: UploadFile, *, kind: str) -> UploadResponse:
        original_name = Path(upload.filename or "uploaded").name
        extension = Path(original_name).suffix.lower()
        allowed_extensions = _allowed_extensions(kind)
        if extension not in allowed_extensions:
            raise ValueError(f"unsupported_extension:{extension}")

        payload = await upload.read()
        max_bytes = settings.upload_max_mb * 1024 * 1024
        if len(payload) > max_bytes:
            raise OverflowError(f"file_too_large:{settings.upload_max_mb}")

        file_id = str(uuid4())
        stored_name = f"{file_id}{extension}"
        target_dir = self.images_dir if kind == "image" else self.files_dir
        target_path = target_dir / stored_name
        target_path.write_bytes(payload)

        record = UploadRecord(
            file_id=file_id,
            kind=kind,
            original_name=original_name,
            stored_name=stored_name,
            relative_path=_relative_upload_path(target_path, self.upload_dir),
            content_type=upload.content_type,
            size_bytes=len(payload),
            metadata={"extension": extension},
            created_at=_now(),
        )
        self.store.save_record(record)
        return self._to_response(record)

    def get_record(self, file_id_or_uri: str) -> UploadRecord | None:
        file_id = _file_id_from_uri(file_id_or_uri)
        return self.store.get_record(file_id) if file_id else None

    def file_path(self, file_id_or_uri: str) -> Path | None:
        record = self.get_record(file_id_or_uri)
        if record is None:
            return None
        path = (self.upload_dir / record.relative_path).resolve()
        upload_root = self.upload_dir.resolve()
        if upload_root not in path.parents and path != upload_root:
            return None
        return path if path.exists() else None

    def load_bytes(self, file_id_or_uri: str) -> tuple[UploadRecord, bytes] | None:
        record = self.get_record(file_id_or_uri)
        if record is None:
            return None
        path = self.file_path(record.file_id)
        if path is None:
            return None
        return record, path.read_bytes()

    def resolve_attachment(self, attachment: dict[str, Any]) -> dict[str, Any]:
        uri = attachment.get("uri")
        if not isinstance(uri, str) or not uri.startswith("upload://"):
            return attachment
        loaded = self.load_bytes(uri)
        if loaded is None:
            return attachment
        record, payload = loaded
        resolved = dict(attachment)
        resolved.setdefault("type", record.kind)
        resolved["filename"] = record.original_name
        resolved["content_type"] = record.content_type
        resolved["content_base64"] = base64.b64encode(payload).decode("ascii")
        resolved.setdefault("metadata", {})["upload_file_id"] = record.file_id
        return resolved

    def delete_upload(self, file_id_or_uri: str) -> bool:
        record = self.get_record(file_id_or_uri)
        if record is None:
            return False
        deleted = self.store.mark_deleted(record.file_id)
        path = self.file_path(record.file_id)
        if path is not None:
            try:
                path.unlink()
            except OSError:
                pass
        return deleted

    def _to_response(self, record: UploadRecord) -> UploadResponse:
        attachment_type = "image" if record.kind == "image" else "file"
        return UploadResponse(
            file_id=record.file_id,
            kind=record.kind,
            original_name=record.original_name,
            size_bytes=record.size_bytes,
            content_type=record.content_type,
            uri=record.uri,
            attachment=Attachment(
                type=attachment_type,
                filename=record.original_name,
                content_type=record.content_type,
                uri=record.uri,
            ),
        )


_upload_service: UploadService | None = None
_service_lock = RLock()


def get_upload_service() -> UploadService:
    global _upload_service
    if _upload_service is None:
        with _service_lock:
            if _upload_service is None:
                _upload_service = UploadService(_build_store(), upload_dir=Path(settings.upload_dir))
    return _upload_service


def reset_upload_service_for_tests(service: UploadService | None = None) -> None:
    global _upload_service
    with _service_lock:
        _upload_service = service


def _build_store() -> UploadStore:
    if settings.postgres_dsn:
        try:
            return PostgreSQLUploadStore(settings.postgres_dsn)
        except Exception:
            if settings.strict_external_stores:
                raise
            return InMemoryUploadStore()
    if settings.strict_external_stores:
        raise RuntimeError("生产环境必须配置 GUSTOBOT_POSTGRES_DSN，不能退回内存上传记录。")
    return InMemoryUploadStore()


def _allowed_extensions(kind: str) -> set[str]:
    raw = settings.allowed_image_extensions if kind == "image" else settings.allowed_file_extensions
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def _relative_upload_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _file_id_from_uri(value: str) -> str | None:
    if value.startswith("upload://"):
        return value.removeprefix("upload://").strip()
    return value.strip() or None


def _record_from_row(row: dict[str, Any]) -> UploadRecord:
    return UploadRecord(
        file_id=row["file_id"],
        kind=row["kind"],
        original_name=row["original_name"],
        stored_name=row["stored_name"],
        relative_path=row["relative_path"],
        content_type=row.get("content_type"),
        size_bytes=int(row["size_bytes"]),
        metadata=dict(row.get("metadata") or {}),
        created_at=row["created_at"],
        deleted_at=row.get("deleted_at"),
    )


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_postgres_driver() -> tuple[Any, Any]:
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ModuleNotFoundError as exc:
        raise RuntimeError("psycopg is required for PostgreSQL upload storage") from exc
    return psycopg, dict_row
