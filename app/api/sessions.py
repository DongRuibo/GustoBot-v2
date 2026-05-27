"""会话管理 API。"""

from fastapi import APIRouter, HTTPException, Query, status

from app.models import MessageItem, SessionCreate, SessionSnapshotItem, SessionSummary, SessionUpdate
from app.sessions.service import get_session_service

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("", response_model=list[SessionSummary])
def list_sessions(
    user_id: str | None = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    active_only: bool = True,
) -> list[SessionSummary]:
    return get_session_service().list_sessions(
        user_id=user_id,
        skip=skip,
        limit=limit,
        active_only=active_only,
    )


@router.post("", response_model=SessionSummary, status_code=status.HTTP_201_CREATED)
def create_session(request: SessionCreate) -> SessionSummary:
    return get_session_service().create_session(user_id=request.user_id, title=request.title)


@router.get("/{session_id}", response_model=SessionSummary)
def get_session(session_id: str) -> SessionSummary:
    session = get_session_service().get_session(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return session


@router.patch("/{session_id}", response_model=SessionSummary)
def update_session(session_id: str, request: SessionUpdate) -> SessionSummary:
    session = get_session_service().update_session(
        session_id,
        title=request.title,
        is_active=request.is_active,
    )
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return session


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(session_id: str) -> None:
    if not get_session_service().soft_delete_session(session_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")


@router.get("/{session_id}/messages", response_model=list[MessageItem])
def list_messages(
    session_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
) -> list[MessageItem]:
    if get_session_service().get_session(session_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return get_session_service().list_messages(session_id, skip=skip, limit=limit)


@router.get("/{session_id}/snapshots", response_model=list[SessionSnapshotItem])
def list_snapshots(
    session_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
) -> list[SessionSnapshotItem]:
    if get_session_service().get_session(session_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return get_session_service().list_snapshots(session_id, skip=skip, limit=limit)


@router.get("/{session_id}/snapshots/{snapshot_id}", response_model=SessionSnapshotItem)
def get_snapshot(session_id: str, snapshot_id: str) -> SessionSnapshotItem:
    if get_session_service().get_session(session_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    snapshot = get_session_service().get_snapshot(session_id, snapshot_id)
    if snapshot is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Snapshot not found")
    return snapshot
