"""测试级应用外壳状态隔离。"""

import shutil
from pathlib import Path
from uuid import uuid4

import pytest

from app.sessions.service import InMemorySessionStore, SessionService, reset_session_service_for_tests
from app.uploads.service import InMemoryUploadStore, UploadService, reset_upload_service_for_tests


@pytest.fixture(autouse=True)
def reset_shell_services():
    # 会话和上传是阶段一新增的应用外壳状态，测试中固定走内存与临时目录，避免污染开发数据。
    upload_root = Path("data/test_uploads").resolve()
    upload_dir = upload_root / str(uuid4())
    reset_session_service_for_tests(SessionService(InMemorySessionStore()))
    reset_upload_service_for_tests(UploadService(InMemoryUploadStore(), upload_dir=upload_dir))
    yield
    reset_session_service_for_tests(None)
    reset_upload_service_for_tests(None)
    resolved_upload_dir = upload_dir.resolve()
    if upload_root in resolved_upload_dir.parents:
        shutil.rmtree(resolved_upload_dir, ignore_errors=True)
