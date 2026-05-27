"""上传文件登记与读取模块。"""

from app.uploads.service import get_upload_service, reset_upload_service_for_tests

__all__ = ["get_upload_service", "reset_upload_service_for_tests"]
