"""multipart 上传 API。"""

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from app.models import UploadResponse
from app.uploads.service import get_upload_service

router = APIRouter(prefix="/upload", tags=["upload"])


@router.post("/file", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)) -> UploadResponse:
    return await _save_upload(file, kind="file")


@router.post("/image", response_model=UploadResponse)
async def upload_image(image: UploadFile = File(...)) -> UploadResponse:
    return await _save_upload(image, kind="image")


@router.get("/{file_id}")
def get_upload(file_id: str) -> FileResponse:
    path = get_upload_service().file_path(file_id)
    record = get_upload_service().get_record(file_id)
    if path is None or record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Upload not found")
    return FileResponse(path, media_type=record.content_type, filename=record.original_name)


@router.delete("/{file_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_upload(file_id: str) -> None:
    if not get_upload_service().delete_upload(file_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Upload not found")


async def _save_upload(file: UploadFile, *, kind: str) -> UploadResponse:
    try:
        return await get_upload_service().save_upload(file, kind=kind)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except OverflowError as exc:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)) from exc
