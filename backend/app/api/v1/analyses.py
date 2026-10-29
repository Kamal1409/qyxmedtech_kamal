"""
Semen analysis endpoints.

POST   /api/v1/analyses              upload a video (+ sample details), queue the analysis
GET    /api/v1/analyses              all analyses, newest first
GET    /api/v1/analyses/{id}         status/progress, and the full result once done
GET    /api/v1/analyses/{id}/overlay tracked sperm paths drawn on a frame (PNG)
GET    /api/v1/analyses/{id}/video   the uploaded video
DELETE /api/v1/analyses/{id}         remove an analysis and its files
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from app.config import get_settings
from app.models.analysis import AnalysisDetail, AnalysisMeta
from app.services.casa import get_store

router = APIRouter()
settings = get_settings()


def _not_found(aid: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Analysis {aid} not found.")


@router.post("", response_model=AnalysisMeta, status_code=status.HTTP_201_CREATED,
             summary="Upload a semen analysis video")
async def create_analysis(
    file: UploadFile = File(..., description="Video recorded by the analyzer (.mp4)"),
    sample_id: str = Form(..., min_length=1, max_length=64),
    patient_ref: str | None = Form(None, max_length=64),
    volume_ml: float | None = Form(None, gt=0, le=20),
    notes: str | None = Form(None, max_length=1000),
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in settings.ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            f"Upload a video file ({', '.join(settings.ALLOWED_VIDEO_EXTENSIONS)}), not '{ext or 'no extension'}'.")
    limit = settings.MAX_UPLOAD_MB * 1024 * 1024
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=settings.DATA_DIR, suffix=ext) as tmp:
        size = 0
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                tmp.close()
                Path(tmp.name).unlink(missing_ok=True)
                raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                                    f"The video is larger than {settings.MAX_UPLOAD_MB} MB.")
            tmp.write(chunk)
    if size == 0:
        Path(tmp.name).unlink(missing_ok=True)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The uploaded file is empty.")
    return get_store().create(Path(tmp.name), file.filename or f"video{ext}", sample_id.strip(),
                              (patient_ref or "").strip() or None, volume_ml, (notes or "").strip() or None)


@router.get("", response_model=list[AnalysisMeta], summary="List analyses")
async def list_analyses():
    return get_store().list()


@router.get("/{aid}", response_model=AnalysisDetail, summary="Analysis status and result")
async def get_analysis(aid: str):
    store = get_store()
    try:
        meta = store.get(aid)
    except KeyError:
        raise _not_found(aid)
    return {**meta, "result": store.result(aid) if meta["status"] == "done" else None}


@router.get("/{aid}/overlay", summary="Trajectory overlay image", response_class=FileResponse)
async def get_overlay(aid: str):
    try:
        p = get_store().file(aid, "overlay")
    except KeyError:
        raise _not_found(aid)
    if not p.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No overlay image for this analysis yet.")
    return FileResponse(p, media_type="image/png")


@router.get("/{aid}/poster", summary="Still frame shown before the video plays", response_class=FileResponse)
async def get_poster(aid: str):
    try:
        p = get_store().file(aid, "poster")
    except KeyError:
        raise _not_found(aid)
    if not p.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No preview frame for this analysis.")
    return FileResponse(p, media_type="image/jpeg")


@router.get("/{aid}/video", summary="Uploaded video", response_class=FileResponse)
async def get_video(aid: str):
    try:
        p = get_store().file(aid, "video")
    except KeyError:
        raise _not_found(aid)
    return FileResponse(p, media_type="video/mp4" if p.suffix == ".mp4" else "application/octet-stream")


@router.delete("/{aid}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete an analysis")
async def delete_analysis(aid: str):
    try:
        get_store().delete(aid)
    except KeyError:
        raise _not_found(aid)
    except RuntimeError:
        raise HTTPException(status.HTTP_409_CONFLICT, "This analysis is running; delete it after it finishes.")
