import os
import uuid
from pathlib import Path
from typing import Optional

import cv2
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from backend.config import (
    MAX_FILE_SIZE,
    SUPPORTED_EXTENSIONS,
    UPLOAD_DIR,
)
from backend.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


class StartRequest(BaseModel):
    video_path: str
    device: str = "cuda"


pipeline_manager = None


def get_pipeline_manager():
    global pipeline_manager
    return pipeline_manager


def set_pipeline_manager(manager):
    global pipeline_manager
    pipeline_manager = manager


@router.get("/model_status")
async def model_status():
    manager = get_pipeline_manager()
    if manager is None:
        return {
            "initialized": False,
            "grounding_dino_loaded": False,
            "clip_loaded": False,
            "active_methods": [],
            "loading": False,
            "load_progress": 0,
            "load_step": "等待初始化",
            "load_steps": [],
            "load_done": False,
            "load_error": None,
        }
    return manager.model_detector.load_status


@router.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="未选择文件")

    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的格式: {ext}",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    file_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}{ext}"

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="文件超过 500MB")

    with open(file_path, "wb") as f:
        f.write(contents)

    cap = cv2.VideoCapture(str(file_path))
    video_info = {}
    if cap.isOpened():
        video_info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / max(cap.get(cv2.CAP_PROP_FPS), 1),
        }
        cap.release()

    logger.info(f"视频上传: {file_path}")
    return {
        "path": str(file_path.relative_to(Path(__file__).resolve().parent.parent.parent)),
        "filename": file.filename,
        "size": len(contents),
        "info": video_info,
    }


@router.post("/start")
async def start_detection(req: StartRequest):
    manager = get_pipeline_manager()
    if manager is None:
        raise HTTPException(status_code=500, detail="Pipeline 未初始化")

    video_path = Path(req.video_path)
    if not video_path.exists():
        raise HTTPException(status_code=400, detail=f"视频文件不存在: {req.video_path}")

    result = manager.processor.start_pipeline(str(video_path), req.device)

    if not manager.model_detector.initialized:
        manager.model_detector.load_models(req.device)

    return result


@router.post("/stop/{pipeline_id}")
async def stop_detection(pipeline_id: str):
    manager = get_pipeline_manager()
    if manager is None:
        raise HTTPException(status_code=500, detail="Pipeline 未初始化")
    return manager.processor.stop_pipeline(pipeline_id)


@router.get("/status/{pipeline_id}")
async def get_status(pipeline_id: str):
    manager = get_pipeline_manager()
    if manager is None:
        raise HTTPException(status_code=500, detail="Pipeline 未初始化")
    return manager.processor.get_status(pipeline_id)


@router.get("/frame_results/{pipeline_id}")
async def get_frame_results(pipeline_id: str):
    manager = get_pipeline_manager()
    if manager is None:
        raise HTTPException(status_code=500, detail="Pipeline 未初始化")
    return manager.processor.get_frame_results(pipeline_id)


@router.get("/stream_video")
async def stream_video(video_path: str):
    from fastapi.responses import StreamingResponse

    path = Path(video_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="视频文件不存在")

    def iter_file():
        with open(path, "rb") as f:
            while True:
                data = f.read(64 * 1024)
                if not data:
                    break
                yield data

    return StreamingResponse(
        iter_file(),
        media_type="video/mp4",
        headers={"Content-Disposition": f"inline; filename={path.name}"},
    )