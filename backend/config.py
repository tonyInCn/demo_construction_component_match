import os
from pathlib import Path

HF_ENDPOINT = "https://hf-mirror.com"
os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)

BASE_DIR = Path(__file__).resolve().parent.parent

CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
]

STATIC_DIR = BASE_DIR / "frontend"

UPLOAD_DIR = BASE_DIR / "video_input"
OUTPUT_DIR = BASE_DIR / "output"
ANNOTATED_DIR = OUTPUT_DIR / "annotated"
ROIS_DIR = OUTPUT_DIR / "rois"

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

MAX_SIDE = 640
IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.45

FRAME_SAMPLE_RATE = 5
INFERENCE_MAX_SIDE = 480
BATCH_SIZE = 8

SUPPORTED_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".wmv",
    ".flv", ".webm", ".m4v", ".mpg", ".mpeg"
}
MAX_FILE_SIZE = 500 * 1024 * 1024