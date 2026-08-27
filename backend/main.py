import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import CORS_ORIGINS, STATIC_DIR
from backend.logger import get_logger
from backend.pipeline.model_detector import ModelDetector
from backend.pipeline.processor import Processor
from backend.routers.pipeline_router import router as pipeline_router, set_pipeline_manager

logger = get_logger(__name__)

logging.getLogger("asyncio").setLevel(logging.CRITICAL)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)


class PipelineManager:
    def __init__(self):
        self.model_detector = ModelDetector()
        self.processor = Processor(self.model_detector)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("应用启动中...")
    manager = PipelineManager()
    set_pipeline_manager(manager)
    logger.info("PipelineManager 初始化完成")
    try:
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, manager.model_detector.load_models, "cuda")
    except Exception as e:
        logger.warning(f"异步加载模型失败: {e}")
    yield
    manager.processor.cleanup()
    logger.info("应用关闭")


def create_app() -> FastAPI:
    app = FastAPI(
        title="建筑构件识别系统",
        description="基于深度学习的装配式建筑构件自动检测与识别系统",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(pipeline_router)

    @app.get("/api/health")
    async def health_check():
        return {"status": "ok", "service": "construction-component-match"}

    if STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

    return app


app = create_app()