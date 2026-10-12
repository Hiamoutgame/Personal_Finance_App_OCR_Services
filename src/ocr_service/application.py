import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from .api.routes import router
from .core.settings import Settings
from .services.ocr import OCRService


logger = logging.getLogger(__name__)


def create_app(settings: Settings, service_factory=OCRService):
    @asynccontextmanager
    async def lifespan(app):
        service = service_factory(settings)
        app.state.ocr_service = service
        try:
            await run_in_threadpool(service.initialize)
            logger.info("OCR service ready; model directory: %s", settings.models_dir)
            yield
        finally:
            await run_in_threadpool(service.close)

    app = FastAPI(
        title="OCR Service",
        summary="Upload file and extract OCR text.",
        description="Upload an image or PDF using multipart/form-data with the field name file.",
        version="1.0.0",
        lifespan=lifespan,
        openapi_tags=[
            {"name": "OCR", "description": "Extract text from uploaded images or PDF files."},
            {"name": "System", "description": "Service health and operational endpoints."},
        ],
        swagger_ui_parameters={"displayRequestDuration": True, "docExpansion": "none", "filter": True, "tryItOutEnabled": True},
    )
    app.state.settings = settings
    app.include_router(router)
    return app
