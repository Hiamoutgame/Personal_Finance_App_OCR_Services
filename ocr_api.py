import io
import logging
import os
import sys
import threading
from contextlib import asynccontextmanager
from typing import Any, Literal

import pdfplumber
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from module.ocr_onnx import OCR
from module import LOCK_KEY_pdfplumber
from module.layout_recognizer import LayoutRecognizer4YOLOv10 as LayoutRecognizer


from t_ocr import boxes_to_text, ocr_image_to_boxes, ocr_image_to_invoice_layout
from utils import get_base_config

os.environ["CUDA_VISIBLE_DEVICES"] = ""

_ocr = None
_ocr_lock = threading.Lock()
_layout_recognizer = None
_layout_lock = threading.Lock()
logger = logging.getLogger("uvicorn.error")


class OCRResponse(BaseModel):
    text: str = Field(
        ...,
        description="Extracted OCR text. PDF pages are separated by a blank line.",
        examples=["Cong ty ABC\nHoa don ban hang\nTong tien: 120.000 VND"],
    )
    layout: Literal["none", "invoice", "document"] | None = Field(
        None,
        description="Structured layout mode used for this response.",
        examples=["invoice", "document"],
    )
    pages: list[dict[str, Any]] | None = Field(
        None,
        description="Per-page structured OCR output. Present when layout=invoice or layout=document.",
    )
    invoice: dict[str, Any] | None = Field(
        None,
        description="Shortcut to the first page invoice layout. Present when layout=invoice.",
    )
    layout_regions: list[dict[str, Any]] | None = Field(
        None,
        description="Shortcut to the first page document layout regions. Present when layout=document.",
    )


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status.", examples=["ok"])


API_DESCRIPTION = """
Upload an image or PDF file and receive extracted OCR text as JSON.

Use multipart/form-data with the form field name `file`.
Supported inputs are PDFs and image formats that Pillow can read, such as JPG, PNG, and WEBP.
"""

OPENAPI_TAGS = [
    {
        "name": "OCR",
        "description": "Extract text from uploaded images or PDF files.",
    },
    {
        "name": "System",
        "description": "Service health and operational endpoints.",
    },
]


def _load_ocr():
    global _ocr
    if _ocr is None:
        with _ocr_lock:
            if _ocr is None:
                _ocr = OCR()
    return _ocr


def _load_layout_recognizer():
    global _layout_recognizer
    if _layout_recognizer is None:
        with _layout_lock:
            if _layout_recognizer is None:
                _layout_recognizer = LayoutRecognizer("layout")
    return _layout_recognizer


def _argv_value(name):
    prefix = f"{name}="
    for index, arg in enumerate(sys.argv):
        if arg == name and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return None


def _docs_host():
    ragflow_config = get_base_config("ragflow", {}) or {}
    host = (
        _argv_value("--host")
        or os.environ.get("UVICORN_HOST")
        or os.environ.get("HOST")
        or ragflow_config.get("host")
        or "127.0.0.1"
    )
    return "127.0.0.1" if host in {"0.0.0.0", "::"} else host


def _docs_port():
    ragflow_config = get_base_config("ragflow", {}) or {}
    return (
        _argv_value("--port")
        or os.environ.get("UVICORN_PORT")
        or os.environ.get("PORT")
        or ragflow_config.get("http_port")
        or 8000
    )


def _log_docs_url(app):
    if not app.docs_url:
        return

    host = _docs_host()
    port = _docs_port()
    logger.info("Local Swagger UI: http://%s:%s%s", host, port, app.docs_url)
    logger.info("Local Swagger port: %s", port)


@asynccontextmanager
async def lifespan(app):
    _log_docs_url(app)
    await run_in_threadpool(_load_ocr)
    yield


app = FastAPI(
    title="OCR Service",
    summary="Upload file and extract OCR text.",
    description=API_DESCRIPTION,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=OPENAPI_TAGS,
    swagger_ui_parameters={
        "displayRequestDuration": True,
        "docExpansion": "none",
        "filter": True,
        "tryItOutEnabled": True,
    },
)


def _pdf_to_images(content):
    images = []
    with sys.modules[LOCK_KEY_pdfplumber]:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                images.append(page.to_image(resolution=216).annotated.convert("RGB"))
    return images


def _image_to_images(content):
    try:
        return [Image.open(io.BytesIO(content)).convert("RGB")]
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a supported image or PDF.") from exc


def _uploaded_file_to_images(content, filename, content_type):
    is_pdf = content_type == "application/pdf" or filename.lower().endswith(".pdf")
    if is_pdf:
        try:
            return _pdf_to_images(content)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Could not read uploaded PDF.") from exc
    return _image_to_images(content)


def _run_ocr(content, filename, content_type, layout):
    images = _uploaded_file_to_images(content, filename, content_type)
    ocr = _load_ocr()

    if layout == "document":
        recognizer = _load_layout_recognizer()
        layouts = recognizer.forward(images, thr=float(0.5))
        pages = []
        for index, image in enumerate(images):
            boxes = ocr_image_to_boxes(ocr, image)
            page_layouts = layouts[index] if index < len(layouts) else []
            pages.append(
                {
                    "page": index + 1,
                    "text": boxes_to_text(boxes),
                    "layout_regions": page_layouts,
                }
            )
        response = {
            "text": "\n\n".join(page["text"] for page in pages if page["text"]),
            "layout": layout,
            "pages": pages,
        }
        if len(pages) == 1:
            response["layout_regions"] = pages[0]["layout_regions"]
        return response

    with _ocr_lock:
        if layout == "invoice":
            pages = []
            for index, image in enumerate(images):
                boxes = ocr_image_to_boxes(ocr, image)
                invoice_layout = ocr_image_to_invoice_layout(
                    ocr,
                    image,
                    boxes=boxes,
                )
                pages.append(
                    {
                        "page": index + 1,
                        "text": boxes_to_text(boxes),
                        "invoice": invoice_layout,
                    }
                )

            response = {
                "text": "\n\n".join(page["text"] for page in pages if page["text"]),
                "layout": layout,
                "pages": pages,
            }
            if len(pages) == 1:
                response["invoice"] = pages[0]["invoice"]
            return response

        texts = []
        for image in images:
            boxes = ocr_image_to_boxes(ocr, image)
            texts.append(boxes_to_text(boxes))
        return {"text": "\n\n".join(text for text in texts if text)}


@app.get(
    "/health",
    tags=["System"],
    summary="Check service health",
    response_model=HealthResponse,
    responses={
        200: {
            "description": "The OCR service is running.",
            "content": {"application/json": {"example": {"status": "ok"}}},
        },
    },
)
def health():
    return {"status": "ok"}


@app.post(
    "/ocr",
    tags=["OCR"],
    summary="Extract text from an uploaded file",
    description=(
        "Upload one image or PDF file using multipart/form-data. "
        "The file field name must be `file`. Use `layout=invoice` to include structured invoice JSON."
    ),
    response_model=OCRResponse,
    response_model_exclude_none=True,
    responses={
        200: {
            "description": "OCR completed successfully.",
            "content": {
                "application/json": {
                    "examples": {
                        "text_only": {
                            "summary": "Raw text",
                            "value": {
                                "text": "Cong ty ABC\nHoa don ban hang\nTong tien: 120.000 VND"
                            },
                        },
                        "invoice_layout": {
                            "summary": "Structured invoice layout",
                            "value": {
                                "text": "PHIEU TINH TIEN\nTong cong 120.000 VND",
                                "layout": "invoice",
                                "pages": [
                                    {
                                        "page": 1,
                                        "text": "PHIEU TINH TIEN\nTong cong 120.000 VND",
                                        "invoice": {
                                            "fields": {
                                                "total_amount": {
                                                    "value": 120000,
                                                    "raw": "120.000 VND",
                                                }
                                            },
                                            "sections": {
                                                "header": {},
                                                "items": {},
                                                "totals": {},
                                                "footer": {},
                                            },
                                            "items": [],
                                            "boxes": [],
                                        },
                                    }
                                ],
                            },
                        },
                    }
                }
            },
        },
        400: {
            "description": "The uploaded file is empty or cannot be read as an image/PDF.",
            "content": {
                "application/json": {
                    "examples": {
                        "empty_file": {
                            "summary": "Empty file",
                            "value": {"detail": "Uploaded file is empty."},
                        },
                        "unsupported_file": {
                            "summary": "Unsupported file",
                            "value": {"detail": "Uploaded file is not a supported image or PDF."},
                        },
                    }
                }
            },
        },
    },
)
async def ocr_upload(
    file: UploadFile = File(
        ...,
        description="Image or PDF file to OCR. Use the multipart field name `file`.",
    ),
    layout: Literal["none", "invoice", "document"] = Query(
        "none",
        description=(
            "Use `invoice` to return structured invoice JSON with sections, fields, items, and boxes. "
            "Use `document` to return layout regions detected by the ONNX layout model."
        ),
    ),
):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    return await run_in_threadpool(
        _run_ocr,
        content,
        file.filename or "",
        file.content_type or "",
        layout,
    )


def _run_local_server():
    import uvicorn

    host = _argv_value("--host") or os.environ.get("HOST") or "127.0.0.1"
    port = int(_argv_value("--port") or os.environ.get("PORT") or _docs_port())
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    _run_local_server()
