from typing import Literal

from fastapi import APIRouter, File, Query, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from .schemas import HealthResponse, OCRResponse

router = APIRouter()


@router.get(
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
async def health():
    return {"status": "ok"}


@router.post(
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
    request: Request,
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
    return await run_in_threadpool(
        request.app.state.ocr_service.process,
        file.file,
        file.filename or "",
        file.content_type or "",
        layout,
    )
