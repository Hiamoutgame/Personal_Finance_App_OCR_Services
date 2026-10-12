from typing import Any, Literal

from pydantic import BaseModel, Field


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
