import logging
import threading
import time
from contextlib import closing

from fastapi import HTTPException

from ocr_service.core.settings import Settings
from .images import iter_images
from .invoice import boxes_to_text, ocr_image_to_boxes, ocr_image_to_invoice_layout


logger = logging.getLogger(__name__)


class OCRService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.lock = threading.Lock()
        self.ocr = None
        self.layout_recognizer = None

    def initialize(self):
        from ocr_service.engines.ocr_onnx import OCR

        with self.lock:
            if self.ocr is None:
                root = self.settings.models_dir
                self.ocr = OCR(str(root / "onnx"), str(root / "vietocr" / "weight"))

    def close(self):
        with self.lock:
            self.layout_recognizer = None
            self.ocr = None

    def process(self, stream, filename, content_type, layout):
        start = time.perf_counter()
        with self.lock:
            if self.ocr is None:
                raise RuntimeError("OCR service has not been initialized")
            stream.seek(0)
            if not stream.read(1):
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            stream.seek(0)
            pages = []
            with closing(iter_images(stream, filename, content_type)) as images:
                for index, image in enumerate(images):
                    if layout == "document" and self.layout_recognizer is None:
                        from ocr_service.engines.layout_recognizer import LayoutRecognizer4YOLOv10
                        self.layout_recognizer = LayoutRecognizer4YOLOv10(model_dir=str(self.settings.models_dir / "onnx"))
                    regions = self.layout_recognizer.forward([image], thr=0.5)[0] if layout == "document" else None
                    boxes = ocr_image_to_boxes(self.ocr, image)
                    page = {"page": index + 1, "text": boxes_to_text(boxes)}
                    if layout == "invoice":
                        page["invoice"] = ocr_image_to_invoice_layout(self.ocr, image, boxes=boxes)
                    elif layout == "document":
                        page["layout_regions"] = regions
                    pages.append(page)
            response = {"text": "\n\n".join(p["text"] for p in pages if p["text"])}
            if layout != "none":
                response.update(layout=layout, pages=pages)
                if len(pages) == 1:
                    key = "invoice" if layout == "invoice" else "layout_regions"
                    response[key] = pages[0][key]
        logger.info("OCR complete: mode=%s pages=%d elapsed=%.2fs", layout, len(pages), time.perf_counter() - start)
        return response
