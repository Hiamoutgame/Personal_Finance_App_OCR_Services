import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ocr_service.application import create_app
from ocr_service.core.settings import Settings
from ocr_service.services.ocr import OCRService


ROOT = Path(__file__).resolve().parents[1]


class FakeOCR:
    def __call__(self, image, device_id=0):
        return [([[10, 10], [200, 10], [200, 30], [10, 30]], ("Tong cong 120.000 VND", 1.0))]


class FakeLayout:
    def forward(self, images, thr=0.5):
        return [[{"type": "text", "bbox": [0, 0, 220, 40], "score": 0.9}] for _ in images]


class FakeService(OCRService):
    def initialize(self):
        self.ocr = FakeOCR()
        self.layout_recognizer = FakeLayout()


@pytest.fixture
def client():
    with TestClient(create_app(Settings(ROOT), service_factory=FakeService)) as client:
        yield client


@pytest.fixture
def image_bytes():
    stream = io.BytesIO()
    Image.new("RGB", (240, 60), "white").save(stream, format="PNG")
    return stream.getvalue()


@pytest.fixture
def pdf_bytes():
    stream = io.BytesIO()
    first = Image.new("RGB", (240, 60), "white")
    second = Image.new("RGB", (240, 60), "white")
    first.save(stream, format="PDF", save_all=True, append_images=[second])
    first.close()
    second.close()
    return stream.getvalue()
