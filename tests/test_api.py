import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from PIL import Image


@pytest.mark.parametrize("layout", ["none", "invoice", "document"])
def test_image_contract(client, image_bytes, layout):
    response = client.post(f"/ocr?layout={layout}", files={"file": ("image.png", image_bytes, "image/png")})
    assert response.status_code == 200
    payload = response.json()
    assert payload["text"] == "Tong cong 120.000 VND"
    if layout == "none":
        assert set(payload) == {"text"}
    else:
        assert payload["layout"] == layout
        assert payload["pages"][0]["page"] == 1
        key = "invoice" if layout == "invoice" else "layout_regions"
        assert payload[key] == payload["pages"][0][key]
        if layout == "invoice":
            assert payload[key]["fields"]["total_amount"]["value"] == 120000
            assert set(payload[key]) >= {"boxes", "items", "fields", "sections", "image"}


@pytest.mark.parametrize("layout", ["none", "invoice", "document"])
def test_pdf_pages(client, pdf_bytes, layout):
    response = client.post(f"/ocr?layout={layout}", files={"file": ("input.pdf", pdf_bytes, "application/pdf")})
    assert response.status_code == 200
    payload = response.json()
    assert payload["text"] == "Tong cong 120.000 VND\n\nTong cong 120.000 VND"
    if layout != "none":
        assert [p["page"] for p in payload["pages"]] == [1, 2]
        assert "invoice" not in payload and "layout_regions" not in payload


@pytest.mark.parametrize("name,content,mime,detail", [
    ("empty.png", b"", "image/png", "Uploaded file is empty."),
    ("invalid.png", b"broken", "image/png", "Uploaded file is not a supported image or PDF."),
    ("invalid.pdf", b"broken", "application/pdf", "Could not read uploaded PDF."),
])
def test_invalid_files(client, name, content, mime, detail):
    response = client.post("/ocr", files={"file": (name, content, mime)})
    assert response.status_code == 400
    assert response.json()["detail"] == detail


def test_validation_and_docs(client, image_bytes):
    assert client.post("/ocr").status_code == 422
    assert client.post("/ocr?layout=unknown", files={"file": ("a.png", image_bytes)}).status_code == 422
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert schema["paths"]["/ocr"]["post"]["requestBody"]["content"].get("multipart/form-data")


@pytest.mark.parametrize("image_format,mime", [("JPEG", "image/jpeg"), ("WEBP", "image/webp")])
def test_other_image_formats(client, image_bytes, image_format, mime):
    stream = io.BytesIO()
    with Image.open(io.BytesIO(image_bytes)) as image:
        image.save(stream, format=image_format)
    response = client.post("/ocr", files={"file": ("image", stream.getvalue(), mime)})
    assert response.status_code == 200
    assert response.json() == {"text": "Tong cong 120.000 VND"}


def test_serialized_requests_and_responsive_health(client, image_bytes):
    service = client.app.state.ocr_service
    original = service.ocr
    entered, release = threading.Event(), threading.Event()
    state_lock = threading.Lock()
    active = peak = 0

    def slow_ocr(image, device_id=0):
        nonlocal active, peak
        with state_lock:
            active += 1
            peak = max(peak, active)
        entered.set()
        assert release.wait(5)
        try:
            return original(image, device_id)
        finally:
            with state_lock:
                active -= 1

    service.ocr = slow_ocr
    def upload(layout):
        return client.post(f"/ocr?layout={layout}", files={"file": ("a.png", image_bytes, "image/png")})

    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = [pool.submit(upload, mode) for mode in ["none", "invoice", "document"]]
        try:
            assert entered.wait(5)
            start = time.perf_counter()
            assert client.get("/health").status_code == 200
            assert time.perf_counter() - start < 1
        finally:
            release.set()
        assert all(f.result().status_code == 200 for f in pending)
    assert peak == 1
