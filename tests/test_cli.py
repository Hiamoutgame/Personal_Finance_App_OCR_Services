from argparse import Namespace
from pathlib import Path
import sys
from types import SimpleNamespace

from conftest import FakeOCR
from ocr_service.cli import ocr, recognizer, pipeline


def fake_backend(monkeypatch):
    monkeypatch.setitem(sys.modules, "ocr_service.engines.ocr_pytorch", SimpleNamespace(OCR=FakeOCR))


def test_ocr_cli_outputs(tmp_path, image_bytes, monkeypatch):
    fake_backend(monkeypatch)
    source = tmp_path / "receipt.png"
    source.write_bytes(image_bytes)
    output = tmp_path / "output"
    ocr.main(Namespace(inputs=str(source), output_dir=str(output), layout="invoice"))
    assert {p.name for p in output.iterdir()} == {"receipt.png", "receipt.png.txt", "receipt.png.text.json", "receipt.png.json"}
    assert (output / "receipt.png.txt").read_text(encoding="utf-8") == "Tong cong 120.000 VND"


def test_layout_cli_output(tmp_path, image_bytes, monkeypatch):
    class Layout:
        labels = ["text"]
        def __init__(self, domain):
            pass
        def forward(self, images, thr):
            return [[{"type": "text", "bbox": [0, 0, 220, 40], "score": 0.9}] for _ in images]
    monkeypatch.setattr(recognizer, "LayoutRecognizer", Layout)
    source = tmp_path / "receipt.png"
    source.write_bytes(image_bytes)
    output = tmp_path / "output"
    recognizer.main(Namespace(inputs=str(source), output_dir=str(output), mode="layout", threshold=0.5))
    assert (output / "receipt.png").is_file()


def test_table_cli_outputs(tmp_path, image_bytes, monkeypatch):
    fake_backend(monkeypatch)
    class Table(recognizer.TableStructureRecognizer):
        labels = ["table row", "table column"]
        def __init__(self):
            pass
        def __call__(self, images, thr):
            return [[
                {"label": "table row", "x0": 0, "x1": 220, "top": 0, "bottom": 40, "score": 0.9},
                {"label": "table column", "x0": 0, "x1": 220, "top": 0, "bottom": 40, "score": 0.9},
            ] for _ in images]
    monkeypatch.setattr(recognizer, "TableStructureRecognizer", Table)
    source = tmp_path / "table.png"
    source.write_bytes(image_bytes)
    output = tmp_path / "output"
    recognizer.main(Namespace(inputs=str(source), output_dir=str(output), mode="tsr", threshold=0.5))
    assert (output / "table.png").is_file()
    assert "| Tong cong 120.000 VND |" in (output / "table.png.md").read_text(encoding="utf-8")


def test_pipeline_output(tmp_path, image_bytes, monkeypatch):
    fake_backend(monkeypatch)
    class Layout:
        def __init__(self, domain):
            pass
        def forward(self, images, thr):
            return [[] for _ in images]
    monkeypatch.setattr(pipeline, "LayoutRecognizer", Layout)
    source = tmp_path / "receipt.png"
    source.write_bytes(image_bytes)
    output = tmp_path / "output"
    pipeline.main(Namespace(inputs=str(source), output_dir=str(output), threshold=0.5))
    assert (output / "receipt.png_full.md").read_text(encoding="utf-8") == "Tong cong 120.000 VND"


def test_empty_table_pages_keep_positions(monkeypatch):
    from ocr_service.engines.recognizer import Recognizer
    from ocr_service.engines.table_structure_recognizer import TableStructureRecognizer
    monkeypatch.setattr(Recognizer, "__call__", lambda self, images, thr: [[], []])
    engine = object.__new__(TableStructureRecognizer)
    assert engine([object(), object()]) == [[], []]
    assert TableStructureRecognizer.construct_table([], markdown=True) == ""
