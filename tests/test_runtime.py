import importlib.abc
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from ocr_service.application import create_app
from ocr_service.core.settings import Settings, parse_server_args
from ocr_service.services.ocr import OCRService
from ocr_service.services.images import init_in_out


ROOT = Path(__file__).resolve().parents[1]


def test_settings_precedence(monkeypatch):
    monkeypatch.delenv("HOST", raising=False)
    monkeypatch.delenv("PORT", raising=False)
    assert Settings.from_environment(ROOT).port == 6969
    monkeypatch.setenv("HOST", "127.0.0.2")
    monkeypatch.setenv("PORT", "7777")
    settings = Settings.from_environment(ROOT)
    assert (settings.host, settings.port) == ("127.0.0.2", 7777)
    settings = parse_server_args(settings, ["--host", "127.0.0.1", "--port", "8888"])
    assert (settings.host, settings.port) == ("127.0.0.1", 8888)


def test_main_forces_one_worker_and_cli_overrides_env(monkeypatch):
    import importlib.util
    from types import SimpleNamespace
    captured = {}
    monkeypatch.setenv("HOST", "127.0.0.2")
    monkeypatch.setenv("PORT", "7777")
    monkeypatch.setenv("WEB_CONCURRENCY", "4")
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=lambda app, **kwargs: captured.update(kwargs)))
    spec = importlib.util.spec_from_file_location("entry_test", ROOT / "ocr_api.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "configure_logging", lambda: None)
    module.main(["--host", "127.0.0.1", "--port", "8888"])
    assert captured == {"host": "127.0.0.1", "port": 8888, "workers": 1}


def test_import_without_optional_dependencies(tmp_path):
    code = """
import importlib.abc, importlib.util, sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','torchvision','matplotlib','huggingface_hub'}:
            raise ImportError('Optional dependency imported: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
spec = importlib.util.spec_from_file_location('entry', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert not any('onnxruntime' == name for name in sys.modules)
assert module.app.title == 'OCR Service'
"""
    result = subprocess.run([sys.executable, "-B", "-c", code, str(ROOT / "ocr_api.py")], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("script,command", [
    ("ocr_api.py", []), ("ocr_cli.py", []), ("ocr_cli.py", ["ocr"]), ("ocr_cli.py", ["recognizer"]), ("ocr_cli.py", ["pipeline"]),
])
def test_entrypoint_help_from_other_directory(script, command, tmp_path):
    result = subprocess.run([sys.executable, "-B", str(ROOT / script), *command, "--help"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_multiple_pdfs_keep_all_pages(tmp_path, pdf_bytes, image_bytes):
    source = tmp_path / "inputs"
    source.mkdir()
    (source / "a.pdf").write_bytes(pdf_bytes)
    (source / "b.pdf").write_bytes(pdf_bytes)
    (source / "c.png").write_bytes(image_bytes)
    from argparse import Namespace
    images, outputs = init_in_out(Namespace(inputs=str(source), output_dir=str(tmp_path / "outputs")))
    try:
        assert len(images) == len(outputs) == 5
        assert [Path(p).name for p in outputs] == ["a.pdf_0.jpg", "a.pdf_1.jpg", "b.pdf_0.jpg", "b.pdf_1.jpg", "c.png"]
    finally:
        for image in images:
            image.close()


def test_missing_model_fails_startup(tmp_path):
    with pytest.raises(FileNotFoundError, match="det.onnx"):
        with TestClient(create_app(Settings(tmp_path))):
            pass


def test_lfs_pointer_fails_with_actionable_error(tmp_path):
    model_dir = tmp_path / "models" / "onnx"
    model_dir.mkdir(parents=True)
    (model_dir / "det.onnx").write_text("version https://git-lfs.github.com/spec/v1\n", encoding="ascii")
    with pytest.raises(ValueError, match="Git LFS pointer"):
        with TestClient(create_app(Settings(tmp_path))):
            pass


@pytest.mark.integration
def test_bundled_models_match_original():
    expected = json.loads((ROOT / "tests/fixtures/hoadon2_baseline.json").read_text(encoding="utf-8"))
    service = OCRService(Settings(ROOT))
    service.initialize()
    try:
        for mode in ["none", "invoice", "document"]:
            with (ROOT / expected["image"]).open("rb") as stream:
                payload = service.process(stream, expected["image"], "image/png", mode)
            assert payload["text"] == expected["invoice"]["text"]
            if mode == "invoice":
                assert json.loads(json.dumps(payload["invoice"])) == expected["invoice"]
                assert payload["invoice"]["boxes"] == expected["boxes"]
            elif mode == "document":
                assert payload["layout_regions"] == expected["layout_regions"]
        assert "torch" not in sys.modules
    finally:
        service.close()
