from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def set_project_root(path):
    global PROJECT_ROOT
    PROJECT_ROOT = Path(path).resolve()


def get_project_base_directory(*parts):
    return str(PROJECT_ROOT.joinpath(*parts))


def onnx_directory():
    return get_project_base_directory("models", "onnx")


def vietocr_directory(*parts):
    return get_project_base_directory("models", "vietocr", *parts)
