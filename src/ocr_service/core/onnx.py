import logging
from pathlib import Path

import onnxruntime as ort


def load_model(model_dir, name, device_id=None):
    path = Path(model_dir) / (name + ".onnx")
    if not path.is_file():
        raise FileNotFoundError(f"Required ONNX model is missing: {path}")
    with path.open("rb") as stream:
        if stream.read(64).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"Model is a Git LFS pointer: {path}. Run git lfs pull before building the image.")
    options = ort.SessionOptions()
    # OCR crops have varying widths; avoid retaining per-shape allocation plans.
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 2
    options.log_severity_level = 3
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    run_options = ort.RunOptions()
    logging.getLogger(__name__).info("Loaded CPU model: %s", path.name)
    return session, run_options
