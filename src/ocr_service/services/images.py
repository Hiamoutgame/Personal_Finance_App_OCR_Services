import logging
import threading
from pathlib import Path

import pdfplumber
from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError


PDF_LOCK = threading.Lock()


def iter_images(stream, filename, content_type):
    stream.seek(0)
    is_pdf = content_type == "application/pdf" or filename.lower().endswith(".pdf")
    if is_pdf:
        try:
            with PDF_LOCK, pdfplumber.open(stream) as pdf:
                for page in pdf.pages:
                    image = page.to_image(resolution=216).annotated.convert("RGB")
                    try:
                        yield image
                    finally:
                        image.close()
                        page.close()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Could not read uploaded PDF.") from exc
    else:
        try:
            with Image.open(stream) as source:
                image = source.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Uploaded file is not a supported image or PDF.") from exc
        try:
            yield image
        finally:
            image.close()


def init_in_out(args):
    images, outputs = [], []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.inputs)
    paths = sorted(p for p in input_path.rglob("*") if p.is_file()) if input_path.is_dir() else [input_path]
    for path in paths:
        try:
            with path.open("rb") as stream:
                for index, image in enumerate(iter_images(stream, str(path), "")):
                    images.append(image.copy())
                    name = f"{path.name}_{index}.jpg" if path.suffix.lower() == ".pdf" else path.name
                    outputs.append(str(output_dir / name))
        except (HTTPException, OSError):
            logging.getLogger(__name__).exception("Could not read input: %s", path)
    return images, outputs
