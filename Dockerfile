# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.12.10
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=6969
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home appuser

COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']))" > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt
COPY src/ ./src/
RUN pip install --no-deps .
COPY ocr_api.py ./
COPY models/onnx/det.onnx models/onnx/layout.onnx ./models/onnx/
COPY models/vietocr/weight/cnn.onnx models/vietocr/weight/encoder.onnx models/vietocr/weight/decoder.onnx ./models/vietocr/weight/
COPY models/vietocr/config/base.yml ./models/vietocr/config/

USER appuser
EXPOSE 6969
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','6969')+'/health',timeout=4)"
CMD ["python", "ocr_api.py"]
