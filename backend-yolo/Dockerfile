FROM python:3.10-slim

# System dependencies for OpenCV / Ultralytics (provides libGL.so.1 and X libs)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    python-dotenv \
    ultralytics \
    opencv-python-headless \
    onnx \
    onnxruntime


COPY server.py .

RUN mkdir -p uploaded_models

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]