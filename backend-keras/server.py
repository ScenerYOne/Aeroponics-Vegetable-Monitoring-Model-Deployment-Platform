from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import cv2
import numpy as np
import base64
from pathlib import Path
import uuid
import time
import gc
import os

MODEL_DIR = os.getenv("MODEL_DIR", "uploaded_models")
MAX_LOADED_MODELS = int(os.getenv("MAX_LOADED_MODELS", 2))

app = FastAPI(title="Keras Universal Model API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(MODEL_DIR)
UPLOAD_DIR.mkdir(exist_ok=True)

loaded_models = {}

def manage_memory():
    if len(loaded_models) > MAX_LOADED_MODELS:
        oldest_id = min(
            loaded_models,
            key=lambda k: loaded_models[k]["last_used"]
        )
        print(f"🧹 Unloading model {oldest_id}")
        del loaded_models[oldest_id]
        tf.keras.backend.clear_session()
        gc.collect()

def infer_input_size(model):
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    if len(shape) >= 3:
        return (shape[2], shape[1])
    return None  

def detect_task(preds):
    """
    Detect task type from model output
    """
    if preds.ndim == 2 and preds.shape[1] > 1:
        return "classification"
    if preds.ndim == 2 and preds.shape[1] == 1:
        return "regression"
    if preds.ndim == 3:
        return "timeseries"
    return "unknown"

# ================= API =================
@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in [".h5", ".keras"]:
        raise HTTPException(400, "รองรับเฉพาะ .h5 และ .keras")

    model_id = str(uuid.uuid4())
    model_path = UPLOAD_DIR / f"{model_id}{suffix}"

    with open(model_path, "wb") as f:
        f.write(await file.read())

    try:
        manage_memory()
        model = tf.keras.models.load_model(model_path)
        input_size = infer_input_size(model)

        loaded_models[model_id] = {
            "model": model,
            "input_size": input_size,
            "last_used": time.time(),
        }

        return {
            "model_id": model_id,
            "model_format": suffix.replace(".", ""),
            "message": "Keras model loaded"
        }

    except Exception as e:
        if model_path.exists():
            model_path.unlink()
        raise HTTPException(500, f"โหลดโมเดลไม่สำเร็จ: {e}")

@app.post("/predict")
async def predict(
    model_id: str = Form(...),
    file: UploadFile | None = File(None)
):
    if model_id not in loaded_models:
        raise HTTPException(404, "Model not loaded")

    model_info = loaded_models[model_id]
    model = model_info["model"]
    input_size = model_info["input_size"]
    model_info["last_used"] = time.time()

    if input_size is not None:
        if file is None:
            raise HTTPException(400, "Image file required")

        img_bytes = await file.read()
        img = cv2.imdecode(
            np.frombuffer(img_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )
        if img is None:
            raise HTTPException(400, "อ่านรูปไม่สำเร็จ")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, input_size)
        x = np.expand_dims(img_resized / 255.0, axis=0)

        preds = model.predict(x, verbose=0)
        task = detect_task(preds)

        # ---------- Classification ----------
        if task == "classification":
            cls_id = int(np.argmax(preds[0]))
            conf = float(preds[0][cls_id])

            label = f"Class {cls_id} ({conf:.2f})"
            cv2.putText(
                img, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2
            )

            _, buffer = cv2.imencode(".jpg", img)
            return {
                "task": "classification",
                "image": base64.b64encode(buffer).decode(),
                "predictions": [
                    {"class": cls_id, "confidence": conf}
                ],
                "model_id": model_id
            }

        # ---------- Regression  ----------
        if task == "regression":
            value = float(preds[0][0])
            return {
                "task": "regression",
                "value": value,
                "model_id": model_id
            }


    preds = model.predict(None, verbose=0)
    task = detect_task(preds)

    if task == "regression":
        return {
            "task": "regression",
            "value": float(preds[0][0]),
            "model_id": model_id
        }

    if task == "timeseries":
        return {
            "task": "timeseries",
            "values": preds[0].tolist(),
            "model_id": model_id
        }

    raise HTTPException(400, "Unsupported model output")
