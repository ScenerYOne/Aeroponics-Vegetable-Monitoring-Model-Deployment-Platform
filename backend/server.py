from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import base64
from pathlib import Path
import uuid
import os


HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
MODEL_DIR = os.getenv("MODEL_DIR", "uploaded_models")
YOLO_CONF = float(os.getenv("YOLO_CONF", 0.3))
YOLO_IOU = float(os.getenv("YOLO_IOU", 0.45))


app = FastAPI(title="YOLO Model Tester API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(MODEL_DIR)
UPLOAD_DIR.mkdir(exist_ok=True)


loaded_models = {}
current_model_id: str | None = None

SUPPORTED_EXT = (".pt", ".pth", ".onnx")

CLASS_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255)
]

@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    global current_model_id

    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXT:
        raise HTTPException(
            status_code=400,
            detail="รองรับเฉพาะไฟล์ .pt .pth และ .onnx"
        )

    model_id = str(uuid.uuid4())
    model_path = UPLOAD_DIR / f"{model_id}{suffix}"

    contents = await file.read()
    with open(model_path, "wb") as f:
        f.write(contents)

    try:
        model = YOLO(str(model_path))

        names = getattr(model, "names", None)
        if isinstance(names, dict):
            class_names = list(names.values())
        elif isinstance(names, (list, tuple)):
            class_names = list(names)
        else:
            class_names = []  

        loaded_models[model_id] = {
            "model": model,
            "names": class_names,
            "format": suffix.replace(".", ""),
            "path": str(model_path)
        }

        current_model_id = model_id

        return {
            "model_id": model_id,
            "model_format": suffix.replace(".", ""),
            "class_names": class_names,
            "message": "Model uploaded and activated"
        }

    except Exception as e:
        if model_path.exists():
            model_path.unlink()
        raise HTTPException(
            status_code=500,
            detail=f"โหลดโมเดลไม่สำเร็จ {str(e)}"
        )

@app.get("/current-model")
async def get_current_model():
    if not current_model_id or current_model_id not in loaded_models:
        return {
            "current_model_id": None,
            "class_names": [],
            "model_format": None
        }

    info = loaded_models[current_model_id]
    return {
        "current_model_id": current_model_id,
        "class_names": info["names"],
        "model_format": info["format"]
    }

@app.post("/use-model")
async def use_model(model_id: str = Form(...)):
    global current_model_id

    if model_id not in loaded_models:
        raise HTTPException(404, detail="ไม่พบโมเดล")

    current_model_id = model_id
    return {
        "message": "Model switched",
        "current_model_id": model_id,
        "model_format": loaded_models[model_id]["format"]
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_id: str = Form(...)
):
    if model_id not in loaded_models:
        raise HTTPException(
            status_code=404,
            detail="ไม่พบโมเดลที่เลือก"
        )

    model_info = loaded_models[model_id]
    model = model_info["model"]
    class_names = model_info["names"]

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(400, detail="ไม่สามารถอ่านรูปภาพได้")

    results = model(img, conf=YOLO_CONF, iou=YOLO_IOU)[0]

    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        label_name = (
            class_names[cls_id]
            if cls_id < len(class_names)
            else f"Class {cls_id}"
        )

        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
        label = f"{label_name} {conf:.0%}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
        cv2.putText(
            img, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
            (255, 255, 255), 2
        )

        detections.append({
            "cls": cls_id,
            "name": label_name,
            "conf": conf
        })

    _, buffer = cv2.imencode(".jpg", img)
    img_base64 = base64.b64encode(buffer).decode()

    return {
        "image": img_base64,
        "detections": detections,
        "class_names": class_names,
        "model_format": model_info["format"],
        "model_id": model_id
    }
