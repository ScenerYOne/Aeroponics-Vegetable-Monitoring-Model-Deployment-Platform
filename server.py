from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import base64
from pathlib import Path
import uuid

app = FastAPI(title="YOLO Model Tester API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploaded_models")
UPLOAD_DIR.mkdir(exist_ok=True)

# เก็บโมเดลทั้งหมด
loaded_models = {}

# เก็บโมเดลที่กำลังใช้งานอยู่ (โมเดลล่าสุดที่อัปโหลด)
current_model_id: str | None = None  # <-- ตัวแปรนี้คือหัวใจสำคัญ!

CLASS_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (255, 165, 0), (128, 0, 128),
    (255, 192, 203), (0, 128, 128)
]

@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    global current_model_id
    
    if not file.filename.lower().endswith((".pt", ".pth")):
        raise HTTPException(400, detail="รองรับเฉพาะไฟล์ .pt หรือ .pth")

    model_id = str(uuid.uuid4())
    model_path = UPLOAD_DIR / f"{model_id}.pt"

    contents = await file.read()
    with open(model_path, "wb") as f:
        f.write(contents)

    try:
        model = YOLO(str(model_path))
        names = model.names
        class_names = list(names.values()) if isinstance(names, dict) else list(names)

        loaded_models[model_id] = {
            "path": str(model_path),
            "model": model,
            "names": class_names
        }

        # อัปเดตโมเดลปัจจุบันทันที
        current_model_id = model_id

        return {
            "model_id": model_id,
            "class_names": class_names,
            "message": f"อัปโหลดสำเร็จ! พบ {len(class_names)} คลาส (ใช้งานอัตโนมัติ)"
        }
    except Exception as e:
        if model_path.exists():
            model_path.unlink()
        raise HTTPException(500, detail=f"โหลดโมเดลไม่สำเร็จ: {str(e)}")

@app.get("/current-model")
async def get_current_model():
    """เช็คว่าใช้โมเดลไหนอยู่ตอนนี้"""
    if not current_model_id or current_model_id not in loaded_models:
        return {"current_model_id": None, "class_names": []}
    info = loaded_models[current_model_id]
    return {
        "current_model_id": current_model_id,
        "class_names": info["names"]
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global current_model_id
    
    if not current_model_id or current_model_id not in loaded_models:
        raise HTTPException(404, detail="ยังไม่ได้อัปโหลดโมเดล กรุณาอัปโหลด .pt ก่อน")

    model_info = loaded_models[current_model_id]
    model = model_info["model"]
    class_names = model_info["names"]

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, detail="ไม่สามารถอ่านรูปภาพได้")

    results = model(img, conf=0.3, iou=0.45)[0]

    # วาดกรอบสีตามคลาส
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 5)
        label = f"{class_names[cls_id]} {conf:.0%}"
        (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
        cv2.rectangle(img, (x1, y1 - h_text - 20), (x1 + w_text + 20, y1), color, -1)
        cv2.putText(img, label, (x1 + 10, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

    _, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    img_base64 = base64.b64encode(buffer).decode()

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        detections.append({
            "cls": cls_id,
            "name": class_names[cls_id],
            "conf": float(box.conf[0]),
        })

    return {
        "image": img_base64,
        "detections": detections,
        "class_names": class_names,
        "current_model_id": current_model_id
    }