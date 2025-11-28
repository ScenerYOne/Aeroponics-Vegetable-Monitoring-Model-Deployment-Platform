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

loaded_models = {}

# สีสวย ๆ สำหรับแต่ละคลาส (B
CLASS_COLORS = [
    (255, 0, 0),      # แดง      → Caramel Romaine
    (0, 255, 0),      # เขียว    → Italian
    (0, 0, 255),      # น้ำเงิน   → Red Coral
    (255, 255, 0),    # เหลือง   → Butterhead
    (255, 0, 255),    # ชมพู     → Frillice
    (0, 255, 255),    # ฟ้า      → Green Oak
    (255, 165, 0),    # ส้ม      → Red Oak
    (128, 0, 128),    # ม่วง
    (255, 192, 203),  # ชมพูอ่อน
    (0, 128, 128),    # เขียวเข้ม
]

@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
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

        return {
            "model_id": model_id,
            "class_names": class_names,
            "message": f"อัปโหลดสำเร็จ! พบ {len(class_names)} คลาส"
        }
    except Exception as e:
        if model_path.exists():
            model_path.unlink()
        raise HTTPException(500, detail=f"โหลดโมเดลไม่สำเร็จ: {str(e)}")


@app.post("/predict")
async def predict(model_id: str = "", file: UploadFile = File(...)):
    if not model_id or model_id not in loaded_models:
        raise HTTPException(404, detail="ไม่พบโมเดล กรุณาอัปโหลดใหม่ก่อน")

    model_info = loaded_models[model_id]
    model = model_info["model"]
    class_names = model_info["names"]

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, detail="ไม่สามารถอ่านรูปภาพได้")

    results = model(img, conf=0.3, iou=0.45)[0]

    # วาดกรอบ + ข้อความสีตามคลาส
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        # สีตามคลาส (วนสีใหม่ถ้าเกิน)
        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]

        # กรอบหนา 5 พิกเซล
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 5)

        # ข้อความพื้นหลัง + ข้อความสีขาว
        label = f"{class_names[cls_id]} {conf:.0%}"
        (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
        cv2.rectangle(img, (x1, y1 - h_text - 20), (x1 + w_text + 20, y1), color, -1)
        cv2.putText(img, label, (x1 + 10, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

    # แปลงเป็น base64
    _, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    img_base64 = base64.b64encode(buffer).decode()

    # ส่ง detections กลับไปให้ frontend
    detections = []
    h, w = img.shape[:2]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        detections.append({
            "cls": cls_id,
            "name": class_names[cls_id],
            "conf": float(box.conf[0]),
        })

    return {
        "image": img_base64,
        "detections": detections,
        "class_names": class_names
    }