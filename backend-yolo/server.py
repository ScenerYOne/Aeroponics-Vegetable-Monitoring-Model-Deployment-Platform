from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import base64
from pathlib import Path
import uuid
import os
import gc  
import time
import torch  


MAX_LOADED_MODELS = int(os.getenv("MAX_LOADED_MODELS", 3)) 
MODEL_DIR = os.getenv("MODEL_DIR", "uploaded_models")
YOLO_CONF = 0.3
YOLO_IOU = 0.45

app = FastAPI(title="Multi-user YOLO API")


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

def manage_memory():
    if len(loaded_models) > MAX_LOADED_MODELS:
        oldest_id = min(loaded_models, key=lambda k: loaded_models[k]['last_used'])
        print(f"--- Unloading model {oldest_id} to save RAM ---")
        
        del loaded_models[oldest_id]
        gc.collect() 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

CLASS_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255)
]

@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in [".pt", ".onnx"]:
        raise HTTPException(400, "Unsupported format")

    model_id = str(uuid.uuid4())
    model_path = UPLOAD_DIR / f"{model_id}{suffix}"

    with open(model_path, "wb") as f:
        f.write(await file.read())

    try:
        manage_memory() 
        
        model = YOLO(str(model_path))
        names = getattr(model, "names", {})
        
        loaded_models[model_id] = {
            "model": model,
            "names": names,
            "format": suffix.replace(".", ""),
            "last_used": time.time()
        }

        return {
            "model_id": model_id,
            "class_names": names,
            "message": "Model uploaded and loaded successfully"
        }
    except Exception as e:
        if model_path.exists(): model_path.unlink()
        raise HTTPException(500, f"Error loading model: {e}")

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_id: str = Form(...) 
):
    if model_id not in loaded_models:
        raise HTTPException(404, "Model not in RAM. Please re-upload or select again.")

    loaded_models[model_id]["last_used"] = time.time()
    model_info = loaded_models[model_id]
    model = model_info["model"]
    class_names = model_info["names"]
    
    img_bytes = await file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    
    results = model(img, conf=YOLO_CONF, iou=YOLO_IOU)[0]
    
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = class_names.get(cls_id, f"Class {cls_id}")
        
        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3) 
        
        label = f"{name} {conf:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        detections.append({"cls": cls_id, "name": name, "conf": conf})

    _, buffer = cv2.imencode(".jpg", img)
    return {
        "image": base64.b64encode(buffer).decode(),
        "detections": detections,
        "model_id": model_id 
    }

@app.get("/list-models")
async def list_models():
    return {
        "loaded_count": len(loaded_models),
        "models": [
            {"id": k, "format": v["format"], "last_used": v["last_used"]} 
            for k, v in loaded_models.items()
        ]
    }


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000)) 
    uvicorn.run(app, host="0.0.0.0", port=port)