# 🚀 AI Model Deployment & Evaluation Platform

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLO-Ultralytics-blueviolet)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?logo=onnx&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Supported-2496ED?logo=docker&logoColor=white)

A **universal web-based platform** for testing and evaluating AI models across multiple frameworks. Originally developed for hydroponic vegetable monitoring, this system has evolved into a **generic AI model tester** that supports:
- **YOLO-based object detection** (`.pt` and `.onnx` formats)
- **Keras/TensorFlow models** (`.h5` and `.keras` formats) for classification, regression, and time-series tasks

All without code modification.

**sever:** https://ai-model-deployment-evaluation-platform-1.onrender.com
**datatest:** https://drive.google.com/drive/folders/13plJAkF3wHLZFqVrNljizwsa96obp50y?usp=drive_link
---

##  Project Evolution & Purpose

**AppTestModel** was initially created out of necessity for rapid testing and performance evaluation of models in the **Hydroponic Vegetable Detection** project. The goal was to quickly analyze Confidence Scores, verify Bounding Box accuracy, and fine-tune model parameters in real-time.

However, during development, we realized this system could extend its capabilities to support other tasks and frameworks without code modification. This led to architectural improvements, transforming it into a **Universal Model Testing Platform** that works with:
- **YOLO models** for object detection tasks
- **Keras/TensorFlow models** for classification, regression, and time-series prediction tasks

###  Core Design Philosophy

The system is built on the **"Zero Configuration for New Models"** principle, meaning:

-  **No Code Modification Required:** Upload a new model and use it immediately
-  **Auto Class Detection:** System automatically extracts class names from model metadata (YOLO) or infers task type (Keras)
-  **Dynamic Color Mapping:** Bounding box colors are automatically assigned based on Class ID (YOLO models)
-  **Multi-Format Support:** 
  - **YOLO Models:** `.pt` (PyTorch) and `.onnx` (Optimized Runtime)
  - **Keras Models:** `.h5` (HDF5) and `.keras` (Keras 3.0+ format)
-  **Multi-Framework Support:** Dual backend architecture supporting both Ultralytics YOLO and TensorFlow/Keras

---

## 🔗 Connected Projects (The End-to-End Pipeline)

This project represents the final stage of a complete AI development workflow, covering everything from data preparation to model deployment:

### 1️⃣ Image Preprocessing & Dataset Generation  
[ScenerYOne/Aeroponics-Vegetable-Monitoring-Image-Preprocessing](https://github.com/ScenerYOne/Aeroponics-Vegetable-Monitoring-Image-Preprocessing.git)

- Perspective Transformation for camera angle correction  
- Image Standardization for data consistency  
- Dataset Preparation for YOLO training  
- Manual Labeling Workflow with LabelImg/Roboflow  

---

### 2️⃣ Model Training & Evaluation  
[ScenerYOne/Aeroponics-Vegetable-Monitoring-Model-Training-Evaluation](https://github.com/ScenerYOne/Aeroponics-Vegetable-Monitoring-Model-Training-Evaluation)

- Dataset Cleaning & Normalization  
- Multi-Dataset Integration for Transfer Learning  
- YOLOv8/YOLOv11 Training and Hyperparameter Tuning  
- Automated Training Reports (mAP, Precision, Recall, Confusion Matrix)  
- ONNX Export for high-performance deployment  

---

### 3️⃣ Model Deployment Platform (This Repository)
[ScenerYOne/AI-Model-Deployment-Evaluation-Platform](https://github.com/ScenerYOne/AI-Model-Deployment-Evaluation-Platform)

- **Web-Based Model Testing:** Test YOLO and Keras models through browser without software installation  
- **Dual Backend Architecture:** Separate FastAPI services for YOLO (object detection) and Keras (classification/regression)  
- **Automatic Inference & Memory Management:** Smart LRU cache for both model types  
- **React Frontend:** Fast-responding and user-friendly UI with unified interface  
- **Real-Time Results:** Instant display of Bounding Boxes (YOLO) or Classification Results (Keras)  
- **Multi-User Support:** Multiple users can test different models simultaneously  

---

## 🔁 Full System Workflow

### YOLO Workflow
```
Raw Images → Preprocessing → Labeled Dataset → YOLO Training → ONNX Export → Deployment Testing (This Platform)
```

### Keras Workflow
```
Raw Data → Data Preprocessing → Model Training (Keras/TensorFlow) → Model Export (.h5/.keras) → Deployment Testing (This Platform)
```

---

##  Key Features

###  Universal Compatibility

- **Multi-Format Support:** 
  - **YOLO Models:** Supports **PyTorch (`.pt`)** files for development and debugging, and **ONNX (`.onnx`)** for high-speed inference testing suitable for production deployment
  - **Keras Models:** Supports **HDF5 (`.h5`)** format (legacy) and **Keras 3.0+ (`.keras`)** format (modern) for classification, regression, and time-series tasks
- **Automatic Task Detection:** 
  - **YOLO:** System reads metadata from models and automatically extracts class names for display, regardless of how many classes or what their names are (e.g., `['car', 'person']` or `['healthy', 'disease_a', 'disease_b']`)
  - **Keras:** System automatically detects task type (classification, regression, time-series) from model output shape and adapts visualization accordingly
- **Zero Code Modification:** Upload new models and use immediately without modifying code or config files

###  Performance & Scalability

- **Multi-User Performance:** Designed to support multiple simultaneous users by separating model loading based on individual Session IDs, allowing multiple developers to test different models concurrently without interference
- **Smart Memory Management (LRU Cache):** When memory approaches capacity, the system automatically unloads the least recently used models from RAM (Least Recently Used Algorithm) to maintain server stability
- **GPU/CPU Auto Detection:** System automatically detects available hardware; if GPU is available, CUDA inference is enabled; otherwise, it automatically falls back to CPU

###  Visualization & Analysis

- **YOLO Models:**
  - **Dynamic Bounding Box Colors:** Automatically separates box colors by Class ID from the predefined `CLASS_COLORS` list for clear result visualization
  - **Confidence Score Display:** Shows confidence values on each Bounding Box for model confidence analysis
  - **Detection Summary:** Displays count of detected objects per class
- **Keras Models:**
  - **Classification Results:** Shows predicted class with confidence score overlaid on image
  - **Regression Results:** Displays predicted numerical value
  - **Time-Series Results:** Returns predicted sequence values
- **Instant Visual Feedback:** Results display immediately as Base64-encoded images with task-specific summaries

---

##  System Architecture & Workflow

The system operates under a highly flexible **Stateless Backend** architecture that supports system scaling:

### Inference Pipeline

#### YOLO Backend (Object Detection)
```
1. Upload Phase
   User uploads .pt/.onnx model → Backend saves to /uploaded_models/
   → Loads model into RAM/GPU → Extracts class names from metadata
   → Returns model_id and class_names to frontend

2. Prediction Phase
   User uploads image + model_id → Backend retrieves model from cache
   → Runs YOLO inference → Draws bounding boxes + labels
   → Returns Base64-encoded image + detection summary

3. Memory Management
   Monitor RAM usage → If threshold exceeded
   → Unload least recently used model → Free up memory
```

#### Keras Backend (Classification/Regression/Time-Series)
```
1. Upload Phase
   User uploads .h5/.keras model → Backend saves to /uploaded_models/
   → Loads model into RAM → Infers input size and task type from model architecture
   → Returns model_id and model_format to frontend

2. Prediction Phase
   User uploads image/data + model_id → Backend retrieves model from cache
   → Preprocesses input (resize, normalize) → Runs Keras inference
   → Detects task type (classification/regression/time-series) → Formats output accordingly
   → Returns Base64-encoded image (classification) or numerical results (regression/time-series)

3. Memory Management
   Monitor RAM usage → If threshold exceeded
   → Unload least recently used model → Clear TensorFlow session → Free up memory
```

### Technical Stack

**Backend (YOLO Service):**
- **FastAPI:** High-performance async API framework
- **Ultralytics YOLO:** Native support for YOLOv8/YOLOv11
- **ONNX Runtime:** Optimized inference engine for production
- **OpenCV:** Image processing and visualization
- **PyTorch:** GPU acceleration support
- **Python 3.9+:** Core runtime environment

**Backend (Keras Service):**
- **FastAPI:** High-performance async API framework
- **TensorFlow/Keras:** Native support for Keras models (.h5, .keras)
- **OpenCV:** Image preprocessing and visualization
- **NumPy:** Numerical operations
- **Python 3.11+:** Core runtime environment

**Frontend:**
- **React 18:** Modern component-based UI
- **Vite:** Fast build tool with HMR (Hot Module Replacement)
- **Axios:** HTTP client for API communication
- **CSS3:** Responsive styling with Flexbox/Grid

**DevOps:**
- **Docker & Docker Compose:** Containerized deployment with multi-service architecture
- **Nginx Reverse Proxy:** Routes requests to appropriate backend (YOLO or Keras) based on API path
- **Railway/Render:** Cloud deployment ready

---

##  Versatility: Beyond Hydroponic Detection

While this project originated from Hydroponic Vegetable Monitoring, the system architecture was designed as a **Generic Platform** that can be immediately applied to other object detection tasks without code modification.

###  Tested Use Cases (Ready-to-Use Applications)

####  Agriculture & Farming
- **Plant Disease Detection:** Detect plant diseases from leaves
- **Crop Counting:** Count produce (tomatoes, peppers, cucumbers)
- **Weed Classification:** Classify weeds for automated spraying
- **Fruit Ripeness:** Assess fruit ripeness levels

####  Industrial & Manufacturing
- **Defect Detection:** Inspect workpiece defects (scratches, cracks)
- **Quality Control:** Verify product size/shape
- **Assembly Verification:** Verify correct component assembly

####  Transportation & Safety
- **Vehicle Detection & Counting:** Count vehicles at various points
- **License Plate Recognition:** Detect and recognize license plates
- **Safety Gear Detection:** Verify hard hat/reflective vest usage
- **Intrusion Detection:** Detect unauthorized persons in restricted areas

####  Healthcare & Medical
- **X-Ray Analysis:** Detect abnormalities in X-ray images
- **Cell Classification:** Classify cell types in microscopy
- **Medical Equipment Detection:** Verify medical equipment presence

####  Retail & E-Commerce
- **Product Recognition:** Recognize products on shelves (Shelf Monitoring)
- **Inventory Management:** Count remaining stock
- **Customer Behavior Analysis:** Analyze in-store customer behavior

###  Why It Works Universally

1. **Metadata-Driven System:** 
   - **YOLO:** System reads class names directly from model files, not hard-coded in the source
   - **Keras:** System infers task type and input requirements from model architecture
2. **Flexible Color Palette:** `CLASS_COLORS` list contains enough colors for multi-class YOLO models; for very large class counts (20-80 classes), colors can be easily added
3. **Format Agnostic:** 
   - **YOLO:** Supports both `.pt` and `.onnx`, the standard YOLO formats
   - **Keras:** Supports both `.h5` (legacy) and `.keras` (modern) formats
4. **Session-Based Isolation:** Each user has their own `model_id`, enabling simultaneous testing of different models across both backends
5. **Dual Backend Architecture:** Separate services for YOLO and Keras allow independent scaling and optimization

---

## 📂 Project Structure

```text
APP_TESTMODEL/
├── backend-yolo/
│   ├── uploaded_models/      # Storage for uploaded .pt/.onnx models
│   ├── server.py             # FastAPI backend for YOLO models
│   ├── requirements.txt      # Python dependencies (FastAPI, Ultralytics, ONNX, OpenCV, PyTorch)
│   └── Dockerfile            # YOLO backend container image
│
├── backend-keras/
│   ├── uploaded_models/      # Storage for uploaded .h5/.keras models
│   ├── server.py             # FastAPI backend for Keras models
│   ├── requirements.txt      # Python dependencies (FastAPI, TensorFlow, OpenCV, NumPy)
│   └── Dockerfile            # Keras backend container image
│
├── frontend/app_testmodel/
│   ├── src/
│   │   ├── assets/           # Static assets (images, icons)
│   │   ├── App.jsx           # Main React component (upload + predict logic)
│   │   ├── App.css           # Component styling
│   │   ├── index.css         # Global styles
│   │   └── main.jsx          # React entry point (ReactDOM.render)
│   ├── public/
│   │   └── vite.svg          # Vite logo
│   ├── .env                  # Frontend environment variables (VITE_API_BASE)
│   ├── vite.config.js        # Vite configuration with proxy settings
│   ├── package.json          # Node.js dependencies (React, Axios, Vite)
│   ├── eslint.config.js      # ESLint rules
│   ├── index.html            # HTML entry point
│   └── Dockerfile            # Frontend container image
│
├── nginx/
│   ├── nginx.conf            # Nginx reverse proxy configuration
│   └── Dockerfile            # Custom nginx image
│
├── .dist/                    # Production build output (compiled frontend)
├── .pycache/                 # Python bytecode cache
├── node_modules/             # Node.js dependencies
├── .gitignore                # Git ignore rules
├── docker-compose.yml        # Docker orchestration (frontend + backend-yolo + backend-keras + nginx)
├── railway.json              # Railway deployment configuration
└── README.md                 # Project documentation (this file)
```

---

##  Installation & Quick Start

### 1. Backend Setup (Python)

We recommend using Conda for environment management to avoid dependency conflicts:

```bash
# Create a new environment
conda create -n app_testmodel python=3.9
conda activate app_testmodel

# Install YOLO backend dependencies
cd backend-yolo
pip install fastapi uvicorn ultralytics opencv-python numpy python-multipart onnxruntime torch

# Install Keras backend dependencies
cd ../backend-keras
pip install fastapi uvicorn tensorflow opencv-python-headless numpy python-multipart
```

### 2. Frontend Setup (React)

Navigate to the frontend directory and install dependencies:

```bash
cd app_testmodel  # or frontend/app_testmodel depending on your structure
npm install
```

---

##  Usage

### Option A: Manual Mode (Two Terminals)

Suitable for development and code debugging:

#### Terminal 1: Start YOLO Backend Server

```bash
# Run from the backend-yolo directory
cd backend-yolo
conda activate app_testmodel
python -m uvicorn server:app --reload --host 127.0.0.1 --port 8000
```

#### Terminal 2: Start Keras Backend Server

```bash
# Run from the backend-keras directory
cd backend-keras
conda activate app_testmodel
python -m uvicorn server:app --reload --host 127.0.0.1 --port 8001
```

#### Terminal 3: Start Frontend Client

```bash
# Run from the app_testmodel directory (D:\APP_TESTMODEL\frontend\app_testmodel)
cd frontend/app_testmodel
npm run dev
```

**Access Points:**
- Frontend UI: [http://localhost:5173](http://localhost:5173)
- YOLO Backend API: [http://localhost:8000](http://localhost:8000)
- Keras Backend API: [http://localhost:8001](http://localhost:8001)
- YOLO Swagger Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Keras Swagger Docs: [http://localhost:8001/docs](http://localhost:8001/docs)

---

### Option B: Docker Mode (Single Command)

**Recommended method for deployment and production:**

```bash
# Clean previous builds and start fresh
docker compose down
docker builder prune -af
docker compose up --build
```

**Why Docker?**
- ✅ **One-Command Deployment:** No need to manually install dependencies
- ✅ **Consistent Environment:** Runs identically on all machines (Windows/Mac/Linux)
- ✅ **Auto-Install ONNX Runtime:** Docker image includes ONNX Runtime
- ✅ **Isolated Network:** Frontend and backend communicate via Docker internal network

**Access Points:**
- Frontend UI: [http://localhost](http://localhost) (via Nginx on port 80)
- YOLO Backend API: [http://localhost/api/yolo/](http://localhost/api/yolo/)
- Keras Backend API: [http://localhost/api/keras/](http://localhost/api/keras/)
- YOLO API Docs: [http://localhost/api/yolo/docs](http://localhost/api/yolo/docs)
- Keras API Docs: [http://localhost/api/keras/docs](http://localhost/api/keras/docs)

---

##  How to Use

### For YOLO Models (Object Detection)

1. **Open Browser:** Navigate to [http://localhost](http://localhost) (Docker) or [http://localhost:5173](http://localhost:5173) (Manual)
2. **Select Backend:** Choose "YOLO Backend" from the backend selector
3. **Upload Model:** Click "Upload Model" button to select your YOLO model file (`.pt` or `.onnx`)
   - System will load the model into RAM and display extracted class names
4. **Upload Image:** Select test image (supports JPG, PNG, BMP)
5. **Run Inference:** Click "Predict" button to process
6. **View Results:** Image displays with:
   - Color-coded Bounding Boxes by class
   - Class Names and Confidence Scores
   - Detection Summary (count of detected objects per class)

### For Keras Models (Classification/Regression/Time-Series)

1. **Open Browser:** Navigate to [http://localhost](http://localhost) (Docker) or [http://localhost:5173](http://localhost:5173) (Manual)
2. **Select Backend:** Choose "Keras Backend" from the backend selector
3. **Upload Model:** Click "Upload Model" button to select your Keras model file (`.h5` or `.keras`)
   - System will load the model and detect task type (classification/regression/time-series)
4. **Upload Image/Data:** 
   - For image models: Select test image (supports JPG, PNG, BMP)
   - For non-image models: System will use default input
5. **Run Inference:** Click "Predict" button to process
6. **View Results:**
   - **Classification:** Image with predicted class and confidence score
   - **Regression:** Numerical value prediction
   - **Time-Series:** Sequence of predicted values

---

##  API Reference

### YOLO Backend API (`/api/yolo/`)

The YOLO backend exposes the following REST API endpoints:

| Method | Endpoint | Description | Request Body | Response |
|:---|:---|:---|:---|:---|
| **POST** | `/upload-model` | Upload `.pt` or `.onnx` model file | `file: File` (multipart/form-data) | `{ model_id: str, class_names: List[str] }` |
| **POST** | `/predict` | Process image with specified model | `file: File, model_id: str` | `{ image: Base64, detections: List[Dict] }` |
| **GET** | `/list-models` | List models currently loaded in memory | - | `{ loaded_count: int, models: List[Dict] }` |

### Keras Backend API (`/api/keras/`)

The Keras backend exposes the following REST API endpoints:

| Method | Endpoint | Description | Request Body | Response |
|:---|:---|:---|:---|:---|
| **POST** | `/upload-model` | Upload `.h5` or `.keras` model file | `file: File` (multipart/form-data) | `{ model_id: str, model_format: str }` |
| **POST** | `/predict` | Process image/data with specified model | `file: File (optional), model_id: str` | `{ task: str, image: Base64 (if classification), value: float (if regression), values: List[float] (if time-series) }` |

### Example: Upload YOLO Model (cURL)

```bash
curl -X POST "http://localhost:8000/upload-model" \
  -F "file=@best.pt"
```

**Response:**
```json
{
  "model_id": "abc123",
  "class_names": ["healthy", "disease_powdery_mildew", "disease_downy_mildew"]
}
```

### Example: Run YOLO Inference (Python)

```python
import requests

# Upload image and run inference
files = {'file': open('test_image.jpg', 'rb')}
data = {'model_id': 'abc123'}
response = requests.post('http://localhost:8000/predict', files=files, data=data)

result = response.json()
print(result['detections'])  # [{'cls': 0, 'name': 'healthy', 'conf': 0.95}, ...]
```

### Example: Upload Keras Model (cURL)

```bash
curl -X POST "http://localhost:8001/upload-model" \
  -F "file=@model.h5"
```

**Response:**
```json
{
  "model_id": "def456",
  "model_format": "h5",
  "message": "Keras model loaded"
}
```

### Example: Run Keras Inference (Python)

```python
import requests

# For classification model
files = {'file': open('test_image.jpg', 'rb')}
data = {'model_id': 'def456'}
response = requests.post('http://localhost:8001/predict', files=files, data=data)

result = response.json()
print(result['task'])  # "classification"
print(result['predictions'])  # [{'class': 0, 'confidence': 0.92}]
```

---

## 🐳 Docker Support (Production-Ready)

### Docker-Based Architecture

```
Browser (localhost:80)
   ↓
Nginx Reverse Proxy
   ↓
   ├─→ /api/yolo/* → YOLO Backend (backend-yolo:8000)
   │                    ↓
   │                 YOLO Model Inference (CPU/GPU)
   │
   ├─→ /api/keras/* → Keras Backend (backend-keras:8000)
   │                     ↓
   │                  Keras Model Inference (CPU)
   │
   └─→ / → Frontend (frontend:5173)
              ↓
           React App (Vite Dev Server)
```

### Environment Variables

#### Frontend (`app_testmodel/.env`)

```env
VITE_API_BASE=/api
```

#### YOLO Backend (`backend-yolo/.env`)

```env
MAX_LOADED_MODELS=3
MODEL_DIR=uploaded_models
YOLO_CONF=0.3
YOLO_IOU=0.45
```

#### Keras Backend (`backend-keras/.env`)

```env
MAX_LOADED_MODELS=2
MODEL_DIR=uploaded_models
```

### Vite Proxy Configuration

API requests from browser are automatically proxied to backend:

**Configuration (`vite.config.js`):**

```javascript
export default defineConfig({
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/api/yolo': {
        target: 'http://localhost:8000',  // YOLO backend
        changeOrigin: true,
        rewrite: path => path.replace('/api/yolo', '')
      },
      '/api/keras': {
        target: 'http://localhost:8001',  // Keras backend
        changeOrigin: true,
        rewrite: path => path.replace('/api/keras', '')
      }
    }
  }
})
```

**Note:** In Docker mode, Nginx handles routing automatically. The above proxy config is for manual development mode.

**Benefits:**
- ✅ Avoids CORS issues
- ✅ Browser doesn't need to know container DNS
- ✅ Single config works for both dev and production

---

##  CPU & GPU Compatibility

System automatically detects hardware without configuration:

```python
# Backend auto-detection logic
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('uploaded_models/model.pt').to(device)
```

**Supported Configurations:**
- ✅ **NVIDIA GPU + CUDA:** Automatically enables GPU acceleration
- ✅ **CPU Only:** Automatically falls back to CPU
- ✅ **Docker GPU:** Supports NVIDIA Container Toolkit

### Enable GPU in Docker

Edit `docker-compose.yml`:

```yaml
services:
  yolo-backend:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

##  Running Mode Comparison

| Mode | Frontend | Backend | Terminals | Dependencies | Use Case |
|:---|:---|:---|:---:|:---|:---|
| **Manual** | `npm run dev` | `uvicorn server:app` (2 backends) | 3 | Manual installation | Development & Debugging |
| **Docker** | Docker Compose | Docker Compose (all services) | 1 | Auto-installed | Production & Deployment |

**Recommendation:** Use Manual Mode when debugging code, but use Docker Mode for actual deployment

---

##  Troubleshooting

### Common Issues & Solutions

#### 1. "Failed to fetch" errors
**Cause:** Frontend cannot connect to backend

**Solution:**
- Verify backend is running at `http://localhost:8000`
- Check that frontend uses `/api` prefix in API calls
- Try restarting Docker Compose: `docker compose restart`

#### 2. Docker build failures
**Cause:** Network issues or old cache

**Solution:**
```bash
docker compose down
docker builder prune -af
docker compose up --build
```

#### 3. "Model loading error"
**Cause:** Invalid model file or insufficient memory

**Solution:**
- **YOLO:** Verify file is actually `.pt` or `.onnx`
- **Keras:** Verify file is actually `.h5` or `.keras`
- Increase RAM for Docker Desktop (Settings → Resources → Memory)
- Reduce `MAX_LOADED_MODELS` in backend `.env` files

#### 4. "ONNX Runtime not found"
**Cause:** ONNX Runtime not installed

**Solution:**
```bash
pip install onnxruntime  # For CPU
pip install onnxruntime-gpu  # For GPU
```

#### 5. Duplicate bounding box colors
**Cause:** Model has more classes than available colors

**Solution:** Add more colors to `CLASS_COLORS` in `server.py`:

```python
CLASS_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    # Add more colors based on class count
    (128, 0, 128), (255, 165, 0), (255, 192, 203),
    # ... (recommend at least 20-30 colors)
]
```

---

##  Current Status & Roadmap

###  Completed Features

#### YOLO Backend
- [x] Support for YOLOv8 and YOLOv11
- [x] Support for `.pt` (PyTorch) and `.onnx` (ONNX Runtime) files
- [x] Automatic RAM management system (LRU Cache)
- [x] Color-coded boxes by Class ID
- [x] Auto Class Name Detection
- [x] Confidence Score Display
- [x] GPU/CPU auto-detection

#### Keras Backend
- [x] Support for `.h5` (HDF5) and `.keras` (Keras 3.0+) files
- [x] Automatic task detection (classification/regression/time-series)
- [x] Automatic input size inference from model architecture
- [x] Support for image-based classification models
- [x] Support for regression models
- [x] Support for time-series prediction models
- [x] Automatic RAM management system (LRU Cache with TensorFlow session clearing)

#### Infrastructure
- [x] Multi-User Support (both backends)
- [x] Docker Deployment Ready with multi-service architecture
- [x] Nginx reverse proxy for routing
- [x] FastAPI Documentation (Swagger UI) for both backends
- [x] Unified frontend interface for both model types


##  Contributing

Contributions are welcome! If you'd like to improve this project:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -m "Add new feature"`
4. Push to branch: `git push origin feature/new-feature`
5. Open a Pull Request

---


##  Authors

- **ScenerYOne** - Initial work and architecture design

---

## 📚 References & Documentation

### Core AI Frameworks
* **Ultralytics YOLO**: [https://docs.ultralytics.com/](https://docs.ultralytics.com/) 
* **TensorFlow/Keras**: [https://www.tensorflow.org/](https://www.tensorflow.org/)
* **ONNX Runtime**: [https://onnxruntime.ai/docs/](https://onnxruntime.ai/docs/) 
* **OpenCV Python**: [https://docs.opencv.org/](https://docs.opencv.org/) 

### Backend (FastAPI)
* **FastAPI Framework**: [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/) 
* **Uvicorn ASGI Server**: [https://www.uvicorn.org/](https://www.uvicorn.org/) 

### Frontend & UI
* **React.js**: [https://react.dev/](https://react.dev/) 
* **Vite Build Tool**: [https://vitejs.dev/](https://vitejs.dev/) 
* **Lucide Icons**: [https://lucide.dev/](https://lucide.dev/)

### Infrastructure
* **Docker Documentation**: [https://docs.docker.com/](https://docs.docker.com/) 
---

**AppTestModel** - *The open bridge between trained models and real-world evaluation.*
