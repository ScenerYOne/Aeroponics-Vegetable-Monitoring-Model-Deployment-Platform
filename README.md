#  AI Model Deployment Platform - Hydroponic Vegetable Detection

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLO-Ultralytics-blueviolet)
![Docker](https://img.shields.io/badge/Docker-Supported-2496ED?logo=docker&logoColor=white)

A web-based tool for testing YOLO models on hydroponic vegetable images.
**AppTestModel** allows users to upload a custom `.pt` model file and test images to instantly visualize detection results with bounding boxes and class names.

## 🔗 Connected Projects (End-to-End AI Pipeline)

This project is part of a complete AI workflow, covering data preparation, model training, and deployment.

### 1️⃣ Image Preprocessing & Dataset Generation  
[ScenerYOne/Aeroponics-Vegetable-Monitoring-Image-Preprocessing](https://github.com/ScenerYOne/Aeroponics-Vegetable-Monitoring-Image-Preprocessing.git)

- Perspective Transformation for camera correction  
- Image standardization  
- Dataset preparation for YOLO training  
- Manual labeling workflow  

---

### 2️⃣ Model Training & Evaluation  
[ScenerYOne/Aeroponics-Vegetable-Monitoring-Model-Training-Evaluation](https://github.com/ScenerYOne/Aeroponics-Vegetable-Monitoring-Model-Training-Evaluation)

- Dataset cleaning & normalization  
- Multi-dataset integration  
- YOLOv8 model training and fine-tuning  
- Automated training reports (mAP, Precision, Recall)  
- ONNX export  

---

### 3️⃣ Model Deployment Platform (This Repository)
[ScenerYOne/Aeroponics-Vegetable-Monitoring-Model-Deployment-Platform](https://github.com/ScenerYOne/Aeroponics-Vegetable-Monitoring-Model-Deployment-Platform)

- Web-based YOLO model testing  
- FastAPI backend for inference  
- React frontend for visualization  
- Real-time detection with bounding boxes and class labels  

---

## 🔁 Full System Workflow

##  Features

- **Model Upload:** Upload your custom YOLO (`.pt`) model via the web interface.
- **Instant Inference:** Upload an image and get real-time detection results.
- **Visual Feedback:** Displays the processed image with drawn bounding boxes and class labels directly in the browser.
- **FastAPI Backend:** Efficient model handling and image processing.
- **React Frontend:** Clean and responsive user interface.
- **Docker Support:** Run the entire system with a single command using Docker Compose.

##  Project Structure

```text
APP_TESTMODEL/
├── backend/
│   ├── uploaded_models/      # Storage for uploaded .pt models
│   ├── server.py             # Backend entry point (FastAPI + Ultralytics)
│   ├── .env                  # Backend environment variables
│   ├── requirements.txt      # Python dependencies
│   └── Dockerfile            # Backend container image
│
├── frontend\ app_testmodel/
│   ├── src/
│   │   ├── assets/           # Static assets
│   │   ├── App.jsx           # Main application logic
│   │   ├── App.css           # Styling
│   │   ├── index.css         # Global styles
│   │   └── main.jsx          # React entry point
│   ├── public/
│   │   └── vite.svg          # Vite logo
│   ├── .env                  # Frontend environment variables (VITE_API_BASE)
│   ├── vite.config.js        # Vite configuration with proxy
│   ├── package.json          # Node.js dependencies
│   ├── eslint.config.js      # ESLint configuration
│   ├── index.html            # HTML entry point
│   └── Dockerfile            # Frontend container image
│
├── .dist/                    # Compiled frontend assets (production build)
├── .pycache_/                # Python cache
├── node_modules/             # Node.js dependencies
├── .gitignore                # Git ignore rules
├── docker-compose.yml        # Docker orchestration
├── railway.json              # Railway deployment config
└── README.md                 # Project documentation
```

##  Installation

### 1. Backend Setup (Python)
It is recommended to use Conda for environment management.

```bash
# Create a new environment
conda create -n app_testmodel python=3.9
conda activate app_testmodel

# Install dependencies
pip install fastapi uvicorn ultralytics opencv-python numpy multipart
```

### 2. Frontend Setup (React)
Navigate to the frontend directory and install Node.js dependencies.

```bash
cd app_testmodel # Name project React
npm install
```

## Usage

### Option A: Manual Mode (Two Terminals)

To run the application manually, you need to open **two terminal windows**.

#### Terminal 1: Start Backend Server
Ensure your Conda environment is activated.

```bash
# Run from the root directory (D:\APP_TESTMODEL)
conda activate app_testmodel
python -m uvicorn server:app --reload --host 127.0.0.1 --port 8000
```

#### Terminal 2: Start Frontend Client

```bash
# Run from the app_testmodel directory (D:\APP_TESTMODEL\app_testmodel)
cd app_testmodel
npm run dev
```

### Option B: Docker Mode (Single Command)

Run the entire system using Docker Compose:

```bash
docker compose up --build
```

Once running:
- **Frontend UI**: [http://localhost:5173](http://localhost:5173)
- **Backend API**: [http://localhost:8000](http://localhost:8000)
- **Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

##  How to Use

1. Open Browser: Go to [http://localhost:5173](http://localhost:5173).
2. Upload Model: Click the upload button to select your YOLO `.pt` file (e.g., best.pt).
3. Upload Image: Select a test image of hydroponic vegetables.
4. Run Inference: Click the button to process the image.
5. View Results: The image will appear with detection boxes and class labels.

## API Reference

The backend exposes the following endpoints:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| **POST** | `/upload-model` | Uploads a `.pt` file. Returns `model_id` and loaded `class_names`. |
| **POST** | `/predict` | Accepts an image + `model_id`. Returns the processed image (Base64) + detections JSON. |

---

## 🐳 Docker Support (Additional Feature)

This project now supports **Docker & Docker Compose** to simplify running the full system (Frontend + Backend) together.

Docker helps reduce environment issues, dependency conflicts, and manual multi-terminal execution during development and deployment.

---

## 🧱 Docker-Based Architecture

- **Frontend**: React + Vite (Node.js)
- **Backend**: FastAPI + Ultralytics YOLO
- **Container Network**: Docker internal bridge
- **Hardware Support**: CPU & GPU compatible

**System Workflow:**

```
Browser → React (Vite) → FastAPI → YOLO Inference
```

---

## ⚙️ Environment Variables

### Frontend Environment (`app_testmodel/.env`)

```env
VITE_API_BASE=/api
```

The frontend communicates with the backend through a Vite proxy, avoiding direct container DNS access from the browser.

---

## 🔁 Vite Proxy Mechanism

API requests from the browser are proxied internally:

```
Browser (localhost:5173)
   ↓
Vite Dev Server (/api/*)
   ↓
FastAPI Backend (yolo-backend:8000)
```

**Proxy Configuration** (`vite.config.js`):

```javascript
server: {
  proxy: {
    '/api': {
      target: 'http://yolo-backend:8000',
      changeOrigin: true,
      rewrite: path => path.replace(/^\/api/, '')
    }
  }
}
```

This design prevents CORS issues and keeps the frontend configuration clean.

---

## 🚀 Running with Docker Compose

### Prerequisites

- Docker Desktop
- (Optional) NVIDIA GPU with NVIDIA Container Toolkit

### Start All Services

```bash
docker compose up --build
```

After startup:
- **Frontend**: [http://localhost:5173](http://localhost:5173)
- **Backend API**: [http://localhost:8000](http://localhost:8000)
- **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🧠 CPU & GPU Compatibility

The backend container automatically detects available hardware:

- **GPU available** → CUDA inference enabled
- **No GPU** → Automatic CPU fallback

No additional configuration is required.

---

## 🧪 Running Mode Comparison

| Mode | Frontend | Backend | Terminals | Use Case |
| :--- | :--- | :--- | :---: | :--- |
| **Manual** | `npm run dev` | `uvicorn server:app` | 2 | Development & debugging |
| **Docker** | Docker Compose | Docker Compose | 1 | Deployment & reproducibility |

---

## 🛠 Troubleshooting Notes

- **Failed to fetch errors**: Ensure the frontend uses `/api` prefix instead of direct backend URLs.
- **Docker build failures**: Network issues may require rebuilding:
  ```bash
  docker compose build backend
  ```
- **Node.js version errors**: Ensure Docker image uses Node **20+**.

---

## 📌 Current Status

- ✔ Fully Dockerized (frontend + backend)
- ✔ Stable browser-to-backend communication
- ✔ Large YOLO models supported
- ✔ Ready for production deployment

---
