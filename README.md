#  AI Model Deployment Platform - Hydroponic Vegetable Detection

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLO-Ultralytics-blueviolet)

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

##  Project Structure

```text
AppTestModel/
├── uploaded_models/      # Storage for uploaded .pt models
├── server.py             # Backend entry point (FastAPI + Ultralytics)
├── app_testmodel/        # Frontend source code (React + Vite)
│   ├── src/
│   │   ├── App.jsx       # Main application logic
│   │   └── App.css       # Styling
│   ├── vite.config.js
│   └── package.json
└── README.md
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

To run the application, you need to open **two terminal windows**.

### Terminal 1: Start Backend Server
Ensure your Conda environment is activated.

```bash
# Run from the root directory (D:\APP_TESTMODEL)
conda activate app_testmodel
python -m uvicorn server:app --reload --host 127.0.0.1 --port 8000

```
### Terminal 2: Start Frontend Client

```bash
# Run from the app_testmodel directory (D:\APP_TESTMODEL\app_testmodel)
cd app_testmodel
npm run dev

```
##  How to Use
  1. Open Browser: Go to (`[.pt](http://localhost:5173)`).
  2. Upload Model: Click the upload button to select your YOLO .pt file (e.g., best.pt).
  3. Upload Image: Select a test image of hydroponic vegetables.
  4. Run Inference: Click the button to process the image.
  5. View Results: The image will appear with detection boxes and class labels.

## API Reference

The backend exposes the following endpoints:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| **POST** | `/upload-model` | Uploads a `.pt` file. Returns `model_id` and loaded `class_names`. |
| **POST** | `/predict` | Accepts an image + `model_id`. Returns the processed image (Base64) + detections JSON. |
