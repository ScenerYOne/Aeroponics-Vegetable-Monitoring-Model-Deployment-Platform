#  AppTestModel - Hydroponic Vegetable Detection

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLO-Ultralytics-blueviolet)

A web-based tool for testing YOLO models on hydroponic vegetable images.
**AppTestModel** allows users to upload a custom `.pt` model file and test images to instantly visualize detection results with bounding boxes and class names.

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
