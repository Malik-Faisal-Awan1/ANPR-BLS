# ANPR-BLS (Automatic Number Plate Recognition)

An enterprise-grade, lean API for Automated Number Plate Recognition (ANPR). This repository contains the highly optimized production API for inference and the full developer environment for research, model training, and data preparation.

## 🏛️ Architecture Overview
The system relies on a unified, two-stage inference pipeline:
1. **Detection (YOLOv9)**: A highly accurate YOLOv9 model (`models/best.pt`) detects the bounding box of license plates in high-resolution images.
2. **OCR / Recognition (Compact Convolutional Transformer - CCT)**: A lightweight, custom-trained CCT model (`models/fast-plate/best.onnx`) processes the cropped plate. It expects a strict tensor contract: a native `uint8` `[Batch, 64, 128, 3]` NHWC tensor, bypassing legacy normalization (no CLAHE, no float casting).

Both models are served through a lean FastAPI backend.

---

## 📂 Monorepo Structure

We employ a unified monorepo architecture to keep the production API lean while preserving the developer context (training code, datasets, evaluation scripts).

```text
ANPR-BLS/
├── api/                   # 🚀 PRODUCTION: Lean FastAPI inference environment
│   ├── app.py             # Main FastAPI server entry point
│   ├── config.py          # Environment & Path configurations
│   ├── core/              # Core inference logic (detector & ocr_engine)
│   ├── requirements.txt   # Lean dependencies (FastAPI, ONNX Runtime, OpenCV, etc.)
│   └── models/            # The exported YOLOv9 and fast-plate ONNX weights
│
├── training/              # 🔬 DEVELOPER / RESEARCH: Model training & evaluation
│   ├── data_prep/         # Scripts to generate synthetic plates, crops, and augmentations
│   ├── evaluate.py        # Benchmark scripts for accuracy and latency
│   └── train_onnx.py      # Scripts used to train the fast-plate-ocr CCT model
│
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started (Production / Inference)

The `api/` directory contains exactly what is needed for production. No GUI bloat, no heavy training libraries.

### 1. Installation
Ensure you have Python 3.10+ installed.
```bash
cd api/
python -m venv venv
# Windows: venv\Scripts\activate | Unix: source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the API Server
Start the lean FastAPI server:
```bash
python app.py
```
*The server binds to `0.0.0.0:5001` by default.*

### 3. Usage
You can test the pipeline using curl or Python:
```bash
# Upload an image for processing
curl -X POST -F "file=@sample_car.jpg" http://127.0.0.1:5001/api/process
```
**Response Format:**
```json
{
  "success": true,
  "numberplate": "LEU509",
  "execution_time_ms": 62.5
}
```

---

## 🔬 Developer Guide (Retraining & Research)

The `training/` environment preserves all context needed to adapt the system for new license plate formats, different camera angles, or domain-specific augmentations.

### 1. Dev Setup
If you need to train models, you will need a separate, heavier environment containing PyTorch, Albumentations, and evaluation frameworks.
```bash
cd training/
# Install training requirements (assuming you maintain a dev_requirements.txt)
pip install torch torchvision albumentations wandb
```

### 2. Retraining the YOLO Detector
To train a new YOLOv9 model for detection, update your dataset YAML and run:
```bash
yolo task=detect mode=train data=plate_dataset.yaml model=yolov9c.pt epochs=100 imgsz=640
```
Export the model to `.pt` or `.onnx` and place it in `api/models/`.

### 3. Retraining the Fast-Plate OCR (CCT)
Our OCR engine uses a CCT. The dataset should consist of tight `128x64` crops of plates.
1. Place training data crops in `training/data/`.
2. Update configuration in `training/configs/`.
3. Run the trainer:
   ```bash
   python train_onnx.py --config configs/fast_plate.yaml
   ```
4. Export the resulting model to `best.onnx` and replace `api/models/fast-plate/best.onnx`.

### 4. Tensor Contract (Important)
If you update the OCR model, ensure you do not break the API's tensor contract. The FastAPI `ocr_engine.py` currently sends the ONNX model a pure `uint8` BGR-to-RGB converted image, without float normalization. If your new training pipeline requires mean/std normalization, you must update the preprocessing step in `api/core/ocr_engine.py`.
