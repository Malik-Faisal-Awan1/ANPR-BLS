import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# ── Model Paths ───────────────────────────────────────────────────────────────
MODEL_PATH = str(MODEL_DIR / "best.onnx")           # YOLOv9 detector
OCR_MODEL_PATH = str(MODEL_DIR / "fast-plate" / "best.onnx")  # CCT OCR

# ── YOLO Classes ──────────────────────────────────────────────────────────────
# 0=license, 1=car, 2=bike, 3=rickshaw, 4=truck
YOLO_CLASSES = {0: "license", 1: "car", 2: "bike", 3: "rickshaw", 4: "truck"}
LICENSE_PLATE_CLASS_ID = 0

# ── Detection Thresholds ──────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.55
IOU_THRESHOLD = 0.45
YOLO_IMGSZ = 640
DEVICE = "cpu"

# ── ROI Filtering ─────────────────────────────────────────────────────────────
ENABLE_FOREGROUND_FILTERING = True
SELECT_PRIMARY_FOREGROUND_ONLY = True
ROI_TOP_IGNORE_RATIO = 0.12
ROI_BOTTOM_IGNORE_RATIO = 0.05
MIN_PLATE_AREA = 200
MIN_PLATE_WIDTH = 15
MIN_PLATE_HEIGHT = 10
FOREGROUND_AREA_RATIO_CUTOFF = 0.25

# ── IO Paths ──────────────────────────────────────────────────────────────────
INPUT_IMAGE_PATH = str(BASE_DIR / "data" / "images")
OUTPUT_DIR = str(BASE_DIR / "output")
FAILED_DIR = str(BASE_DIR / "data" / "failed")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Output Behaviour ──────────────────────────────────────────────────────────
SAVE_RESULT_IMAGES = True
SAVE_CROPPED_PLATES = False
DELETE_PROCESSED_IMAGES = False

# ── API Limits ────────────────────────────────────────────────────────────────
API_MAX_UPLOAD_BYTES = int(os.environ.get("ANPR_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
API_MAX_IMAGE_PIXELS = int(os.environ.get("ANPR_MAX_IMAGE_PIXELS", str(25_000_000)))
API_UPLOAD_READ_CHUNK_BYTES = int(os.environ.get("ANPR_UPLOAD_READ_CHUNK_BYTES", str(1024 * 1024)))

# ── OCR ───────────────────────────────────────────────────────────────────────
OCR_INTRA_OP_THREADS = min(4, os.cpu_count() or 4)

# ── Logging ───────────────────────────────────────────────────────────────────
VERBOSE_LOGGING = False