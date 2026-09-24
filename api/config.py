import os
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent

# Model directory
MODEL_DIR = os.path.join(BASE_DIR, "models")
# Repointed from v3_best.onnx (legacy artifact, same 9-channel layout) to best.onnx per THS v4 §4 Phase 0.
MODEL_PATH = os.path.join(MODEL_DIR, "best.onnx")
CUSTOM_OCR_MODEL_DIR = os.path.join(MODEL_DIR, "custom_ocr")

# ── Detection ────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.55
IOU_THRESHOLD = 0.45
DEVICE = "cpu"

# YOLO inference image size (YOLOv9 exported at 640)
YOLO_IMGSZ = 640

# Frames larger than this long edge are downscaled BEFORE YOLO inference.
# Coordinates are automatically mapped back so crops are taken from the FULL RESOLUTION image.
# This gives a major speed boost while maintaining crystal-clear OCR sharpness. Set 0 to disable.
MAX_FRAME_DIM = 640

# YOLOv9 Class names (from classes_yolov9.txt)
YOLO_CLASSES = {
    0: "license",
    1: "car",
    2: "bike",
    3: "rickshaw",
    4: "truck"
}
LICENSE_PLATE_CLASS_ID = 0

# ── Foreground Vehicle & ROI Filtering ───────────────────────────────────────
# Enable filtering to eliminate distant/background vehicles and focus on foreground vehicle
ENABLE_FOREGROUND_FILTERING = True
# If True, selects only the primary foreground plate (largest area + lower frame position)
SELECT_PRIMARY_FOREGROUND_ONLY = True
# Ignore plate detections in top percentage of frame (sky / horizon / background distant traffic)
ROI_TOP_IGNORE_RATIO = 0.12
# Ignore plate detections in bottom percentage of frame (camera timestamp / extreme edge)
ROI_BOTTOM_IGNORE_RATIO = 0.05
# Minimum bounding box area (w * h) to consider a valid plate
MIN_PLATE_AREA = 200
MIN_PLATE_WIDTH = 15
MIN_PLATE_HEIGHT = 10
# Background plate area ratio cutoff (ignore plates < 25% area of primary foreground plate)
FOREGROUND_AREA_RATIO_CUTOFF = 0.25

# ── IO Paths ─────────────────────────────────────────────────────────────────
INPUT_IMAGE_PATH = os.path.join(BASE_DIR, "data", "images")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
SAVE_CROPPED_PLATES = False
SAVE_RESULT_IMAGES = True
DELETE_PROCESSED_IMAGES = False

# API / server safety limits.
API_MAX_UPLOAD_BYTES = int(os.environ.get("ANPR_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
API_MAX_IMAGE_PIXELS = int(os.environ.get("ANPR_MAX_IMAGE_PIXELS", str(25_000_000)))
API_MAX_IN_FLIGHT_REQUESTS = int(os.environ.get("ANPR_API_MAX_IN_FLIGHT", "1"))
API_PROCESS_LOCK_TIMEOUT_SECONDS = float(os.environ.get("ANPR_PROCESS_LOCK_TIMEOUT_SECONDS", "0"))
API_UPLOAD_READ_CHUNK_BYTES = int(os.environ.get("ANPR_UPLOAD_READ_CHUNK_BYTES", str(1024 * 1024)))

# ── OCR & Multi-Core ──────────────────────────────────────────────────────────
# CTC model softmax scores are naturally diffuse — set to 0.0 to accept all results.
# Raise gradually (e.g., 0.15, 0.20) only once you have confirmed real output to tune against.
OCR_MIN_CONFIDENCE = 0.0
# Multi-core CPU thread count for ONNX Runtime OCR inference
OCR_INTRA_OP_THREADS = min(4, os.cpu_count() or 4)

# OCR model input dimensions (from inference.onnx: [None, 3, 48, 320])
OCR_INPUT_HEIGHT = 48
OCR_INPUT_WIDTH = 320

# ── Plate Color Detection ──────────────────────────────────────────────────
ENABLE_COLOR_DETECTION = True

# HSV thresholds for background color classification
COLOR_WHITE_SAT_MAX = 50
COLOR_WHITE_VAL_MIN = 180
COLOR_GREEN_HUE_MIN = 35
COLOR_GREEN_HUE_MAX = 85
COLOR_GREEN_SAT_MIN = 40
COLOR_YELLOW_HUE_MIN = 15
COLOR_YELLOW_HUE_MAX = 35
COLOR_YELLOW_SAT_MIN = 50
COLOR_BLUE_HUE_MIN = 90
COLOR_BLUE_HUE_MAX = 130
COLOR_BLUE_SAT_MIN = 40
COLOR_BLACK_VAL_MAX = 60
COLOR_BLACK_SAT_MAX = 50
COLOR_INVERT_ENABLED = True

# ── Display / UI ──────────────────────────────────────────────────────────────
DISPLAY_RESULTS = False

# INFO logging inside per-frame loops is expensive on Windows terminals.
VERBOSE_LOGGING = False