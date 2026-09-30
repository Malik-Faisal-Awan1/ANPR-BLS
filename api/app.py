import cv2
import os
import logging
import time
import asyncio
import numpy as np
import re
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from PIL import Image, UnidentifiedImageError
from core.detector import PlateDetector
from core.ocr_engine import TextExtractor
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

detector: PlateDetector = None
ocr_engine: TextExtractor = None


# ── Errors ────────────────────────────────────────────────────────────────────

class UploadTooLargeError(ValueError):
    pass

class InvalidImageError(ValueError):
    pass


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_name(filename: str) -> str:
    name = os.path.basename(filename or "").replace("\x00", "").strip() or "upload.jpg"
    stem, ext = os.path.splitext(name)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "upload"
    ext = ext.lower() if ext.lower() in {".jpg", ".jpeg", ".png"} else ".jpg"
    return f"{stem[:80]}{ext}"


def _unique_name(filename: str) -> str:
    stem, ext = os.path.splitext(_safe_name(filename))
    return f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000:06d}{ext}"


async def _read_limited(file: UploadFile) -> bytes:
    max_bytes = config.API_MAX_UPLOAD_BYTES
    chunk_size = config.API_UPLOAD_READ_CHUNK_BYTES
    chunks, total = [], 0
    while chunk := await file.read(chunk_size):
        total += len(chunk)
        if total > max_bytes:
            raise UploadTooLargeError(f"File exceeds {max_bytes} bytes")
        chunks.append(chunk)
    if total == 0:
        raise InvalidImageError("Empty image file")
    return b"".join(chunks)


def _validate_image_bytes(contents: bytes) -> None:
    try:
        with Image.open(BytesIO(contents)) as img:
            w, h = img.size
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("Invalid image file") from exc
    if w * h > config.API_MAX_IMAGE_PIXELS:
        raise InvalidImageError(f"Image too large: {w}x{h}")




def _run_inference(frame: np.ndarray, img_name: str) -> dict:
    start = time.time()
    display = frame.copy() if config.SAVE_RESULT_IMAGES else None
    detections = detector.detect(frame)

    crops, meta = [], []
    for det in sorted(detections, key=lambda d: d["conf"], reverse=True):
        if det["conf"] < config.CONFIDENCE_THRESHOLD:
            continue
        crop = PlateDetector.crop_detection(frame, det["bbox"], expand=True)
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            continue
        crops.append(crop)
        meta.append({"det": det, "box": det["bbox"]})

    texts = ocr_engine.extract_plate_texts(crops) if crops else []
    results = []
    for m, text in zip(meta, texts):
        if not text:
            continue
        det = m["det"]
        x1, y1, x2, y2 = m["box"]
        results.append({"plate": text, "vehicle": det.get("vehicle_type", "unknown"), "conf": round(float(det["conf"]), 4)})
        if display is not None:
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = f"{text} ({det['conf']*100:.1f}%)"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display, (x1, y1 - th - 10), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(display, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    success = bool(results)
    out_path = ""
    if success and display is not None:
        out_path = os.path.join(config.OUTPUT_DIR, f"result_{_safe_name(img_name)}")
        if not cv2.imwrite(out_path, display):
            logger.warning("Could not save result image: %s", out_path)
            out_path = ""

    return {
        "success": success,
        "detections": results,
        "processed_image_path": os.path.abspath(out_path) if out_path else "",
        "execution_time_ms": round((time.time() - start) * 1000, 2),
    }


# ── App ───────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector, ocr_engine
    logger.info("Loading models...")
    detector = PlateDetector()
    ocr_engine = TextExtractor()
    logger.info("Models ready.")
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:1420", "http://127.0.0.1:1420", "tauri://localhost", "https://tauri.localhost"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/api/process")
async def api_process(file: UploadFile = File(...)):
    if not file.filename:
        return JSONResponse(status.HTTP_400_BAD_REQUEST, {"success": False, "error": "No file provided"})
    try:
        contents = await _read_limited(file)
        _validate_image_bytes(contents)
        frame = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return JSONResponse(status.HTTP_400_BAD_REQUEST, {"success": False, "error": "Cannot decode image"})
        img_name = _unique_name(file.filename)
        result = await asyncio.get_running_loop().run_in_executor(None, _run_inference, frame, img_name)
        logger.info("Processed %s | %d detection(s) | %.1f ms", img_name, len(result["detections"]), result["execution_time_ms"])
        return JSONResponse(status.HTTP_200_OK, result)
    except UploadTooLargeError as e:
        return JSONResponse(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, {"success": False, "error": str(e)})
    except InvalidImageError as e:
        return JSONResponse(status.HTTP_400_BAD_REQUEST, {"success": False, "error": str(e)})
    except Exception as e:
        logger.error("Inference error: %s", e, exc_info=True)
        return JSONResponse(status.HTTP_500_INTERNAL_SERVER_ERROR, {"success": False, "error": str(e)})


@app.get("/api/status")
async def api_status():
    return {"status": "running", "input": config.INPUT_IMAGE_PATH, "output": config.OUTPUT_DIR}


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5001))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False, workers=1)
