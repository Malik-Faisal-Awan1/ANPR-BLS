import cv2
import os
import glob
import json
import logging
import config
import threading
import time
import asyncio
import numpy as np
import re
import uuid
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
from PIL import Image, UnidentifiedImageError
from core.detector import PlateDetector
from core.ocr_engine import TextExtractor

# logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables for models and locks
detector = None
ocr_engine = None


class UploadTooLargeError(ValueError):
    pass


class InvalidImageError(ValueError):
    pass


class ProcessingBusyError(RuntimeError):
    pass


def safe_image_name(filename: str) -> str:
    name = os.path.basename(filename or "").replace("\x00", "").strip()
    if not name:
        name = "upload.jpg"

    stem, ext = os.path.splitext(name)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    ext = re.sub(r"[^A-Za-z0-9.]+", "", ext.lower())
    if ext not in {".jpg", ".jpeg", ".png"}:
        ext = ".jpg"
    if not stem:
        stem = "upload"
    return f"{stem[:80]}{ext}"


def unique_image_name(filename: str) -> str:
    name = safe_image_name(filename)
    stem, ext = os.path.splitext(name)
    suffix = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000:06d}"
    return f"{stem}_{suffix}{ext}"


async def read_upload_limited(file: UploadFile) -> bytes:
    max_bytes = int(getattr(config, "API_MAX_UPLOAD_BYTES", 50 * 1024 * 1024))
    chunk_size = int(getattr(config, "API_UPLOAD_READ_CHUNK_BYTES", 1024 * 1024))
    chunks = []
    total = 0

    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise UploadTooLargeError(f"Uploaded file exceeds {max_bytes} bytes")
        chunks.append(chunk)

    if total == 0:
        raise InvalidImageError("Empty image file")
    return b"".join(chunks)


def validate_encoded_image(contents: bytes) -> None:
    max_pixels = int(getattr(config, "API_MAX_IMAGE_PIXELS", 25_000_000))
    try:
        with Image.open(BytesIO(contents)) as image:
            width, height = image.size
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("Invalid image file") from exc

    if width <= 0 or height <= 0:
        raise InvalidImageError("Invalid image dimensions")
    if width * height > max_pixels:
        raise InvalidImageError(f"Image is too large: {width}x{height} exceeds {max_pixels} pixels")


def validate_frame_dimensions(frame: np.ndarray) -> None:
    max_pixels = int(getattr(config, "API_MAX_IMAGE_PIXELS", 25_000_000))
    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        raise InvalidImageError("Invalid image dimensions")
    if width * height > max_pixels:
        raise InvalidImageError(f"Image is too large: {width}x{height} exceeds {max_pixels} pixels")


def write_image_or_raise(path: str, image: np.ndarray) -> None:
    if not cv2.imwrite(path, image):
        raise OSError(f"Failed to write image: {path}")


def init_models():
    global detector, ocr_engine
    if detector is None or ocr_engine is None:
        logger.info("Initializing ANPR Models (once at startup)...")
        try:
            detector = PlateDetector()
            ocr_engine = TextExtractor(lang="en")
            logger.info("ANPR Models initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline components. Error: {str(e)}")
            raise e


process_lock = threading.RLock()
api_request_slots = threading.BoundedSemaphore(
    max(1, int(getattr(config, "API_MAX_IN_FLIGHT_REQUESTS", 1)))
)


def process_single_frame_with_timeout(frame, img_name, lock_timeout, isolated_request=False):
    timeout = float(lock_timeout or 0)
    if timeout <= 0:
        process_lock.acquire()
    elif not process_lock.acquire(timeout=timeout):
        raise ProcessingBusyError("ANPR processing engine is busy")
    try:
        return process_single_frame(frame, img_name)
    finally:
        process_lock.release()


def process_single_frame(frame, img_name):
    # Time monitoring start
    start_time = time.time()

    display_frame = frame.copy() if getattr(config, "SAVE_RESULT_IMAGES", True) else None

    # Run detector
    detections = detector.detect(frame)
    logger.info(f"[DETECTOR] {img_name}: {len(detections)} plate(s) detected by YOLO")

    plate_found_in_frame = False
    detected_texts = []

    # Collect primary crops
    crop_metadata = []

    # Sort detections by confidence descending
    sorted_detections = sorted(detections, key=lambda x: x['conf'], reverse=True)
    for idx, det in enumerate(sorted_detections):
        x1, y1, x2, y2 = det['bbox']
        conf = det['conf']

        if conf < config.CONFIDENCE_THRESHOLD:
            continue

        # Add minimal padding to left and right (2%)
        bw = x2 - x1
        pad_2pct = int(bw * 0.02)
        x1 = max(0, int(x1 - pad_2pct))
        x2 = min(frame.shape[1], int(x2 + pad_2pct))
        y1 = max(0, int(y1))
        y2 = min(frame.shape[0], int(y2))

        # Minimal uniform padding: 4% horizontal, 4% vertical
        pad_x = max(3, int((x2 - x1) * 0.04))
        pad_y = max(2, int((y2 - y1) * 0.04))
        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(frame.shape[1], x2 + pad_x)
        crop_y2 = min(frame.shape[0], y2 + pad_y)

        cropped_plate = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        plate_number = ""
        vehicle_type = det.get('vehicle_type', 'unknown')

        crop_metadata.append({
            "idx": idx,
            "det": det,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cropped_plate": cropped_plate,
            "plate_number": plate_number,
            "vehicle_type": vehicle_type
        })

    # Run batch OCR on all crops
    ocr_inputs = []
    ocr_indices = []
    for i, meta in enumerate(crop_metadata):
        if not meta["plate_number"] and meta["cropped_plate"].size > 0:
            h, w = meta["cropped_plate"].shape[:2]
            if h >= 10 and w >= 10:
                ocr_inputs.append(meta["cropped_plate"])
                ocr_indices.append(i)

    if ocr_inputs:
        ocr_results = ocr_engine.extract_plate_texts(ocr_inputs, batch_size=8)
        for i, text in zip(ocr_indices, ocr_results):
            crop_metadata[i]["plate_number"] = text

    # Post-process and draw
    for meta in crop_metadata:
        det = meta["det"]
        plate_number = meta["plate_number"]
        vehicle_type = meta["vehicle_type"]
        x1, y1, x2, y2 = meta["x1"], meta["y1"], meta["x2"], meta["y2"]
        conf = det['conf']
        idx = meta["idx"]
        cropped_plate = meta["cropped_plate"]

        if plate_number:
            plate_found_in_frame = True
            detected_texts.append({"plate": plate_number, "vehicle": vehicle_type, "conf": float(conf)})

            if config.SAVE_CROPPED_PLATES:
                crop_name = f"crop_candidate_{os.path.splitext(img_name)[0]}_{idx}.jpg"
                crop_path = os.path.join(config.OUTPUT_DIR, crop_name)
                try:
                    write_image_or_raise(crop_path, cropped_plate)
                except OSError as e:
                    logger.warning(f"Could not save cropped plate image {crop_path}: {e}")

            if display_frame is not None:
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                label = f"{plate_number} ({conf*100:.1f}%)"
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(display_frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 255, 0), -1)
                cv2.putText(display_frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        else:
            if display_frame is not None:
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    if not plate_found_in_frame:
        failed_filename = f"failed_{safe_image_name(img_name)}"
        failed_path = os.path.join(config.OUTPUT_DIR, failed_filename)
        try:
            # Save display_frame (has red boxes from detected-but-unread plates) for diagnosis
            save_frame = display_frame if display_frame is not None else frame
            write_image_or_raise(failed_path, save_frame)
        except OSError as e:
            logger.warning(f"Could not save failed image {failed_path}: {e}")

    output_path = ""
    if plate_found_in_frame:
        output_filename = f"result_{safe_image_name(img_name)}"
        output_path = os.path.join(config.OUTPUT_DIR, output_filename)
        if getattr(config, "SAVE_RESULT_IMAGES", True):
            write_image_or_raise(output_path, display_frame)
        output_path = os.path.abspath(output_path)

    elapsed_time_ms = (time.time() - start_time) * 1000
    execution_time = round(elapsed_time_ms, 2)
    
    plates_log = ", ".join([f"{d['plate']} ({d['vehicle']})" for d in detected_texts]) if detected_texts else "NONE"

    logger.info(
        f"[CORE ENGINE] File: {img_name} | Found: {plate_found_in_frame} | "
        f"Plates: {plates_log} | Speed: {execution_time} ms"
    )

    result = {
        "success": plate_found_in_frame,
        "detections": detected_texts,
        "processed_image_path": output_path,
        "execution_time_ms": execution_time
    }

    return result


def cleanup_old_files(directory, max_age_seconds=86400, max_files=1000):
    """
    Cleans up files older than max_age_seconds or keeps only the most recent max_files
    to prevent disk space exhaustion in production environments.
    """
    try:
        if not os.path.exists(directory):
            return
        files = []
        for entry in os.scandir(directory):
            if entry.is_file():
                stat = entry.stat()
                files.append((entry.path, stat.st_mtime))
        
        # Sort oldest first
        files.sort(key=lambda x: x[1])
        
        now = time.time()
        # 1. Age-based cleanup
        for path, mtime in files:
            if now - mtime > max_age_seconds:
                try:
                    os.remove(path)
                except OSError:
                    pass
        
        # 2. Count-based cleanup (keep under max_files limit)
        current_files = [f for f in glob.glob(os.path.join(directory, "*")) if os.path.isfile(f)]
        if len(current_files) > max_files:
            files_info = [(f, os.path.getmtime(f)) for f in current_files]
            files_info.sort(key=lambda x: x[1])
            for i in range(len(files_info) - max_files):
                try:
                    os.remove(files_info[i][0])
                except OSError:
                    pass
    except Exception as e:
        logger.error(f"Error cleaning up directory {directory}: {e}")


def watch_directory(stop_event: threading.Event):
    input_dir = config.INPUT_IMAGE_PATH
    failed_dir = getattr(config, 'FAILED_DIR', os.path.join(os.path.dirname(input_dir), "failed"))
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(failed_dir, exist_ok=True)
    
    logger.info(f"Directory watcher started. Monitoring: {input_dir}")
    logger.info(f"Failed images will be saved to: {failed_dir}")
    
    last_cleanup_time = 0.0
    
    while not stop_event.is_set():
        try:
            now = time.time()
            if now - last_cleanup_time > 3600:
                cleanup_old_files(config.OUTPUT_DIR)
                cleanup_old_files(failed_dir)
                last_cleanup_time = now

            image_paths = []
            seen = set()
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
                for p in glob.glob(os.path.join(input_dir, ext)):
                    norm = os.path.normcase(os.path.abspath(p))
                    if norm not in seen:
                        seen.add(norm)
                        image_paths.append(p)
            
            if not image_paths:
                time.sleep(0.1)  # Faster polling for real-time
                continue
                
            image_paths.sort()
            
            for img_path in image_paths:
                if not os.path.exists(img_path):
                    continue
                img_name = os.path.basename(img_path)
                if img_name.startswith(("result_", "crop_", "failed_")):
                    continue
                    
                frame = cv2.imread(img_path)
                if frame is None:
                    logger.warning(f"Watcher: Could not read {img_path}. Moving to failed folder.")
                    try:
                        os.replace(img_path, os.path.join(failed_dir, img_name))
                    except:
                        pass
                    continue
                    
                try:
                    validate_frame_dimensions(frame)
                except InvalidImageError as e:
                    logger.warning(f"Watcher: Skipping oversized/invalid image {img_path}: {e}")
                    try:
                        os.replace(img_path, os.path.join(failed_dir, img_name))
                    except:
                        pass
                    continue
                    
                logger.info(f"--- Watcher: Processing single image: {img_name} ---")
                try:
                    result = process_single_frame_with_timeout(frame, img_name, getattr(config, "API_PROCESS_LOCK_TIMEOUT_SECONDS", 0), False)
                    if result.get("success"):
                        detections = result.get("detections", [])
                        plates_log = ", ".join([f"{d.get('plate', 'UNKNOWN')} ({d.get('vehicle', 'unknown')})" for d in detections])
                        logger.info(f"\n[WATCHER SUCCESS] Plate found in {img_name} -> {plates_log}")
                        if getattr(config, "DELETE_PROCESSED_IMAGES", True):
                            try:
                                os.remove(img_path)
                            except OSError:
                                pass
                    else:
                        logger.info(f"\n[WATCHER INFO] No plate detected in {img_name}")
                        if getattr(config, "DELETE_PROCESSED_IMAGES", True):
                            try:
                                os.replace(img_path, os.path.join(failed_dir, img_name))
                            except OSError:
                                pass
                except Exception as e:
                    logger.error(f"Watcher: Error processing {img_name}: {e}")
                    try:
                        os.replace(img_path, os.path.join(failed_dir, img_name))
                    except:
                        pass
                
                # Proactive GC hint
                del frame

        except Exception as e:
            logger.error(f"Error in watcher loop: {e}")
            time.sleep(1.0)
            
    logger.info("Watcher thread stopped cleanly.")


# Lifespan context manager for FastAPI startup and shutdown events
stop_event = threading.Event()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize models
    init_models()
    logger.info("Starting directory watcher thread...")
    watcher_thread = threading.Thread(target=watch_directory, args=(stop_event,), daemon=True)
    watcher_thread.start()
    
    yield
    
    # Shutdown
    logger.info("Shutting down directory watcher...")
    stop_event.set()
    watcher_thread.join(timeout=3.0)


# Initialize FastAPI app with lifespan handler
app = FastAPI(lifespan=lifespan)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:1420",
        "http://127.0.0.1:1420",
        "tauri://localhost",
        "https://tauri.localhost",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Endpoint: Process an uploaded image
@app.post("/api/process")
async def api_process(file: UploadFile = File(...)):
    if not file.filename:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={'success': False, 'error': 'No selected file'}
        )

    acquired = False
    try:
        try:
            await asyncio.to_thread(api_request_slots.acquire)
            acquired = True
        except Exception as e:
            if isinstance(e, ProcessingBusyError) or "busy" in str(e).lower():
                raise ProcessingBusyError("ANPR server is busy") from e
            raise

        contents = await read_upload_limited(file)
        validate_encoded_image(contents)
        file_bytes = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if frame is None:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={'success': False, 'error': 'Invalid image file'}
            )

        validate_frame_dimensions(frame)
        img_name = unique_image_name(file.filename)

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            process_single_frame_with_timeout,
            frame,
            img_name,
            getattr(config, "API_PROCESS_LOCK_TIMEOUT_SECONDS", 0),
            True
        )
        
        # API execution log inside terminal
        logger.info(
            f"--- API RESPONSE: Processed {img_name} | Detections: {len(result['detections'])} | "
            f"Speed: {result['execution_time_ms']} ms ---"
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=result)
    except UploadTooLargeError as e:
        return JSONResponse(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            content={'success': False, 'error': str(e)}
        )
    except InvalidImageError as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={'success': False, 'error': str(e)}
        )
    except ProcessingBusyError as e:
        logger.warning(f"API request rejected: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={'success': False, 'error': str(e)}
        )
    except Exception as e:
        logger.error(f"API Error processing image: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={'success': False, 'error': str(e)}
        )
    finally:
        if acquired:
            api_request_slots.release()


@app.get("/api/status")
async def api_status():
    failed_dir = getattr(config, 'FAILED_DIR', os.path.join(os.path.dirname(config.INPUT_IMAGE_PATH), "failed"))
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            'status': 'running',
            'watcher_directory': config.INPUT_IMAGE_PATH,
            'output_directory': config.OUTPUT_DIR,
            'failed_directory': failed_dir
        }
    )

def run_server():
    import uvicorn
    port = int(os.environ.get("PORT", 5001))
    logger.info(f"Starting Uvicorn API Server on port {port}...")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False, workers=1)


if __name__ == "__main__":
    run_server()
