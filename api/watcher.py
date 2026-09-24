import cv2
import os
import glob
import logging
import time
import sys

# Allow running from any directory
sys.path.insert(0, os.path.dirname(__file__))

import config
from core.detector import PlateDetector
from core.ocr_engine import TextExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _write_image(path: str, img) -> None:
    if not cv2.imwrite(path, img):
        raise OSError(f"Failed to write: {path}")


def _cleanup(directory, max_age_seconds=86400, max_files=1000):
    if not os.path.exists(directory):
        return
    try:
        entries = [(e.path, e.stat().st_mtime) for e in os.scandir(directory) if e.is_file()]
        entries.sort(key=lambda x: x[1])
        now = time.time()
        for path, mtime in entries:
            if now - mtime > max_age_seconds:
                try:
                    os.remove(path)
                except OSError:
                    pass
        current = [e[0] for e in entries if os.path.exists(e[0])]
        for path in current[: max(0, len(current) - max_files)]:
            try:
                os.remove(path)
            except OSError:
                pass
    except Exception as e:
        logger.error("Cleanup error in %s: %s", directory, e)


def process(frame, img_name: str, detector, ocr_engine) -> dict:
    start = time.time()
    display = frame.copy() if config.SAVE_RESULT_IMAGES else None
    detections = detector.detect(frame)

    crops, crop_meta = [], []
    for det in sorted(detections, key=lambda d: d["conf"], reverse=True):
        if det["conf"] < config.CONFIDENCE_THRESHOLD:
            continue
        x1, y1, x2, y2 = det["bbox"]
        bw = x2 - x1
        x1 = max(0, x1 - int(bw * 0.06))
        x2 = min(frame.shape[1], x2 + int(bw * 0.06))
        y1 = max(0, y1 - int((y2 - y1) * 0.04))
        y2 = min(frame.shape[0], y2 + int((y2 - y1) * 0.04))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            continue
        crops.append(crop)
        crop_meta.append({"det": det, "box": (x1, y1, x2, y2), "crop": crop})

    texts = ocr_engine.extract_plate_texts(crops) if crops else []
    results = []
    for meta, text in zip(crop_meta, texts):
        if not text:
            continue
        det = meta["det"]
        x1, y1, x2, y2 = meta["box"]
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
        out_path = os.path.join(config.OUTPUT_DIR, f"result_{img_name}")
        try:
            _write_image(out_path, display)
        except OSError as e:
            logger.warning("Could not save result image: %s", e)

    return {"success": success, "detections": results,
            "processed_image_path": out_path, "execution_time_ms": round((time.time() - start) * 1000, 2)}


def run():
    input_dir = config.INPUT_IMAGE_PATH
    output_dir = config.OUTPUT_DIR
    failed_dir = os.path.join(os.path.dirname(input_dir), "failed")

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(failed_dir, exist_ok=True)

    logger.info("Loading models...")
    detector = PlateDetector()
    ocr_engine = TextExtractor()
    logger.info("Watcher ready. Monitoring: %s", input_dir)

    last_cleanup = 0.0
    while True:
        try:
            now = time.time()
            if now - last_cleanup > 3600:
                _cleanup(output_dir)
                _cleanup(failed_dir)
                last_cleanup = now

            seen, paths = set(), []
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
                for p in glob.glob(os.path.join(input_dir, ext)):
                    norm = os.path.normcase(os.path.abspath(p))
                    if norm not in seen:
                        seen.add(norm)
                        paths.append(p)

            if not paths:
                time.sleep(0.1)
                continue

            for img_path in sorted(paths):
                if not os.path.exists(img_path):
                    continue
                img_name = os.path.basename(img_path)
                if img_name.startswith(("result_", "crop_", "failed_")):
                    continue

                frame = cv2.imread(img_path)
                if frame is None:
                    logger.warning("Unreadable image, moving to failed: %s", img_name)
                    try:
                        os.replace(img_path, os.path.join(failed_dir, img_name))
                    except OSError:
                        pass
                    continue

                logger.info("Processing: %s", img_name)
                try:
                    result = process(frame, img_name, detector, ocr_engine)
                    if result["success"]:
                        plates = ", ".join(f"{d['plate']} ({d['vehicle']})" for d in result["detections"])
                        logger.info("[SUCCESS] %s -> %s", img_name, plates)
                    else:
                        logger.info("[NO PLATE] %s", img_name)
                    if config.DELETE_PROCESSED_IMAGES:
                        dest = None if result["success"] else os.path.join(failed_dir, img_name)
                        try:
                            if dest:
                                os.replace(img_path, dest)
                            else:
                                os.remove(img_path)
                        except OSError:
                            pass
                except Exception as e:
                    logger.error("Error processing %s: %s", img_name, e)
                    try:
                        os.replace(img_path, os.path.join(failed_dir, img_name))
                    except OSError:
                        pass
                finally:
                    del frame

        except KeyboardInterrupt:
            logger.info("Watcher stopped.")
            break
        except Exception as e:
            logger.error("Watcher loop error: %s", e)
            time.sleep(1.0)


if __name__ == "__main__":
    run()
