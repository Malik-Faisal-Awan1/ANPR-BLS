"""
test_pipeline.py - End-to-end ANPR pipeline validation
=======================================================
Tests: YOLOv9 ONNX detector -> plate crop -> fast-plate ONNX OCR

Run from the project root:
    python test_pipeline.py [image_path]

If no image is given the first .jpg in data/images/ is used.

Transparency guarantees:
  - Every failure prints the actual exception + full traceback.
  - Tensor shapes and dtypes are printed at each stage.
  - PASS / FAIL is stated explicitly for each contract check.
  - INCONCLUSIVE is reported clearly when detector finds 0 plates.
  - No mocked data; real models + real images from data/images/.
"""

import sys
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. Environment check (before any project import)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SEP = "=" * 60
print(f"\n{SEP}")
print("ANPR PIPELINE TEST")
print(SEP)
print(f"[ENV] Working directory : {ROOT}")

YOLO_ONNX = ROOT / "models" / "best.onnx"
OCR_ONNX  = ROOT / "models" / "fast-plate" / "best.onnx"

for label, path in [("YOLO ONNX", YOLO_ONNX), ("OCR ONNX", OCR_ONNX)]:
    status = "EXISTS" if path.exists() else "MISSING"
    print(f"[ENV] {label:12s}: {path}  [{status}]")
    if not path.exists():
        print(f"\n[ABORT] Required model not found: {path}")
        sys.exit(1)

# ---------------------------------------------------------------------------
# 1. Resolve test image
# ---------------------------------------------------------------------------
if len(sys.argv) > 1:
    img_path = Path(sys.argv[1])
else:
    candidates = sorted((ROOT / "data" / "images").glob("*.jpg"))
    if not candidates:
        print("\n[ABORT] No .jpg files in data/images/. Pass image path as argument.")
        sys.exit(1)
    img_path = candidates[0]

print(f"[ENV] Test image       : {img_path}")
if not img_path.exists():
    print(f"\n[ABORT] Image not found: {img_path}")
    sys.exit(1)

print(f"{SEP}\n")

# ---------------------------------------------------------------------------
# 2. Import dependencies
# ---------------------------------------------------------------------------
print("STEP 0 - Import dependencies")
try:
    import cv2
    import numpy as np
    print(f"  [OK] cv2 {cv2.__version__},  numpy {np.__version__}")
except ImportError as exc:
    print(f"  [FAIL] {exc}")
    sys.exit(1)

try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"  [OK] onnxruntime {ort.__version__}  |  providers: {providers}")
except ImportError as exc:
    print(f"  [FAIL] onnxruntime not installed: {exc}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 3. Load detector
# ---------------------------------------------------------------------------
print("\nSTEP 1 - Load YOLOv9 detector")
try:
    sys.path.insert(0, str(ROOT))
    import config
    from core.detector import PlateDetector

    t0 = time.perf_counter()
    detector = PlateDetector()
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  [OK] PlateDetector loaded in {load_ms:.1f} ms")
    print(f"       Model : {config.MODEL_PATH}")
except Exception:
    print("  [FAIL] Could not load PlateDetector:")
    traceback.print_exc()
    sys.exit(1)

# ---------------------------------------------------------------------------
# 4. Load OCR engine
# ---------------------------------------------------------------------------
print("\nSTEP 2 - Load OCR engine (fast-plate ONNX)")
try:
    from core.ocr_engine import TextExtractor

    t0 = time.perf_counter()
    ocr = TextExtractor()
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  [OK] TextExtractor loaded in {load_ms:.1f} ms")
    print(f"       Model : {OCR_ONNX}")

    inp = ocr.session.get_inputs()[0]
    print(f"       Input  name  : {inp.name}")
    print(f"       Input  shape : {inp.shape}")
    print(f"       Input  dtype : {inp.type}")

    expected_type = "tensor(uint8)"
    if inp.type != expected_type:
        print(f"  [WARN] Expected dtype '{expected_type}', got '{inp.type}'")
    else:
        print(f"       [OK] dtype contract verified: {expected_type}")

    outp = ocr.session.get_outputs()[0]
    print(f"       Output name  : {outp.name}")
    print(f"       Output shape : {outp.shape}  (expected [B, 10, 37])")
except Exception:
    print("  [FAIL] Could not load OCR engine:")
    traceback.print_exc()
    sys.exit(1)

# ---------------------------------------------------------------------------
# 5. Read image
# ---------------------------------------------------------------------------
print(f"\nSTEP 3 - Read test image: {img_path.name}")
frame = cv2.imread(str(img_path))
if frame is None:
    print(f"  [FAIL] cv2.imread returned None for {img_path}")
    sys.exit(1)
print(f"  [OK] Frame shape: {frame.shape}  dtype: {frame.dtype}")

# ---------------------------------------------------------------------------
# 6. Detect
# ---------------------------------------------------------------------------
print("\nSTEP 4 - Run YOLOv9 detector")
try:
    t0 = time.perf_counter()
    detections = detector.detect(frame)
    det_ms = (time.perf_counter() - t0) * 1000
    print(f"  [OK] Detection finished in {det_ms:.1f} ms")
    print(f"       Plates detected: {len(detections)}")
    for i, d in enumerate(detections):
        print(f"       [{i}] bbox={d['bbox']}  conf={d['conf']:.3f}")
except Exception:
    print("  [FAIL] Detector raised an exception:")
    traceback.print_exc()
    sys.exit(1)

if not detections:
    print("\n[INCONCLUSIVE] Detector found 0 plates -- cannot test OCR crop path.")
    print("  The detector itself did not crash, but no crop exists to pass to OCR.")
    print("  This is NOT a PASS.")
    print("  Try a different image: python test_pipeline.py <path_to_image>")
    sys.exit(0)

# ---------------------------------------------------------------------------
# 7. Crop
# ---------------------------------------------------------------------------
print("\nSTEP 5 - Crop best detection")
best = detections[0]
x1, y1, x2, y2 = best["bbox"]
crop = frame[y1:y2, x1:x2]
if crop.size == 0:
    print(f"  [FAIL] Crop is empty for bbox {best['bbox']} on frame {frame.shape}")
    sys.exit(1)
print(f"  [OK] Crop shape: {crop.shape}  dtype: {crop.dtype}")

# ---------------------------------------------------------------------------
# 8. Tensor contract check (before inference)
# ---------------------------------------------------------------------------
print("\nSTEP 6 - Inspect OCR preprocess tensor (before inference)")
tensor = ocr._preprocess(crop)
print(f"  shape : {tensor.shape}   (expected (1, 64, 128, 3))")
print(f"  dtype : {tensor.dtype}   (expected uint8)")
print(f"  min   : {int(tensor.min())}   max: {int(tensor.max())}   (expected 0-255)")

shape_ok = tensor.shape == (1, 64, 128, 3)
dtype_ok = tensor.dtype == np.uint8
range_ok = int(tensor.min()) >= 0 and int(tensor.max()) <= 255

print(f"  shape contract : {'PASS' if shape_ok else 'FAIL'}")
print(f"  dtype contract : {'PASS' if dtype_ok else 'FAIL'}")
print(f"  range contract : {'PASS' if range_ok else 'FAIL'}")

if not (shape_ok and dtype_ok and range_ok):
    print("\n[ABORT] Tensor contract violated -- halting before inference.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 9. OCR inference
# ---------------------------------------------------------------------------
print("\nSTEP 7 - Run OCR inference")
try:
    t0 = time.perf_counter()
    text = ocr.extract_text(crop)
    ocr_ms = (time.perf_counter() - t0) * 1000
    print(f"  [OK] OCR finished in {ocr_ms:.1f} ms")
    print(f"  Decoded text : {repr(text)}")
    if not text:
        print("  [WARN] Empty string returned.")
        print("         Likely cause: all 10 slots decoded to '_' (padding).")
        print("         The pipeline ran correctly; the model may need fine-tuning")
        print("         for this plate style, or the crop is very low quality.")
except Exception:
    print("  [FAIL] OCR inference raised an exception:")
    traceback.print_exc()
    sys.exit(1)

# ---------------------------------------------------------------------------
# 10. Summary
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("SUMMARY")
print(SEP)
print(f"  Detector : {det_ms:.1f} ms  |  {len(detections)} plate(s) found")
print(f"  OCR      : {ocr_ms:.1f} ms  |  result = {repr(text)}")
print(f"  Total    : {det_ms + ocr_ms:.1f} ms")
print(f"  Tensor   : shape {tensor.shape}  dtype {tensor.dtype}")
print(SEP + "\n")
