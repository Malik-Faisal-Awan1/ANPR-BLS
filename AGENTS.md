# AGENTS.md -- ANPR Root

## 1. Summary
Automatic Number Plate Recognition (ANPR) system for Pakistani vehicles: FastAPI REST API + folder-watcher,
YOLOv9t ONNX plate detector, CCT-S-v2 (fast-plate-ocr) ONNX OCR engine. Stack: Python 3.14, ONNX Runtime, OpenCV, FastAPI/Uvicorn.

---

## 2. Commands

### Install
```
pip install -r api/requirements.txt
```

### Run API server
```
# From repo root:
python api/app.py

# From api/ directory:
python app.py

# Dev (auto-reload) -- from api/:
uvicorn app:app --host 0.0.0.0 --port 5001 --reload

# Prod (no reload) -- from api/:
uvicorn app:app --host 0.0.0.0 --port 5001 --workers 1
```

### Run folder-watcher
```
# From repo root:
python api/watcher.py

# From api/ directory:
python watcher.py
```

### End-to-end pipeline test
```
# From repo root:
python api/test_pipeline.py
python api/test_pipeline.py path/to/plate.jpg

# From api/ directory:
python test_pipeline.py
python test_pipeline.py path/to/plate.jpg
```

> **CWD does not matter.** All paths in config.py are resolved via
> `Path(__file__).resolve().parent` (config.py line 4) -- absolute, anchored
> to api/. Verified: both api/ and repo root work for all three entry points.

---

## 3. Definition of Done
- `test_pipeline.py` prints `PASS` for all 3 tensor contracts (shape, dtype, range).
- API `GET /api/status` returns HTTP 200.
- API `POST /api/process` returns `{"success": true}` with >=1 detection on a valid plate image.
- No `[FAIL]` or `[ABORT]` in `test_pipeline.py` output.
- `api/models/best.onnx` (YOLO) and `api/models/fast-plate/best.onnx` (OCR) exist before any run.

---

## 4. Lookup Map

| Path | Purpose |
|---|---|
| `api/app.py` | FastAPI entrypoint; `POST /api/process`, `GET /api/status`; `_run_inference()` is the core pipeline |
| `api/watcher.py` | Folder-poll loop; watches `api/data/images/`, writes annotated results to `api/output/`, unreadable images to `api/data/failed/` |
| `api/config.py` | Single source of truth for all paths, thresholds, and limits. All tunables live here. Paths are absolute, resolved from config.py location. |
| `api/core/detector.py` | `PlateDetector` -- YOLOv9t ONNX; letterbox pre-proc + manual sigmoid NMS + ROI/foreground filter; vehicle_type populated via intersection-area matching in `_postprocess()` |
| `api/core/ocr_engine.py` | `TextExtractor` -- CCT ONNX; `_CHARSET` must match trained alphabet exactly |
| `api/test_pipeline.py` | Integration harness; validates tensor contracts; prints PASS/FAIL/INCONCLUSIVE explicitly; requires >=1 jpg in `api/data/images/` |
| `api/models/best.onnx` | YOLOv9t detector (5 output classes: license=0, car=1, bike=2, rickshaw=3, truck=4) |
| `api/models/fast-plate/best.onnx` | Fine-tuned CCT-S-v2 OCR model (ONNX input name: "input", shape: NHWC uint8 [N,64,128,3], output name: "plate", shape: [N,10,37]) |
| `api/models/v3_best.onnx` | Previous YOLO checkpoint kept for rollback; not loaded at runtime |
| `training/` | Colab artefacts only (.keras, .onnx, CSVs). Not used at runtime. |
| `anpr_migration/` | Legacy migration artefacts + old PaddleOCR ONNX (inference.onnx). Not used at runtime. |
| `colab_train_fast_plate_ocr_v3.md` | Authoritative Colab runbook for re-training and ONNX export of the OCR model |

### vehicle_type in API/watcher output
`vehicle_type` is **always present** in detection output dicts. It is populated by `_postprocess()` in `detector.py`: for each detected plate, the vehicle bounding box with maximum intersection area is found; if none overlaps, it is set to `"unknown"`. Intersection area (not IoU ratio) is the comparison metric. This spatial matching is the only source -- no separate vehicle classification step exists.

---

## 5. Conventions & Gotchas

- **CORS whitelist is hard-coded** in `app.py` to Tauri origins (`localhost:1420`). Add new origins there, not in `config.py`.
- **Inference runs in `run_in_executor`** in `app.py` because ONNX Runtime releases the GIL. Do not convert to bare `async` calls.
- **`SELECT_PRIMARY_FOREGROUND_ONLY = True`** returns only the single highest-scoring plate per frame even when multiple plates exist. Intentional for the primary use-case.
- **Letterbox padding fill = 114** in `_preprocess` in `detector.py`: matches Ultralytics default. Do not change without retraining.
- **YOLO raw output is `[batch, 9, N_anchors]`**: indices `[4:9]` are raw class logits; sigmoid is applied manually in `_postprocess`. Not standard Ultralytics postprocessing format.
- **`watcher.py` skips files prefixed `result_`, `crop_`, `failed_`** to avoid re-processing its own outputs.
- **numpy pin updated to `>=2.4.6`** in requirements.txt. Verified: numpy 2.4.6 passes all three entry points and `pip check` shows no numpy conflicts.
- **`config.OCR_INTRA_OP_THREADS`** is wired into `SessionOptions.intra_op_num_threads` in `TextExtractor.__init__`. Change thread count in config.py only. (Tuned at 2 threads on one machine at batch 5; benchmark before changing.)
- **`_preprocess` uses letterbox (black padding)**: crop is scaled to fit 128x64 while preserving aspect ratio; black strips fill the remainder. Do NOT revert to plain resize -- square/two-line plates were distorted without this.
- **6% horizontal + 4% vertical bbox expansion** is applied before cropping in both `_run_inference` in `app.py` and `process` in `watcher.py`. Keep both in sync if changed.
- **`_decode` uses `rstrip("_")`**: CCT pads trailing slots with `_`; mid-string `_` does not occur in valid model output.

---

## 6. Hard Boundaries

- **Never modify** `api/models/best.onnx` or `api/models/fast-plate/best.onnx` directly -- binary model weights. Replace by copying a new export.
- **Never commit** `api/data/images/`, `api/data/failed/`, or `api/output/` contents -- runtime I/O only.
- **Do not touch** `anpr_migration/` -- legacy reference artefacts, not active code.
- **Do not modify** `training/` from running code -- Colab-produced artefacts copied here manually.
- **`training/plate_dict_36.txt`** is the legacy PaddleOCR dictionary. The current CCT model does NOT use it; alphabet is baked into `_CHARSET` in `api/core/ocr_engine.py`.
- **`graphify-out/`** is generated output from the graphify tool -- never edit by hand.

---

## 7. Past Mistakes

- **Never use `fast_plate_ocr.LicensePlateRecognizer`** in production -- it downloads/uses the pre-trained global model, ignoring `config.OCR_MODEL_PATH` and the fine-tuned model at `api/models/fast-plate/`.
- **Never apply float normalisation (÷255, mean/std) to OCR input** -- the CCT model handles normalisation internally. ONNX input dtype must be `uint8` (0-255). Passing float32 silently produces garbage output.
- **Never resize OCR crops to a size other than width=128, height=64** -- the CCT model is fixed at this resolution.
- **Never use CTC decoding for OCR output** -- the model uses per-slot argmax over 37 classes, not CTC.
- **For all training-specific pitfalls** (plate_region CSV/config mismatch, PositionEmbedding ONNX export patch for PyTorch 2.6+, etc.) -- see `colab_train_fast_plate_ocr_v3.md` Cell 5. Those notes live there so they stay co-located with the code they describe.

---

## 8. Upgrade Paths (Planned)

> **Status: NOT implemented.** Never reference outbox tables, adapters, or correction storage as if they exist.
> Full design spec: `docs/ROADMAP.md`.

**Behavior:** Mention once per session, one sentence, attributed to maintainers, only when a trigger matches below. Before designing any outbox/correction schema, read docs/ROADMAP.md and follow it.
> Skip if the user declines. If the user chooses a different approach, follow the user.
> Avoid changes that make the upgrade harder (e.g. hard-wiring printing into `watcher.py`), but never block the user's request.

### a) Delivery / printing / external emission of plate reads
Maintainers plan an SQLite outbox + pluggable delivery adapters to send plate reads to printers or external systems without coupling delivery to detection.

### b) Plate correction, labeled-sample storage, retraining, model versioning
Maintainers plan a human-in-the-loop correction flow that feeds a validated retraining pipeline with versioned ONNX rollback.

## Skipped nested AGENTS.md
Intentionally skipped (no custom rules): `api/data/`, `api/output/`, `api/inference/`, `configs/`, `graphify-out/`, `.agent/`.
