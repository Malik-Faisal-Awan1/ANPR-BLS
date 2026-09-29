# AGENTS.md -- api/core/

## detector.py

- YOLO output tensor is `[batch, 9, N_anchors]`: indices `[0:4]` = cx/cy/w/h, `[4:9]` = raw class logits. Sigmoid applied manually in `_postprocess`. Not standard Ultralytics postprocessing format.
- NMS is done per-class via `cv2.dnn.NMSBoxes` which expects `[x,y,w,h]` format (not `[x1,y1,x2,y2]`).
- `vehicle_type` is populated in `_postprocess` by finding the vehicle bbox with maximum **intersection area** (not IoU ratio) with each plate bbox. Plates with no overlapping vehicle bbox get `"unknown"`. This is the only source of vehicle_type.
- `_calculate_foreground_score` biases toward lower-in-frame plates (`y_weight = 0.25 + 0.75 * y_norm`). Intentional for approaching-vehicle scenarios.
- `SELECT_PRIMARY_FOREGROUND_ONLY=True` in config.py truncates results to 1 plate per frame. Change only via config, not code.
- `detect` is a thin wrapper around `detect_batch`. Add batch-level logic to `detect_batch`, not `detect`.

## ocr_engine.py

- `_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"` (37 chars). Must stay byte-for-byte identical to the alphabet baked into `api/models/fast-plate/best.onnx`. Any mismatch silently corrupts all readings.
- Input pipeline: BGR crop -> `cv2.resize(crop, (128, 64))` -> `cv2.cvtColor(BGR2RGB)` -> `[np.newaxis]` adds batch dim -> dtype stays `uint8`. No float conversion ever.
- `_decode` uses `rstrip("_")`: CCT pads trailing slots with `_`; mid-string `_` does not occur in valid model output, so we strip trailing `_` only (not all `_`). Verified by simulation.
- `OCR_INTRA_OP_THREADS` from config.py is wired into `SessionOptions.intra_op_num_threads` in `__init__`. Verified accepted by onnxruntime.
- Output shape from ONNX session: `[N, 10, 37]`. Decoded by argmax over axis=-1 per slot, then charset index lookup, then rstrip `_`.
- All four method names must be kept for call-site compatibility: `extract_text`, `extract_texts`, `extract_plate` (alias of extract_text), `extract_plate_texts` (alias of extract_texts).
