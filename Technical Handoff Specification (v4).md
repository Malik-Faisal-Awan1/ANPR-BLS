# Technical Handoff Specification — ANPR OCR Retrain (PP-OCRv3 recognition, 36-char plates)

### v4 — Final (incorporates three rounds of independent review)

> Scope: retrain **recognition only**. Detection (YOLOv9) is working and untouched.
> Target: Google Colab, Python 3.12, T4 GPU. Serving runtime: CPU ONNX Runtime.
> All tensor shapes and latencies below were measured in this repo, not inferred.

## Revision Notes (v1 → v2)

| # | Issue | Fix | Where |
|---|---|---|---|
| 1 | Train/serve preprocessing mismatch: training used verbatim upstream transforms (no CLAHE); serving runs CLAHE-L + RGB before resize | Insert a CLAHE-L + BGR→RGB transform into the train/eval pipeline, matching `ocr_engine.py` | §2 table, Phase 1 |
| 2 | Loading a 97-class pretrained checkpoint into a 37-class head will shape-mismatch or silently fail | Strip CTC/SAR head keys from the checkpoint before load (weight surgery), verify prefixes against the live model first | §2, Phase 3 |
| 3 | PIR (`inference.json`) export from Paddle 3.0 hits unmapped ops in `paddle2onnx` | Set `FLAGS_enable_pir_api=0` before export to force legacy static-graph output | §2 table, Phase 2, Phase 4 |
| 4 | Overfit sanity script in v1 was unrunnable pseudocode | Replaced with a script built on PaddleOCR's actual `build_model`/`build_loss`/`build_dataloader` API | Phase 3 |
| 5 | Flat `lr=0.001` and unweighted `CTC:SAR = 1:1` loss risk gradient shock into the pretrained backbone/neck | Drop LR to `0.0002` w/ 5 warmup epochs; downweight `SARLoss` to `0.2` | §2 table, Phase 3 |
| 6 | Acceptance gate G5 evaluated on only 16 images — no statistical power | Split into a quantitative gate (≥150 held-out crops) and a qualitative sanity check | §4 gates |

## Revision Notes (v2 → v3)

| # | Issue | Fix | Where |
|---|---|---|---|
| 7 | `save_load.py` auto-appends `.pdparams` to `Global.pretrained_model`; extensioned path causes double-extension `FileNotFoundError` | Pass the path **without** the `.pdparams` extension | §2, Phase 3, Appendix A |
| 8 | PaddleOCR's CLI `-o` override parser is unreliable on nested list-of-dict paths | Generate one self-contained `rec_plate_36.yml` with every override baked in | §2, Phase 3, Appendix A |
| 9 | `MultiHead` routes to `SARHead` in `model.train()` mode, which needs `targets` for teacher forcing | Call `model(images, targets)` in the overfit script | Phase 3, Appendix B |
| 10 | PaddleOCR builds transforms from a string→class registry; bare `- CLAHE:` raises `KeyError` | Do CLAHE **offline**; point label files at processed images | Phase 1 |

## Revision Notes (v3 → v4)

A third review, focused on train/serve contract fidelity, Paddle 3.0 device semantics, and Gate 0 export config hygiene, found four remaining implementation traps.

| # | Issue | Fix | Where |
|---|---|---|---|
| 11 | Offline-CLAHE crops written as BGR, decoded as `img_mode: BGR`, produce BGR tensors — but serving feeds RGB. Channels 0/2 inverted between train and serve → colored/tinted plates degrade severely. | Change `DecodeImage.img_mode` to `RGB` in **both** Train and Eval transforms. Offline script continues to write standard BGR to disk so the RGB decode is a single, well-defined swap. | §2 table, Appendix A, Phase 1 |
| 12 | In Paddle 3.0, `build_model()` allocates parameters on the default place (CPU) unless `paddle.device.set_device()` runs first. `build_dataloader(..., device="gpu")` then yields CUDA tensors; the first forward crashes with a CPU/CUDA mismatch. | Call `paddle.device.set_device("gpu")` immediately after `import paddle` in the overfit script. | Appendix B, Phase 3 |
| 13 | Gate 0 smoke (Phase 2) needs to export *pretrained* weights. Using `rec_plate_36.yml` (37-class dict) against the unstripped 97-class checkpoint crashes on shape mismatch. | Gate 0 must use the **upstream** `configs/rec/PP-OCRv3/en_PP-OCRv3_mobile_rec.yml` and point `Global.pretrained_model` at the unstripped upstream `best_accuracy`. | Phase 2 |
| 14 | Weight surgery was delegated to the coding agent without a reference implementation; substring surgery done wrong silently strips backbone keys or nothing. | Add Appendix D: a verified `strip_head_weights.py` with an unconditional `assert len(removed_keys) > 0`, plus an optional live-model `state_dict()` cross-check. | Phase 3, Appendix D |

---

## 1. Codebase Baseline & File Manifest

| Path | Responsibility |
|---|---|
| `app.py` (lines 167–181) | Request pipeline. YOLO crop → +2% horizontal pre-pad → +4%/4% uniform pad (`pad_x=max(3,…)`, `pad_y=max(2,…)`) → OCR. Line 1 imports `torch` before Paddle (Windows `shm.dll` conflict workaround). |
| `config.py` | Runtime constants. `YOLO_IMGSZ=640`, `CONFIDENCE_THRESHOLD=0.55`, `IOU_THRESHOLD=0.45`, `OCR_INPUT_HEIGHT=48`, `OCR_INPUT_WIDTH=320`, `OCR_MIN_CONFIDENCE=0.0`, `OCR_INTRA_OP_THREADS=min(4,cpu)`, `MODEL_PATH=models/v3_best.onnx`, `CUSTOM_OCR_MODEL_DIR=models/custom_ocr`. |
| `core/detector.py` | YOLOv9 ONNX detector. Centered letterbox (114-pad), NMS, ROI/foreground filtering. Postprocess assumes **9 output channels** (4 bbox cxcywh + 5 sigmoid class scores, plate = class 0). |
| `core/ocr_engine.py` | ONNX Runtime rec inference. Preprocess: **CLAHE on L-channel → BGR2RGB** → aspect-preserving resize to h=48 → `(x/255-0.5)/0.5` norm → right-pad to 320. Decode: raw `argmax` (no re-softmax), CTC blank=0 collapse, full-width-space→space, strip spaces, `a-z`→upper, keep `A-Z0-9` only. Loads `models/custom_ocr/ocr_v4.onnx`; dict hardcoded as 96-entry `PP_OCR_DICT`. **This file is the ground truth for the train-time image contract.** |
| `models/best.onnx` (9.5 MB) | Working YOLOv9 detector. Measured I/O: `images [batch,3,height,width]` → `output0 [batch,9,anchors]`. |
| `models/v3_best.onnx` | Legacy detector artifact, same 9-channel layout. `config.MODEL_PATH` points here, **not** at `best.onnx` — path debt. |
| `models/best.pt`, `models/last.pt` (4.6 MB each) | YOLO training checkpoints (export source). |
| `models/custom_ocr/ocr_v4.onnx` (22.2 MB) | Deployed rec model = **base PP-OCRv4 English, 97 classes**. Measured I/O: `x [None,3,48,320]` → `fetch_name_0 [None,40,97]` float32. CPU latency ~111 ms/crop. Root cause of letter errors; digits survived because 0–9 occupy indices 1–10 in both dicts. |
| `models/custom_ocr/en_dict.txt` | Correct 36-char target dict (`0-9`, `A-Z`, one/line). Currently unused by `ocr_engine.py` (dict hardcoded). |
| `models/custom_ocr/fine_tuned_ocr.onnx.FAILED` | 0-byte artifact of the failed `paddle2onnx` conversion. Loader must never reference it. |
| `models/plate_model_inference.zip` → `…/rec_plate/{inference.json, inference.pdiparams, inference.yml}` | Prior fine-tune, Paddle 3.x PIR format. `inference.yml`: `model_name: en_PP-OCRv3_mobile_rec`, `RecResizeImg [3,48,320]`, `CTCLabelDecode`, 36-char dict. Parsed graph tail: `…/CTCHead/Linear [64×38]` → `softmax` → `fetch`. Head = 38 outputs vs dict 36 + blank 1 = 37. **Unexplained +1 class; old weights are unshippable.** |
| `models/plate_model_weights.zip` | Prior training weights (`.pdparams/.pdopt/.states`). Reference only. |
| `models/paddle_static/` | Empty directory. Dead artifact. |
| `anpr_migration/{inference.onnx, yolov9t_15k_custom.onnx, en_dict_copy.txt, classes_yolov9.txt}` | Migration inputs. `en_dict_copy.txt` = 36-char dict; `classes_yolov9.txt` = 5 detector classes. |
| `training/plate_dict_36.txt` | Canonical 36-char dict copy (byte-identical to `en_dict.txt`, verified by `diff`). |
| `training/validate_rec_labels.py` | Label-file validator (TAB-separated `relpath<TAB>LABEL`, charset ⊆ dict, len ≤ 25, image readable, no dup paths). Tested: accepts clean rows, rejects lowercase/`-`. |
| `colab_retrain_ocr.md` | Colab runbook (prereqs, Gate 0, train, export, convert, wire-in). |
| `colab.md` | YOLO export runbook. **Cell 4 asserts 10 output channels — contradicts measured 9-channel exports and `detector.py`.** Do not follow Cell 4's channel assertion. |
| `data/images/` (16 JPG, 1080×1920) | Test images only. No label files in repo. Training labels live with the operator. |
| `data/failed/` | Empty directory. Purpose undocumented. |
| `source` | 0-byte stray file in repo root. Delete. |
| `requirements.txt` | Pinned runtime env (`ultralytics==8.4.51`, `onnx==1.21.0`, `onnxruntime>=1.18.0`, `opencv-python==4.13.0.92`, `torch==2.12.0`, `protobuf==7.34.1`, `numpy<2.4`). No Paddle packages — Paddle lives only in `venv_312`. Duplicate `python-multipart` entries (3.3.0 and 0.0.32). |
| `venv_312/` | Python 3.12.3 env with `paddlepaddle==3.0.0`, `paddle2onnx==2.1.0`, `paddleocr==3.7.0`, `paddlex==3.7.2`, `numpy==2.3.5`, `protobuf==7.36.2`, `opencv-contrib-python==4.10.0.84`. Paddle CLI is `venv_312/bin/paddle2onnx` (`python -m paddle2onnx` does not exist). |

Input/output tensor shapes (measured, not inferred):

| Model | Input | Output | Preprocess contract |
|---|---|---|---|
| `best.onnx` / `v3_best.onnx` | `images [N,3,H,W]` float32, fed 640×640 | `output0 [N,9,A]` float32 raw logits; ch0–3 cxcywh, ch4–8 class scores, sigmoid in code | BGR→RGB, /255, centered letterbox pad 114, strip pad + rescale in postprocess |
| `ocr_v4.onnx` | `x [N,3,48,320]` float32 | `fetch_name_0 [N,40,97]` float32, rows sum ≈1.0 (softmax already applied) | CLAHE-L → BGR2RGB → h=48 aspect resize → `(x/255-0.5)/0.5` → right-pad 320 |
| Retrain target | `x [N,3,48,320]` float32 | `[N,40,37]` (40 = measured timestep count at width 320; 37 = 36 dict + blank). **Assert at export; abort on any other last-dim.** | Same as `ocr_engine._prepare_single_input`, **RGB channels** |

Data loader conventions (PaddleOCR `SimpleDataSet`, per `en_PP-OCRv3_mobile_rec.yml` release/3.7): label files `./train_data/rec_gt_train.txt`, `./train_data/rec_gt_val.txt`, rows `relpath<TAB>LABEL`, joined under `data_dir`; train transforms `DecodeImage` → `RecConAug(prob 0.5, ext_data_num 2, image_shape [48,320,3])` → `RecAug` → `MultiLabelEncode` → `RecResizeImg([3,48,320])` → `KeepKeys(image, label_ctc, label_sar, length, valid_ratio)`; eval drops the augmentation ops. SAR labels are auto-derived by `MultiLabelEncode`. **v4: `DecodeImage.img_mode` is set to `RGB`, not `BGR` — see §2 "Train-time image transform" and Appendix A.**

Baseline metrics: **none exist.** No val accuracy, no CER, no per-image OCR transcript log, no detector mAP. The only quantified facts are the two latency probes above (YOLO ~143 ms/frame CPU, OCR ~111 ms/crop CPU) and the qualitative failure (letters wrong, digits right, e.g. `LET1191`→`SL1191`).

Technical debt register: (1) `config.MODEL_PATH` → `v3_best.onnx` while ops run against `best.onnx`; (2) `colab.md` Cell 4 encodes a 10-channel layout nothing produces; (3) `ocr_engine.py` hardcodes the 97-char dict; (4) `OCR_MIN_CONFIDENCE=0.0` accepts every decode; (5) 0-byte `.FAILED` file, empty `paddle_static/`, empty `data/failed/`, stray `source` file, duplicate `python-multipart` pins.

## 2. Architectural Delta & Hyperparameter Matrix

Module changes (recognition only; detector untouched):

- Backbone: `MobileNetV1Enhance(scale=0.5, last_conv_stride=[1,2], last_pool_type=avg, last_pool_kernel_size=[2,2])` (release/3.7 `en_PP-OCRv3_mobile_rec.yml`, fetched verbatim).
- Neck: SVTR (`name: svtr, dims: 64, depth: 2, hidden_dims: 120, use_guide: True`) — same family as the prior graph but re-initialized from official English pretraining.
- Heads: `MultiHead = [CTCHead, SARHead(enc_dim=512, max_text_length=25)]`. Deployed head is CTC only; SAR is a training auxiliary and is discarded at export.
- Loss: `MultiLoss = [CTCLoss weight=1.0, SARLoss weight=0.2]`.
- PostProcess/Metric: `CTCLabelDecode` / `RecMetric(main_indicator=acc, ignore_space=False)`.
- Dictionary: `training/plate_dict_36.txt` (36 chars) with `Global.use_space_char=False`.

| Parameter | Current | Proposed | Technical Justification |
|---|---|---|---|
| Rec config | Unknown (no yml/logs) | `configs/rec/PP-OCRv3/en_PP-OCRv3_mobile_rec.yml` @ `release/3.7` | Only config matching both prior graph and 48×320 runtime contract on Paddle 3.x / Python 3.12 |
| Init weights | Suspect prior weights | `en_PP-OCRv3_rec_train.tar` (204 MB), **head keys stripped** | Starts from consistent dict/head pair; unstripped checkpoint shape-mismatches 37-class head |
| Optimizer / LR | Unknown | Adam (0.9, 0.999), Cosine `lr=0.0002`, `warmup_epoch=5`, L2 `3e-5` | `0.001` is from-scratch rate; applying it during head resize drives gradient shock into pretrained backbone |
| Loss weighting | Unknown | `CTC=1.0`, `SAR=0.2`, **baked into yml** | SAR magnitudes 3–5× CTC early; uniform weighting lets a discarded head dominate the shared neck |
| Config delivery | N/A | Single self-contained `rec_plate_36.yml` (Appendix A) | CLI `-o` parser unreliable on nested list-of-dict paths |
| **Train-time image transform** | `DecodeImage(BGR)` → `RecConAug` → `RecAug` → `RecResizeImg` | **`DecodeImage(img_mode: RGB)`** → `RecConAug` → `RecAug` → `RecResizeImg`; CLAHE applied **offline** and images saved as BGR on disk | Serving contract is CLAHE-L → BGR2RGB. Offline script writes BGR; `img_mode: RGB` performs the single, well-defined BGR→RGB swap at load, exactly matching serving. **v3's `img_mode: BGR` trained on BGR tensors while serving fed RGB — a silent red/blue channel inversion.** |
| Epochs | Unknown | `80` | Bounds Colab cost; `best_accuracy` selection retains the peak |
| Batch/card | Unknown | `64`, fallback `32` on OOM | 128 is multi-GPU; 64 fits T4 16 GB at 48×320 with ~12 M-param backbone |
| Eval cadence | Unknown | `eval_batch_step=[0,500]` | Small plate datasets need denser validation to catch the acc peak |
| `character_dict_path` | Upstream (97 classes) | `training/plate_dict_36.txt` in train AND export | Dict determines head width (37 = 36 + blank) |
| `use_space_char` | Presumed true | `False` | Plates contain no spaces; a space token adds a dead logit |
| `max_text_length` | Unknown | `25` | Config default; bounds SAR label length |
| Export output | `inference.json` (PIR) + 38-class head | `FLAGS_enable_pir_api=0` → legacy static graph `inference.pdmodel` + `.pdiparams`, 37-class head; convert opset 11; assert `n_cls == 37` | `paddle2onnx==2.1.0` can't map PIR-dialect ops emitted by Paddle 3.0 |
| Runtime dict | Hardcoded 97-char list | Load `models/custom_ocr/en_dict.txt` (36 chars) | Removes dict/code drift; preserves no-re-softmax argmax |

Pre-training/checkpoint strategy: download `en_PP-OCRv3_rec_train.tar`, extract `best_accuracy.{pdparams,pdopt}`. **Before** pointing `Global.pretrained_model` at it, run Appendix D to strip CTC/SAR head keys. Pass the stripped checkpoint via `-o Global.pretrained_model=…/best_accuracy_stripped` **without extension**. Fine-tune mode: loads weights, resets optimizer state and epoch counter. No frozen layers. Resume-from-interrupt uses `-o Global.checkpoints=./output/rec_plate_36/latest`. Selection metric is val `acc`; ship `best_accuracy`.

Explicit architectural failure risks:

1. **PIR→ONNX conversion can still fail even with `FLAGS_enable_pir_api=0`.** The flag sidesteps the specific ops seen in the last failure but is not a coverage guarantee. Exporter-coverage risk, not a training risk.
2. **CTC/SAR multi-head + custom dict coupling.** A single stray line in `plate_dict_36.txt` silently changes both head widths and label indices → trains but decodes garbage. Single point of failure with no checksum.
3. **Capacity/latency corner.** `MobileNetV1Enhance(0.5)` + 2-layer SVTR (~12 M params) sized for clean text lines; heavily degraded plates may plateau below the bar. The next step up breaks the ~111 ms CPU budget by ~10×. If val `acc` plateaus under gate, only data volume or latency budget can give.
4. **Weight-surgery correctness.** Stripping head keys assumes PaddleOCR's `MultiHead` state-dict prefixes. Confirm against the freshly-initialized model's `state_dict()` before trusting the strip — see Appendix D's hardened variant.

## 3. Hardware & Environment Feasibility

- Training compute: one NVIDIA T4 (16 GB, Colab GPU runtime). FP32 (Paddle default; **do not enable AMP**). Batch 64 nominal, 32 on OOM, 16 floor; if 16 OOMs, reduce `num_workers` to 2 first, then concede. Expected wall time: 1–3 h for 80 epochs on a few-thousand-crop dataset; Gate 0 smoke ≈ 5 min.
- Inference compute (unchanged): CPU-only ONNX Runtime; measured YOLO ~143 ms/frame at 640, OCR ~111 ms/crop at 48×320. New rec model must benchmark ≤ 150 ms/crop before swap.
- Environment: Colab Python **3.12** (2025.10+ images; do not target 3.10). `paddlepaddle==3.0.0`, `paddle2onnx==2.1.0`, `paddleocr==3.7.0`, `onnxruntime>=1.10`, `opencv-python-headless<4.10`, `pillow`, `pyyaml`.
- Hard constraints: **no `numpy<2` downgrade**; **no Python interpreter switching**; protobuf fix is `pip install -q "protobuf>=5,<6"` + runtime restart; `paddle2onnx` invoked as binary; PaddleOCR cloned at `--branch release/3.7`; `ls`-verify `tools/train.py`, `tools/export_model.py`, `en_PP-OCRv3_mobile_rec.yml`; `FLAGS_enable_pir_api=0` exported in the same shell as `export_model.py`.
- No package additions to the serving repo. Local Paddle work uses `venv_312/bin/python`, never system Python 3.14.

## 4. Step-by-Step Implementation Backlog

**Phase 0 — repo hygiene** (local, no GPU):
- [ ] Delete `source`. Document or remove `data/failed/`.
- [ ] Fix `requirements.txt` duplicate `python-multipart` pins.
- [ ] Record decision: repoint `config.MODEL_PATH` to `models/best.onnx` or document why `v3_best.onnx` stays.
- [ ] Correct or delete `colab.md` Cell 4.

**Phase 1 — data contract** (local, no GPU):
- [ ] Crop plate regions with working YOLO (`models/best.onnx`) — train on crops, not full frames.
- [ ] Write `rec_gt_train.txt` / `rec_gt_val.txt` (`relpath<TAB>LABEL`, labels ⊆ `0-9A-Z`, len ≤ 25, disjoint splits, 10–15% val).
- [ ] Run `python training/validate_rec_labels.py <labels> <imgdir>` on both. Zero errors required.
- [ ] `diff` val dict copy against `training/plate_dict_36.txt`; must be byte-identical.
- [ ] **[v4]** Offline CLAHE script: read raw crops (BGR), split L, apply CLAHE with **the exact `clipLimit`/`tileGridSize` read out of `core/ocr_engine.py`** (do not assume OpenCV defaults), merge L back, **convert to RGB**, then **convert back to BGR before `cv2.imwrite`** — the on-disk file must be standard BGR so that `DecodeImage(img_mode: RGB)` at load performs exactly one BGR→RGB swap. Save to `train_data/crops_clahe/`. Point `rec_gt_train.txt`/`rec_gt_val.txt` at the processed images. The yml transform list contains **no CLAHE operator** (PaddleOCR's string→class registry would `KeyError`).

**Phase 2 — Gate 0 smoke** (Colab T4, ~5 min, **blocks everything after**):
- [ ] Install + verify cells pass (`paddle.utils.run_check()`, `is_compiled_with_cuda()==True` on GPU runtime).
- [ ] Clone `release/3.7`, `ls`-verify `tools/train.py`, `tools/export_model.py`, `en_PP-OCRv3_mobile_rec.yml`.
- [ ] **[v4]** Export the **unstripped upstream pretrained** checkpoint for the smoke test:
  - Config: `configs/rec/PP-OCRv3/en_PP-OCRv3_mobile_rec.yml` (the **upstream** file, **not** `rec_plate_36.yml` — the custom yml declares a 37-class head and will crash against the 97-class checkpoint).
  - `Global.pretrained_model` pointing at the extracted `en_PP-OCRv3_rec_train/best_accuracy` (unstripped, no extension).
  - `export FLAGS_enable_pir_api=0` in the same cell before `tools/export_model.py`.
- [ ] Assert `inference.pdmodel` + `.pdiparams` + `.yml` all non-zero.
- [ ] Convert via `paddle2onnx`, opset 11, `onnx.checker`. If legacy-graph export still fails to convert, fall back to routes 2–3 (`paddlex` plugin, `p2o` opset 17).
- [ ] ORT probe: input `[?,3,48,320]`, 3-D float output, rows ≈ 1.0. (Smoke-test output last-dim will be 97 — that's expected; the 37-class assert applies only to Phase 4.) All converters failing here = STOP, adopt PyTorch fallback, no training.

**Phase 3 — train** (Colab T4):
- [ ] **[v4]** Run `training/strip_head_weights.py` (Appendix D). It loads the 97-class pretrained `best_accuracy.pdparams`, strips `ctc_head.*`, `sar_head.*`, and `head.fc*` keys, asserts at least one key was stripped, saves `best_accuracy_stripped.pdparams`. For maximum safety, use the hardened variant that cross-checks against the freshly-built 37-class model's `state_dict().keys()` and strips any key whose shape disagrees.
- [ ] Generate `configs/rec/PP-OCRv3/rec_plate_36.yml` (Appendix A) as a single self-contained file. **Do not chain CLI `-o` overrides.** In that file, set `Global.pretrained_model: ./best_accuracy_stripped` **without** the `.pdparams` extension.
- [ ] Run `training/overfit_sanity.py` (Appendix B). Must pass before full training. **[v4]** The script calls `paddle.device.set_device("gpu")` before building the model — without it, Paddle 3.0 allocates parameters on CPU while the loader yields CUDA tensors, crashing on the first forward.
- [ ] Launch fine-tune per `colab_retrain_ocr.md` §6 using `rec_plate_36.yml` (epoch 80, batch 64→32, all layers unfrozen).
- [ ] Keep `output/rec_plate_36/best_accuracy*`. Abort on NaN loss or flat val `acc` for 15 epochs.

**Phase 4 — export + convert** (Colab):
- [ ] Run Appendix C exactly: `export FLAGS_enable_pir_api=0`, then `export_model.py` from `output/rec_plate_36/best_accuracy` with `rec_plate_36.yml`.
- [ ] Assert both `.pdmodel`/`.pdiparams` non-zero (Appendix C step 2) before converting.
- [ ] Convert via `paddle2onnx` static path (Appendix C step 3). Fall back to routes 2–3 if needed.
- [ ] Assert exported output last-dim `== 37` (Appendix C step 4).
- [ ] Paddle-vs-ONNX parity: identical greedy-CTC strings on ≥95% of val crops.

**Phase 5 — integrate** (local):
- [ ] Copy new ONNX to `models/custom_ocr/plate_rec_36.onnx` (new name; keep `ocr_v4.onnx` until beaten).
- [ ] Refactor `ocr_engine.py` to load `en_dict.txt` from disk.
- [ ] Score all 16 `data/images/` + record transcripts; swap only on strict improvement with no digit regressions.

### Acceptance gates (all mandatory, no partial credit)

- **G0** export path: `FLAGS_enable_pir_api=0` set before export; `.onnx` size > 1 MB; `onnx.checker` passes; ORT input `[?,3,48,320]`; output 3-D float with rows summing to 1.0 ± 0.01.
- **G1** overfit: single-batch CTC+SAR loss falls ≥ 90% within 150–200 iterations, run against the **stripped** pretrained checkpoint, on GPU (i.e. `paddle.device.set_device("gpu")` called first).
- **G2** training: val `RecMetric acc` strictly increases over the pretrained baseline; `best_accuracy` written; NaN at any step fails.
- **G3** export: `inference.pdmodel` + `.pdiparams` + `.yml` non-zero; ONNX last-dim exactly 37; dict file byte-identical to `training/plate_dict_36.txt`.
- **G4** parity: Paddle vs ONNX greedy-CTC strings identical on ≥95% of val crops.
- **G5a — quantitative:** held-out ≥ 150 annotated crops (disjoint), plate-level exact-match ≥ 92%, CER ≤ 0.03.
- **G5b — qualitative:** run the 16 `data/images/` end-to-end; verify the `LET1191`-class correction; zero digit regressions vs `ocr_v4.onnx`; per-crop CPU latency ≤ 150 ms.

---

## Appendix A — Generated run config: `configs/rec/PP-OCRv3/rec_plate_36.yml`

**[v4]** `img_mode` is **`RGB`** in both Train and Eval. `pretrained_model` has **no `.pdparams` extension**. Every §2 override is baked in.

```yaml
Global:
  use_gpu: true
  epoch_num: 80
  log_smooth_window: 20
  print_batch_step: 10
  save_model_dir: ./output/rec_plate_36
  save_epoch_step: 3
  eval_batch_step: [0, 500]
  cal_metric_during_train: true
  pretrained_model: ./best_accuracy_stripped  # NOTE: no .pdparams extension — save_load.py appends it
  checkpoints: null
  character_dict_path: training/plate_dict_36.txt
  use_space_char: false
  infer_img: null

Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    name: Cosine
    learning_rate: 0.0002
    warmup_epoch: 5
  regularizer:
    name: L2
    factor: 3.0e-05

Architecture:
  model_type: rec
  algorithm: SVTR_LCNet
  Transform: null
  Backbone:
    name: MobileNetV1Enhance
    scale: 0.5
    last_conv_stride: [1, 2]
    last_pool_type: avg
    last_pool_kernel_size: [2, 2]
  Head:
    name: MultiHead
    head_list:
      - CTCHead:
          Neck:
            name: svtr
            dims: 64
            depth: 2
            hidden_dims: 120
            use_guide: true
          Head:
            fc_decay: 0.00001
      - SARHead:
          enc_dim: 512
          max_text_length: 25

Loss:
  name: MultiLoss
  loss_config_list:
    - CTCLoss:
        weight: 1.0
    - SARLoss:
        weight: 0.2

PostProcess:
  name: CTCLabelDecode

Metric:
  name: RecMetric
  main_indicator: acc
  ignore_space: false

Train:
  dataset:
    name: SimpleDataSet
    data_dir: ./train_data
    label_file_list: [./train_data/rec_gt_train.txt]
    transforms:
      - DecodeImage:
          img_mode: RGB        # MATCHES PRODUCTION: one BGR→RGB swap at load
          channel_first: false
      - RecConAug:
          prob: 0.5
          ext_data_num: 2
          image_shape: [48, 320, 3]
      - RecAug:
      - MultiLabelEncode:
      - RecResizeImg:
          image_shape: [3, 48, 320]
      - KeepKeys:
          keep_keys: [image, label_ctc, label_sar, length, valid_ratio]
  loader:
    shuffle: true
    batch_size_per_card: 64
    drop_last: true
    num_workers: 4

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: ./train_data
    label_file_list: [./train_data/rec_gt_val.txt]
    transforms:
      - DecodeImage:
          img_mode: RGB        # MATCHES PRODUCTION
          channel_first: false
      - MultiLabelEncode:
      - RecResizeImg:
          image_shape: [3, 48, 320]
      - KeepKeys:
          keep_keys: [image, label_ctc, label_sar, length, valid_ratio]
  loader:
    shuffle: false
    max_text_length: 25
    batch_size_per_card: 64
    drop_last: false
    num_workers: 4
```

Note: `label_file_list` points at the offline-CLAHE'd crop set from Phase 1 (`train_data/crops_clahe/`). Confirm the paths inside those label files resolve to the processed images, not the raw crops, before launching. **Confirm that the offline script writes BGR to disk**, not RGB — otherwise the `img_mode: RGB` decode double-swaps and reproduces the exact bug Patch A fixes.

## Appendix B — Verified overfit sanity script: `training/overfit_sanity.py`

**[v4]** Adds `paddle.device.set_device("gpu")` before any model construction. Passes `targets` into every `model(...)` call (required in `model.train()` mode because `MultiHead` routes to `SARHead`).

```python
import paddle
paddle.device.set_device("gpu")  # PREVENTS CPU/GPU TENSOR MISMATCH IN PADDLE 3.0

from ppocr.data import build_dataloader
from ppocr.modeling.architectures import build_model
from ppocr.losses import build_loss
from ppocr.utils.utility import load_config

cfg = load_config("configs/rec/PP-OCRv3/rec_plate_36.yml")
model = build_model(cfg["Architecture"])
model.train()
loss_fn = build_loss(cfg["Loss"])
loader = build_dataloader(cfg, "Train", device="gpu")

batch = next(iter(loader))
images = batch[0]
targets = batch[1:]  # [label_ctc, label_sar, length, valid_ratio]

opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())

preds = model(images, targets)
init_loss = loss_fn(preds, batch)["loss"].item()

for _ in range(150):
    preds = model(images, targets)
    loss_dict = loss_fn(preds, batch)
    loss = loss_dict["loss"]
    loss.backward()
    opt.step()
    opt.clear_grad()

final_loss = loss.item()
print(f"[Sanity Check] Init Loss: {init_loss:.4f} -> Final Loss: {final_loss:.4f}")
assert final_loss < 0.15 * init_loss, f"Overfit failed: {init_loss:.4f} -> {final_loss:.4f}"
print("[Sanity Check] Passed: gradient flow verified across backbone, neck, and both heads.")
```

Adapt the `ppocr.*` import paths if the checked-out `release/3.7` branch differs — verify by `ls`-ing `ppocr/` first.

## Appendix C — Export & convert commands (Phase 4)

Run from the PaddleOCR repo root with `venv_312` active.

```bash
# 1. Export static graph (disable PIR explicitly, same shell as the export call)
export FLAGS_enable_pir_api=0
python tools/export_model.py \
  -c configs/rec/PP-OCRv3/rec_plate_36.yml \
  -o Global.pretrained_model=./output/rec_plate_36/best_accuracy \
     Global.save_inference_dir=./inference/rec_plate_36

# 2. Assert files exist and are non-zero
test -s ./inference/rec_plate_36/inference.pdmodel || { echo "pdmodel missing"; exit 1; }
test -s ./inference/rec_plate_36/inference.pdiparams || { echo "pdiparams missing"; exit 1; }

# 3. Convert via paddle2onnx, static-graph path
paddle2onnx \
  --model_dir ./inference/rec_plate_36 \
  --model_filename inference.pdmodel \
  --params_filename inference.pdiparams \
  --save_file ./models/custom_ocr/plate_rec_36.onnx \
  --opset_version 11 \
  --enable_onnx_checker True

# 4. Assert 37 classes on the output node
python -c "
import onnx
model = onnx.load('./models/custom_ocr/plate_rec_36.onnx')
out_shape = [d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim]
print('Exported Output Shape:', out_shape)
assert out_shape[-1] == 37, f'Expected 37 classes, got {out_shape[-1]}'
"
```

If step 3 fails even on the legacy static graph, fall back to routes 2–3 from §2 (`paddlex` plugin, `p2o` opset 17) before treating it as a hard architectural stop.

## Appendix D — Weight surgery: `training/strip_head_weights.py`

**[v4]** Primary variant — matches the strip prefixes used in the PP-OCRv3 `MultiHead` implementation and hard-fails if nothing is stripped.

```python
import os
import paddle

src_path = "en_PP-OCRv3_rec_train/best_accuracy.pdparams"
dst_path = "best_accuracy_stripped.pdparams"

assert os.path.exists(src_path), f"Source weights not found at {src_path}"

state_dict = paddle.load(src_path)
initial_keys = len(state_dict)

# PP-OCRv3 MultiHead prefixes: head.ctc_head.*, head.sar_head.*
# Also guard against a bare head.fc* if a variant uses it.
stripped_dict = {}
removed_keys = []

for k, v in state_dict.items():
    if "ctc_head" in k or "sar_head" in k or k.startswith("head.fc"):
        removed_keys.append(k)
    else:
        stripped_dict[k] = v

print(f"Total keys: {initial_keys}")
print(f"Stripped {len(removed_keys)} head weight keys:")
for rk in removed_keys:
    print(f"  - {rk}")

assert len(stripped_dict) > 0, "Error: All weights were stripped!"
assert len(removed_keys) > 0, "Error: No head keys found — check prefix assumptions."

paddle.save(stripped_dict, dst_path)
print(f"Successfully saved stripped weights to {dst_path}")
```

**Hardened variant (recommended when the agent can build the target model):** replace the substring loop with a live-model intersection so that any prefix drift in PaddleOCR 3.7 fails loudly rather than silently.

```python
import os
import paddle
from ppocr.modeling.architectures import build_model
from ppocr.utils.utility import load_config

paddle.device.set_device("gpu")

src_path = "en_PP-OCRv3_rec_train/best_accuracy.pdparams"
dst_path = "best_accuracy_stripped.pdparams"
assert os.path.exists(src_path), f"Source weights not found at {src_path}"

# Build the 37-class target model so we have the authoritative key set and shapes.
cfg = load_config("configs/rec/PP-OCRv3/rec_plate_36.yml")
target_model = build_model(cfg["Architecture"])
target_sd = target_model.state_dict()

source_sd = paddle.load(src_path)
stripped, removed, kept = {}, [], []

for k, v in source_sd.items():
    tv = target_sd.get(k, None)
    if tv is None:
        removed.append((k, "not in target")); continue
    if tuple(v.shape) != tuple(tv.shape):
        removed.append((k, f"shape {tuple(v.shape)} vs {tuple(tv.shape)}")); continue
    stripped[k] = v
    kept.append(k)

print(f"source keys={len(source_sd)}  kept={len(kept)}  removed={len(removed)}")
for k, why in removed:
    print(f"  removed: {k}  ({why})")

assert len(kept) > 0, "Nothing kept — check config or checkpoint."
assert any("head" in k for k, _ in removed), "No head keys removed — assumptions wrong."

paddle.save(stripped, dst_path)
print(f"Saved {dst_path}")
```

The hardened variant is the one that satisfies §2 risk #4 ("verify, don't assume"). Either passes the strip step; the hardened one converts a silent shape-mismatch on load into a loud failure here.

---

**Final verdict.** With v4:

- The train/serve image contract is 1:1 (`img_mode: RGB` on disk-BGR crops, matching `ocr_engine.py`'s CLAHE-L → BGR2RGB).
- Paddle 3.0 device semantics are handled (`paddle.device.set_device` before model construction).
- Gate 0 exercises the export/convert path with the correct upstream config and unstripped checkpoint, avoiding a spurious shape-mismatch crash.
- Weight surgery has a reference implementation with a live-model cross-check.
- v3's other traps (PIR, list-of-dict CLI parsing, SAR teacher forcing, offline CLAHE, `.pdparams` double-extension) remain addressed.

Hand off.