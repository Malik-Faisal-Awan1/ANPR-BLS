# ROADMAP.md -- Planned Upgrade Paths

> Status: PLANNED, not implemented. Nothing below exists in the codebase today.
> Owner: maintainers. Last updated: 2026-09-29.

---

## (a) ANPR-to-Legacy-Printer Integration

### Trigger
Any task that delivers plate reads to printers, external APIs, webhooks, or any system outside this process.

### Motivation
Detection latency must not block delivery reliability. Network/printer outages must not cause lost reads.

### Approach: SQLite Outbox + Pluggable Delivery Adapters

**Outbox event schema**

| Field | Type | Notes |
|---|---|---|
| `event_id` | TEXT (UUID) | Primary key; idempotency key for delivery |
| `plate` | TEXT | Decoded plate text |
| `vehicle_type` | TEXT | car/bike/rickshaw/truck/unknown |
| `confidence` | REAL | Detector confidence score |
| `timestamp` | TEXT (ISO-8601) | Time of detection |
| `image_ref` | TEXT | Absolute path to result image, or NULL |
| `status` | TEXT | pending / sent / failed / acked |
| `retry_count` | INTEGER | Incremented on each failed delivery attempt |
| `last_error` | TEXT | Last delivery error message, or NULL |

**Idempotency and deduplication**
- Each detection write is idempotent by `event_id` (INSERT OR IGNORE).
- Same vehicle detected across consecutive frames: deduplicate within a configurable time window before writing. TODO: decide dedup window and key (e.g. plate text per camera).

**Below-confidence routing**
- **OCR Confidence Definition:** The minimum over all non-pad slots of the maximum per-slot probability.
- Reads where this OCR confidence is below `TODO: decide from labeled data` are NOT written to the outbox for auto-delivery. Route them to a human-review queue instead. Do not print uncertain reads automatically.
- **Note:** At 0.30, a known wrong read (0.3047) would pass. Confidence is not calibrated, and confident misreads (e.g. 0/O, 8/B) still need format validation.
- **Explicit rule:** Detector confidence (`config.CONFIDENCE_THRESHOLD`) must NOT gate printing. It is only used to decide if a plate exists in the image.

**Delivery process (separate from API/watcher)**
- `api/app.py` and `api/watcher.py` are already separate processes; the delivery adapter is a third process.
- SQLite must be opened in WAL mode (`PRAGMA journal_mode=WAL`) so the writer and reader do not block each other.
- Retry with exponential backoff on `failed` rows; after N retries, log and leave `failed` for manual review. TODO: decide max retries and backoff ceiling.
- Printer-offline behavior: rows stay `pending`; delivery resumes automatically when connectivity restores. No data loss.

**Adapter interface**
- Adapters are pluggable: webhook / REST API call / JSON file drop.
- TODO: verify which protocol the target legacy printer system can actually consume. Do not implement all three speculatively; ask the operator first.
- Active adapter is selected via config (not hardcoded in `watcher.py` or `app.py`).

---

## (b) Human-in-the-Loop Correction -> Automated OCR Retraining

### Trigger
Any task that stores operator corrections, labeled image samples, retraining orchestration, or model versioning.

### Scope
- This pipeline retrains the **OCR recognizer** (`api/models/fast-plate/best.onnx`) only.
- The **detector** (`api/models/best.onnx`) requires bounding-box labels; that is a separate labeling and training flow not covered here.

### Correction record schema

| Field | Notes |
|---|---|
| `image_path` | Path to the cropped plate image |
| `predicted_text` | What the model read |
| `corrected_text` | Operator's correction |
| `model_version` | Version string of the model that made the prediction |
| `confidence` | Per-character or overall confidence at prediction time |
| `timestamp` | When the correction was submitted |
| `operator_id` | Who submitted the correction. TODO: define auth/identity scheme. |

**Input validation**
- Validate `corrected_text` against expected plate format before storing (regex or format check).
- Reject corrections with invalid characters or implausible length to prevent typos poisoning training data.

### Retraining trigger and promotion gate

- 100 (owner-specified) confirmed corrections produce a **CANDIDATE model**, not a deployment.
- The held-out validation set must never include images the model was trained on, and must never include the new corrections used for fine-tuning.
- Mix new correction samples with a sample of original training data to avoid catastrophic forgetting. TODO: decide mixing ratio.
- Promote the candidate ONNX only if validation accuracy is strictly better than the current production model.
- Keep the previous `api/models/fast-plate/best.onnx` as a named rollback (e.g. `best_vN-1.onnx`) until the new one is verified in staging.
- Rollback procedure: copy previous file back and restart the inference process. No code change required.

### Fine-tuning entry point
- Fine-tune from the current `best.keras` checkpoint using the Colab runbook (`colab_train_fast_plate_ocr_v3.md`).
- Export a versioned ONNX and run `api/test_pipeline.py` before promoting to production.

### Offline / distributed sites
- Sites without reliable connectivity queue corrections locally.
- A sync process uploads the queue to a central system that holds the labeled image store.
- Central system runs the promotion gate; it pushes the new ONNX back to edge sites.
- TODO: verify network topology and whether edge sites can reach the central system.

### Privacy / plate retention
- TODO: decide retention policy for plate images stored as correction samples.
- Consider whether storing plate images falls under local data-protection regulations.
- Until policy is decided: do not store raw plate images outside the existing `api/output/` directory already in use.
