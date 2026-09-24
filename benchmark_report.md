# Pre-trained `fast-plate-ocr` Benchmark Report

**Model:** `cct-s-v2-global-model` (Pre-trained)
**Dataset:** Validation Set (1012 images)

| Preprocessing | Accuracy | Avg Latency (CPU) |
|---|---|---|
| **Raw Crops** | 77.47% | 77.10 ms |
| **CLAHE Crops** | 77.17% | 72.31 ms |

**Conclusion on Preprocessing:**
Raw preprocessing yields better accuracy with the pre-trained model. We will proceed with this preprocessing strategy for fine-tuning.
