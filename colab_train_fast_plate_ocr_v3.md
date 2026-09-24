# Colab Runbook v3 — fast-plate-ocr Fine-Tuning (Pre-trained CCT-S-v2)
**Target: Google Colab T4 GPU, Python 3.10+**

> **Before you start:**
> 1. Select **Runtime → Change runtime type → T4 GPU**
> 2. Ensure your `train_data` folder on Google Drive has `images/`, `rec_gt_train.txt`, and `rec_gt_val.txt`.

---

## Phase 1: Installation & Data Setup

### Cell 1 — Mount Google Drive & Install Dependencies
```bash
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

```bash
%%bash
# Pin protobuf<6 to avoid warnings with Colab pre-installed libraries
pip install -q "protobuf<6.0.0dev"
pip install -q "fast-plate-ocr[train,onnx-gpu]==1.1.0"
```

---

### Cell 2 — Dataset Preparation (Fast Local SSD & CSV Conversion)
> **Why copy to `/content/train_data`?**  
> Google Drive FUSE mount is notoriously slow for thousands of small image files during training epochs. Copying them once to Colab's local NVMe SSD increases training speed by **10x–20x**.

```python
import os
import csv
import shutil

DRIVE_DATA_DIR = "/content/drive/MyDrive/train_data"
LOCAL_DATA_DIR = "/content/train_data"

if not os.path.exists(DRIVE_DATA_DIR):
    raise FileNotFoundError(f"⚠️ Could not find {DRIVE_DATA_DIR}. Ensure Google Drive is mounted correctly.")

# Copy dataset to local Colab SSD for maximum I/O performance
if not os.path.exists(LOCAL_DATA_DIR):
    print("Copying dataset to local Colab SSD for high-speed training...")
    shutil.copytree(DRIVE_DATA_DIR, LOCAL_DATA_DIR)
    print("Dataset copied successfully!")

DATA_DIR = LOCAL_DATA_DIR if os.path.exists(LOCAL_DATA_DIR) else DRIVE_DATA_DIR

def convert_tsv_to_csv(tsv_rel_path, csv_out_path):
    tsv_path = os.path.join(DATA_DIR, tsv_rel_path)
    count = 0
    with open(tsv_path, 'r', encoding='utf-8') as fin, \
         open(csv_out_path, 'w', newline='', encoding='utf-8') as fout:
        writer = csv.writer(fout)
        writer.writerow(['image_path', 'plate_text'])
        for line in fin:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                img_path, label = parts
                # Point to raw images (benchmark proved raw beats CLAHE)
                img_path = img_path.replace('crops_clahe/', 'images/')
                writer.writerow([img_path, label])
                count += 1
    print(f"Generated {csv_out_path} ({count} samples)")

# CSVs are stored inside DATA_DIR so image paths ('images/...') resolve cleanly
convert_tsv_to_csv('rec_gt_train.txt', os.path.join(DATA_DIR, 'train_annotations.csv'))
convert_tsv_to_csv('rec_gt_val.txt', os.path.join(DATA_DIR, 'val_annotations.csv'))
print("==> Annotations generated OK!")
```

---

## Phase 2: Official Pre-trained Configs & Fine-Tuning

### Cell 3 — Download Official CCT-S-v2 Architecture, Plate Config & Weights
We download the exact official configuration files and pre-trained weights from the `fast-plate-ocr` release assets:
- `cct_s_v2_global_model_config.yaml` (Compact Convolutional Transformer architecture)
- `cct_s_v2_global_plate_config.yaml` (Valid alphabet `'0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'`, `pad_char: '_'`, 10 slots)
- `cct_s_v2_global.keras` (Pre-trained weights checkpoint for transfer learning)

```bash
%%bash
wget -q -O /content/model_config.yaml \
  https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/cct_s_v2_global_model_config.yaml

wget -q -O /content/plate_config.yaml \
  https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/cct_s_v2_global_plate_config.yaml

wget -q -O /content/cct_s_v2_global.keras \
  https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/cct_s_v2_global.keras

echo "==> Downloaded model_config.yaml, plate_config.yaml, and cct_s_v2_global.keras!"
```

#### Sanity Check (Run to verify configs):
```python
from fast_plate_ocr.train.model.config import load_plate_config_from_yaml
from fast_plate_ocr.train.model.model_schema import load_model_config_from_yaml

plate_cfg = load_plate_config_from_yaml('/content/plate_config.yaml')
model_cfg = load_model_config_from_yaml('/content/model_config.yaml')

print(f"✅ PlateConfig Validated: alphabet_size={plate_cfg.vocabulary_size}, pad_char={repr(plate_cfg.pad_char)}, slots={plate_cfg.max_plate_slots}, shape=({plate_cfg.img_height}, {plate_cfg.img_width})")
print(f"✅ ModelConfig Validated: model={model_cfg.model}")
```

---

### Cell 4 — Launch Fine-Tuning (Live Real-Time Streaming Output)
This runs the fine-tuning with PyTorch backend and streams output continuously with live progress:

```python
import os
import sys
import subprocess

# Ensure PyTorch backend and disable Python stdout buffering
os.environ["KERAS_BACKEND"] = "torch"
os.environ["PYTHONUNBUFFERED"] = "1"

# Dataset paths
annotations = "/content/train_data/train_annotations.csv"
val_annotations = "/content/train_data/val_annotations.csv"

if not os.path.exists(annotations):
    print("⚠️ Local /content/train_data not found, falling back to Google Drive...")
    annotations = "/content/drive/MyDrive/train_data/train_annotations.csv"
    val_annotations = "/content/drive/MyDrive/train_data/val_annotations.csv"
    print("💡 TIP: Copying to /content/train_data first makes training 15x faster!")
else:
    print(f"✅ Using local dataset at /content/train_data")

cmd = [
    "fast-plate-ocr", "train",
    "--model-config-file", "/content/model_config.yaml",
    "--plate-config-file", "/content/plate_config.yaml",
    "--annotations", annotations,
    "--val-annotations", val_annotations,
    "--weights-path", "/content/cct_s_v2_global.keras",
    "--epochs", "80",
    "--batch-size", "64",
    "--lr", "0.0001",
    "--warmup-fraction", "0.05",
    "--final-lr-factor", "0.01",
    "--weight-decay", "0.01",
    "--output-dir", "/content/output"
]

print("🚀 Starting training... Live terminal output below:\n" + "="*60)

# Use pty (pseudo-terminal) on Linux/Colab for live interactive progress bar streaming
try:
    import pty
    master, slave = pty.openpty()
    proc = subprocess.Popen(
        cmd,
        stdin=slave,
        stdout=slave,
        stderr=slave,
        close_fds=True
    )
    os.close(slave)
    
    while True:
        try:
            data = os.read(master, 1024)
            if not data:
                break
            sys.stdout.write(data.decode("utf-8", errors="replace"))
            sys.stdout.flush()
        except OSError:
            break
            
    os.close(master)
    proc.wait()
except (ImportError, AttributeError):
    # Fallback for non-pty environments
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in iter(proc.stdout.readline, ''):
        print(line, end='', flush=True)
    proc.wait()

if proc.returncode == 0:
    print("\n" + "="*60 + "\n🎉 Training finished successfully!")
else:
    print(f"\n⚠️ Process exited with code {proc.returncode}")
```

*(Alternative 1-line bash equivalent using `!` instead of `%%bash`)*:
```bash
!export KERAS_BACKEND=torch && PYTHONUNBUFFERED=1 fast-plate-ocr train --model-config-file /content/model_config.yaml --plate-config-file /content/plate_config.yaml --annotations /content/train_data/train_annotations.csv --val-annotations /content/train_data/val_annotations.csv --weights-path /content/cct_s_v2_global.keras --epochs 80 --batch-size 64 --lr 0.0001 --warmup-fraction 0.05 --final-lr-factor 0.01 --weight-decay 0.01 --output-dir /content/output
```


---

## Phase 3: Export to ONNX

### Cell 5 — Export Trained Model to ONNX
After training completes, find the best checkpoint and export it directly to ONNX:

```python
import os
import glob
import pathlib

# Ensure PyTorch backend
os.environ["KERAS_BACKEND"] = "torch"

import keras
import fast_plate_ocr.train.model.layers as layers
from fast_plate_ocr.train.model.config import load_plate_config_from_yaml
from fast_plate_ocr.train.utilities.utils import load_keras_model
from fast_plate_ocr.cli.export import export_onnx

# 1. Patch the data-dependent slice in PositionEmbedding to make it compatible with PyTorch 2.6's ONNX exporter
layers.PositionEmbedding.call = lambda self, inputs, start_index=0: keras.ops.broadcast_to(
    keras.ops.convert_to_tensor(self.position_embeddings), keras.ops.shape(inputs)
)

# 2. Locate the best checkpoint produced by training
keras_files = sorted(glob.glob("/content/output/**/best.keras", recursive=True))
if not keras_files:
    raise FileNotFoundError("Could not find best.keras in /content/output")
model_path = pathlib.Path(keras_files[-1])
print(f"Loading checkpoint: {model_path}")

plate_config = load_plate_config_from_yaml("/content/plate_config.yaml")
model = load_keras_model(model_path, plate_config)

out_file = pathlib.Path("/content/best.onnx")

print("Exporting model to ONNX...")
export_onnx(
    model=model,
    plate_config=plate_config,
    out_file=out_file,
    simplify=True,
    dynamic_batch=True,
    skip_validation=False,
    onnx_input_dtype="uint8",
    onnx_data_format="channels_last",
)

if out_file.exists():
    print(f"\n🎉 SUCCESS! Exported model saved at: {out_file} ({out_file.stat().st_size / (1024*1024):.2f} MB)")
else:
    print("\n⚠️ Export failed: file not created.")
```

### Next Steps:
1. Download `/content/best.onnx` from Colab (right-click in Colab's file browser → Download).
2. Rename or place it in your local project folder: `models/custom_ocr/best.onnx`.
3. Update `config.py` if needed to point to your new fine-tuned model weights!

