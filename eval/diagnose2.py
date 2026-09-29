"""
Tasks 1, 3, 4, 5 from the diagnosis request.
Run from repo root: python eval/diagnose2.py
"""
import os, sys, glob, re, random, cv2
import numpy as np

os.environ["KERAS_BACKEND"] = "torch"

sys.path.insert(0, os.path.abspath("api"))
from core.detector import PlateDetector
from core.ocr_engine import TextExtractor

detector = PlateDetector()
ocr = TextExtractor()

CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"

def onnx_forward(crop):
    batch = ocr._preprocess(crop)
    return ocr.session.run(None, {ocr.input_name: batch})[0]  # [1,10,37]

# ---------- helpers ----------
def decode_logits(logits):
    return "".join(CHARSET[i] for i in np.argmax(logits[0], axis=-1)).rstrip("_")

def laplacian_sharpness(bgr):
    resized = cv2.resize(bgr, (128, 64))
    return cv2.Laplacian(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

# ---------- Task 1: Keras/ONNX parity ----------
print("\n=== 1. Keras/ONNX parity ===")
keras_path = "api/models/fast-plate/best.keras"
eval_images = glob.glob("api/data/images/*.jpg")
labeled_eval = [(img, re.match(r'^([A-Z0-9]+)(?:[_-]\d+)?$',
                 os.path.splitext(os.path.basename(img))[0]).group(1))
                for img in eval_images
                if re.match(r'^([A-Z0-9]+)(?:[_-]\d+)?$',
                   os.path.splitext(os.path.basename(img))[0])]

eval_crops = []
eval_truths = []
eval_onnx_reads = []
for img_path, truth in labeled_eval:
    frame = cv2.imread(img_path)
    dets = detector.detect(frame)
    if not dets:
        eval_crops.append(None); eval_truths.append(truth); eval_onnx_reads.append("NONE")
        continue
    crop = PlateDetector.crop_detection(frame, dets[0]["bbox"], expand=True)
    eval_crops.append(crop)
    eval_truths.append(truth)
    logits = onnx_forward(crop)
    eval_onnx_reads.append(decode_logits(logits))

try:
    from fast_plate_ocr.train.utilities.utils import load_keras_model
    from fast_plate_ocr.train.model.config import load_plate_config_from_yaml

    plate_cfg_path = "api/models/fast-plate/plate_config.yaml"
    if not os.path.exists(plate_cfg_path):
        import urllib.request
        urllib.request.urlretrieve(
            "https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/cct_s_v2_global_plate_config.yaml",
            plate_cfg_path)

    plate_cfg = load_plate_config_from_yaml(plate_cfg_path)
    k_model = load_keras_model(keras_path, plate_cfg)

    import torch
    differing = 0
    max_diff = 0.0
    for crop, truth, onnx_read in zip(eval_crops, eval_truths, eval_onnx_reads):
        if crop is None:
            continue
        batch_np = ocr._preprocess(crop)
        batch_t = torch.tensor(batch_np)
        out = k_model(batch_t)
        # The model may return a dict or a tensor
        if isinstance(out, dict):
            # find the plate key
            key = [k for k in out if "plate" in k.lower() or "output" in k.lower()][0]
            k_logits = out[key]
        else:
            k_logits = out
        if hasattr(k_logits, "detach"):
            k_logits = k_logits.detach().numpy()
        elif hasattr(k_logits, "numpy"):
            k_logits = k_logits.numpy()

        o_logits = onnx_forward(crop)
        d = float(np.max(np.abs(o_logits - k_logits)))
        max_diff = max(max_diff, d)

        k_read = decode_logits(k_logits)
        if k_read != onnx_read:
            differing += 1
            print(f"  DIFFER: truth={truth} keras={k_read} onnx={onnx_read}")

    print(f"Keras/ONNX parity: {differing} differing reads, max output diff = {max_diff:.6f}")

except Exception as e:
    print(f"Keras parity skipped: {e}")

# ---------- Task 3: Visual inspection ----------
print("\n=== 3. Visual inspection of wrong + uncertain crops ===")
wrong_names = ["AQM759","AWT763","BDG097","BTA605","BYJ117","LEE6509","LEU2700","LEU5094","LEU6463"]
uncertain_images = [(img, os.path.splitext(os.path.basename(img))[0])
                    for img in eval_images
                    if not re.match(r'^([A-Z0-9]+)(?:[_-]\d+)?$',
                       os.path.splitext(os.path.basename(img))[0])]

print("Wrong reads crops (from detected+expanded crop):")
wrong_aspects = []
for w in wrong_names:
    matches = [(c, t) for c, t in zip(eval_crops, eval_truths) if t == w]
    if not matches:
        print(f"  {w}: crop not found"); continue
    crop = matches[0][0]
    if crop is None:
        print(f"  {w}: no detection"); continue
    h, ww = crop.shape[:2]
    asp = ww / h
    wrong_aspects.append(asp)
    tag = "two-line/square" if asp < 1.5 else "single-line"
    read = eval_onnx_reads[[t for _, t in labeled_eval].index(w)] if w in [t for _, t in labeled_eval] else "?"
    print(f"  {w}: {ww}x{h} asp={asp:.2f} -> {tag} | ONNX read: {read}")
    cv2.imwrite(f"eval/inspect_{w}.jpg", crop)

print("\nUncertain crops:")
for img_path, name in uncertain_images:
    frame = cv2.imread(img_path)
    dets = detector.detect(frame)
    if not dets:
        print(f"  {name}: no detection")
        continue
    crop = PlateDetector.crop_detection(frame, dets[0]["bbox"], expand=True)
    h, ww = crop.shape[:2]
    asp = ww / h
    tag = "two-line/square" if asp < 1.5 else "single-line"
    logits = onnx_forward(crop)
    read = decode_logits(logits)
    print(f"  {name}: {ww}x{h} asp={asp:.2f} -> {tag} | ONNX read: {read}")
    cv2.imwrite(f"eval/inspect_unc_{name}.jpg", crop)

# Count two-line in training (sample 200 by aspect ratio of stored crops)
print("\nTraining crop aspect ratio sample (200):")
train_txt = "train_data/rec_gt_train.txt"
with open(train_txt, "r") as f:
    t_entries = [(l.strip().split("\t")) for l in f if len(l.strip().split("\t")) == 2]
random.seed(42)
sample = random.sample(t_entries, min(200, len(t_entries)))
two_line_count = 0
inspected = 0
for img_rel, label in sample:
    p = os.path.join("train_data", img_rel.replace("crops_clahe/", "images/"))
    img = cv2.imread(p)
    if img is None: continue
    h, w = img.shape[:2]
    asp = w / h
    if asp < 1.5:
        two_line_count += 1
    inspected += 1
print(f"  Inspected: {inspected}, two-line/square (asp<1.5): {two_line_count} ({100*two_line_count/inspected:.1f}%)")

# ---------- Task 4: Blur at 128x64 ----------
print("\n=== 4. Sharpness at 128x64 ===")
correct_crops = [(c, t) for c, t, r in zip(eval_crops, eval_truths, eval_onnx_reads) if t == r and c is not None]
wrong_crops   = [(c, t) for c, t, r in zip(eval_crops, eval_truths, eval_onnx_reads) if t != r and c is not None]

s_correct = [laplacian_sharpness(c) for c, _ in correct_crops]
s_wrong   = [laplacian_sharpness(c) for c, _ in wrong_crops]
print(f"  Correct (N={len(s_correct)}): median sharpness@128x64 = {np.median(s_correct):.1f}")
print(f"  Wrong   (N={len(s_wrong)}): median sharpness@128x64 = {np.median(s_wrong):.1f}")

# Train crops at 128x64
t_sharp = []
for img_rel, label in sample:
    p = os.path.join("train_data", img_rel.replace("crops_clahe/", "images/"))
    img = cv2.imread(p)
    if img is None: continue
    t_sharp.append(laplacian_sharpness(img))
print(f"  Train (N={len(t_sharp)}): median sharpness@128x64 = {np.median(t_sharp):.1f}")

# ---------- Task 5: Label audit ----------
print("\n=== 5. Label audit (100 random training crops) ===")
audit_sample = random.sample(t_entries, min(100, len(t_entries)))
mismatch = 0
audited = 0
blank = 0
for img_rel, label in audit_sample:
    p = os.path.join("train_data", img_rel.replace("crops_clahe/", "images/"))
    img = cv2.imread(p)
    if img is None: continue
    audited += 1
    if img.size == 0:
        blank += 1; continue
    # run ONNX on it
    logits = onnx_forward(img)
    pred = decode_logits(logits)
    # A mismatch is when the model prediction doesn't match the label at all
    # (since we can't humanly inspect all here, we flag cases where model is very
    # confident but label differs -- confidence metric)
    o_probs = logits[0]
    slot_mask = np.argmax(o_probs, axis=-1) != 36
    if slot_mask.any():
        conf = float(np.min(np.max(o_probs[slot_mask], axis=-1)))
    else:
        conf = 0.0
    if conf > 0.90 and pred != label:
        mismatch += 1
        print(f"  MISMATCH conf={conf:.3f}: label={label} pred={pred}  [{img_rel}]")
print(f"  High-conf label mismatch: {mismatch}/{audited} ({100*mismatch/audited:.1f}%)")
print("  (Note: low-conf disagreements not flagged; only conf>0.90 treated as likely label noise)")
