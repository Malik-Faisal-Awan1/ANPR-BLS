import cv2
import glob
import os
import re
import numpy as np
from datetime import datetime
from collections import defaultdict
import onnxruntime as ort
import Levenshtein

# Ensure we can import from api
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../api')))
from core.detector import PlateDetector
from core.ocr_engine import TextExtractor
import config

detector = PlateDetector()
ocr = TextExtractor()

def get_conf(batch):
    out = ocr.session.run(None, {ocr.input_name: batch})[0]
    p = out[0]
    pred = np.argmax(p, axis=-1)
    mask = pred != 36
    conf = np.min(np.max(p, axis=-1)[mask]) if np.any(mask) else 0.0
    text = ''.join('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'[idx] for idx in pred).rstrip('_')
    return text, conf

images = glob.glob(os.path.join(os.path.dirname(__file__), '../api/data/images/*.jpg'))
labeled = []
uncertain = []

for img in images:
    name = os.path.splitext(os.path.basename(img))[0]
    # Labeled: filename (minus extension and any _N/-N duplicate suffix) = true plate, uppercase.
    # We remove _1 or -1
    m = re.match(r'^([A-Z0-9]+)(?:[_-]\d+)?$', name)
    if m:
        labeled.append((img, m.group(1)))
    else:
        uncertain.append((img, name))

def evaluate(expand):
    correct = 0
    wrong = []
    char_errors = 0
    no_plate = 0
    correct_confs = []
    wrong_confs = []
    
    for img_path, truth in labeled:
        frame = cv2.imread(img_path)
        dets = detector.detect(frame)
        if not dets:
            no_plate += 1
            char_errors += len(truth)
            continue
        det = dets[0]
        crop = PlateDetector.crop_detection(frame, det['bbox'], expand=expand)
        if crop.size == 0:
            no_plate += 1
            char_errors += len(truth)
            continue
            
        batch = ocr._preprocess(crop)
        text, conf = get_conf(batch)
        
        if text == truth:
            correct += 1
            correct_confs.append(conf)
        else:
            wrong.append({'truth': truth, 'read': text, 'conf': conf})
            char_errors += Levenshtein.distance(truth, text)
            wrong_confs.append(conf)
            
    return {
        'expand': expand,
        'correct': correct,
        'no_plate': no_plate,
        'wrong': wrong,
        'char_errors': char_errors,
        'total': len(labeled),
        'correct_confs': correct_confs,
        'wrong_confs': wrong_confs
    }

res_on = evaluate(True)
res_off = evaluate(False)

# Evaluate uncertain set with expansion ON
unc_results = []
for img_path, name in uncertain:
    frame = cv2.imread(img_path)
    dets = detector.detect(frame)
    if not dets:
        unc_results.append({'name': name, 'text': 'NONE', 'conf': 0.0})
        continue
    det = dets[0]
    crop = PlateDetector.crop_detection(frame, det['bbox'], expand=True)
    batch = ocr._preprocess(crop)
    text, conf = get_conf(batch)
    unc_results.append({'name': name, 'text': text, 'conf': conf})

# Threshold table for expansion ON
thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99]
table = []
all_confs = res_on['correct_confs'] + res_on['wrong_confs']
for t in thresholds:
    auto_printed = sum(1 for c in all_confs if c >= t)
    coverage = auto_printed / res_on['total'] * 100 if res_on['total'] > 0 else 0
    wrong_printed = sum(1 for w in res_on['wrong'] if w['conf'] >= t)
    precision = (auto_printed - wrong_printed) / auto_printed * 100 if auto_printed > 0 else 0
    table.append({'t': t, 'cov': coverage, 'wrong': wrong_printed, 'prec': precision, 'auto': auto_printed})

# Recommendation
recommended_t = None
for row in reversed(table):
    if row['wrong'] == 0:
        recommended_t = row['t']
    else:
        break

# Confident misreads >= 0.9
confident_misreads = [w for w in res_on['wrong'] if w['conf'] >= 0.9]

# Generate Markdown
md = []
md.append(f"# Evaluation Report ({datetime.now().strftime('%Y-%m-%d')})\n")
md.append("## Dataset Setup")
md.append(f"- Labeled images: {len(labeled)} (Distinct plates: {len(set(t for _, t in labeled))})")
md.append(f"- Uncertain images: {len(uncertain)}")

for res in [res_on, res_off]:
    md.append(f"\n## 1. Exact-match Accuracy (Expansion {'ON' if res['expand'] else 'OFF'})")
    md.append(f"- Total labeled: {res['total']}")
    md.append(f"- Correct: {res['correct']}")
    md.append(f"- No plate detected: {res['no_plate']}")
    md.append(f"- Character-level errors: {res['char_errors']}")
    md.append("\n**Wrong Reads:**")
    if res['wrong']:
        for w in res['wrong']:
            md.append(f"  - `{w['truth']}` -> `{w['read']}` (conf: {w['conf']:.4f})")
    else:
        md.append("  - None")
        
    md.append(f"\n## 2. OCR Confidence Distribution (Expansion {'ON' if res['expand'] else 'OFF'})")
    def get_stats(confs):
        if not confs: return 'N/A'
        return f"Min: {np.min(confs):.4f}, Median: {np.median(confs):.4f}, Max: {np.max(confs):.4f}"
    md.append(f"- Correct reads ({len(res['correct_confs'])}): {get_stats(res['correct_confs'])}")
    md.append(f"- Wrong reads ({len(res['wrong_confs'])}): {get_stats(res['wrong_confs'])}")

md.append("\n## 3. Threshold Table (Expansion ON)")
md.append("| Threshold | Coverage (%) | Auto-Printed | Wrong Auto-Printed | Precision (%) |")
md.append("|---|---|---|---|---|")
for r in table:
    md.append(f"| {r['t']:.2f} | {r['cov']:.1f}% | {r['auto']} | {r['wrong']} | {r['prec']:.1f}% |")

md.append(f"\n## 4. Uncertain Set (Expansion ON)")
md.append("| Image | Read | Confidence |")
md.append("|---|---|---|")
unc_auto = sum(1 for u in unc_results if recommended_t is not None and u['conf'] >= recommended_t)
for u in unc_results:
    md.append(f"| {u['name']} | `{u['text']}` | {u['conf']:.4f} |")
md.append(f"\n**Note:** {unc_auto} out of {len(uncertain)} uncertain images would auto-print at the recommended threshold.")

md.append("\n## 5. Confident Misreads (conf >= 0.9, Expansion ON)")
if confident_misreads:
    for w in confident_misreads:
        md.append(f"- `{w['truth']}` -> `{w['read']}` (conf: {w['conf']:.4f})")
else:
    md.append("- None")

md.append(f"\n## Recommendation")
if recommended_t is not None:
    rec_row = next(r for r in table if r['t'] == recommended_t)
    near_misses = sum(1 for w in res_on['wrong'] if w['conf'] >= recommended_t - 0.05 and w['conf'] < recommended_t)
    md.append(f"**Recommended Threshold:** {recommended_t:.2f}")
    md.append(f"- Coverage: {rec_row['cov']:.1f}%")
    md.append(f"- Zero wrong auto-prints on the labeled set.")
    md.append(f"- There were {near_misses} wrong reads sitting within 0.05 below this threshold.")
else:
    md.append("No safe threshold found that yields zero wrong auto-prints.")
md.append(f"\n*Note: This recommendation is provisional based on {res_on['total']} samples.*")

with open(os.path.join(os.path.dirname(__file__), '../docs/EVAL_2026-09-29.md'), 'w') as f:
    f.write('\n'.join(md))
print('Evaluation complete. Results written to docs/EVAL_2026-09-29.md')
