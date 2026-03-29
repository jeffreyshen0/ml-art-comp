"""
evaluate.py — Run pred.py on the held-out test split and report metrics.
Uses the same 70/15/15 split (random_state=42) as all other experiments.
"""
import sys
import os
import numpy as np
import pandas as pd

# ── Import predict_all from pred.py ───────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pred import predict_all

# ── Build the held-out test CSV (same split as training) ─────────────────────
from sklearn.model_selection import train_test_split

DATA_PATH = "new_data2.csv"
data = pd.read_csv(DATA_PATH)

_, tmp = train_test_split(data, test_size=0.30, random_state=42)
_, test_df = train_test_split(tmp, test_size=0.50, random_state=42)

# Save test set (without 'painting' label) to a temp CSV for predict_all
TEST_CSV    = "_eval_test_input.csv"
ANSWERS_CSV = "_eval_test_answers.csv"

test_df.drop(columns=["painting"]).to_csv(TEST_CSV, index=False)
y_true = test_df["painting"].reset_index(drop=True)

# ── Run inference ─────────────────────────────────────────────────────────────
print("Running predict_all on test set...")
import time
t0 = time.time()
preds = predict_all(TEST_CSV)
elapsed = time.time() - t0

y_pred = pd.Series(preds, name="predicted")
painting_classes = sorted(y_true.unique())

# ── Accuracy ──────────────────────────────────────────────────────────────────
correct = sum(p == t for p, t in zip(preds, y_true))
total   = len(y_true)
accuracy = correct / total

# ── Per-class breakdown ───────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  Test set evaluation  ({total} examples, {elapsed:.3f}s)")
print(f"{'='*55}")
print(f"  Overall Accuracy : {accuracy:.4f}  ({correct}/{total} correct)")
print(f"{'='*55}")

print(f"\n{'Class':<30} {'Correct':>8} {'Total':>8} {'Acc':>8}")
print("-" * 55)
for cls in painting_classes:
    mask     = y_true == cls
    cls_preds = y_pred[mask.values]
    cls_true  = y_true[mask]
    n_total   = mask.sum()
    n_correct = (cls_preds.values == cls_true.values).sum()
    print(f"{cls:<30} {n_correct:>8} {n_total:>8} {n_correct/n_total:>8.4f}")

# ── Confusion matrix ──────────────────────────────────────────────────────────
print(f"\n{'Confusion Matrix (rows=true, cols=pred)':}")
header = f"{'':30}" + "".join(f"{c[-8:]:>10}" for c in painting_classes)
print(header)
print("-" * (30 + 10 * len(painting_classes)))
for true_cls in painting_classes:
    row = f"{true_cls:<30}"
    for pred_cls in painting_classes:
        count = sum(
            1 for p, t in zip(preds, y_true) if t == true_cls and p == pred_cls
        )
        row += f"{count:>10}"
    print(row)

# ── Most common errors ────────────────────────────────────────────────────────
errors = [(t, p) for t, p in zip(y_true, preds) if t != p]
if errors:
    print(f"\n  Misclassifications ({len(errors)} total):")
    from collections import Counter
    for (true_lbl, pred_lbl), cnt in Counter(errors).most_common():
        print(f"    True: {true_lbl:<32} Pred: {pred_lbl:<32} x{cnt}")
else:
    print("\n  No misclassifications!")

print(f"\n{'='*55}")

# ── Cleanup ───────────────────────────────────────────────────────────────────
os.remove(TEST_CSV)
