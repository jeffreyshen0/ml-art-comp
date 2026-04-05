"""
test.py — Evaluate pred.py on the held-out test split (15% of new_data2.csv).
Uses the same 70/15/15 split (random_state=42) as train_and_save.py so the
reported accuracy is a clean, unbiased estimate of generalisation performance.
"""
import sys
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pred import predict_all

DATA_PATH = "new_data2.csv"
TMP_CSV   = "_test_input.csv"

# ── Reproduce the same 70/15/15 split used in train_and_save.py ──────────────
data = pd.read_csv(DATA_PATH)
_, _tmp    = train_test_split(data, test_size=0.30, random_state=42)
_, test_df = train_test_split(_tmp, test_size=0.50, random_state=42)   # 15% of total

y = test_df["painting"].reset_index(drop=True)
test_df.drop(columns=["painting"]).to_csv(TMP_CSV, index=False)

# ── Run inference ─────────────────────────────────────────────────────────────
print(f"Running pred.py on held-out test set ({len(y)} examples)...")
t0    = time.time()
preds = predict_all(TMP_CSV)
elapsed = time.time() - t0

# ── Accuracy ──────────────────────────────────────────────────────────────────
correct = sum(p == t for p, t in zip(preds, y))
total   = len(y)

print(f"\nTest Accuracy : {correct}/{total} = {correct / total:.4f}  ({elapsed:.2f}s)")

# ── Cleanup ───────────────────────────────────────────────────────────────────
os.remove(TMP_CSV)
