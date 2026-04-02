"""
test.py — Evaluate overall accuracy of pred.py on the full dataset.
pred.py is a pure inference script (no train/test split needed).
"""
import sys
import os
import time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pred import predict_all

DATA_PATH = "new_data2.csv"
TMP_CSV   = "_test_input.csv"

# Load full dataset and strip the label column for inference
data  = pd.read_csv(DATA_PATH)
y     = data["painting"].reset_index(drop=True)
data.drop(columns=["painting"]).to_csv(TMP_CSV, index=False)

# Run inference
print("Running pred.py on full dataset...")
t0    = time.time()
preds = predict_all(TMP_CSV)
elapsed = time.time() - t0

# Compute accuracy
correct = sum(p == t for p, t in zip(preds, y))
total   = len(y)

print(f"\nOverall Accuracy : {correct}/{total} = {correct / total:.4f}  ({elapsed:.2f}s)")

# Cleanup
os.remove(TMP_CSV)
