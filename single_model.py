"""
single_model.py

Trains a single GradientBoostingClassifier on the full feature set
(structured + TF-IDF text), using the same 70/15/15 train/val/test split
and preprocessing pipeline as train_and_save.py.

Reports train, val, and test accuracies.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split as _tts

# ── Hyperparameters ───────────────────────────────────────────────────────────
RANDOM_STATE  = 42
N_ESTIMATORS  = 100
LEARNING_RATE = 0.1
MAX_DEPTH     = 5

# ── Load data ─────────────────────────────────────────────────────────────────
data = pd.read_csv("new_data2.csv")

# ── 70 / 15 / 15  train / val / test split ───────────────────────────────────
train_df, _tmp_df = _tts(data, test_size=0.30, random_state=RANDOM_STATE)
val_df,   test_df = _tts(_tmp_df, test_size=0.50, random_state=RANDOM_STATE)
print(f"Split sizes  → train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")

# ── Multi-label encode ────────────────────────────────────────────────────────
def fit_multilabel(df, col):
    mlb = MultiLabelBinarizer()
    split = df[col].str.split(",").apply(lambda x: [s.strip() for s in x])
    mlb.fit(split)
    return mlb

def apply_multilabel(mlb, df, col):
    split = df[col].str.split(",").apply(lambda x: [s.strip() for s in x])
    return mlb.transform(split)

mlb_room   = fit_multilabel(train_df, "room")
mlb_season = fit_multilabel(train_df, "season")
mlb_view   = fit_multilabel(train_df, "view_with")

# ── Structured features ───────────────────────────────────────────────────────
NUMERICAL_COLS = [
    "emotion_intensity", "feel_sombre", "feel_content", "feel_calm",
    "feel_uneasy", "prominent_colours", "objects_noticed", "willingness_to_pay",
]

X_num_tr   = train_df[NUMERICAL_COLS].values.astype(float)
struct_mean = X_num_tr.mean(0)
struct_std  = X_num_tr.std(0)
struct_std[struct_std == 0] = 1.0

def build_structured(df):
    X_num      = df[NUMERICAL_COLS].values.astype(float)
    X_num_norm = (X_num - struct_mean) / struct_std
    room_enc   = apply_multilabel(mlb_room,   df, "room")
    season_enc = apply_multilabel(mlb_season, df, "season")
    view_enc   = apply_multilabel(mlb_view,   df, "view_with")
    return np.hstack([X_num_norm, room_enc, season_enc, view_enc]).astype(float)

X_struct_tr = build_structured(train_df)
X_struct_va = build_structured(val_df)
X_struct_te = build_structured(test_df)

# ── TF-IDF text features ──────────────────────────────────────────────────────
TEXT_COLS = ["feeling_description", "food_association", "soundtrack"]
for col in TEXT_COLS:
    data[col] = data[col].fillna("")

tfidf_vecs = {}
for col in TEXT_COLS:
    vec = TfidfVectorizer(
        max_features=100, sublinear_tf=True,
        strip_accents="unicode", ngram_range=(1, 2),
    )
    vec.fit(train_df[col].fillna(""))
    tfidf_vecs[col] = vec

def build_text(df):
    return np.hstack([
        tfidf_vecs[col].transform(df[col].fillna("")).toarray()
        for col in TEXT_COLS
    ])

X_text_tr = build_text(train_df)
X_text_va = build_text(val_df)
X_text_te = build_text(test_df)

# ── Combine all features ──────────────────────────────────────────────────────
X_train = np.hstack([X_struct_tr, X_text_tr])
X_val   = np.hstack([X_struct_va, X_text_va])
X_test  = np.hstack([X_struct_te, X_text_te])

print(f"Feature dimensionality → {X_train.shape[1]} "
      f"({X_struct_tr.shape[1]} structured + {X_text_tr.shape[1]} TF-IDF)")

# ── Labels ────────────────────────────────────────────────────────────────────
y_train = train_df["painting"].values
y_val   = val_df["painting"].values
y_test  = test_df["painting"].values

# ── Train single model ────────────────────────────────────────────────────────
print("\nTraining single GradientBoostingClassifier on full feature set...")
model = GradientBoostingClassifier(
    n_estimators=N_ESTIMATORS,
    learning_rate=LEARNING_RATE,
    max_depth=MAX_DEPTH,
    random_state=RANDOM_STATE,
)
model.fit(X_train, y_train)

# ── Report accuracies ─────────────────────────────────────────────────────────
train_acc = accuracy_score(y_train, model.predict(X_train))
val_acc   = accuracy_score(y_val,   model.predict(X_val))
test_acc  = accuracy_score(y_test,  model.predict(X_test))

print(f"\n{'='*45}")
print(f"  Train accuracy : {train_acc:.4f}  ({train_acc*100:.2f}%)")
print(f"  Val   accuracy : {val_acc:.4f}  ({val_acc*100:.2f}%)")
print(f"  Test  accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")
print(f"{'='*45}")
