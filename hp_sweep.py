"""
Hyperparameter sweep for the structured + text ensemble.

Sweeps over all combinations of:
  n_estimators  ∈ {50, 100, 200}
  learning_rate ∈ {0.05, 0.1, 0.2}
  max_depth     ∈ {1, 3, 5}
  w_struct      ∈ {0.3, 0.5, 0.7}   (ensemble weight for structured model;
                                       text weight = 1 − w_struct)

Uses the same feature pipeline as train_and_save.py (numerical + multi-label
categorical + TF-IDF text features) and the same 70/15/15 split.
"""

import itertools
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# ── Data ──────────────────────────────────────────────────────────────────────
data = pd.read_csv("new_data2.csv")
RANDOM_STATE = 42

# ── 70 / 15 / 15  train / val / test split ───────────────────────────────────
train_df, _tmp_df = train_test_split(data, test_size=0.30, random_state=RANDOM_STATE)
val_df,   test_df = train_test_split(_tmp_df, test_size=0.50, random_state=RANDOM_STATE)
print(f"Split sizes  → train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")

# ── Multi-label encode (fit on train only) ────────────────────────────────────
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

NUMERICAL_COLS = [
    "emotion_intensity", "feel_sombre", "feel_content", "feel_calm",
    "feel_uneasy", "prominent_colours", "objects_noticed", "willingness_to_pay",
]

X_num_tr = train_df[NUMERICAL_COLS].values.astype(float)
struct_mean = X_num_tr.mean(0)
struct_std  = X_num_tr.std(0)
struct_std[struct_std == 0] = 1.0

def build_structured(df):
    X_num = df[NUMERICAL_COLS].values.astype(float)
    X_num_norm = (X_num - struct_mean) / struct_std
    room_enc   = apply_multilabel(mlb_room,   df, "room")
    season_enc = apply_multilabel(mlb_season, df, "season")
    view_enc   = apply_multilabel(mlb_view,   df, "view_with")
    return np.hstack([X_num_norm, room_enc, season_enc, view_enc]).astype(float)

X_struct_tr = build_structured(train_df)
X_struct_va = build_structured(val_df)

# ── TF-IDF text features (fit on train only) ─────────────────────────────────
TEXT_COLS = ["feeling_description", "food_association", "soundtrack"]
for col in TEXT_COLS:
    data[col] = data[col].fillna("")

tfidf_vecs = {}
for col in TEXT_COLS:
    vec = TfidfVectorizer(max_features=100, sublinear_tf=True,
                          strip_accents="unicode", ngram_range=(1, 2))
    vec.fit(train_df[col].fillna(""))
    tfidf_vecs[col] = vec

def build_text(df):
    return np.hstack([
        tfidf_vecs[col].transform(df[col].fillna("")).toarray()
        for col in TEXT_COLS
    ])

X_text_tr = build_text(train_df)
X_text_va = build_text(val_df)

# ── Labels ────────────────────────────────────────────────────────────────────
y_train = train_df["painting"].values
y_val   = val_df["painting"].values

# ── Hyperparameter grid ───────────────────────────────────────────────────────
N_ESTIMATORS_LIST  = [50, 100, 200]
LEARNING_RATE_LIST = [0.05, 0.1, 0.2]
MAX_DEPTH_LIST     = [1, 3, 5]
W_STRUCT_LIST      = [0.3, 0.5, 0.7]   # weight for structured model probs

grid = list(itertools.product(
    N_ESTIMATORS_LIST, LEARNING_RATE_LIST, MAX_DEPTH_LIST, W_STRUCT_LIST
))
print(f"\nTotal configurations: {len(grid)}\n")

# ── Sweep ─────────────────────────────────────────────────────────────────────
header = f"{'n_est':>5}  {'lr':>5}  {'depth':>5}  {'w_str':>5}  {'Train':>7}  {'Valid':>7}"
print(header)
print("-" * len(header))

results = []

for n_est, lr, depth, w_struct in grid:
    # Train structured model
    m_struct = GradientBoostingClassifier(
        n_estimators=n_est, learning_rate=lr,
        max_depth=depth, random_state=RANDOM_STATE,
    )
    m_struct.fit(X_struct_tr, y_train)

    # Train text model (same GBM hyper-parameters)
    m_text = GradientBoostingClassifier(
        n_estimators=n_est, learning_rate=lr,
        max_depth=depth, random_state=RANDOM_STATE,
    )
    m_text.fit(X_text_tr, y_train)

    # Weighted ensemble prediction
    w_text = 1.0 - w_struct
    def ensemble_predict(Xs, Xt):
        sp = m_struct.predict_proba(Xs)
        tp = m_text.predict_proba(Xt)
        combined = w_struct * sp + w_text * tp
        return m_struct.classes_[combined.argmax(axis=1)]

    train_acc = accuracy_score(y_train, ensemble_predict(X_struct_tr, X_text_tr))
    val_acc   = accuracy_score(y_val,   ensemble_predict(X_struct_va, X_text_va))

    print(f"{n_est:>5}  {lr:>5}  {depth:>5}  {w_struct:>5}  {train_acc:>7.4f}  {val_acc:>7.4f}")
    results.append((n_est, lr, depth, w_struct, train_acc, val_acc))

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
best = max(results, key=lambda r: r[5])  # best by validation accuracy
print(f"Best config (by val acc):")
print(f"  n_estimators  = {best[0]}")
print(f"  learning_rate = {best[1]}")
print(f"  max_depth     = {best[2]}")
print(f"  w_struct      = {best[3]}")
print(f"  Train acc     = {best[4]:.4f}")
print(f"  Val   acc     = {best[5]:.4f}")
