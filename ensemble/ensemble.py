"""
Feature-Partitioned Ensemble
────────────────────────────
Trains one model on structured features and a different model on text (BoW)
features, then combines their class probabilities via soft voting or stacking.

Feature groups
  • Structured : numerical ratings + multi-hot room/season/view_with
  • Text       : TF-IDF BoW on feeling_description, food_association, soundtrack
"""

import pandas as pd
import numpy as np
from itertools import product

from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

RANDOM_STATE = 42

# ── Load data ──────────────────────────────────────────────────────────────
data = pd.read_csv("../new_data2.csv")

# ── Multi-label categorical encoding ──────────────────────────────────────
def encode_multilabel(df, col):
    mlb = MultiLabelBinarizer()
    split = df[col].str.split(",").apply(lambda x: [s.strip() for s in x])
    return pd.DataFrame(mlb.fit_transform(split),
                        columns=[f"{col}_{c}" for c in mlb.classes_])

room_enc    = encode_multilabel(data, "room")
season_enc  = encode_multilabel(data, "season")
view_enc    = encode_multilabel(data, "view_with")

NUMERICAL_COLS = [
    "emotion_intensity", "feel_sombre", "feel_content",
    "feel_calm", "feel_uneasy", "prominent_colours",
    "objects_noticed", "willingness_to_pay",
]

X_structured = pd.concat([
    data[NUMERICAL_COLS].reset_index(drop=True),
    room_enc, season_enc, view_enc,
], axis=1)

# ── Text BoW ───────────────────────────────────────────────────────────────
TEXT_COLS = ["feeling_description", "food_association", "soundtrack"]
for col in TEXT_COLS:
    data[col] = data[col].fillna("")

text_frames = []
for col in TEXT_COLS:
    vec = TfidfVectorizer(max_features=100, sublinear_tf=True,
                          strip_accents="unicode", ngram_range=(1, 2))
    mat = vec.fit_transform(data[col]).toarray()
    text_frames.append(pd.DataFrame(mat,
                       columns=[f"{col}_{f}" for f in vec.get_feature_names_out()]))

X_text = pd.concat(text_frames, axis=1)

y = data["painting"]

# ── Train / valid / test split ─────────────────────────────────────────────
def make_splits(*Xs):
    idx = np.arange(len(y))
    idx_tr, idx_tmp = train_test_split(idx, test_size=0.30, random_state=RANDOM_STATE)
    idx_va, idx_te  = train_test_split(idx_tmp, test_size=0.50, random_state=RANDOM_STATE)
    return [(X.iloc[idx_tr], X.iloc[idx_va], X.iloc[idx_te]) for X in Xs]

(Xs_tr, Xs_va, Xs_te), (Xt_tr, Xt_va, Xt_te) = make_splits(X_structured, X_text)
y_tr = y.iloc[train_test_split(np.arange(len(y)), test_size=0.30, random_state=RANDOM_STATE)[0]]
_, y_tmp = train_test_split(y, test_size=0.30, random_state=RANDOM_STATE)
y_va, y_te = train_test_split(y_tmp, test_size=0.50, random_state=RANDOM_STATE)
y_tr_idx = np.array([i for i in range(len(y)) if i not in y_tmp.index])
y_tr = y.iloc[y_tr_idx] if False else y_tr  # keep simple version

# Redo cleanly
all_idx = np.arange(len(y))
tr_idx, tmp_idx = train_test_split(all_idx, test_size=0.30, random_state=RANDOM_STATE)
va_idx, te_idx  = train_test_split(tmp_idx, test_size=0.50, random_state=RANDOM_STATE)

def split_by_idx(X):
    return X.iloc[tr_idx].reset_index(drop=True), \
           X.iloc[va_idx].reset_index(drop=True), \
           X.iloc[te_idx].reset_index(drop=True)

Xs_tr, Xs_va, Xs_te = split_by_idx(X_structured)
Xt_tr, Xt_va, Xt_te = split_by_idx(X_text)
y_tr = y.iloc[tr_idx].reset_index(drop=True)
y_va = y.iloc[va_idx].reset_index(drop=True)
y_te = y.iloc[te_idx].reset_index(drop=True)

# ── Normalise only structured numerical cols ───────────────────────────────
num_end = len(NUMERICAL_COLS)

def norm_structured(tr, va, te):
    tr, va, te = tr.values.copy().astype(float), va.values.copy().astype(float), te.values.copy().astype(float)
    mean = tr[:, :num_end].mean(0); std = tr[:, :num_end].std(0); std[std == 0] = 1
    for X in [tr, va, te]:
        X[:, :num_end] = (X[:, :num_end] - mean) / std
    return tr, va, te

Xs_tr_n, Xs_va_n, Xs_te_n = norm_structured(Xs_tr, Xs_va, Xs_te)
# Text is already TF-IDF scaled – no further normalisation needed
Xt_tr_n = Xt_tr.values.astype(float)
Xt_va_n = Xt_va.values.astype(float)
Xt_te_n = Xt_te.values.astype(float)

# ── Model catalogue ────────────────────────────────────────────────────────
MODELS = {
    "GradBoost":  lambda: GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    "ExtraTrees": lambda: ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    "AdaBoost":   lambda: AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=RANDOM_STATE),
    "LogReg":     lambda: LogisticRegression(C=10, fit_intercept=False, max_iter=1000),
    "MLP":        lambda: MLPClassifier(hidden_layer_sizes=(150, 50), learning_rate_init=0.1,
                                         max_iter=1000, random_state=RANDOM_STATE, verbose=False),
    "DecisionTree": lambda: DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
}

# ── Soft-voting combiner ───────────────────────────────────────────────────
def soft_vote(proba_list, classes):
    avg = np.mean(proba_list, axis=0)
    return classes[np.argmax(avg, axis=1)]

# ── Run all structured × text combos ──────────────────────────────────────
header = f"{'Structured model':<15} {'Text model':<15} {'Train Acc':>10} {'Valid Acc':>10}"
print("\n" + "=" * len(header))
print(header)
print("=" * len(header))

results = []
classes = None

for struct_name, text_name in product(MODELS.keys(), MODELS.keys()):
    # ── Structured branch ──────────────────────────────────────────────────
    m_struct = MODELS[struct_name]()
    m_struct.fit(Xs_tr_n, y_tr)
    p_struct_tr = m_struct.predict_proba(Xs_tr_n)
    p_struct_va = m_struct.predict_proba(Xs_va_n)
    if classes is None:
        classes = m_struct.classes_

    # ── Text branch ───────────────────────────────────────────────────────
    m_text = MODELS[text_name]()
    m_text.fit(Xt_tr_n, y_tr)
    p_text_tr = m_text.predict_proba(Xt_tr_n)
    p_text_va = m_text.predict_proba(Xt_va_n)

    # Align class order between models
    def align_proba(m, proba):
        order = [list(m.classes_).index(c) for c in classes]
        return proba[:, order]

    p_struct_tr = align_proba(m_struct, p_struct_tr)
    p_struct_va = align_proba(m_struct, p_struct_va)
    p_text_tr   = align_proba(m_text,   p_text_tr)
    p_text_va   = align_proba(m_text,   p_text_va)

    # ── Soft vote ─────────────────────────────────────────────────────────
    y_tr_pred = soft_vote([p_struct_tr, p_text_tr], classes)
    y_va_pred = soft_vote([p_struct_va, p_text_va], classes)

    train_acc = accuracy_score(y_tr, y_tr_pred)
    val_acc   = accuracy_score(y_va, y_va_pred)

    results.append({
        "structured": struct_name,
        "text":       text_name,
        "train_acc":  train_acc,
        "valid_acc":  val_acc,
    })
    print(f"{struct_name:<15} {text_name:<15} {train_acc:>10.4f} {val_acc:>10.4f}")

print("=" * len(header))

# ── Summary ────────────────────────────────────────────────────────────────
df = pd.DataFrame(results).sort_values("valid_acc", ascending=False)

print("\n📊 Top-10 combinations by Validation Accuracy:")
print(df.head(10).to_string(index=False))

best = df.iloc[0]
print(
    f"\n🏆 Best combo: {best['structured']} (structured) + {best['text']} (text) "
    f"→ Train: {best['train_acc']:.4f}, Valid: {best['valid_acc']:.4f}"
)

# ── Baselines for comparison ───────────────────────────────────────────────
print("\n📌 Baselines (single model, best from prior runs):")
print(f"  GradientBoosting alone (No BoW) → Valid: 0.9118")
print(f"  GradientBoosting alone (BoW)    → Valid: 0.8950")
