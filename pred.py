"""
Prediction script for the ML art competition.
Uses a soft-voting ensemble of two GradientBoosting models:
  - Model 1: trained on structured features (numerical + categorical)
  - Model 2: trained on TF-IDF Bag-of-Words text features

Only uses: numpy, pandas, re (all stdlib/allowed).
No sklearn, pytorch, tensorflow, etc.
Weights are hardcoded inline (generated from model_params.npz via _gen_pred.py).
"""
import re
import os
import numpy as np
import pandas as pd

# ── Hardcoded model parameters ──────────────────────────────────────────────
_DIR    = os.path.dirname(os.path.abspath(__file__))
_PARAMS = np.load(os.path.join(_DIR, "model_params.npz"), allow_pickle=True)

CLASSES        = _PARAMS["classes"]
INIT_RAW       = _PARAMS["init_raw"]          # log-prior per class (n_classes,)

STRUCT_MEAN    = _PARAMS["struct_mean"]
STRUCT_STD     = _PARAMS["struct_std"]
ROOM_CLASSES   = _PARAMS["room_classes"]
SEASON_CLASSES = _PARAMS["season_classes"]
VIEW_CLASSES   = _PARAMS["view_classes"]

STRUCT_LR = float(_PARAMS["struct_lr"][0])
S_FEAT    = _PARAMS["s_feat"];  S_THRESH = _PARAMS["s_thresh"]
S_LEFT    = _PARAMS["s_left"];  S_RIGHT  = _PARAMS["s_right"];  S_VAL = _PARAMS["s_val"]

TEXT_LR = float(_PARAMS["text_lr"][0])
T_FEAT  = _PARAMS["t_feat"];  T_THRESH = _PARAMS["t_thresh"]
T_LEFT  = _PARAMS["t_left"];  T_RIGHT  = _PARAMS["t_right"];  T_VAL = _PARAMS["t_val"]

TEXT_COLS = ["feeling_description", "food_association", "soundtrack"]
TFIDF_VOCAB = {}   # col -> {term: index}
TFIDF_IDF   = {}   # col -> idf array
for _col in TEXT_COLS:
    _terms = _PARAMS[f"vocab_{_col}"]
    TFIDF_VOCAB[_col] = {str(t): i for i, t in enumerate(_terms)}
    TFIDF_IDF[_col]   = _PARAMS[f"idf_{_col}"]

NUMERICAL_COLS = [
    "emotion_intensity","feel_sombre","feel_content","feel_calm",
    "feel_uneasy","prominent_colours","objects_noticed","willingness_to_pay",
]

# ── TF-IDF transform (pure numpy) ────────────────────────────────────────────
_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")

def _tfidf_row(text, vocab, idf):
    tokens = _TOKEN_RE.findall(text.lower())
    ngrams = tokens[:]
    for i in range(len(tokens) - 1):
        ngrams.append(tokens[i] + " " + tokens[i + 1])
    counts = {}
    for g in ngrams:
        if g in vocab:
            counts[vocab[g]] = counts.get(vocab[g], 0) + 1
    vec = np.zeros(len(vocab))
    for idx, cnt in counts.items():
        vec[idx] = (1.0 + np.log(cnt)) * idf[idx]
    norm = np.sqrt((vec ** 2).sum())
    if norm > 0:
        vec /= norm
    return vec

def _build_text_features(df):
    mats = []
    for col in TEXT_COLS:
        texts = df[col].fillna("").astype(str).tolist()
        vocab, idf = TFIDF_VOCAB[col], TFIDF_IDF[col]
        mats.append(np.vstack([_tfidf_row(t, vocab, idf) for t in texts]))
    return np.hstack(mats)

# ── Multi-label binarise ──────────────────────────────────────────────────────
def _mlb_row(value, classes):
    tags = {s.strip() for s in str(value).split(",")}
    return np.array([1.0 if c in tags else 0.0 for c in classes])

def _build_structured_features(df):
    X_num = df[NUMERICAL_COLS].values.astype(float)
    X_num = (X_num - STRUCT_MEAN) / STRUCT_STD
    room_mat   = np.vstack([_mlb_row(v, ROOM_CLASSES)   for v in df["room"].fillna("")])
    season_mat = np.vstack([_mlb_row(v, SEASON_CLASSES) for v in df["season"].fillna("")])
    view_mat   = np.vstack([_mlb_row(v, VIEW_CLASSES)   for v in df["view_with"].fillna("")])
    return np.hstack([X_num, room_mat, season_mat, view_mat])

# ── Gradient Boosting inference (vectorised over samples) ─────────────────────
def _gb_proba(X, feat, thresh, left, right, val, lr, init_raw):
    n     = X.shape[0]
    n_est = feat.shape[0]
    n_cls = feat.shape[1]
    raw   = np.tile(init_raw, (n, 1))
    for k in range(n_est):
        for c in range(n_cls):
            nodes = np.zeros(n, dtype=np.int32)
            for _ in range(20):                          # max tree depth = 3
                is_leaf = left[k, c, nodes] == -1
                if is_leaf.all():
                    break
                fidx  = np.where(is_leaf, 0, feat[k, c, nodes])
                go_l  = X[np.arange(n), fidx] <= thresh[k, c, nodes]
                new   = np.where(go_l, left[k, c, nodes], right[k, c, nodes])
                nodes = np.where(is_leaf, nodes, new)
            raw[:, c] += lr * val[k, c, nodes]
    raw -= raw.max(1, keepdims=True)
    e = np.exp(raw)
    return e / e.sum(1, keepdims=True)

# ── Public API ────────────────────────────────────────────────────────────────
def predict_all(filename):
    """
    Read a CSV test file and return a list of painting predictions.
    Paintings are one of:
      'The Persistence of Memory', 'The Starry Night', 'The Water Lily Pond'
    """
    df = pd.read_csv(filename)
    df = df.reset_index(drop=True)

    X_struct = _build_structured_features(df)
    X_text   = _build_text_features(df)

    p_struct = _gb_proba(X_struct, S_FEAT, S_THRESH, S_LEFT, S_RIGHT, S_VAL,
                         STRUCT_LR, INIT_RAW)
    p_text   = _gb_proba(X_text,   T_FEAT, T_THRESH, T_LEFT, T_RIGHT, T_VAL,
                         TEXT_LR,  INIT_RAW)

    # Soft vote: average probabilities from both models
    combined = p_struct + p_text
    idx      = combined.argmax(axis=1)
    return [str(CLASSES[i]) for i in idx]
