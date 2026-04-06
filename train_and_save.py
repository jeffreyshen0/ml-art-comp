"""
Train GradientBoosting ensemble and save all parameters for pure-numpy
inference in pred.py. Run once (requires sklearn). Produces model_params.npz.

Uses a 70/15/15 train/val/test split (random_state=42) so the saved weights
generalise to held-out data rather than being overfit to the full dataset.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42
N_ESTIMATORS = 200
data = pd.read_csv("new_data2.csv")

# ── 70 / 15 / 15  train / val / test split ───────────────────────────────────
from sklearn.model_selection import train_test_split as _tts
train_df, _tmp_df = _tts(data, test_size=0.30, random_state=RANDOM_STATE)
val_df,   test_df = _tts(_tmp_df, test_size=0.50, random_state=RANDOM_STATE)
print(f"Split sizes  → train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")

# ── Multi-label encode ────────────────────────────────────────────────────
def fit_multilabel(df, col):
    mlb = MultiLabelBinarizer()
    split = df[col].str.split(",").apply(lambda x: [s.strip() for s in x])
    return mlb.fit_transform(split), mlb.classes_

room_enc,   room_classes   = fit_multilabel(data, "room")
season_enc, season_classes = fit_multilabel(data, "season")
view_enc,   view_classes   = fit_multilabel(data, "view_with")

NUMERICAL_COLS = [
    "emotion_intensity","feel_sombre","feel_content","feel_calm",
    "feel_uneasy","prominent_colours","objects_noticed","willingness_to_pay",
]
X_num = data[NUMERICAL_COLS].values.astype(float)
struct_mean = X_num.mean(0); struct_std = X_num.std(0); struct_std[struct_std==0] = 1.0
X_num_norm  = (X_num - struct_mean) / struct_std
X_structured = np.hstack([X_num_norm, room_enc, season_enc, view_enc]).astype(float)

# ── TF-IDF text features ──────────────────────────────────────────────────
TEXT_COLS = ["feeling_description", "food_association", "soundtrack"]
for col in TEXT_COLS:
    data[col] = data[col].fillna("")

tfidf_vecs, text_mats = {}, []
for col in TEXT_COLS:
    vec = TfidfVectorizer(max_features=100, sublinear_tf=True,
                          strip_accents="unicode", ngram_range=(1, 2))
    mat = vec.fit_transform(data[col]).toarray()
    tfidf_vecs[col] = vec
    text_mats.append(mat)
X_text = np.hstack(text_mats)

# ── Build per-split labels ────────────────────────────────────────────────
y_all   = data["painting"].values
y_train = train_df["painting"].values
y_val   = val_df["painting"].values
y_test  = test_df["painting"].values

# Align feature matrices to each split using index
train_idx = train_df.index
val_idx   = val_df.index
test_idx  = test_df.index

X_struct_tr = X_structured[train_idx]
X_struct_va = X_structured[val_idx]
X_struct_te = X_structured[test_idx]
X_text_tr   = X_text[train_idx]
X_text_va   = X_text[val_idx]
X_text_te   = X_text[test_idx]

# ── Train on train split only ─────────────────────────────────────────────
print("Training structured model...")
m_struct = GradientBoostingClassifier(n_estimators=N_ESTIMATORS, learning_rate=0.1,
                                       max_depth=3, random_state=RANDOM_STATE)
m_struct.fit(X_struct_tr, y_train)

print("Training text model...")
m_text = GradientBoostingClassifier(n_estimators=N_ESTIMATORS, learning_rate=0.1,
                                     max_depth=3, random_state=RANDOM_STATE)
m_text.fit(X_text_tr, y_train)

def _ensemble_acc(Xs, Xt, y_true):
    sp = m_struct.predict_proba(Xs)
    tp = m_text.predict_proba(Xt)
    preds = m_struct.classes_[(sp + tp).argmax(1)]
    return accuracy_score(y_true, preds)

print(f"Ensemble train accuracy : {_ensemble_acc(X_struct_tr, X_text_tr, y_train):.4f}")
print(f"Ensemble val   accuracy : {_ensemble_acc(X_struct_va, X_text_va, y_val):.4f}")
print(f"Ensemble test  accuracy : {_ensemble_acc(X_struct_te, X_text_te, y_test):.4f}")

# Use the full dataset arrays for index alignment in tree extraction
y = y_all

# ── Extract tree arrays ───────────────────────────────────────────────────
def extract_trees(model):
    n_est = len(model.estimators_)
    n_cls = len(model.classes_)
    max_n = max(model.estimators_[k][c].tree_.node_count
                for k in range(n_est) for c in range(n_cls))
    feat   = np.full((n_est, n_cls, max_n), -2, dtype=np.int32)
    thresh = np.full((n_est, n_cls, max_n), -2.0)
    left   = np.full((n_est, n_cls, max_n), -1, dtype=np.int32)
    right  = np.full((n_est, n_cls, max_n), -1, dtype=np.int32)
    val    = np.zeros((n_est, n_cls, max_n))
    for k in range(n_est):
        for c in range(n_cls):
            t = model.estimators_[k][c].tree_; n = t.node_count
            feat[k,c,:n]=t.feature; thresh[k,c,:n]=t.threshold
            left[k,c,:n]=t.children_left; right[k,c,:n]=t.children_right
            val[k,c,:n]=t.value[:,0,0]
    return feat, thresh, left, right, val

print("Extracting trees...")
s_feat,s_thresh,s_left,s_right,s_val = extract_trees(m_struct)
t_feat,t_thresh,t_left,t_right,t_val = extract_trees(m_text)

# Init raw prediction = log(class priors) — must use *training* labels to
# match sklearn's internal GBM initialiser (which sees only the train split)
classes = m_struct.classes_
priors  = np.array([(y_train==c).sum() for c in classes], float) / len(y_train)
priors  = np.clip(priors, np.finfo(float).eps, 1-np.finfo(float).eps)
init_raw = np.log(priors)

# ── Validate numpy implementation matches sklearn ─────────────────────────
def numpy_gb_proba(X, feat, thresh, left, right, val, lr, init_r):
    n, n_est, n_cls = X.shape[0], feat.shape[0], feat.shape[1]
    raw = np.tile(init_r, (n, 1))
    for k in range(n_est):
        for c in range(n_cls):
            nodes = np.zeros(n, dtype=np.int32)
            for _ in range(20):
                is_leaf = left[k,c,nodes] == -1
                if is_leaf.all(): break
                fidx = np.where(is_leaf, 0, feat[k,c,nodes])
                go_l = X[np.arange(n), fidx] <= thresh[k,c,nodes]
                new  = np.where(go_l, left[k,c,nodes], right[k,c,nodes])
                nodes = np.where(is_leaf, nodes, new)
            raw[:,c] += lr * val[k,c,nodes]
    raw -= raw.max(1, keepdims=True)
    e = np.exp(raw); return e / e.sum(1, keepdims=True)

check_n = min(50, len(X_struct_tr))
my_s = numpy_gb_proba(X_struct_tr[:check_n], s_feat,s_thresh,s_left,s_right,s_val,
                       m_struct.learning_rate, init_raw)
sk_s = m_struct.predict_proba(X_struct_tr[:check_n])
assert np.allclose(my_s, sk_s, atol=1e-4), "Structured model mismatch!"

my_t = numpy_gb_proba(X_text_tr[:check_n], t_feat,t_thresh,t_left,t_right,t_val,
                       m_text.learning_rate, init_raw)
sk_t = m_text.predict_proba(X_text_tr[:check_n])
assert np.allclose(my_t, sk_t, atol=1e-4), "Text model mismatch!"
print("✅ Numpy implementation validated!")

# ── Save ──────────────────────────────────────────────────────────────────
tfidf_save = {}
for col in TEXT_COLS:
    vec = tfidf_vecs[col]
    tfidf_save[f"vocab_{col}"]   = np.array(vec.get_feature_names_out())
    tfidf_save[f"idf_{col}"]     = vec.idf_

np.savez_compressed("model_params.npz",
    classes=classes, init_raw=init_raw,
    struct_mean=struct_mean, struct_std=struct_std,
    room_classes=room_classes, season_classes=season_classes, view_classes=view_classes,
    struct_lr=np.array([m_struct.learning_rate]),
    s_feat=s_feat, s_thresh=s_thresh, s_left=s_left, s_right=s_right, s_val=s_val,
    text_lr=np.array([m_text.learning_rate]),
    t_feat=t_feat, t_thresh=t_thresh, t_left=t_left, t_right=t_right, t_val=t_val,
    **tfidf_save,
)
print("Saved model_params.npz")
