"""
Train GradientBoosting ensemble and save all parameters for pure-numpy
inference in pred.py. Run once (requires sklearn). Produces model_params.npz.
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

y = data["painting"].values

# ── Train ─────────────────────────────────────────────────────────────────
print("Training structured model...")
m_struct = GradientBoostingClassifier(n_estimators=N_ESTIMATORS, learning_rate=0.1,
                                       max_depth=3, random_state=RANDOM_STATE)
m_struct.fit(X_structured, y)

print("Training text model...")
m_text = GradientBoostingClassifier(n_estimators=N_ESTIMATORS, learning_rate=0.1,
                                     max_depth=3, random_state=RANDOM_STATE)
m_text.fit(X_text, y)

sp = m_struct.predict_proba(X_structured)
tp = m_text.predict_proba(X_text)
ens = m_struct.classes_[(sp + tp).argmax(1)]
print(f"Ensemble train accuracy: {accuracy_score(y, ens):.4f}")

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

# Init raw prediction = log(class priors)
classes = m_struct.classes_
priors  = np.array([(y==c).sum() for c in classes], float) / len(y)
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

check_n = min(50, len(y))
my_s = numpy_gb_proba(X_structured[:check_n], s_feat,s_thresh,s_left,s_right,s_val,
                       m_struct.learning_rate, init_raw)
sk_s = m_struct.predict_proba(X_structured[:check_n])
assert np.allclose(my_s, sk_s, atol=1e-4), "Structured model mismatch!"

my_t = numpy_gb_proba(X_text[:check_n], t_feat,t_thresh,t_left,t_right,t_val,
                       m_text.learning_rate, init_raw)
sk_t = m_text.predict_proba(X_text[:check_n])
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
