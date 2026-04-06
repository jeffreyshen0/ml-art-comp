import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Load data ──────────────────────────────────────────────────────────────
data = pd.read_csv("../new_data2.csv")

# ── Encode multi-label categorical columns ─────────────────────────────────
def encode_multilabel(df, col):
    mlb = MultiLabelBinarizer()
    split = df[col].str.split(",").apply(lambda x: [s.strip() for s in x])
    return pd.DataFrame(mlb.fit_transform(split), columns=[f"{col}_{c}" for c in mlb.classes_])

room_encoded     = encode_multilabel(data, "room")
season_encoded   = encode_multilabel(data, "season")
view_with_encoded = encode_multilabel(data, "view_with")

numerical_cols = [
    "emotion_intensity",
    "feel_sombre",
    "feel_content",
    "feel_calm",
    "feel_uneasy",
    "prominent_colours",
    "objects_noticed",
    "willingness_to_pay",
]

X_base = pd.concat([
    data[numerical_cols].reset_index(drop=True),
    room_encoded,
    season_encoded,
    view_with_encoded,
], axis=1)

# ── Bag of Words via TF-IDF on text columns ────────────────────────────────
TEXT_COLS = ["feeling_description", "food_association", "soundtrack"]
for col in TEXT_COLS:
    data[col] = data[col].fillna("")

vectorizers = {}
text_frames = []
for col in TEXT_COLS:
    vec = TfidfVectorizer(max_features=100, sublinear_tf=True,
                          strip_accents="unicode", ngram_range=(1, 2))
    mat = vec.fit_transform(data[col]).toarray()
    vectorizers[col] = vec
    text_frames.append(pd.DataFrame(mat, columns=[f"{col}_{f}" for f in vec.get_feature_names_out()]))

X_text = pd.concat(text_frames, axis=1)
X_with_bow = pd.concat([X_base.reset_index(drop=True), X_text], axis=1)

y = data["painting"]

RANDOM_STATE = 42

def make_splits(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE)
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def normalize(X_train, X_valid, X_test, num_end):
    X_train = X_train.values.copy().astype(float)
    X_valid = X_valid.values.copy().astype(float)
    X_test  = X_test.values.copy().astype(float)
    mean = X_train[:, :num_end].mean(axis=0)
    std  = X_train[:, :num_end].std(axis=0)
    std[std == 0] = 1  # avoid divide-by-zero
    X_train[:, :num_end] = (X_train[:, :num_end] - mean) / std
    X_valid[:, :num_end] = (X_valid[:, :num_end] - mean) / std
    X_test[:,  :num_end] = (X_test[:,  :num_end] - mean) / std
    return X_train, X_valid, X_test

# ── Run both feature sets ──────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"{'Features':<15} {'Valid Acc':>12} {'Test Acc':>12}")
print(f"{'='*60}")

for feat_label, X in [("No BoW", X_base), ("With BoW", X_with_bow)]:
    X_train, X_valid, X_test, y_train, y_valid, y_test = make_splits(X, y)
    X_train_n, X_valid_n, X_test_n = normalize(X_train, X_valid, X_test, len(numerical_cols))

    model = LogisticRegression(C=10, fit_intercept=False, max_iter=1000)
    model.fit(X_train_n, y_train)

    val_acc  = accuracy_score(y_valid, model.predict(X_valid_n))
    test_acc = accuracy_score(y_test,  model.predict(X_test_n))
    print(f"{'Logistic Reg ' + feat_label:<15} {val_acc:>12.4f} {test_acc:>12.4f}")

print(f"{'='*60}")


# Hyperparameter tuning using Grid Search
from sklearn.model_selection import GridSearchCV

param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],           # inverse regularization strength
    "penalty": ["l1", "l2"],                 # regularization type
    "solver": ["liblinear", "saga"],         # solvers that support both l1/l2
    "fit_intercept": [True, False],
}

base_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

search = GridSearchCV(
    base_model,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=2,
)

search.fit(X_train_n, y_train)

print(f"Best params: {search.best_params_}")
print(f"Best CV score: {search.best_score_:.4f}")

best_model = search.best_estimator_
val_acc = accuracy_score(y_valid, best_model.predict(X_valid_n))
test_acc = accuracy_score(y_test, best_model.predict(X_test_n))
print(f"Validation: {val_acc:.4f}")
print(f"Test: {test_acc:.4f}")
