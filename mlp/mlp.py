import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
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

room_encoded      = encode_multilabel(data, "room")
season_encoded    = encode_multilabel(data, "season")
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

    model = MLPClassifier(
        hidden_layer_sizes=(150, 50),
        random_state=RANDOM_STATE,
        learning_rate_init=0.1,
        max_iter=1000,
        verbose=False,
    )
    model.fit(X_train_n, y_train)

    val_acc  = accuracy_score(y_valid, model.predict(X_valid_n))
    test_acc = accuracy_score(y_test,  model.predict(X_test_n))
    print(f"{'MLP ' + feat_label:<15} {val_acc:>12.4f} {test_acc:>12.4f}")

print(f"{'='*60}")

# Hyperparameter Tuning
from itertools import product

results = []
for lr in [0.001, 0.01, 0.1]:
    for alpha in [1e-4, 1e-3, 1e-2]:
        for layers in [(128,), (150, 50), (128, 64, 32)]:
            model = MLPClassifier(hidden_layer_sizes=layers, learning_rate_init=lr,
                                  alpha=alpha, max_iter=1000, random_state=RANDOM_STATE)
            model.fit(X_train_n, y_train)
            val_acc = accuracy_score(y_valid, model.predict(X_valid_n))
            results.append({"lr": lr, "alpha": alpha, "layers": layers, "val_acc": val_acc})

best = max(results, key=lambda x: x["val_acc"])
print(best)

best_params = {"lr": 0.01, "alpha": 0.0001, "layers": (128, 64, 32)}

final_model = MLPClassifier(
    hidden_layer_sizes=best_params["layers"],
    learning_rate_init=best_params["lr"],
    alpha=best_params["alpha"],
    max_iter=1000,
    random_state=RANDOM_STATE,
)
final_model.fit(X_train_n, y_train)

print(f"Validation: {accuracy_score(y_valid, final_model.predict(X_valid_n)):.4f}")
print(f"Test:       {accuracy_score(y_test, final_model.predict(X_test_n)):.4f}")
