import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

# ── Load data ─────────────────────────────────────────────────────────────
data = pd.read_csv("../new_data2.csv")

# ── Encode multi-label categorical columns ─────────────────────────────────
def encode_multilabel(df, col):
    mlb = MultiLabelBinarizer()
    split = df[col].str.split(",").apply(lambda x: [s.strip() for s in x])
    return pd.DataFrame(
        mlb.fit_transform(split),
        columns=[f"{col}_{c}" for c in mlb.classes_],
    )

room_encoded    = encode_multilabel(data, "room")
season_encoded  = encode_multilabel(data, "season")
view_encoded    = encode_multilabel(data, "view_with")

# ── Numerical features ─────────────────────────────────────────────────────
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

X_base = pd.concat(
    [data[numerical_cols].reset_index(drop=True),
     room_encoded, season_encoded, view_encoded],
    axis=1,
)

# ── Text columns (Bag of Words via TF-IDF) ─────────────────────────────────
TEXT_COLS = ["feeling_description", "food_association", "soundtrack"]

# Fill NaN in text columns with empty string
for col in TEXT_COLS:
    data[col] = data[col].fillna("")

vectorizers = {}
text_matrices = []
for col in TEXT_COLS:
    vec = TfidfVectorizer(
        max_features=100,
        sublinear_tf=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
    )
    mat = vec.fit_transform(data[col])
    vectorizers[col] = vec
    text_matrices.append(mat)

X_text_sparse = hstack(text_matrices)                # sparse (n, total_vocab)
X_text_dense  = pd.DataFrame(
    X_text_sparse.toarray(),
    columns=[
        f"{col}_{feat}"
        for col, vec in vectorizers.items()
        for feat in vec.get_feature_names_out()
    ],
)

# ── Two feature sets: with & without BoW ──────────────────────────────────
X_no_bow  = X_base.copy()
X_with_bow = pd.concat([X_base.reset_index(drop=True), X_text_dense], axis=1)

y = data["painting"]

RANDOM_STATE = 42

def make_splits(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test

# ── Model zoo ─────────────────────────────────────────────────────────────
models = {
    "DecisionTree (depth=5)": DecisionTreeClassifier(
        max_depth=5, random_state=RANDOM_STATE
    ),
    "DecisionTree (depth=None)": DecisionTreeClassifier(
        max_depth=None, random_state=RANDOM_STATE
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=200, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE
    ),
    "AdaBoost": AdaBoostClassifier(
        n_estimators=200, learning_rate=0.5, random_state=RANDOM_STATE
    ),、
}

# ── Run experiments ────────────────────────────────────────────────────────
header = f"{'Model':<35} {'Features':<12} {'Valid Acc':>10} {'Test Acc':>10}"
print("\n" + "=" * len(header))
print(header)
print("=" * len(header))

results = []

for feat_label, X in [("No BoW", X_no_bow), ("With BoW", X_with_bow)]:
    X_train, X_valid, X_test, y_train, y_valid, y_test = make_splits(X, y)

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        val_acc  = accuracy_score(y_valid, clf.predict(X_valid))
        test_acc = accuracy_score(y_test,  clf.predict(X_test))
        results.append({
            "model": name,
            "features": feat_label,
            "valid_acc": val_acc,
            "test_acc": test_acc,
        })
        print(f"{name:<35} {feat_label:<12} {val_acc:>10.4f} {test_acc:>10.4f}")

print("=" * len(header))

# ── Summary: top-5 by test accuracy ────────────────────────────────────────
df_results = pd.DataFrame(results).sort_values("test_acc", ascending=False)
print("\n📊 Top-5 by Test Accuracy:")
print(df_results.head(5).to_string(index=False))

best = df_results.iloc[0]
print(
    f"\n🏆 Best model: {best['model']} ({best['features']}) "
    f"→ Valid: {best['valid_acc']:.4f}, Test: {best['test_acc']:.4f}"
)

# Hyperparameter tuning using Randomized Search
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define param distributions for each model type
param_dists = {
    "DecisionTree": {
        "max_depth": [3, 5, 10, 15, 20, None],
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 10),
        "criterion": ["gini", "entropy"],
    },
    "RandomForest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15, 20, None],
        "min_samples_split": randint(2, 15),
        "min_samples_leaf": randint(1, 8),
        "max_features": ["sqrt", "log2", 0.3, 0.5],
    },
    "ExtraTrees": {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15, 20, None],
        "min_samples_split": randint(2, 15),
        "min_samples_leaf": randint(1, 8),
        "max_features": ["sqrt", "log2", 0.3, 0.5],
    },
    "GradientBoosting": {
        "n_estimators": [100, 200, 300],
        "learning_rate": uniform(0.01, 0.29),  # 0.01 to 0.3
        "max_depth": [3, 4, 5, 6],
        "min_samples_split": randint(2, 15),
        "subsample": uniform(0.7, 0.3),  # 0.7 to 1.0
    },
    "AdaBoost": {
        "n_estimators": [50, 100, 200, 300],
        "learning_rate": uniform(0.01, 0.99),  # 0.01 to 1.0
    },
}

base_models = {
    "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    "ExtraTrees": ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "AdaBoost": AdaBoostClassifier(random_state=RANDOM_STATE),
}

# Run tuning
print(f"\n{'=' * 70}")
print(f"{'Model':<20} {'Valid Acc':>12} {'Test Acc':>12} {'Best Params'}")
print(f"{'=' * 70}")

best_results = []

# Use With BoW since it performed better
X = X_with_bow
X_train, X_valid, X_test, y_train, y_valid, y_test = make_splits(X, y)

for name, base_model in base_models.items():
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dists[name],
        n_iter=30,
        cv=3,
        scoring="accuracy",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    val_acc = accuracy_score(y_valid, best_model.predict(X_valid))
    test_acc = accuracy_score(y_test, best_model.predict(X_test))

    best_results.append({
        "model": name,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "params": search.best_params_,
    })

    print(f"{name:<20} {val_acc:>12.4f} {test_acc:>12.4f}")
    print(f"   └─ {search.best_params_}")

print(f"{'=' * 70}")

# Show winner
best = max(best_results, key=lambda x: x["val_acc"])
print(f"\n🏆 Best: {best['model']} → Val: {best['val_acc']:.4f}, Test: {best['test_acc']:.4f}")
