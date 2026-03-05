import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()

data = pd.read_csv("../new_data2.csv")
print(data.head())

def encode_multilabel(data, col):
    mlb = MultiLabelBinarizer()
    split = data[col].str.split(",").apply(lambda x: [s.strip() for s in x])
    return pd.DataFrame(mlb.fit_transform(split), columns=[f"{col}_{c}" for c in mlb.classes_])

room_encoded = encode_multilabel(data, "room")
season_encoded = encode_multilabel(data, "season")
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

X = pd.concat([
    data[numerical_cols].reset_index(drop=True),
    room_encoded,
    season_encoded,
    view_with_encoded,
], axis=1)

y = data["painting"]

print(X.shape)
print(y.shape)
# Splitting the data into 70/15/15
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)

# Second split: split the 30% temp into 15% valid, 15% test
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

print("Train:", X_train.shape)
print("Valid:", X_valid.shape)
print("Test:", X_test.shape)

# Normalizing the data
X_train_norm = X_train.values.copy().astype('float')
X_valid_norm = X_valid.values.copy().astype('float')
X_test_norm = X_test.values.copy().astype('float')

# Extract numerical columns
numerical_end = len(numerical_cols)

# Training Data
mean = X_train_norm[:, :numerical_end].mean(axis=0)
std = X_train_norm[:, :numerical_end].std(axis=0)

X_train_norm[:,:numerical_end] = (X_train_norm[:, : numerical_end] - mean) / std

# Validation Data
X_valid_norm[:, :numerical_end] = (X_valid_norm[:, :numerical_end] - mean) / std

# Test Data
X_test_norm[:, :numerical_end] = (X_test_norm[:, :numerical_end] - mean) / std

model = MLPClassifier(hidden_layer_sizes=(150,50),
                      random_state=42,
                      verbose=True,
                      learning_rate_init=0.1,
                      max_iter=1000)

model.fit(X_train_norm, y_train)

y_valid_pred = model.predict(X_valid_norm)
print("Validation Accuracy:", accuracy_score(y_valid, y_valid_pred))

y_test_pred = model.predict(X_test_norm)
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
