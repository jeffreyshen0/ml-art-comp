import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score

data = pd.read_csv('new_data2.csv')

def encode_multilabel(df, col):
    mlb = MultiLabelBinarizer()
    split = df[col].str.split(',').apply(lambda x: [s.strip() for s in x])
    return pd.DataFrame(mlb.fit_transform(split), columns=[f'{col}_{c}' for c in mlb.classes_])

room_enc = encode_multilabel(data, 'room')
season_enc = encode_multilabel(data, 'season')
view_enc = encode_multilabel(data, 'view_with')

NUMERICAL_COLS = ['emotion_intensity','feel_sombre','feel_content','feel_calm',
                  'feel_uneasy','prominent_colours','objects_noticed','willingness_to_pay']
X = pd.concat([data[NUMERICAL_COLS].reset_index(drop=True), room_enc, season_enc, view_enc], axis=1)
y = data['painting']

X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42)

configs = [
    (50,  0.10, 3),
    (100, 0.10, 3),
    (200, 0.10, 3),
    (200, 0.05, 3),
    (200, 0.20, 3),
    (200, 0.10, 5),
    (200, 0.10, 1),
]

print("n_est    lr  depth    Train    Valid     Test")
print("-" * 50)
for n_est, lr, depth in configs:
    m = GradientBoostingClassifier(n_estimators=n_est, learning_rate=lr,
                                   max_depth=depth, random_state=42)
    m.fit(X_train, y_train)
    tr = accuracy_score(y_train, m.predict(X_train))
    va = accuracy_score(y_valid, m.predict(X_valid))
    te = accuracy_score(y_test,  m.predict(X_test))
    print(f"{n_est:>5}  {lr:>5}  {depth:>5}  {tr:.4f}  {va:.4f}  {te:.4f}")
