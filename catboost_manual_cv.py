import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from catboost import CatBoostClassifier

TRAIN_PATH = "data/processed/train.csv"   # change to "Data/processed/train.csv" if needed
TARGET_COL = "Label"
RANDOM_STATE = 42
N_SPLITS = 10

df = pd.read_csv(TRAIN_PATH)

y = df[TARGET_COL].astype(int).to_numpy()
X = df.drop(columns=[TARGET_COL])

# numeric only
X = X.select_dtypes(include=[np.number])

# drop all-missing
all_missing_cols = X.columns[X.isna().all()].tolist()
if all_missing_cols:
    print("Dropping all-missing columns:", all_missing_cols)
    X = X.drop(columns=all_missing_cols)

X = X.to_numpy()
print("Features used:", X.shape[1])

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

accs, precs, recs, f1s = [], [], [], []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Fit preprocessing on TRAIN fold only (no leakage)
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    model = CatBoostClassifier(
        iterations=800,
        learning_rate=0.05,
        depth=8,
        loss_function="Logloss",
        random_seed=RANDOM_STATE,
        verbose=False
    )

    model.fit(X_train_imp, y_train)
    preds = model.predict(X_test_imp).astype(int)

    accs.append(accuracy_score(y_test, preds))
    precs.append(precision_score(y_test, preds, zero_division=0))
    recs.append(recall_score(y_test, preds, zero_division=0))
    f1s.append(f1_score(y_test, preds, zero_division=0))

    print(
        "Fold", fold,
        "| acc=", round(accs[-1], 4),
        "prec=", round(precs[-1], 4),
        "rec=", round(recs[-1], 4),
        "f1=", round(f1s[-1], 4)
    )

print("\nCatBoost (Manual Stratified 10-fold CV, train only):")
print(" accuracy: mean=", round(float(np.mean(accs)), 4), " std=", round(float(np.std(accs)), 4))
print("precision: mean=", round(float(np.mean(precs)), 4), " std=", round(float(np.std(precs)), 4))
print("   recall: mean=", round(float(np.mean(recs)), 4), " std=", round(float(np.std(recs)), 4))
print("       f1: mean=", round(float(np.mean(f1s)), 4), " std=", round(float(np.std(f1s)), 4))

print("\n✅ CatBoost CV complete (manual CV avoids sklearn tag compatibility issue).")

