import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from catboost import CatBoostClassifier

TRAIN_PATH = "data/processed/train.csv"   # change to Data/processed/train.csv if needed
TARGET_COL = "Label"
RANDOM_STATE = 42
N_SPLITS = 10

df = pd.read_csv(TRAIN_PATH)

y = df[TARGET_COL].astype(int)
X = df.drop(columns=[TARGET_COL])

X = X.select_dtypes(include=[np.number])
all_missing_cols = X.columns[X.isna().all()].tolist()
if all_missing_cols:
    print("Dropping all-missing columns:", all_missing_cols)
    X = X.drop(columns=all_missing_cols)

print("Features used:", X.shape[1])

pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("model", CatBoostClassifier(
        iterations=800,
        learning_rate=0.05,
        depth=8,
        loss_function="Logloss",
        random_seed=RANDOM_STATE,
        verbose=False
    ))
])

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1"
}

results = cross_validate(
    pipe, X, y,
    cv=cv,
    scoring=scoring,
    n_jobs=-1,
    error_score="raise"
)

print("\nCatBoost (Stratified 10-fold CV, train only):")
for metric in scoring:
    vals = results[f"test_{metric}"]
    print(f"{metric:9s}: mean={vals.mean():.4f}  std={vals.std():.4f}")
