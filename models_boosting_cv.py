# models_boosting_cv.py
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# -----------------------
# Config
# -----------------------
TRAIN_PATH = "data/processed/train.csv"   # <-- change to "Data/processed/train.csv" if needed
TARGET_COL = "Label"
RANDOM_STATE = 42
N_SPLITS = 10

# -----------------------
# Load train data
# -----------------------
df = pd.read_csv(TRAIN_PATH)

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found. Columns: {list(df.columns)}")

y = df[TARGET_COL].astype(int)
X = df.drop(columns=[TARGET_COL])

# Keep numeric columns only (your dataset is mostly numeric, but this is safe)
X = X.select_dtypes(include=[np.number])

# Drop columns that are entirely missing (fixes the warnings you saw earlier)
all_missing_cols = X.columns[X.isna().all()].tolist()
if all_missing_cols:
    print(f"Dropping all-missing columns: {all_missing_cols}")
    X = X.drop(columns=all_missing_cols)

print(f"Features used: {X.shape[1]}")

# -----------------------
# CV + scoring
# -----------------------
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1"
}

def evaluate_model(name: str, estimator):
    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])

    results = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    print(f"\n{name} (Stratified {N_SPLITS}-fold CV, train only):")
    for k in ["test_accuracy", "test_precision", "test_recall", "test_f1"]:
        vals = results[k]
        metric_name = k.replace("test_", "")
        print(f"{metric_name:>9}: mean={vals.mean():.4f}  std={vals.std():.4f}")

# -----------------------
# Models
# -----------------------
xgb = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    eval_metric="logloss",
    n_jobs=-1
)

lgbm = LGBMClassifier(
    n_estimators=800,
    learning_rate=0.05,
    num_leaves=63,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

cat = CatBoostClassifier(
    iterations=800,
    learning_rate=0.05,
    depth=8,
    loss_function="Logloss",
    random_seed=RANDOM_STATE,
    verbose=False
)

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    evaluate_model("XGBoost", xgb)
    evaluate_model("LightGBM", lgbm)
    evaluate_model("CatBoost", cat)

    print("\n✅ Completed XGB + LightGBM + CatBoost (no leakage).")
