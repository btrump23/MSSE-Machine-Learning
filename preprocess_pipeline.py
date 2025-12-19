import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ----------------------------
# Config
# ----------------------------
TRAIN_PATH = "Data/processed/train.csv"
TARGET_COL = "Label"

RANDOM_STATE = 42
N_SPLITS = 10

# ID/hash/metadata fields to exclude (ignored if not present)
DROP_COLS = ["Identify", "MD5"]


def main():
    # ----------------------------
    # Load training data ONLY
    # ----------------------------
    df = pd.read_csv(TRAIN_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found. "
            f"Available columns (first 30): {list(df.columns)[:30]}"
        )

    # Drop obvious non-feature columns
    df = df.drop(columns=DROP_COLS, errors="ignore")

    # Split X/y
    X = df.drop(columns=[TARGET_COL]).copy()
    y = df[TARGET_COL].copy()

    # ----------------------------
    # Coerce/clean features
    # ----------------------------
    # Convert any object columns to numeric where possible (bad parses -> NaN)
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # Replace inf values with NaN so imputer can handle them
    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop columns that are entirely missing (no observed values)
    all_missing_cols = X.columns[X.isna().all()].tolist()
    if all_missing_cols:
        print("Dropping all-missing columns:", all_missing_cols)
        X = X.drop(columns=all_missing_cols)

    # Keep numeric columns only (anything still non-numeric is dropped)
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    dropped_non_numeric = [c for c in X.columns if c not in numeric_features]

    print(f"Total feature columns after cleaning: {X.shape[1]}")
    print(f"Numeric feature columns used: {len(numeric_features)}")
    if dropped_non_numeric:
        print("Dropped non-numeric columns:", dropped_non_numeric)

    if len(numeric_features) == 0:
        raise ValueError("No numeric feature columns available after cleaning.")

    # ----------------------------
    # Pipeline: impute + scale + model
    # ----------------------------
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocess = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)],
        remainder="drop"
    )

    model = LogisticRegression(max_iter=5000)

    clf = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", model),
    ])

    # ----------------------------
    # Cross-validation (NO leakage)
    # ----------------------------
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    results = cross_validate(
        clf, X, y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        error_score="raise"
    )

    print("\n10-fold CV results (train only):")
    for metric in scoring.keys():
        scores = results[f"test_{metric}"]
        print(f"{metric:>9}: mean={scores.mean():.4f}  std={scores.std():.4f}")

    print("\n✅ Pipeline + CV completed (no leakage).")


if __name__ == "__main__":
    main()
