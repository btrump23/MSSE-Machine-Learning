import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# ----------------------------
# Config
# ----------------------------
TRAIN_PATH = "Data/processed/train.csv"
TARGET_COL = "Label"
RANDOM_STATE = 42
N_SPLITS = 10
DROP_COLS = ["Identify", "MD5"]


def load_and_clean_train():
    df = pd.read_csv(TRAIN_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found.")

    df = df.drop(columns=DROP_COLS, errors="ignore")

    X = df.drop(columns=[TARGET_COL]).copy()
    y = df[TARGET_COL].copy()

    # Coerce objects to numeric
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # Replace inf with NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop all-missing columns
    all_missing_cols = X.columns[X.isna().all()].tolist()
    if all_missing_cols:
        print("Dropping all-missing columns:", all_missing_cols)
        X = X.drop(columns=all_missing_cols)

    numeric_features = X.select_dtypes(include="number").columns.tolist()
    if not numeric_features:
        raise ValueError("No numeric features found after cleaning.")

    return X, y, numeric_features


def make_preprocess(numeric_features):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocess = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)],
        remainder="drop"
    )
    return preprocess


def evaluate_model(name, model, preprocess, X, y, cv, scoring):
    clf = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", model),
    ])

    results = cross_validate(
        clf, X, y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        error_score="raise"
    )

    row = {"model": name}
    for metric in scoring.keys():
        vals = results[f"test_{metric}"]
        row[f"{metric}_mean"] = float(vals.mean())
        row[f"{metric}_std"] = float(vals.std())
    return row


def main():
    X, y, numeric_features = load_and_clean_train()
    preprocess = make_preprocess(numeric_features)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    models = [
        ("LogisticRegression",
         LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),

        ("DecisionTree",
         DecisionTreeClassifier(random_state=RANDOM_STATE)),

        ("RandomForest",
         RandomForestClassifier(
             n_estimators=300,
             random_state=RANDOM_STATE,
             n_jobs=-1
         )),
    ]

    rows = []
    for name, model in models:
        print(f"\nRunning 10-fold CV for: {name}")
        rows.append(evaluate_model(name, model, preprocess, X, y, cv, scoring))

    results_df = pd.DataFrame(rows).sort_values(by="f1_mean", ascending=False)

    print("\n=== 10-fold CV Comparison (train only, no leakage) ===")
    with pd.option_context("display.max_columns", None):
        print(results_df.to_string(index=False))

    results_df.to_csv("models_cv_results.csv", index=False)
    print("\nSaved: models_cv_results.csv")


if __name__ == "__main__":
    main()

