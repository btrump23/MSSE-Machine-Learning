# train_save_pipeline.py
import os
import joblib
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from lightgbm import LGBMClassifier

TRAIN_PATH = "data/processed/train.csv"
ARTIFACT_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "lgbm_pipeline.joblib")

TARGET_COL = "Label"
DROP_ALL_MISSING = ["FormatedTimeDateStamp", "ImportedDlls", "ImportedSymbols", "SHA1"]

def load_train():
    df = pd.read_csv(TRAIN_PATH)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' not found in {TRAIN_PATH}")

    # Drop all-missing columns (known from your earlier checks)
    cols_to_drop = [c for c in DROP_ALL_MISSING if c in df.columns]
    if cols_to_drop:
        print(f"Dropping all-missing columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    # Drop obvious non-feature columns if present (safe)
    for c in ["MD5", "Identify"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL])

    # Ensure numeric only (your dataset appears numeric)
    X = X.apply(pd.to_numeric, errors="coerce")

    print(f"Train rows: {len(df)} | Features used: {X.shape[1]}")
    return X, y, list(X.columns)

def build_pipeline():
    # NOTE: scaling is not needed for tree models; keeping it simple + robust:
    # just impute + model.
    imputer = SimpleImputer(strategy="median")

    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline([
        ("imputer", imputer),
        ("model", model)
    ])
    return pipe

def main():
    X, y, feature_names = load_train()
    pipe = build_pipeline()

    # Optional: quick CV sanity check (train only, no leakage)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1"}

    print("\nRunning 10-fold CV (train only)...")
    cv_results = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    for k in scoring.keys():
        print(f"{k:>9}: mean={cv_results['test_'+k].mean():.4f} std={cv_results['test_'+k].std():.4f}")

    # Fit final model on ALL training data (still no hold-out touched)
    print("\nTraining final model on full training set...")
    pipe.fit(X, y)

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # Save pipeline + metadata together
    bundle = {
        "pipeline": pipe,
        "feature_names": feature_names,
        "dropped_all_missing": DROP_ALL_MISSING,
        "target_col": TARGET_COL
    }

    joblib.dump(bundle, MODEL_PATH)
    print(f"\n✅ Saved packaged pipeline to: {MODEL_PATH}")

if __name__ == "__main__":
    main()

