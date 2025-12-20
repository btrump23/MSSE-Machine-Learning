import argparse
from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Silence harmless LightGBM feature-name warning
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)

# ----------------------------
# Config
# ----------------------------
ARTIFACTS_DIR = Path("artifacts")
DEFAULT_MODEL_PATH = ARTIFACTS_DIR / "lgbm_pipeline.joblib"
FEATURE_NAMES_PATH = ARTIFACTS_DIR / "feature_names.joblib"
DEFAULT_OUTPUT_PATH = ARTIFACTS_DIR / "test_predictions.csv"


# ----------------------------
# Load model / pipeline
# ----------------------------
def load_pipeline(model_path: Path = DEFAULT_MODEL_PATH):
    """
    Load a saved sklearn Pipeline.
    Supports either:
      - pipeline saved directly
      - dict wrapper { "pipeline": pipe, ... }
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    obj = joblib.load(model_path)

    # Unwrap if dict
    if isinstance(obj, dict):
        for key in ("pipeline", "model", "estimator", "clf"):
            if key in obj:
                obj = obj[key]
                break

    if not hasattr(obj, "predict"):
        raise TypeError(
            f"Loaded object is not a model/pipeline with predict(): {type(obj)}"
        )

    return obj


# ----------------------------
# Prepare features
# ----------------------------
def prepare_features(df: pd.DataFrame):
    """
    Prepare X and y (if present).
    - Drops non-feature columns
    - Coerces any remaining non-numeric values to NaN
    - Reindexes to training feature order
    """
    df = df.copy()

    # Extract label if present
    y = None
    if "Label" in df.columns:
        y = df["Label"].astype(int)
        df = df.drop(columns=["Label"])

    # Drop known non-feature/string columns
    drop_cols = [
        "Identify",
        "Name",
        "MD5",
        "SHA1",
        "FormatedTimeDateStamp",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Coerce remaining object columns to numeric
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Align to training feature list
    if FEATURE_NAMES_PATH.exists():
        feature_names = joblib.load(FEATURE_NAMES_PATH)

        # Ensure missing columns exist
        for c in feature_names:
            if c not in df.columns:
                df[c] = np.nan

        # Drop extras + enforce order
        df = df[feature_names]

    return df, y


# ----------------------------
# Metrics
# ----------------------------
def compute_metrics_if_available(y_true, y_pred):
    if y_true is None:
        return None

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, digits=4),
    }


# ----------------------------
# Flask helper
# ----------------------------
def predict_dataframe(pipe, df: pd.DataFrame):
    """
    Used by Flask routes.
    Returns (output_df, metrics_or_None)
    """
    X, y_true = prepare_features(df)
    y_pred = pipe.predict(X).astype(int)

    out_df = df.copy()
    out_df["Prediction"] = y_pred

    metrics = compute_metrics_if_available(y_true, y_pred)
    return out_df, metrics


# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Run predictions using trained model pipeline")
    parser.add_argument("csv", help="Path to input CSV file")
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to joblib model",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output CSV path",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)
    model_path = Path(args.model)
    out_path = Path(args.out)

    if not csv_path.exists():
        print(f"Input CSV not found: {csv_path}", file=sys.stderr)
        return 1

    print(f"Loading data: {csv_path}")
    df = pd.read_csv(csv_path)

    print("Loading model...")
    pipe = load_pipeline(model_path)

    print("Preparing features...")
    X, y_true = prepare_features(df)

    print("Running predictions...")
    y_pred = pipe.predict(X).astype(int)

    out_df = df.copy()
    out_df["Prediction"] = y_pred

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved predictions to: {out_path.resolve()}")

    metrics = compute_metrics_if_available(y_true, y_pred)
    if metrics:
        print("\nMetrics (Label present in input)")
        print(f"accuracy : {metrics['accuracy']:.4f}")
        print(f"precision: {metrics['precision']:.4f}")
        print(f"recall   : {metrics['recall']:.4f}")
        print(f"f1       : {metrics['f1']:.4f}")

        print("\nConfusion Matrix (rows=true, cols=pred):")
        print(metrics["confusion_matrix"])

        print("\nClassification Report:")
        print(metrics["classification_report"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
