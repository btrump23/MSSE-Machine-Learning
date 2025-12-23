"""
predict.py

Batch prediction + optional metrics for the malware detector project.

Exports for app.py:
- load_pipeline()
- prepare_features(df)
- predict_dataframe(df, pipe=None)
- compute_metrics_if_available(y_true, y_pred)

CLI:
  python predict.py path/to/input.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd


# ---------------------------
# Config
# ---------------------------

DEFAULT_MODEL_PATH = Path("artifacts") / "lgbm_pipeline.joblib"
DEFAULT_OUTPUT_DIR = Path("artifacts")
DEFAULT_OUTPUT_NAME = "test_predictions.csv"

# Common label column names we may see in input CSVs
LABEL_CANDIDATES = ("Label", "label", "y", "target", "class")


# ---------------------------
# Core helpers
# ---------------------------

def load_pipeline(model_path: Optional[str | Path] = None):
    """
    Load the trained sklearn-compatible pipeline from joblib.
    """
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Model pipeline not found at: {path.resolve()}\n"
            f"Expected a joblib file. If you used a different name, pass --model PATH."
        )
    return joblib.load(path)


def _extract_label(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[str]]:
    """
    If a label column exists, return (X_df, y_series, label_col_name).
    Otherwise return (df, None, None).
    """
    for col in LABEL_CANDIDATES:
        if col in df.columns:
            y = df[col]
            X = df.drop(columns=[col])
            return X, y, col
    return df, None, None


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for inference:
    - Drop label column if present (handled elsewhere)
    - Coerce non-numeric columns to numeric (errors -> NaN) so numeric pipelines won't crash
    - Keep column names (helps with sklearn warnings / consistency)
    """
    X = df.copy()

    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        # Coerce to numeric. Non-parsable values become NaN and are handled by the pipeline's imputer.
        for c in non_numeric:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    return X


def predict_dataframe(df: pd.DataFrame, pipe=None) -> pd.DataFrame:
    """
    Run predictions on a dataframe.
    Returns a new dataframe that includes:
    - prediction (0/1)
    - (optional) proba_1 if pipeline supports predict_proba
    """
    if pipe is None:
        pipe = load_pipeline()

    X, y_true, label_col = _extract_label(df)
    X = prepare_features(X)

    # Predict
    y_pred = pipe.predict(X)
    # Ensure int for clean output
    y_pred = np.asarray(y_pred).astype(int)

    out = df.copy()
    out["prediction"] = y_pred

    # Optional probability
    if hasattr(pipe, "predict_proba"):
        try:
            proba = pipe.predict_proba(X)
            if proba is not None and proba.shape[1] >= 2:
                out["proba_1"] = proba[:, 1]
        except Exception:
            # If predict_proba isn't supported by the final estimator, ignore
            pass

    return out


def compute_metrics_if_available(y_true, y_pred) -> Optional[dict]:
    """
    Compute basic classification metrics if labels are provided.
    """
    if y_true is None:
        return None

    # Lazy import so predict can run even if sklearn isn't installed in minimal environments
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report,
    )

    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true_arr, y_pred_arr).tolist(),
        "classification_report": classification_report(y_true_arr, y_pred_arr, digits=4),
    }
    return metrics


# ---------------------------
# CLI
# ---------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run malware detector predictions on a CSV.")
    parser.add_argument("csv_path", type=str, help="Path to input CSV (may include Label).")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to joblib pipeline.")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUTPUT_DIR / DEFAULT_OUTPUT_NAME),
                        help="Path to output predictions CSV.")
    args = parser.parse_args(argv)

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"Input file not found: {csv_path}")
        return 1

    print(f"Loading data: {csv_path}")
    df = pd.read_csv(csv_path)

    print("Loading model...")
    pipe = load_pipeline(args.model)

    print("Preparing features...")
    X, y_true, label_col = _extract_label(df)
    X_prepped = prepare_features(X)

    print("Running predictions...")
    y_pred = pipe.predict(X_prepped)
    y_pred = np.asarray(y_pred).astype(int)

    out_df = df.copy()
    out_df["prediction"] = y_pred

    # Optional probability
    if hasattr(pipe, "predict_proba"):
        try:
            proba = pipe.predict_proba(X_prepped)
            if proba is not None and proba.shape[1] >= 2:
                out_df["proba_1"] = proba[:, 1]
        except Exception:
            pass

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved predictions to: {out_path.resolve()}")

    # Metrics if label present
    if y_true is not None:
        metrics = compute_metrics_if_available(y_true, y_pred)
        if metrics:
            print("\nMetrics (Label present in input)")
            print(f"accuracy : {metrics['accuracy']:.4f}")
            print(f"precision: {metrics['precision']:.4f}")
            print(f"recall   : {metrics['recall']:.4f}")
            print(f"f1       : {metrics['f1']:.4f}")
            print("\nConfusion Matrix (rows=true, cols=pred):")
            print(np.array(metrics["confusion_matrix"]))
            print("\nClassification Report:")
            print(metrics["classification_report"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
