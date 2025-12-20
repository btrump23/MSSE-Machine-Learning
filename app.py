# app.py
from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple, Optional

import joblib
import pandas as pd
from flask import Flask, render_template, request, send_file, jsonify

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

APP_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = APP_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "lgbm_pipeline.joblib"

# Columns you should never feed into the model as features
DROP_ALWAYS = {"Label", "Identify", "MD5", "SHA1"}  # keep safe + consistent with your pipeline
# If these exist and are non-numeric, we coerce them; your dataset sometimes includes Name strings
COERCE_NUMERIC_IF_PRESENT = {"Name", "ImportedDlls", "ImportedSymbols", "FormatedTimeDateStamp"}


def load_pipeline():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Prepares X for the model. If Label exists, returns y too.
    - Drops non-feature columns (Label/Identify/MD5/SHA1) if present
    - Coerces any non-numeric columns to numeric (errors->NaN)
    """
    y = None
    if "Label" in df.columns:
        y = pd.to_numeric(df["Label"], errors="coerce")

    X = df.copy()

    # Drop always-drop columns if present
    for col in list(DROP_ALWAYS):
        if col in X.columns:
            X = X.drop(columns=[col])

    # Coerce any object columns (or known columns) to numeric
    non_numeric_cols = [c for c in X.columns if X[c].dtype == "object"]
    to_coerce = sorted(set(non_numeric_cols).union(COERCE_NUMERIC_IF_PRESENT).intersection(set(X.columns)))
    if to_coerce:
        for c in to_coerce:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    return X, y


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_true = y_true.dropna().astype(int)
    y_pred = pd.Series(y_pred).iloc[y_true.index].astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, digits=4),
    }
    return metrics


app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.get("/")
def index():
    return render_template("index.html", model_path=str(MODEL_PATH))


@app.post("/predict-csv")
def predict_csv():
    if "file" not in request.files:
        return render_template("index.html", error="No file part in request", model_path=str(MODEL_PATH)), 400

    f = request.files["file"]
    if not f or f.filename.strip() == "":
        return render_template("index.html", error="No file selected", model_path=str(MODEL_PATH)), 400

    try:
        df = pd.read_csv(f)
    except Exception as e:
        return render_template("index.html", error=f"Could not read CSV: {e}", model_path=str(MODEL_PATH)), 400

    pipe = load_pipeline()
    X, y = prepare_features(df)

    # Predict
    y_pred = pipe.predict(X).astype(int)

    # Attach predictions
    out_df = df.copy()
    out_df["Prediction"] = y_pred

    # Optional probability if model supports it
    if hasattr(pipe, "predict_proba"):
        try:
            proba = pipe.predict_proba(X)[:, 1]
            out_df["Prob_Malware"] = proba
        except Exception:
            pass

    # Metrics only if Label exists
    metrics = None
    if y is not None:
        metrics = compute_metrics(y, y_pred)

    # Build output CSV in-memory
    buf = io.StringIO()
    out_df.to_csv(buf, index=False)
    buf.seek(0)

    return render_template(
        "results.html",
        filename=f.filename,
        rows=len(out_df),
        metrics=metrics,
        download_ready=True,
    ), 200, {
        "X-Download-Filename": "predictions.csv"
    }


@app.get("/download-latest")
def download_latest():
    """
    Simple helper: returns a tiny CSV placeholder if you want a stable download route later.
    You can expand this to store last output in session / disk.
    """
    return jsonify(error="Not implemented: use the browser download prompt from /predict-csv"), 400


if __name__ == "__main__":
    # For local dev
    app.run(host="127.0.0.1", port=5000, debug=True)
