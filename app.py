import os
import uuid
import pickle
import datetime as dt

import numpy as np
import pandas as pd

from flask import (
    Flask, request, render_template, jsonify,
    send_file, redirect, url_for
)

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# IMPORTANT:
# Your pickled model is a custom wrapper class, so this import MUST exist
# on Render at import-time, before pickle.load() runs.
# (If you remove this, Render will throw: ModuleNotFoundError: model_wrapper)
from model_wrapper import LinearModelWrapper  # noqa: F401


app = Flask(__name__)

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")     # expects model.pkl at repo root
DOWNLOADS_DIR = os.path.join(BASE_DIR, "downloads")
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

# ---------------------------
# Load model once
# ---------------------------
def load_model(path: str):
    with open(path, "rb") as f:
        obj = pickle.load(f)

    # If you accidentally loaded a numpy array or something else, fail loudly.
    if not hasattr(obj, "predict"):
        raise TypeError(
            f"Loaded object from {path} is {type(obj)} and has no predict(). "
            "This usually means you deployed the wrong model.pkl."
        )
    return obj


model = load_model(MODEL_PATH)

# ---------------------------
# Helpers
# ---------------------------
def _safe_predict_proba_or_score(X: pd.DataFrame):
    """
    Returns a float score per row used for ROC AUC.
    Prefer predict_proba (class 1 prob), else decision_function.
    If neither exists, returns None.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        proba = np.asarray(proba)
        # binary: use column for class 1 if available
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.ravel()

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return np.asarray(scores).ravel()

    return None


def _make_confusion_matrix_table(cm: np.ndarray, labels=("0", "1")):
    """Return a simple HTML table string for the confusion matrix."""
    return f"""
    <table border="1" cellpadding="6" cellspacing="0">
      <tr>
        <th></th><th>Pred {labels[0]}</th><th>Pred {labels[1]}</th>
      </tr>
      <tr>
        <th>True {labels[0]}</th><td>{cm[0,0]}</td><td>{cm[0,1]}</td>
      </tr>
      <tr>
        <th>True {labels[1]}</th><td>{cm[1,0]}</td><td>{cm[1,1]}</td>
      </tr>
    </table>
    """


def _normalize_columns_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes label + common junk columns and returns features-only df.
    """
    df = df.copy()

    # drop common label column
    if "Label" in df.columns:
        df = df.drop(columns=["Label"])

    # drop unnamed index columns from CSV exports
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    return df


def _coerce_features_numeric(X: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure every feature column is numeric (float).
    Converts strings -> numbers where possible, otherwise NaN, then fills NaN.
    This prevents errors like:
      - "can't multiply sequence by non-int of type 'float'"
      - object dtype inference issues
    """
    X = X.copy()

    # Ensure stable column order for the model if it exposes feature_names_in_
    if hasattr(model, "feature_names_in_"):
        needed = list(getattr(model, "feature_names_in_"))
        # add missing columns as 0
        for c in needed:
            if c not in X.columns:
                X[c] = 0.0
        # keep only the needed columns (ignore extras)
        X = X[needed]

    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return X


def _predict_dataframe(df_features: pd.DataFrame) -> pd.Series:
    preds = model.predict(df_features)
    preds = np.asarray(preds).ravel()
    return pd.Series(preds, name="Prediction")


# ---------------------------
# Routes
# ---------------------------
@app.get("/health")
def health():
    info = {
        "status": "ok",
        "model_type": str(type(model)),
        "has_predict_proba": bool(hasattr(model, "predict_proba")),
        "has_decision_function": bool(hasattr(model, "decision_function")),
    }
    return jsonify(info)


@app.get("/")
def home():
    return redirect(url_for("predict_csv"))


# ---------- Manual single-row prediction ----------
@app.get("/predict-manual")
def predict_manual():
    # Pre-filled demo row (adjust values as you like)
    demo = {
        "BaseOfCode": 4096,
        "BaseOfData": 8192,
        "Characteristics": 258,
        "DllCharacteristics": 0,
        "Entropy": 6.5,
        "FileAlignment": 512,
        "ImageBase": 4194304,
        "ImportedDlls": 12,
        "ImportedSymbols": 900,
        "Machine": 332,
        "Magic": 267,
        "NumberOfRvaAndSizes": 16,
        "NumberOfSections": 5,
        "NumberOfSymbols": 0,
        "PE_TYPE": 0,
        "PointerToSymbolTable": 0,
        "Size": 123456,
        "SizeOfCode": 40960,
        "SizeOfHeaders": 1024,
        "SizeOfImage": 65536,
        "SizeOfInitializedData": 8192,
        "SizeOfOptionalHeader": 224,
        "SizeOfUninitializedData": 0,
        "TimeDateStamp": 1234567890,
    }
    return render_template("predict_manual.html", demo=demo, result=None, error=None)


@app.post("/predict-manual")
def predict_manual_post():
    try:
        row = {}
        for k, v in request.form.items():
            if v is None or str(v).strip() == "":
                row[k] = np.nan
            else:
                # accept ints/floats
                try:
                    row[k] = int(v)
                except ValueError:
                    row[k] = float(v)

        X = pd.DataFrame([row])
        X = _coerce_features_numeric(X)

        pred = int(_predict_dataframe(X).iloc[0])
        label = "malware" if pred == 1 else "goodware"

        return render_template("predict_manual.html", demo=row, result=label, error=None)

    except Exception as e:
        return render_template("predict_manual.html", demo=dict(request.form), result=None, error=str(e))


# ---------- CSV prediction + evaluation ----------
@app.get("/predict-csv")
def predict_csv():
    return render_template("predict_csv.html", metrics=None, preview=None, download_url=None, error=None)


@app.post("/predict-csv")
def predict_csv_post():
    try:
        if "file" not in request.files:
            return render_template("predict_csv.html", metrics=None, preview=None, download_url=None, error="No file uploaded.")

        f = request.files["file"]
        if f.filename == "":
            return render_template("predict_csv.html", metrics=None, preview=None, download_url=None, error="No file selected.")

        df = pd.read_csv(f)

        has_labels = "Label" in df.columns
        y_true = df["Label"].copy() if has_labels else None

        X = _normalize_columns_for_inference(df)
        X = _coerce_features_numeric(X)

        preds = _predict_dataframe(X)

        # attach predictions
        df_out = df.copy()
        df_out["Prediction"] = preds

        # ---- metrics if labels present ----
        metrics = None
        if has_labels:
            # Coerce y_true to numeric 0/1 safely (some files have "0"/"1" as strings)
            y = pd.to_numeric(y_true, errors="coerce").fillna(0).astype(int)
            p = pd.to_numeric(preds, errors="coerce").fillna(0).astype(int)

            acc = float(accuracy_score(y, p))

            scores = _safe_predict_proba_or_score(X)
            auc = None
            if scores is not None:
                try:
                    auc = float(roc_auc_score(y, scores))
                except Exception:
                    auc = None

            cm = confusion_matrix(y, p, labels=[0, 1])
            cm_html = _make_confusion_matrix_table(cm, labels=("0", "1"))

            metrics = {"accuracy": acc, "auc": auc, "confusion_matrix_html": cm_html}

        # ---- save output CSV to downloads dir + create download link ----
        token = uuid.uuid4().hex[:10]
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"predictions_{ts}_{token}.csv"
        out_path = os.path.join(DOWNLOADS_DIR, out_name)
        df_out.to_csv(out_path, index=False)

        # Preview first 10 rows (nice for marking)
        preview = df_out.head(10).to_dict(orient="records")

        download_url = url_for("download_file", filename=out_name)

        return render_template("predict_csv.html", metrics=metrics, preview=preview, download_url=download_url, error=None)

    except Exception as e:
        return render_template("predict_csv.html", metrics=None, preview=None, download_url=None, error=str(e))


@app.get("/download/<path:filename>")
def download_file(filename):
    full_path = os.path.join(DOWNLOADS_DIR, filename)
    if not os.path.isfile(full_path):
        return f"File not found: {filename}", 404
    return send_file(full_path, as_attachment=True, download_name=filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
