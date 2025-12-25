import os
import io
import uuid
import pickle
import datetime as dt

import pandas as pd
import numpy as np

from flask import (
    Flask, request, render_template, jsonify,
    send_file, redirect, url_for
)

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# -------------------------------------------------
# App
# -------------------------------------------------
app = Flask(__name__)

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
DOWNLOADS_DIR = os.path.join(BASE_DIR, "downloads")
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

# -------------------------------------------------
# Model loading (FIXED – tolerant, no crashing)
# -------------------------------------------------
def load_model(model_path: str):
    with open(model_path, "rb") as f:
        obj = pickle.load(f)

    # Direct model
    if hasattr(obj, "predict"):
        return obj

    # Dict bundle
    if isinstance(obj, dict):
        for v in obj.values():
            if hasattr(v, "predict"):
                return v
        return list(obj.values())[0]

    # List / tuple bundle
    if isinstance(obj, (list, tuple)):
        return obj[0]

    return obj


model = load_model(MODEL_PATH)

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def _safe_predict_proba_or_score(X: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.ndim == 2 and p.shape[1] > 1:
            return p[:, 1]
        return p.ravel()

    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X)).ravel()

    return None


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Label" in df.columns:
        df.drop(columns=["Label"], inplace=True)
    return df


def _confusion_matrix_html(cm):
    return f"""
    <table border="1" cellpadding="6">
        <tr><th></th><th>Pred 0</th><th>Pred 1</th></tr>
        <tr><th>True 0</th><td>{cm[0,0]}</td><td>{cm[0,1]}</td></tr>
        <tr><th>True 1</th><td>{cm[1,0]}</td><td>{cm[1,1]}</td></tr>
    </table>
    """

# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.get("/health")
def health():
    return jsonify(status="ok", model=str(type(model)))


@app.get("/")
def index():
    return redirect(url_for("predict_csv"))


# -------------------------------------------------
# Manual Prediction
# -------------------------------------------------
@app.get("/predict-manual")
def predict_manual():
    demo = {
        "BaseOfCode": 4096,
        "BaseOfData": 8192,
        "Entropy": 6.5,
        "FileAlignment": 512,
        "ImageBase": 4194304,
        "ImportedDlls": 12,
        "NumberOfSections": 5,
        "SizeOfImage": 65536,
        "TimeDateStamp": 1234567890
    }
    return render_template("predict_manual.html", demo=demo, result=None, error=None)


@app.post("/predict-manual")
def predict_manual_post():
    try:
        row = {}
        for k, v in request.form.items():
            row[k] = float(v)

        X = pd.DataFrame([row])
        pred = int(model.predict(X)[0])
        label = "malware" if pred == 1 else "goodware"

        return render_template(
            "predict_manual.html",
            demo=row,
            result=label,
            error=None
        )

    except Exception as e:
        return render_template(
            "predict_manual.html",
            demo=request.form,
            result=None,
            error=str(e)
        )


# -------------------------------------------------
# CSV Prediction + Metrics
# -------------------------------------------------
@app.get("/predict-csv")
def predict_csv():
    return render_template(
        "predict_csv.html",
        metrics=None,
        preview=None,
        download_url=None,
        error=None
    )


@app.post("/predict-csv")
def predict_csv_post():
    try:
        file = request.files.get("file")
        if not file or file.filename == "":
            raise ValueError("No file uploaded")

        df = pd.read_csv(file)

        has_labels = "Label" in df.columns
        y_true = df["Label"] if has_labels else None

        X = _normalize_columns(df)
        preds = model.predict(X)

        df_out = df.copy()
        df_out["Prediction"] = preds

        metrics = None
        if has_labels:
            acc = accuracy_score(y_true, preds)
            scores = _safe_predict_proba_or_score(X)
            auc = roc_auc_score(y_true, scores) if scores is not None else None
            cm = confusion_matrix(y_true, preds)

            metrics = {
                "accuracy": acc,
                "auc": auc,
                "confusion_matrix": _confusion_matrix_html(cm)
            }

        fname = f"predictions_{dt.datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}.csv"
        out_path = os.path.join(DOWNLOADS_DIR, fname)
        df_out.to_csv(out_path, index=False)

        return render_template(
            "predict_csv.html",
            metrics=metrics,
            preview=df_out.head(10).to_dict(orient="records"),
            download_url=url_for("download_file", filename=fname),
            error=None
        )

    except Exception as e:
        return render_template(
            "predict_csv.html",
            metrics=None,
            preview=None,
            download_url=None,
            error=str(e)
        )


# -------------------------------------------------
# Download
# -------------------------------------------------
@app.get("/download/<path:filename>")
def download_file(filename):
    path = os.path.join(DOWNLOADS_DIR, filename)
    if not os.path.exists(path):
        return "File not found", 404
    return send_file(path, as_attachment=True)


# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
