import os
import uuid
import pickle
import datetime as dt

import numpy as np
import pandas as pd

from flask import Flask, request, render_template, redirect, url_for, send_file
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# ---------------------------
# App setup
# ---------------------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
DOWNLOADS_DIR = os.path.join(BASE_DIR, "downloads")
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

# ---------------------------
# Load model ONCE
# ---------------------------
with open(MODEL_PATH, "rb") as f:
    MODEL = pickle.load(f)

if not hasattr(MODEL, "predict"):
    raise TypeError("Loaded model does not implement predict()")

# ---------------------------
# Helpers
# ---------------------------
def _coerce_features_numeric(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    # Drop junk index columns
    X = X.loc[:, ~X.columns.astype(str).str.match(r"^Unnamed")]

    # Force numeric
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    return X.fillna(0)


def _predict_dataframe(X: pd.DataFrame) -> pd.Series:
    X = _coerce_features_numeric(X)
    preds = MODEL.predict(X.to_numpy(dtype=float))
    return pd.Series(preds, name="Prediction")


def _safe_auc(X, y_true):
    if hasattr(MODEL, "predict_proba"):
        probs = MODEL.predict_proba(X.to_numpy(dtype=float))[:, 1]
        return roc_auc_score(y_true, probs)
    return None


def _confusion_html(cm):
    return f"""
    <table border="1" cellpadding="6">
      <tr><th></th><th>Pred 0</th><th>Pred 1</th></tr>
      <tr><th>True 0</th><td>{cm[0,0]}</td><td>{cm[0,1]}</td></tr>
      <tr><th>True 1</th><td>{cm[1,0]}</td><td>{cm[1,1]}</td></tr>
    </table>
    """

# ---------------------------
# Routes
# ---------------------------
@app.get("/")
def home():
    return redirect(url_for("predict_csv"))

@app.get("/predict-csv")
def predict_csv():
    return render_template("predict_csv.html", metrics=None, preview=None, download_url=None, error=None)

@app.post("/predict-csv")
def predict_csv_post():
    try:
        f = request.files.get("file")
        if not f or f.filename == "":
            raise ValueError("No CSV file uploaded")

        df = pd.read_csv(f)
        has_labels = "Label" in df.columns

        y_true = df["Label"] if has_labels else None
        X = df.drop(columns=["Label"]) if has_labels else df

        preds = _predict_dataframe(X)
        df_out = df.copy()
        df_out["Prediction"] = preds

        metrics = None
        if has_labels:
            acc = accuracy_score(y_true, preds)
            auc = _safe_auc(X, y_true)
            cm = confusion_matrix(y_true, preds, labels=[0, 1])

            metrics = {
                "accuracy": round(acc, 4),
                "auc": round(auc, 4) if auc is not None else None,
                "confusion_matrix_html": _confusion_html(cm)
            }

        fname = f"predictions_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.csv"
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
        return render_template("predict_csv.html", metrics=None, preview=None, download_url=None, error=str(e))

@app.get("/predict-manual")
def predict_manual():
    return render_template("predict_manual.html", result=None, error=None)

@app.post("/predict-manual")
def predict_manual_post():
    try:
        row = {k: float(v) for k, v in request.form.items()}
        X = pd.DataFrame([row])
        pred = int(_predict_dataframe(X).iloc[0])
        label = "malware" if pred == 1 else "goodware"
        return render_template("predict_manual.html", result=label, error=None)
    except Exception as e:
        return render_template("predict_manual.html", result=None, error=str(e))

@app.get("/download/<filename>")
def download_file(filename):
    return send_file(os.path.join(DOWNLOADS_DIR, filename), as_attachment=True)

# ---------------------------
# Local run
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
