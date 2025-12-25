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


app = Flask(__name__)

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")   # keep as model.pkl like you trained
DOWNLOADS_DIR = os.path.join(BASE_DIR, "downloads")
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

# ---------------------------
# Load model once
# ---------------------------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

if not hasattr(model, "predict"):
    raise TypeError(f"Loaded object from {MODEL_PATH} is {type(model)} and has no predict().")


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
        # binary: use column for class 1 if available
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        # fallback
        return proba.ravel()

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return np.asarray(scores).ravel()

    return None


def _make_confusion_matrix_table(cm: np.ndarray, labels=("0", "1")):
    """Return a simple HTML table string for the confusion matrix."""
    # cm expected [[tn, fp],[fn,tp]]
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
    Ensures we don't include Label in the features,
    and keeps dataframe clean.
    """
    df = df.copy()
    # common label column name in your assignment = Label
    if "Label" in df.columns:
        df = df.drop(columns=["Label"])
    return df


def _predict_dataframe(df_features: pd.DataFrame) -> pd.Series:
    preds = model.predict(df_features)
    return pd.Series(preds, name="Prediction")


# ---------------------------
# Routes
# ---------------------------
@app.get("/health")
def health():
    return jsonify(status="ok", model_type=str(type(model)))


@app.get("/")
def home():
    # simple landing
    return redirect(url_for("predict_csv"))


# ---------- Manual single-row prediction ----------
@app.get("/predict-manual")
def predict_manual():
    # Pre-filled demo row (you can replace values with a known row from dataset)
    demo = {
        # Put a small set OR all 29 features if you want.
        # Minimal approach: show all columns from your training schema if you know them.
        # If you don't know all, keep a "paste full row" style is not ideal for marking.
        # Better: keep the full feature list you used.
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
        "TimeDateStamp": 1234567890
    }
    return render_template("predict_manual.html", demo=demo, result=None, error=None)


@app.post("/predict-manual")
def predict_manual_post():
    try:
        # Build one-row dataframe from form inputs
        row = {}
        for k, v in request.form.items():
            # convert numeric strings safely
            if v is None or v.strip() == "":
                row[k] = np.nan
            else:
                # try int then float
                try:
                    row[k] = int(v)
                except ValueError:
                    row[k] = float(v)

        X = pd.DataFrame([row])
        pred = _predict_dataframe(X).iloc[0]
        label = "malware" if int(pred) == 1 else "goodware"
        return render_template("predict_manual.html", demo=row, result=label, error=None)
    except Exception as e:
        return render_template("predict_manual.html", demo=request.form, result=None, error=str(e))


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
        y_true = None
        if has_labels:
            y_true = df["Label"].copy()

        X = _normalize_columns_for_inference(df)
        preds = _predict_dataframe(X)

        # attach predictions
        df_out = df.copy()
        df_out["Prediction"] = preds

        # ---- metrics if labels present ----
        metrics = None
        if has_labels:
            # accuracy
            acc = float(accuracy_score(y_true, preds))

            # AUC (needs scores/probabilities)
            scores = _safe_predict_proba_or_score(X)
            auc = None
            if scores is not None:
                try:
                    auc = float(roc_auc_score(y_true, scores))
                except Exception:
                    auc = None

            # confusion matrix
            cm = confusion_matrix(y_true, preds, labels=[0, 1])
            cm_html = _make_confusion_matrix_table(cm, labels=("0", "1"))

            metrics = {
                "accuracy": acc,
                "auc": auc,
                "confusion_matrix_html": cm_html
            }

        # ---- save output CSV to downloads dir + create download link ----
        token = uuid.uuid4().hex[:10]
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"predictions_{ts}_{token}.csv"
        out_path = os.path.join(DOWNLOADS_DIR, out_name)
        df_out.to_csv(out_path, index=False)

        # Preview first 10 rows (nice for marking)
        preview = df_out.head(10).to_dict(orient="records")

        download_url = url_for("download_file", filename=out_name)

        return render_template(
            "predict_csv.html",
            metrics=metrics,
            preview=preview,
            download_url=download_url,
            error=None
        )

    except Exception as e:
        return render_template("predict_csv.html", metrics=None, preview=None, download_url=None, error=str(e))


@app.get("/download/<path:filename>")
def download_file(filename):
    # This is what makes the browser download to the user's computer.
    full_path = os.path.join(DOWNLOADS_DIR, filename)
    if not os.path.isfile(full_path):
        return f"File not found: {filename}", 404

    return send_file(full_path, as_attachment=True, download_name=filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
