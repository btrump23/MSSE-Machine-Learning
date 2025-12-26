import os
import uuid
import pickle
import datetime as dt

import numpy as np
import pandas as pd

from flask import (
    Flask,
    request,
    render_template,
    jsonify,
    redirect,
    url_for,
    send_file,
)

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# IMPORTANT:
# We will look for model.pkl in the repo root (same folder as app.py).
# That is the file that MUST unpickle into an object with .predict()
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

DOWNLOADS_DIR = os.path.join(BASE_DIR, "downloads")
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Model loading (NEVER crash the server on import)
# ─────────────────────────────────────────────────────────────
MODEL = None
MODEL_ERROR = None


def load_model(model_path: str):
    """
    Load a pickled model. Must return an object with predict().
    If it doesn't, we keep server alive and show a useful error.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    with open(model_path, "rb") as f:
        obj = pickle.load(f)

    # common case: sklearn estimator/pipeline
    if hasattr(obj, "predict"):
        return obj

    # sometimes people pickle a dict like {"model": estimator, ...}
    if isinstance(obj, dict):
        for k in ("model", "estimator", "clf", "pipeline"):
            if k in obj and hasattr(obj[k], "predict"):
                return obj[k]
        raise TypeError(
            f"Loaded a dict from {model_path} but no key contained a .predict() model. Keys={list(obj.keys())}"
        )

    # if it's an ndarray, it's NOT a model (your exact issue)
    if isinstance(obj, np.ndarray):
        raise TypeError(
            f"Loaded object from {model_path} is numpy.ndarray (len={len(obj)}). "
            f"This is NOT a trained model. You must replace model.pkl with the pickled model object "
            f"(e.g., sklearn Pipeline/Classifier or your LinearModelWrapper instance)."
        )

    raise TypeError(
        f"Loaded object from {model_path} is {type(obj)} and has no predict(). "
        f"Replace model.pkl with the pickled model object."
    )


def ensure_model_loaded():
    global MODEL, MODEL_ERROR
    if MODEL is not None:
        return
    if MODEL_ERROR is not None:
        return
    try:
        MODEL = load_model(MODEL_PATH)
    except Exception as e:
        MODEL_ERROR = str(e)


# Load once at startup, but do NOT crash if broken
ensure_model_loaded()

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _has_labels(df: pd.DataFrame) -> bool:
    return "Label" in df.columns


def _coerce_features_numeric(X: pd.DataFrame) -> pd.DataFrame:
    """
    Force all features numeric and remove common junk columns.
    """
    X = X.copy()

    # Drop common junk columns (pandas index saved to CSV, etc.)
    X = X.loc[:, ~X.columns.astype(str).str.match(r"^Unnamed")]
    X.columns = [str(c).strip() for c in X.columns]

    # Force numeric
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # Replace NaNs with 0
    X = X.fillna(0)

    return X



def _normalize_columns_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Label" in df.columns:
        df = df.drop(columns=["Label"])
    return df


def _predict_dataframe(X: pd.DataFrame) -> pd.Series:
    ensure_model_loaded()
    if MODEL_ERROR:
        raise RuntimeError(MODEL_ERROR)

    X = _coerce_features_numeric(X)

    # IMPORTANT: convert to numeric numpy matrix (prevents object dtype sneaking in)
    X_mat = X.to_numpy(dtype=float)

    preds = MODEL.predict(X_mat)
    return pd.Series(np.asarray(preds).ravel(), name="Prediction")


X = _coerce_features_numeric(X)
X_mat = X.to_numpy(dtype=float)

if hasattr(MODEL, "predict_proba"):
    proba = MODEL.predict_proba(X_mat)
    ...
if hasattr(MODEL, "decision_function"):
    scores = MODEL.decision_function(X_mat)
    ...



def _make_confusion_matrix_table(cm: np.ndarray, labels=("0", "1")) -> str:
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


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    ensure_model_loaded()
    return jsonify(
        status="ok" if MODEL_ERROR is None else "error",
        model_path=MODEL_PATH,
        model_error=MODEL_ERROR,
        model_type=str(type(MODEL)) if MODEL is not None else None,
    )


@app.get("/")
def index():
    """
    Landing page. If you have templates/index.html, we'll use it.
    Otherwise we redirect to CSV.
    """
    # If your index.html exists, show it and include links inside the template.
    # But we will still support direct routes below.
    try:
        return render_template(
            "index.html",
            model_error=MODEL_ERROR,
            csv_url=url_for("predict_csv"),
            manual_url=url_for("predict_manual"),
        )
    except Exception:
        return redirect(url_for("predict_csv"))


# ─────────────────────────────────────────────────────────────
# Manual prediction (supports BOTH /predict-manual and /predict_manual)
# ─────────────────────────────────────────────────────────────
@app.get("/predict-manual")
@app.get("/predict_manual")
def predict_manual():
    ensure_model_loaded()

    # Prefilled demo row (edit to match your exact features if needed)
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

    return render_template(
        "predict_manual.html",
        demo=demo,
        result=None,
        error=MODEL_ERROR,
    )


@app.post("/predict-manual")
@app.post("/predict_manual")
def predict_manual_post():
    ensure_model_loaded()
    if MODEL_ERROR:
        return render_template("predict_manual.html", demo=request.form, result=None, error=MODEL_ERROR)

    try:
        row = {}
        for k, v in request.form.items():
            if v is None or str(v).strip() == "":
                row[k] = np.nan
            else:
                # allow int/float inputs
                try:
                    row[k] = int(v)
                except ValueError:
                    row[k] = float(v)

        X = pd.DataFrame([row])
        pred = int(_predict_dataframe(X).iloc[0])
        label = "malware" if pred == 1 else "goodware"

        return render_template("predict_manual.html", demo=row, result=label, error=None)
    except Exception as e:
        return render_template("predict_manual.html", demo=request.form, result=None, error=str(e))


# ─────────────────────────────────────────────────────────────
# CSV prediction + evaluation (supports BOTH /predict-csv and /predict_csv)
# ─────────────────────────────────────────────────────────────
@app.get("/predict-csv")
@app.get("/predict_csv")
def predict_csv():
    ensure_model_loaded()
    return render_template(
        "predict_csv.html",
        metrics=None,
        preview=None,
        download_url=None,
        error=MODEL_ERROR,
        manual_url=url_for("predict_manual"),
    )


@app.post("/predict-csv")
@app.post("/predict_csv")
def predict_csv_post():
    ensure_model_loaded()
    if MODEL_ERROR:
        return render_template("predict_csv.html", metrics=None, preview=None, download_url=None, error=MODEL_ERROR)

    try:
        if "file" not in request.files:
            return render_template("predict_csv.html", metrics=None, preview=None, download_url=None, error="No file uploaded.")

        f = request.files["file"]
        if not f or f.filename == "":
            return render_template("predict_csv.html", metrics=None, preview=None, download_url=None, error="No file selected.")

        df = pd.read_csv(f)

        # Labels if present
        has_labels = _has_labels(df)
        y_true = df["Label"].copy() if has_labels else None

        # Features
        X = _normalize_columns_for_inference(df)

        # Predict
        preds = _predict_dataframe(X)

        # Output DF (keep original columns + Prediction)
        df_out = df.copy()
        df_out["Prediction"] = preds

        # Metrics (only if labels exist)
        metrics = None
        if has_labels:
            acc = float(accuracy_score(y_true, preds))

            scores = _safe_predict_proba_or_score(X)
            auc = None
            if scores is not None:
                try:
                    auc = float(roc_auc_score(y_true, scores))
                except Exception:
                    auc = None

            cm = confusion_matrix(y_true, preds, labels=[0, 1])
            cm_html = _make_confusion_matrix_table(cm, labels=("0", "1"))

            metrics = {"accuracy": acc, "auc": auc, "confusion_matrix_html": cm_html}

        # Save to downloads
        token = uuid.uuid4().hex[:10]
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"predictions_{ts}_{token}.csv"
        out_path = os.path.join(DOWNLOADS_DIR, out_name)
        df_out.to_csv(out_path, index=False)

        # Preview first 10 rows
        preview = df_out.head(10).to_dict(orient="records")
        download_url = url_for("download_file", filename=out_name)

        # You already have templates/results.html in your repo; if you prefer results.html,
        # swap render_template("predict_csv.html", ...) to render_template("results.html", ...)
        return render_template(
            "predict_csv.html",
            metrics=metrics,
            preview=preview,
            download_url=download_url,
            error=None,
            manual_url=url_for("predict_manual"),
        )

    except Exception as e:
        return render_template("predict_csv.html", metrics=None, preview=None, download_url=None, error=str(e))


@app.get("/download/<path:filename>")
def download_file(filename):
    full_path = os.path.join(DOWNLOADS_DIR, filename)
    if not os.path.isfile(full_path):
        return f"File not found: {filename}", 404
    return send_file(full_path, as_attachment=True, download_name=filename)


# Optional: keep old endpoints alive (if you ever linked them)
@app.get("/predict")
def legacy_predict_redirect():
    return redirect(url_for("predict_csv"))


if __name__ == "__main__":
    # Local dev
    app.run(host="0.0.0.0", port=5000, debug=True)
