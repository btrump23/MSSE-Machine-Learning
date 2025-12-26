import os
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string, send_from_directory

# Try joblib first (common for sklearn pipelines)
try:
    import joblib
except Exception:
    joblib = None

import pickle

app = Flask(__name__)

# -----------------------------
# CONFIG
# -----------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.pkl")

DOWNLOADS_DIR = os.path.join(HERE, "downloads")
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

# Your model’s “manual input” is 27 values (as your UI shows)
# These are the 27 numeric/primary columns (from your dataset header)
FEATURE_COLUMNS = [
    "BaseOfCode",
    "BaseOfData",
    "Characteristics",
    "DllCharacteristics",
    "Entropy",
    "FileAlignment",
    "FormatedTimeDateStamp",
    "Identify",
    "ImageBase",
    "ImportedDlls",
    "ImportedSymbols",
    "Machine",
    "Magic",
    "Name",
    "NumberOfRvaAndSizes",
    "NumberOfSections",
    "NumberOfSymbols",
    "PE_TYPE",
    "PointerToSymbolTable",
    "Size",
    "SizeOfCode",
    "SizeOfHeaders",
    "SizeOfImage",
    "SizeOfInitializedData",
    "SizeOfOptionalHeader",
    "SizeOfUninitializedData",
    "TimeDateStamp",
]

EXPECTED_FEATURES = len(FEATURE_COLUMNS)  # 27

# The pipeline REQUIRES these columns to exist (your error confirms this)
REQUIRED_ID_COLUMNS = ["MD5", "SHA1"]

# Common non-feature columns to drop (DO NOT DROP MD5/SHA1)
DROP_COLS_IF_PRESENT = {
    "Label", "label", "target", "Target",
}

MODEL = None
MODEL_ERROR = None


# -----------------------------
# MODEL LOAD
# -----------------------------
def load_model():
    global MODEL, MODEL_ERROR
    MODEL = None
    MODEL_ERROR = None

    if not os.path.exists(MODEL_PATH):
        MODEL_ERROR = f"model.pkl not found at: {MODEL_PATH}"
        return

    try:
        if joblib is not None:
            MODEL = joblib.load(MODEL_PATH)
        else:
            with open(MODEL_PATH, "rb") as f:
                MODEL = pickle.load(f)
    except Exception as e:
        MODEL = None
        MODEL_ERROR = f"{type(e).__name__}: {e}"


load_model()


# -----------------------------
# HELPERS
# -----------------------------
def _safe_model():
    if MODEL is None:
        raise RuntimeError(f"Model not loaded. {MODEL_ERROR or ''}".strip())
    return MODEL


def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure MD5 and SHA1 exist (pipeline demands them).
    Put blank strings if missing.
    """
    out = df.copy()
    for c in REQUIRED_ID_COLUMNS:
        if c not in out.columns:
            out[c] = ""
        else:
            # force string type (safe for text/vectorizers)
            out[c] = out[c].astype(str)
    return out


def _coerce_numeric_columns_only(df: pd.DataFrame, numeric_cols) -> pd.DataFrame:
    """
    Only coerce our numeric FEATURE_COLUMNS.
    Leave MD5/SHA1 as strings.
    """
    out = df.copy()
    for c in numeric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out[numeric_cols] = out[numeric_cols].fillna(0)
    return out


def _build_df_from_manual_values(values):
    """
    Manual form provides a list of floats.
    We build a DF with FEATURE_COLUMNS, pad/truncate to 27,
    THEN add MD5/SHA1 as blank strings.
    """
    arr = np.asarray(values, dtype=float).reshape(1, -1)

    if arr.shape[1] < EXPECTED_FEATURES:
        pad = np.zeros((1, EXPECTED_FEATURES - arr.shape[1]), dtype=float)
        arr = np.hstack([arr, pad])
    elif arr.shape[1] > EXPECTED_FEATURES:
        arr = arr[:, :EXPECTED_FEATURES]

    df = pd.DataFrame(arr, columns=FEATURE_COLUMNS)

    # Coerce numerics (already float, but safe)
    df = _coerce_numeric_columns_only(df, FEATURE_COLUMNS)

    # Add required ID columns for pipeline
    df = _ensure_required_columns(df)

    return df


def _predict_proba_or_label(model, X_df: pd.DataFrame):
    """
    Returns (labels, probs)
    probs = probability of class 1 if available, else None
    """
    if hasattr(model, "predict_proba"):
        probs_all = model.predict_proba(X_df)
        probs = None
        if probs_all is not None and len(probs_all.shape) == 2 and probs_all.shape[1] >= 2:
            probs = probs_all[:, 1]
        labels = model.predict(X_df)
        return labels, probs

    labels = model.predict(X_df)
    return labels, None


# -----------------------------
# UI
# -----------------------------
BASE_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Malware Detector</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 40px; }
    .container { max-width: 980px; margin: 0 auto; }
    h1 { font-size: 64px; margin-bottom: 8px; }
    .nav a { margin-right: 18px; font-size: 20px; }
    .card { border: 1px solid #ddd; border-radius: 12px; padding: 22px; margin-top: 18px; }
    .error { color: #b00000; font-weight: 700; font-size: 22px; }
    .muted { color: #666; }
    input[type=text] { width: 70%; padding: 12px; font-size: 16px; border-radius: 10px; border: 1px solid #ccc; }
    input[type=file] { font-size: 16px; }
    button { padding: 12px 18px; font-size: 16px; font-weight: 700; border: none; border-radius: 12px; cursor: pointer; }
    button.primary { background: #0e1a2b; color: #fff; }
    pre { white-space: pre-wrap; background: #f7f7f7; padding: 12px; border-radius: 10px; }
  </style>
</head>
<body>
<div class="container">
  <h1>Malware Detector</h1>
  <div class="nav">
    <a href="/predict">Manual prediction (single instance)</a> |
    <a href="/predict-csv">CSV Prediction</a> |
    <a href="/health">Health</a>
  </div>
  <hr>
  {{ body|safe }}
</div>
</body>
</html>
"""


# -----------------------------
# ROUTES
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    body = f"""
    <div class="card">
      <h2>Malware Detector</h2>
      <ul>
        <li><a href="/predict">Manual prediction (single instance)</a></li>
        <li><a href="/predict-csv">CSV Prediction</a></li>
        <li><a href="/health">Health</a></li>
      </ul>
      <p class="muted">Manual expects exactly {EXPECTED_FEATURES} numeric values. The pipeline also requires MD5 and SHA1 columns (auto-added).</p>
    </div>
    """
    return render_template_string(BASE_HTML, body=body)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok" if MODEL is not None else "error",
        "model_loaded": MODEL is not None,
        "model_error": MODEL_ERROR,
        "expected_manual_features": EXPECTED_FEATURES,
        "required_id_columns": REQUIRED_ID_COLUMNS,
        "downloads_dir": DOWNLOADS_DIR,
    })


@app.route("/predict", methods=["GET", "POST"])
def manual_predict():
    result_html = ""

    if request.method == "POST":
        try:
            model = _safe_model()

            raw = (request.form.get("row") or "").strip()
            if not raw:
                raise ValueError("Please paste a comma-separated row of numeric values.")

            parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
            vals = [float(p) for p in parts]

            X_df = _build_df_from_manual_values(vals)

            labels, probs = _predict_proba_or_label(model, X_df)

            label = labels[0] if len(labels) else None
            prob = probs[0] if probs is not None and len(probs) else None

            result_html = "<div class='card'><h3>Result</h3>"
            result_html += f"<p><b>Prediction:</b> {label}</p>"
            if prob is not None:
                result_html += f"<p><b>Probability (class 1):</b> {float(prob):.6f}</p>"
            result_html += "</div>"

        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            tb = traceback.format_exc()
            result_html = f"<div class='card'><div class='error'>Error: {error}</div><pre>{tb}</pre></div>"

    body = f"""
    <div class="card">
      <h2>Manual prediction</h2>
      <p class="muted">
        Paste a single row of numeric feature values (comma-separated). Model expects {EXPECTED_FEATURES} values.
        If fewer, we pad zeros; if more, we truncate. We also auto-add MD5 and SHA1 as blank strings.
      </p>
      <form method="POST">
        <input type="text" name="row" placeholder="e.g. 0,0,0,...">
        <button class="primary" type="submit">Predict</button>
      </form>
    </div>
    {result_html}
    """
    return render_template_string(BASE_HTML, body=body)


@app.route("/predict-csv", methods=["GET", "POST"])
def predict_csv():
    result_html = ""

    if request.method == "POST":
        try:
            model = _safe_model()

            if "file" not in request.files:
                raise ValueError("No file uploaded.")
            f = request.files["file"]
            if not f or f.filename.strip() == "":
                raise ValueError("No file selected.")

            df = pd.read_csv(f)

            # Drop label column if present (keep MD5/SHA1!)
            for c in list(df.columns):
                if c in DROP_COLS_IF_PRESENT:
                    df = df.drop(columns=[c])

            # Ensure numeric feature columns exist and are in correct order
            for c in FEATURE_COLUMNS:
                if c not in df.columns:
                    df[c] = 0

            df = df[FEATURE_COLUMNS + [c for c in REQUIRED_ID_COLUMNS if c in df.columns] + [c for c in REQUIRED_ID_COLUMNS if c not in df.columns]]

            # Coerce only numeric feature columns
            df = _coerce_numeric_columns_only(df, FEATURE_COLUMNS)

            # Ensure MD5/SHA1 exist (blank if missing)
            df = _ensure_required_columns(df)

            labels, probs = _predict_proba_or_label(model, df)

            out = df.copy()
            out["prediction"] = labels
            if probs is not None:
                out["prob_class1"] = probs

            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            out_name = f"predictions_{ts}.csv"
            out_path = os.path.join(DOWNLOADS_DIR, out_name)
            out.to_csv(out_path, index=False)

            result_html = f"""
            <div class="card">
              <h3>Done</h3>
              <p>Wrote output file:</p>
              <p><a href="/downloads/{out_name}">{out_name}</a></p>
            </div>
            """

        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            tb = traceback.format_exc()
            result_html = f"<div class='card'><div class='error'>Error: {err}</div><pre>{tb}</pre></div>"

    body = f"""
    <div class="card">
      <h2>Upload CSV</h2>
      <p class="muted">
        Expected numeric features: {EXPECTED_FEATURES}. We keep MD5 and SHA1 (pipeline requires them).
        Label/target columns are dropped if present. Non-numeric in numeric columns becomes 0.
      </p>
      <form method="POST" enctype="multipart/form-data">
        <input type="file" name="file" accept=".csv">
        <button class="primary" type="submit">Run Predictions</button>
      </form>
    </div>
    {result_html}
    """
    return render_template_string(BASE_HTML, body=body)


@app.route("/downloads/<path:filename>", methods=["GET"])
def download_file(filename):
    return send_from_directory(DOWNLOADS_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=True)
