import os
import io
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

# These must match the columns your pipeline expects (29 features)
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
    # If your model truly expects exactly 29, add the remaining 2 here.
    # Based on your dataset screenshots, these two are often present in processed data:
    "SizeOfInitializedData",  # (duplicate guard removed below if needed)
    "SizeOfUninitializedData" # (duplicate guard removed below if needed)
]

# De-duplicate if you accidentally repeated any names
FEATURE_COLUMNS = list(dict.fromkeys(FEATURE_COLUMNS))

EXPECTED_FEATURES = len(FEATURE_COLUMNS)

# Common non-feature columns to drop from uploaded CSVs
DROP_COLS_IF_PRESENT = {
    "Label", "label", "target", "Target",
    "MD5", "SHA1",
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
def _ensure_dataframe(X):
    """
    Ensure X is a pandas DataFrame with FEATURE_COLUMNS.
    If X is ndarray/list, build DF with those columns.
    """
    if isinstance(X, pd.DataFrame):
        # Ensure correct columns exist and order matches
        df = X.copy()
        for c in FEATURE_COLUMNS:
            if c not in df.columns:
                df[c] = 0
        df = df[FEATURE_COLUMNS]
        return df

    # Otherwise treat as array-like
    arr = np.asarray(X)

    # Force 2D
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    # Pad/truncate to EXPECTED_FEATURES
    if arr.shape[1] < EXPECTED_FEATURES:
        pad = np.zeros((arr.shape[0], EXPECTED_FEATURES - arr.shape[1]), dtype=float)
        arr = np.hstack([arr.astype(float, copy=False), pad])
    elif arr.shape[1] > EXPECTED_FEATURES:
        arr = arr[:, :EXPECTED_FEATURES]

    return pd.DataFrame(arr, columns=FEATURE_COLUMNS)


def _coerce_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert everything to numeric where possible.
    Non-numeric -> NaN -> fill with 0.
    """
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.fillna(0)
    return out


def _predict_proba_or_label(model, X_df: pd.DataFrame):
    """
    Returns (labels, probs)
    probs = probability of class 1 if available, else None
    """
    # sklearn pipeline expects DF with named cols if trained that way
    if hasattr(model, "predict_proba"):
        probs_all = model.predict_proba(X_df)
        # binary: take class-1 probability if exists
        if probs_all is not None and len(probs_all.shape) == 2 and probs_all.shape[1] >= 2:
            probs = probs_all[:, 1]
        else:
            probs = None
        labels = model.predict(X_df)
        return labels, probs

    # fallback: only labels
    labels = model.predict(X_df)
    return labels, None


def _safe_model():
    if MODEL is None:
        raise RuntimeError(f"Model not loaded. {MODEL_ERROR or ''}".strip())
    return MODEL


# -----------------------------
# UI TEMPLATES
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
    body = """
    <div class="card">
      <h2>Malware Detector</h2>
      <ul>
        <li><a href="/predict">Manual prediction (single instance)</a></li>
        <li><a href="/predict-csv">CSV Prediction</a></li>
        <li><a href="/health">Health</a></li>
      </ul>
      <p class="muted">Model expects exactly {{n}} features.</p>
    </div>
    """
    return render_template_string(BASE_HTML, body=render_template_string(body, n=EXPECTED_FEATURES))


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok" if MODEL is not None else "error",
        "model_loaded": MODEL is not None,
        "model_error": MODEL_ERROR,
        "expected_features": EXPECTED_FEATURES,
        "downloads_dir": DOWNLOADS_DIR,
    })


@app.route("/predict", methods=["GET", "POST"])
def manual_predict():
    error = None
    result_html = ""

    if request.method == "POST":
        try:
            model = _safe_model()

            raw = (request.form.get("row") or "").strip()
            if not raw:
                raise ValueError("Please paste a comma-separated row of numeric values.")

            # parse user input -> float list
            parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
            vals = []
            for p in parts:
                vals.append(float(p))

            # IMPORTANT FIX: build DataFrame with named columns
            X_df = _ensure_dataframe(vals)
            X_df = _coerce_numeric_df(X_df)

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
      <p class="muted">Paste a single row of numeric feature values (comma-separated). Model expects {EXPECTED_FEATURES} values. If fewer, we pad zeros; if more, we truncate.</p>
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

            # Read CSV
            df = pd.read_csv(f)

            # Drop label-ish columns if present
            for c in list(df.columns):
                if c in DROP_COLS_IF_PRESENT:
                    df = df.drop(columns=[c])

            # Coerce to numeric (non-numeric -> 0)
            df = _coerce_numeric_df(df)

            # Ensure expected columns exist in correct order
            for c in FEATURE_COLUMNS:
                if c not in df.columns:
                    df[c] = 0

            df = df[FEATURE_COLUMNS]

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
      <p class="muted">Expected features: {EXPECTED_FEATURES}. We drop common label columns (Label/target) and coerce non-numeric values to 0. Then we reorder/pad columns to exactly {EXPECTED_FEATURES} features.</p>
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
    # Local dev
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=True)
