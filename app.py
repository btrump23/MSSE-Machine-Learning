import os
import io
import csv
import zipfile
import tempfile
import traceback
from datetime import datetime
from urllib.request import urlretrieve

import pandas as pd
import joblib
from flask import Flask, request, jsonify, send_file, Response, render_template_string

# =========================
# CONFIG
# =========================
APP_VERSION = "prod-modelzip-v3-form-download"

# GitHub Release asset (must be public or accessible)
MODEL_ZIP_URL = "https://github.com/btrump23/MSSE-Machine-Learning/releases/download/model-v1/model.zip"

# Where we cache the downloaded zip + extracted model on the server
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "model_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

CACHED_ZIP_PATH = os.path.join(CACHE_DIR, "model.zip")
EXTRACT_DIR = os.path.join(CACHE_DIR, "model")
MODEL_PKL_PATH = os.path.join(EXTRACT_DIR, "model.pkl")

# If you run locally and want to use a local zip instead of downloading:
LOCAL_MODEL_ZIP = os.path.join(BASE_DIR, "model.zip")     # optional
LOCAL_MODEL_PKL = os.path.join(BASE_DIR, "model.pkl")     # optional

# =========================
# HTML UI
# =========================
HTML_PAGE = """
<!doctype html>
<html>
<head>
  <title>MSSE ML Predictor</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; }
    h1 { margin-bottom: 10px; }
    .card { border: 1px solid #ddd; border-radius: 8px; padding: 18px; }
    .status { margin-top: 18px; padding: 12px; background:#f7f7f7; border-radius: 6px; }
    button { padding: 10px 14px; cursor: pointer; }
  </style>
  <script>
    function showRunning() {
      const el = document.getElementById("statusText");
      el.textContent = "Running… (may take ~10-60s if Render is waking)";
      return true;
    }
  </script>
</head>
<body>
  <h1>Upload CSV</h1>
  <div class="card">
    <form method="post" enctype="multipart/form-data" action="/predict_csv" onsubmit="return showRunning()">
      <input type="file" name="file" required />
      <button type="submit">Predict (Download CSV)</button>
    </form>

    <div class="status">
      <h2>Status</h2>
      <div id="statusText">Waiting…</div>
    </div>
  </div>
</body>
</html>
"""

# =========================
# APP
# =========================
app = Flask(__name__)


def _json_error(message: str, status_code: int = 500, trace: str | None = None):
    payload = {"status": "error", "message": message}
    if trace:
        payload["trace"] = trace
    return jsonify(payload), status_code


def ensure_model_bundle():
    """
    Ensures model.pkl is available at MODEL_PKL_PATH by:
      1) Using already extracted file if present
      2) Else using local model.zip if exists
      3) Else downloading model.zip from GitHub release
      4) Extracting model.pkl
    """
    # Already extracted?
    if os.path.exists(MODEL_PKL_PATH):
        return MODEL_PKL_PATH

    os.makedirs(EXTRACT_DIR, exist_ok=True)

    # Prefer local model.pkl if present
    if os.path.exists(LOCAL_MODEL_PKL):
        # Copy into extract dir so the rest of the app is consistent
        joblib.dump(joblib.load(LOCAL_MODEL_PKL), MODEL_PKL_PATH)
        return MODEL_PKL_PATH

    # Prefer local zip if present
    zip_path = None
    if os.path.exists(LOCAL_MODEL_ZIP):
        zip_path = LOCAL_MODEL_ZIP
    else:
        # Download once (cache)
        if not os.path.exists(CACHED_ZIP_PATH):
            urlretrieve(MODEL_ZIP_URL, CACHED_ZIP_PATH)
        zip_path = CACHED_ZIP_PATH

    # Extract
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(EXTRACT_DIR)

    if not os.path.exists(MODEL_PKL_PATH):
        raise FileNotFoundError(f"Model file not found after extract: {MODEL_PKL_PATH}")

    return MODEL_PKL_PATH


def load_model():
    """
    Supports:
      - bundle dict with keys incl 'pipeline' (your case)
      - direct sklearn estimator/pipeline
    Returns:
      (predictor, bundle_dict_or_none)
    """
    pkl_path = ensure_model_bundle()
    obj = joblib.load(pkl_path)

    bundle = None
    predictor = obj

    # Your trained artifact is a dict bundle
    if isinstance(obj, dict):
        bundle = obj
        if "pipeline" in obj:
            predictor = obj["pipeline"]
        elif "model" in obj:
            predictor = obj["model"]
        else:
            raise TypeError(f"Loaded dict bundle has no 'pipeline' or 'model' key. Keys={list(obj.keys())}")

    if not hasattr(predictor, "predict"):
        raise TypeError("Loaded object is not a valid sklearn predictor (no predict()).")

    return predictor, bundle


# Load once at startup (Render + local). If it fails, endpoints will show error.
try:
    MODEL, MODEL_BUNDLE = load_model()
except Exception:
    MODEL = None
    MODEL_BUNDLE = None
    STARTUP_ERROR = traceback.format_exc()
else:
    STARTUP_ERROR = None


def _predict_from_df(df: pd.DataFrame):
    if MODEL is None:
        raise RuntimeError(f"Model not loaded.\n{STARTUP_ERROR or ''}")

    # If bundle contains feature_names, align/order columns safely
    if isinstance(MODEL_BUNDLE, dict) and MODEL_BUNDLE.get("feature_names"):
        feature_names = list(MODEL_BUNDLE["feature_names"])
        # Only keep needed columns; add missing with 0
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_names]

    preds = MODEL.predict(df)
    return preds


@app.get("/version")
def version():
    return jsonify({"version": APP_VERSION})


@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_PAGE)


@app.route("/predict_csv", methods=["GET", "POST"])
def predict_csv():
    # GET: show UI
    if request.method == "GET":
        return render_template_string(HTML_PAGE)

    # POST: accept CSV upload and return a downloadable CSV
    try:
        if "file" not in request.files:
            return _json_error("No file uploaded. Use form-data key 'file'.", 400)

        f = request.files["file"]
        if not f.filename:
            return _json_error("Empty filename.", 400)

        # Read CSV into DataFrame
        df = pd.read_csv(f)

        # Predict
        preds = _predict_from_df(df)

        # Output: append predictions column
        out_df = df.copy()
        out_df["prediction"] = list(preds)

        # Create CSV in-memory
        buf = io.StringIO()
        out_df.to_csv(buf, index=False)
        buf.seek(0)

        # Force a browser download
        filename = "predictions.csv"
        return Response(
            buf.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    except Exception as e:
        return _json_error(str(e), 500, traceback.format_exc())


@app.route("/predict_csv_json", methods=["POST"])
def predict_csv_json():
    """
    Optional: POST CSV and return JSON predictions (useful for debugging).
    """
    try:
        if "file" not in request.files:
            return _json_error("No file uploaded. Use form-data key 'file'.", 400)

        f = request.files["file"]
        df = pd.read_csv(f)
        preds = _predict_from_df(df)
        return jsonify({"status": "ok", "predictions": [int(x) for x in preds]})
    except Exception as e:
        return _json_error(str(e), 500, traceback.format_exc())


# For local running: python app.py
if __name__ == "__main__":
    # If you want hot reload locally, set FLASK_ENV=development and run flask, but this is fine.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=True)
