import os
import io
import zipfile
import pickle
import traceback
import urllib.request

from flask import Flask, request, jsonify, send_file, render_template_string

import pandas as pd


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CACHE_DIR = os.path.join(BASE_DIR, ".cache")
ZIP_PATH = os.path.join(CACHE_DIR, "model.zip")
MODEL_DIR = os.path.join(CACHE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

# Known-good public release asset URL
DEFAULT_MODEL_URL = "https://github.com/btrump23/MSSE-Machine-Learning/releases/download/model-v1/model.zip"
MODEL_ZIP_URL = os.getenv("MODEL_ZIP_URL", DEFAULT_MODEL_URL).strip()

# Guard rails
if not MODEL_ZIP_URL.startswith("http"):
    raise RuntimeError(f"MODEL_ZIP_URL invalid: {MODEL_ZIP_URL!r}")

# Ensure cache dir exists
os.makedirs(CACHE_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# MODEL DOWNLOAD + LOAD (single code path)
# ─────────────────────────────────────────────────────────────
def _download_file(url: str, dest_path: str) -> None:
    """
    GitHub-safe downloader:
    - sends User-Agent
    - follows redirects (urllib does by default)
    - writes binary content to disk
    """
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest_path, "wb") as f:
        f.write(r.read())


def ensure_model_present() -> None:
    """
    Ensures MODEL_PATH exists by downloading ZIP_PATH (if needed) and extracting to MODEL_DIR.
    """
    # Already extracted and ready
    if os.path.exists(MODEL_PATH):
        print("[model] already present:", MODEL_PATH)
        return

    # Download zip if missing
    if not os.path.exists(ZIP_PATH):
        print("[model] downloading from:", MODEL_ZIP_URL)
        print("[model] saving to:", ZIP_PATH)
        _download_file(MODEL_ZIP_URL, ZIP_PATH)
    else:
        print("[model] zip already cached:", ZIP_PATH)

    # Extract zip
    print("[model] extracting to:", MODEL_DIR)
    os.makedirs(MODEL_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(MODEL_DIR)

    # Verify expected file exists
    if not os.path.exists(MODEL_PATH):
        # show zip contents to help debugging if this ever happens
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            names = z.namelist()
        raise RuntimeError(f"model.pkl not found after extraction. ZIP contained: {names}")

    print("[model] READY:", MODEL_PATH)


def load_model():
    ensure_model_present()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)

MODEL = None
STARTUP_ERROR = None

try:
    print(f"[startup] MODEL_ZIP_URL={MODEL_ZIP_URL!r}")
    MODEL = load_model()
    print("[startup] model loaded:", type(MODEL).__name__)
except Exception:
    STARTUP_ERROR = traceback.format_exc()
    print("[startup] MODEL LOAD FAILED")
    print(STARTUP_ERROR)


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template_string("""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>MSSE ML Predictor</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 30px; max-width: 900px; }
    .card { border: 1px solid #ddd; padding: 18px; border-radius: 12px; }
    .muted { color: #666; }
    button { padding: 10px 14px; cursor: pointer; }
    code { background: #f6f6f6; padding: 2px 6px; border-radius: 6px; }
  </style>
</head>
<body>
  <div class="card">
    <h2>MSSE ML Predictor</h2>
    <p class="muted">Upload a CSV and download <b>predictions.csv</b>.</p>

    <p>Status endpoint: <a href="/health">/health</a></p>

    <h3>Upload CSV</h3>
    <form action="/predict_csv" method="post" enctype="multipart/form-data">
      <input type="file" name="file" accept=".csv" required />
      <button type="submit">Predict</button>
    </form>

    <p class="muted" style="margin-top: 14px;">
      POST endpoint: <code>/predict_csv</code>
    </p>
  </div>
</body>
</html>
""")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": MODEL is not None,
        "model_loaded": MODEL is not None,
        "model_type": type(MODEL).__name__ if MODEL is not None else None,
        "model_zip_url": MODEL_ZIP_URL,
        "zip_path": ZIP_PATH,
        "model_path": MODEL_PATH,
        "startup_error": STARTUP_ERROR,
    })


@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    if MODEL is None:
        return jsonify({
            "status": "error",
            "message": "Model not loaded.",
            "trace": STARTUP_ERROR
        }), 500

    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded (field name must be 'file')."}), 400

    f = request.files["file"]
    if not f.filename.lower().endswith(".csv"):
        return jsonify({"status": "error", "message": "Please upload a .csv file."}), 400

    try:
        df = pd.read_csv(f)
        if df.empty:
            return jsonify({"status": "error", "message": "CSV is empty."}), 400

        # NOTE: Your model expects the same columns/features as training.
        X = df

        # Classification: write proba columns + prediction
        if hasattr(MODEL, "predict_proba"):
            proba = MODEL.predict_proba(X)
            for i in range(proba.shape[1]):
                df[f"proba_{i}"] = proba[:, i]
            if hasattr(MODEL, "predict"):
                df["prediction"] = MODEL.predict(X)
        else:
            # Regression or classifiers without proba
            df["prediction"] = MODEL.predict(X)

        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)

        return send_file(
            buf,
            mimetype="text/csv",
            as_attachment=True,
            download_name="predictions.csv"
        )

    except Exception:
        return jsonify({
            "status": "error",
            "message": "Prediction failed.",
            "trace": traceback.format_exc()
        }), 500


# ─────────────────────────────────────────────────────────────
# LOCAL RUN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
