import os
import io
import zipfile
import pickle
import traceback
import urllib.request

from flask import Flask, request, jsonify, send_file, render_template_string
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, ".cache")
ZIP_PATH = os.path.join(CACHE_DIR, "model.zip")
MODEL_DIR = os.path.join(CACHE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

# HARD CODED — cannot be overridden, cannot drift
MODEL_ZIP_URL = "https://github.com/btrump23/MSSE-Machine-Learning/releases/download/model-v1/model.zip"


def _download_file(url: str, dest_path: str) -> None:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.HTTPRedirectHandler())
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with opener.open(req, timeout=60) as r, open(dest_path, "wb") as f:
        f.write(r.read())


def ensure_model_present() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        print("[model] already present:", MODEL_PATH)
        return

    if not os.path.exists(ZIP_PATH):
        print("[model] downloading:", MODEL_ZIP_URL)
        _download_file(MODEL_ZIP_URL, ZIP_PATH)

    print("[model] extracting:", ZIP_PATH, "->", MODEL_DIR)
    os.makedirs(MODEL_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(MODEL_DIR)

    if not os.path.exists(MODEL_PATH):
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            raise RuntimeError(f"model.pkl missing after extraction. ZIP contained: {z.namelist()}")

    print("[model] READY:", MODEL_PATH)


def load_model():
    ensure_model_present()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


app = Flask(__name__)

MODEL = None
STARTUP_ERROR = None

try:
    MODEL = load_model()
except Exception:
    STARTUP_ERROR = traceback.format_exc()
    print("[startup] MODEL LOAD FAILED\n", STARTUP_ERROR)


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
        "model_zip_url": MODEL_ZIP_URL,
        "zip_path": ZIP_PATH,
        "model_path": MODEL_PATH,
        "startup_error": STARTUP_ERROR,
    })


@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    if MODEL is None:
        return jsonify({"status": "error", "message": "Model not loaded.", "trace": STARTUP_ERROR}), 500

    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded (field name must be 'file')."}), 400

    f = request.files["file"]
    if not f.filename.lower().endswith(".csv"):
        return jsonify({"status": "error", "message": "Please upload a .csv file."}), 400

    try:
        df = pd.read_csv(f)
        if df.empty:
            return jsonify({"status": "error", "message": "CSV is empty."}), 400

        X = df

        if hasattr(MODEL, "predict_proba"):
            proba = MODEL.predict_proba(X)
            for i in range(proba.shape[1]):
                df[f"proba_{i}"] = proba[:, i]
            if hasattr(MODEL, "predict"):
                df["prediction"] = MODEL.predict(X)
        else:
            df["prediction"] = MODEL.predict(X)

        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)

        return send_file(buf, mimetype="text/csv", as_attachment=True, download_name="predictions.csv")

    except Exception:
        return jsonify({"status": "error", "message": "Prediction failed.", "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
