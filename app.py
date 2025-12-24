import os
import io
import zipfile
import pickle
import urllib.request
import traceback

from flask import Flask, request, jsonify, send_file
import pandas as pd


# ─────────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CACHE_DIR = os.path.join(BASE_DIR, ".cache")
ZIP_PATH = os.path.join(CACHE_DIR, "model.zip")
MODEL_DIR = os.path.join(CACHE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

MODEL_ZIP_URL = "https://github.com/btrump23/MSSE-Machine-Learning/releases/download/model-v1/model.zip"


# ─────────────────────────────────────────────────────────────
# Model loading (SINGLE SOURCE OF TRUTH)
# ─────────────────────────────────────────────────────────────
def ensure_model_loaded() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        print("[model] already loaded:", MODEL_PATH)
        return

    print("[model] DOWNLOADING:", MODEL_ZIP_URL)
    req = urllib.request.Request(
        MODEL_ZIP_URL,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    with urllib.request.urlopen(req, timeout=60) as r, open(ZIP_PATH, "wb") as f:
        f.write(r.read())

    print("[model] EXTRACTING")
    os.makedirs(MODEL_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(MODEL_DIR)

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("model.pkl missing after extraction")

    print("[model] READY:", MODEL_PATH)


def load_model():
    ensure_model_loaded()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)

try:
    MODEL = load_model()
    STARTUP_ERROR = None
except Exception:
    MODEL = None
    STARTUP_ERROR = traceback.format_exc()
    print("[startup] MODEL LOAD FAILED")
    print(STARTUP_ERROR)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": MODEL is not None,
        "model_loaded": MODEL is not None,
        "model_path": MODEL_PATH,
        "error": STARTUP_ERROR,
    })


@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    if MODEL is None:
        return jsonify({
            "status": "error",
            "message": "Model not loaded",
            "trace": STARTUP_ERROR,
        }), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Please upload a CSV file"}), 400

    try:
        df = pd.read_csv(file)

        # Basic sklearn compatibility
        if hasattr(MODEL, "predict_proba"):
            probs = MODEL.predict_proba(df)
            for i in range(probs.shape[1]):
                df[f"proba_{i}"] = probs[:, i]

            if hasattr(MODEL, "predict"):
                df["prediction"] = MODEL.predict(df)
        else:
            df["prediction"] = MODEL.predict(df)

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
            "trace": traceback.format_exc()
        }), 500


# ─────────────────────────────────────────────────────────────
# Local run
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
