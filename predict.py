from __future__ import annotations

"""
MSSE Machine Learning – Prediction API

Local run:
  python predict.py

Render run (Linux):
  gunicorn predict:app

Endpoints:
  GET  /           -> health + model load status
  POST /predict_csv -> upload CSV file under form field name "file"
"""

import os
import pickle
import traceback
from typing import Any

import pandas as pd
from flask import Flask, jsonify, request


# -----------------------------------------------------------------------------
# Flask app
# -----------------------------------------------------------------------------
app = Flask(__name__)

# -----------------------------------------------------------------------------
# Model config
# -----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.pkl")


MODEL: Any | None = None
MODEL_ERROR: str | None = None


def _is_git_lfs_pointer(first_bytes: bytes) -> bool:
    return first_bytes.startswith(b"version https://git-lfs.github.com/spec")


def load_model() -> Any:
    """
    Loads the pickled model once and caches it.
    Always uses binary mode.
    Provides clear error messages (missing file / LFS pointer).
    """
    global MODEL, MODEL_ERROR

    if MODEL is not None:
        return MODEL

    if not os.path.exists(MODEL_PATH):
        MODEL_ERROR = f"Model file not found: {MODEL_PATH}"
        raise FileNotFoundError(MODEL_ERROR)

    try:
        with open(MODEL_PATH, "rb") as f:
            head = f.read(80)
            f.seek(0)

            # Common gotcha: model tracked by Git LFS but not actually present on server
            if _is_git_lfs_pointer(head):
                MODEL_ERROR = (
                    "model.pkl is a Git LFS pointer, not the real binary model.\n"
                    "Fix: ensure the real model.pkl is committed (not just the pointer), "
                    "or configure Git LFS on the build environment and pull LFS files."
                )
                raise RuntimeError(MODEL_ERROR)

            MODEL = pickle.load(f)

        MODEL_ERROR = None
        return MODEL

    except Exception:
        MODEL_ERROR = traceback.format_exc()
        raise


# Try loading at startup (so / tells you immediately)
try:
    load_model()
except Exception:
    # Keep serving / so you can see the error in JSON
    pass


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def health():
    if MODEL is not None:
        return jsonify({"status": "ok", "message": "Model loaded successfully"})
    return jsonify({"status": "error", "message": "Model not loaded", "trace": MODEL_ERROR}), 500


@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    """
    Expects multipart/form-data with a file field named 'file'.
    Example PowerShell test:
      $csv="C:\\path\\to\\test.csv"
      curl.exe -X POST -F "file=@$csv" http://127.0.0.1:5000/predict_csv
    """
    try:
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded. Use form field name 'file'."}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"status": "error", "message": "Empty filename."}), 400

        df = pd.read_csv(file)

        model = load_model()

        # If your model expects preprocessing, it should be inside the pipeline.
        preds = model.predict(df)

        return jsonify({"status": "success", "rows": int(len(df)), "predictions": preds.tolist()})

    except Exception:
        return jsonify({"status": "error", "trace": traceback.format_exc()}), 500


# -----------------------------------------------------------------------------
# Local entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
