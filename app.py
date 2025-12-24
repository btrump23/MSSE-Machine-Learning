import os
import io
import urllib.request
import zipfile
from datetime import datetime

import pandas as pd
from flask import Flask, request, jsonify, render_template_string
import joblib

app = Flask(__name__)

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Local model path (will exist locally; in prod we download it)
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# GitHub Release asset (ZIP) URL (uploads block .pkl, so we host model.zip)
MODEL_ZIP_URL = "https://github.com/btrump23/MSSE-Machine-Learning/releases/download/model-v1/model.zip"

# Optional: if your CSV includes a label column you want dropped before prediction
TARGET_COLUMN = None

# For sanity checking deployment
APP_VERSION = "prod-modelzip-v1"


# ================= MODEL DOWNLOAD/LOAD =================
def ensure_model_exists():
    """If model.pkl is missing (e.g., on Render), download model.zip and extract model.pkl."""
    if os.path.exists(MODEL_PATH):
        return

    zip_path = os.path.join(BASE_DIR, "model.zip")

    # Download ZIP from GitHub Releases
    urllib.request.urlretrieve(MODEL_ZIP_URL, zip_path)

    # Extract into BASE_DIR (expecting model.pkl inside)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(BASE_DIR)

    # Clean up zip
    try:
        os.remove(zip_path)
    except OSError:
        pass

    # Final check
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Downloaded and extracted model.zip but {MODEL_PATH} still not found. "
            f"Ensure model.zip contains model.pkl at the top level."
        )


def load_model():
    ensure_model_exists()

    obj = joblib.load(MODEL_PATH)

    # Handle model saved as a dict/bundle
    if isinstance(obj, dict):
        for key in ("pipeline", "model", "estimator", "clf"):
            if key in obj and obj[key] is not None:
                obj = obj[key]
                break

    if not hasattr(obj, "predict"):
        raise TypeError(f"Loaded object has no predict(): {type(obj)}")

    return obj


# Lazy-load so Render can boot even if model isn't present until first request
MODEL = None


def get_model():
    global MODEL
    if MODEL is None:
        MODEL = load_model()
    return MODEL


# ================= UI =================
HTML_PAGE = """
<!doctype html>
<html>
<head>
  <title>MSSE ML Predictor</title>
  <style>
    body { font-family: Arial; padding: 40px; }
    input, button { margin: 10px 0; }
    pre { background: #f4f4f4; padding: 15px; }
  </style>
</head>
<body>
  <h2>Upload CSV</h2>
  <form id="form" enctype="multipart/form-data">
    <input type="file" name="file" accept=".csv" required />
    <br />
    <button type="submit">Predict</button>
  </form>

  <h3>Result</h3>
  <pre id="output">Waiting...</pre>

<script>
document.getElementById("form").onsubmit = async (e) => {
  e.preventDefault();
  const data = new FormData(e.target);
  const res = await fetch("/predict_csv", { method: "POST", body: data });
  const json = await res.json();
  document.getElementById("output").textContent = JSON.stringify(json, null, 2);
};
</script>
</body>
</html>
"""


@app.get("/")
def index():
    return render_template_string(HTML_PAGE)


@app.get("/version")
def version():
    return {"version": APP_VERSION}


# ================= PREDICT =================
@app.route("/predict_csv", methods=["GET", "POST"])
def predict_csv():
    # GET shows a friendly page instead of 405
    if request.method == "GET":
        return render_template_string(HTML_PAGE)

    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"status": "error", "message": "No file uploaded"}), 400

        df = pd.read_csv(io.BytesIO(file.read()))

        if TARGET_COLUMN and TARGET_COLUMN in df.columns:
            df = df.drop(columns=[TARGET_COLUMN])

        model = get_model()

        preds = model.predict(df)
        preds_list = [p.item() if hasattr(p, "item") else p for p in preds]

        # Write output CSV next to app.py (timestamped)
        out = df.copy()
        out["prediction"] = preds_list
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(BASE_DIR, f"predictions_{ts}.csv")
        out.to_csv(output_path, index=False)

        # Optional probabilities if available
        probs = None
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(df).max(axis=1).tolist()
            except Exception:
                probs = None

        return jsonify(
            {
                "status": "ok",
                "rows": len(preds_list),
                "predictions_preview": preds_list[:20],
                "probabilities_preview": probs[:20] if probs else None,
                "output_file": output_path,
            }
        )

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ================= RUN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
