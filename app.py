import os
import uuid
import pickle

import numpy as np
import pandas as pd
from flask import Flask, request, render_template, send_file, redirect, url_for

# REQUIRED for pickle load
from model_wrapper import LinearModelWrapper  # noqa: F401

app = Flask(__name__)

# -------------------------
# Paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
DOWNLOADS_DIR = os.path.join(BASE_DIR, "downloads")

os.makedirs(DOWNLOADS_DIR, exist_ok=True)

# -------------------------
# Load model ONCE
# -------------------------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

if not hasattr(model, "predict"):
    raise RuntimeError("Loaded object does not implement predict()")

# -------------------------
# Utilities
# -------------------------
def prepare_features(df: pd.DataFrame) -> np.ndarray:
    """
    HARD GUARANTEE:
    Returns a pure float numpy array.
    This permanently fixes:
      - 'can't multiply sequence by non-int of type float'
      - dtype=object issues
    """

    df = df.copy()

    # Drop label if present
    if "Label" in df.columns:
        df = df.drop(columns=["Label"])

    # Drop unnamed junk columns
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

    # Convert everything to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Final sanitation
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 🔑 THE FIX — MODEL GETS NUMPY FLOAT ARRAY ONLY
    return df.to_numpy(dtype=float)

# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    return redirect(url_for("predict_csv"))

@app.route("/predict-csv", methods=["GET", "POST"])
def predict_csv():
    if request.method == "GET":
        return render_template("predict_csv.html", error=None)

    try:
        file = request.files.get("file")
        if not file or file.filename == "":
            raise ValueError("No CSV file uploaded")

        df = pd.read_csv(file)

        X = prepare_features(df)
        preds = model.predict(X).astype(int)

        output = df.copy()
        output["Prediction"] = preds

        filename = f"predictions_{uuid.uuid4().hex[:8]}.csv"
        out_path = os.path.join(DOWNLOADS_DIR, filename)
        output.to_csv(out_path, index=False)

        return render_template(
            "predict_csv.html",
            error=None,
            download_url=url_for("download_file", filename=filename)
        )

    except Exception as e:
        return render_template("predict_csv.html", error=str(e))

@app.route("/download/<filename>")
def download_file(filename):
    path = os.path.join(DOWNLOADS_DIR, filename)
    if not os.path.exists(path):
        return "File not found", 404
    return send_file(path, as_attachment=True)

@app.route("/predict-manual")
def predict_manual():
    return render_template("predict_manual.html")

# -------------------------
# Local dev
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
