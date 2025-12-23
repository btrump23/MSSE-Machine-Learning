import os
import io
import time
import traceback
from pathlib import Path

from flask import Flask, request, jsonify, send_file, render_template_string
import pandas as pd
import joblib

app = Flask(__name__)

# ---- Paths (Render-safe) ----
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "lgbm_pipeline.joblib"
OUT_DIR = Path(os.environ.get("OUT_DIR", "/tmp"))  # Render allows /tmp

# ---- Load model once at startup ----
model = None
model_error = None

try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model_error = f"{type(e).__name__}: {e}"

# ---- Simple HTML GUI ----
INDEX_HTML = """
<!doctype html>
<html>
  <head>
    <title>Malware Detector - CSV Prediction</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 40px; }
      .card { max-width: 700px; padding: 24px; border: 1px solid #ddd; border-radius: 12px; }
      h1 { margin-top: 0; }
      .muted { color: #666; }
      .error { color: #b00020; white-space: pre-wrap; }
      button { padding: 10px 16px; font-size: 16px; cursor: pointer; }
    </style>
  </head>
  <body>
    <div class="card">
      <h1>Malware Detector</h1>
      <p class="muted"><b>Model:</b> {{ model_path }}</p>
      {% if not model_loaded %}
        <p class="error"><b>Model failed to load:</b>\n{{ model_error }}</p>
      {% endif %}

      <h3>Upload CSV</h3>
      <form action="/predict-csv" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept=".csv" required />
        <br><br>
        <button type="submit">Run Predictions</button>
      </form>
    </div>
  </body>
</html>
"""

@app.get("/")
def index():
    # GUI page
    return render_template_string(
        INDEX_HTML,
        model_loaded=(model is not None),
        model_error=model_error,
        model_path=str(MODEL_PATH),
    )

@app.get("/health")
def health():
    # JSON status endpoint
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "out_dir": str(OUT_DIR),
        "error": model_error
    })

@app.post("/predict-csv")
def predict_csv():
    if model is None:
        return jsonify({"status": "error", "error": f"Model not loaded: {model_error}"}), 500

    if "file" not in request.files:
        return jsonify({"status": "error", "error": "No file part in request"}), 400

    f = request.files["file"]
    if f.filename.strip() == "":
        return jsonify({"status": "error", "error": "No selected file"}), 400

    try:
        # Read CSV
        df = pd.read_csv(f)

        # If the CSV contains a target column, drop it (common for test files)
        for possible_target in ["target", "label", "y"]:
            if possible_target in df.columns:
                df = df.drop(columns=[possible_target])

        # Predict
        y_pred = model.predict(df)

        # Attach predictions column
        out_df = df.copy()
        out_df["prediction"] = y_pred

        # Write to /tmp (Render-safe)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = OUT_DIR / f"predictions_{ts}.csv"
        out_df.to_csv(out_path, index=False)

        # Return as downloadable file
        return send_file(
            out_path,
            as_attachment=True,
            download_name="predictions.csv",
            mimetype="text/csv",
        )

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc()
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
