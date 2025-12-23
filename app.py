import os
import traceback
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, request, send_file, render_template_string, jsonify

# ------------------------------------------------------------------------------
# App setup
# ------------------------------------------------------------------------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "lgbm_pipeline.joblib")

# Where to write outputs:
# - On Render (Linux) write to /tmp (writable, ephemeral)
# - Locally on Windows, write to ./artifacts (like your localhost version)
DEFAULT_OUT_DIR = os.path.join(BASE_DIR, "artifacts")
OUT_DIR = "/tmp" if os.name != "nt" else DEFAULT_OUT_DIR

# Ensure local output dir exists (Windows/local)
try:
    os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)
except Exception:
    pass

# ------------------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------------------
pipe = None
model_error = None

try:
    pipe = joblib.load(MODEL_PATH)
except Exception as e:
    model_error = f"{type(e).__name__}: {e}"
    print("❌ Model failed to load:", model_error)
    print(traceback.format_exc())

# ------------------------------------------------------------------------------
# Simple UI template
# ------------------------------------------------------------------------------
UPLOAD_PAGE = """
<!doctype html>
<html>
<head>
  <title>Malware Detector - CSV Prediction</title>
  <style>
    body { font-family: Arial, sans-serif; padding: 40px; }
    .card { max-width: 760px; padding: 30px; border: 1px solid #ddd; border-radius: 8px; }
    .error { background: #fee; border: 1px solid #f00; padding: 10px; margin-bottom: 15px; }
    .ok { background: #eef; border: 1px solid #88f; padding: 10px; margin-bottom: 15px; }
    code { background: #f6f6f6; padding: 2px 6px; border-radius: 4px; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Malware Detector</h1>
    <p><b>Model:</b> {{ model_path }}</p>
    <p><b>Output directory:</b> <code>{{ out_dir }}</code></p>

    {% if error %}
      <div class="error"><b>Error:</b> {{ error }}</div>
    {% endif %}

    {% if message %}
      <div class="ok">{{ message | safe }}</div>
    {% endif %}

    <form action="/predict-csv" method="post" enctype="multipart/form-data">
      <h3>Upload CSV</h3>
      <input type="file" name="file" accept=".csv" required>
      <br><br>
      <button type="submit">Run Predictions</button>
    </form>
  </div>
</body>
</html>
"""

# ------------------------------------------------------------------------------
# Health / status endpoint
# ------------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def status():
    return jsonify({
        "status": "ok",
        "model_loaded": pipe is not None,
        "model_path": MODEL_PATH,
        "out_dir": OUT_DIR,
        "error": model_error
    })

# ------------------------------------------------------------------------------
# UI shortcut
# ------------------------------------------------------------------------------
@app.route("/ui", methods=["GET"])
def ui():
    return render_template_string(
        UPLOAD_PAGE,
        model_path=MODEL_PATH,
        out_dir=OUT_DIR,
        error=model_error,
        message=None
    )

# ------------------------------------------------------------------------------
# CSV prediction endpoint (writes file + returns it)
# ------------------------------------------------------------------------------
@app.route("/predict-csv", methods=["GET", "POST"])
def predict_csv():
    if request.method == "GET":
        return render_template_string(
            UPLOAD_PAGE,
            model_path=MODEL_PATH,
            out_dir=OUT_DIR,
            error=model_error,
            message=None
        )

    try:
        if pipe is None:
            raise RuntimeError(f"Model not loaded: {model_error}")

        if "file" not in request.files:
            raise ValueError("No file part in request.")

        f = request.files["file"]
        if not f or not f.filename:
            raise ValueError("No file selected.")

        # Read CSV
        df = pd.read_csv(f)
        if df.empty:
            raise ValueError("Uploaded CSV is empty.")

        # Build X
        X = df.copy()

        # Drop common target columns if present
        for col in ["target", "label", "y"]:
            if col in X.columns:
                X = X.drop(columns=[col])

        # Determine expected columns + model object
        expected_cols = None
        model = pipe

        if isinstance(pipe, dict):
            if "pipeline" not in pipe:
                raise ValueError("Loaded model bundle is invalid (missing 'pipeline').")
            model = pipe["pipeline"]
            expected_cols = pipe.get("feature_names")

        elif hasattr(model, "feature_names_in_"):
            expected_cols = list(model.feature_names_in_)

        elif hasattr(model, "named_steps"):
            last = list(model.named_steps.values())[-1]
            if hasattr(last, "feature_names_in_"):
                expected_cols = list(last.feature_names_in_)

        # Align columns
        if expected_cols:
            missing = [c for c in expected_cols if c not in X.columns]
            if missing:
                raise ValueError(f"CSV missing required columns: {missing[:20]}")
            # reorder & drop extras
            X = X[expected_cols]

        # Force numeric + fill NaNs
        for c in X.columns:
            if X[c].dtype == "object":
                X[c] = pd.to_numeric(X[c], errors="coerce")
        X = X.fillna(0)

        # Predict
        y_pred = model.predict(X)

        # Output dataframe
        out_df = df.copy()
        out_df["prediction"] = y_pred

        # Ensure output directory exists
        os.makedirs(OUT_DIR, exist_ok=True)

        # Write file (timestamped so multiple runs don’t clash)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_filename = f"predictions_{ts}.csv"
        out_path = os.path.join(OUT_DIR, out_filename)

        out_df.to_csv(out_path, index=False)

        # Show message + return file download
        # (Render can't "let you browse the server files", but this confirms it wrote successfully.)
        print(f"✅ Wrote predictions to: {out_path}")

        return send_file(
            out_path,
            mimetype="text/csv",
            as_attachment=True,
            download_name="predictions.csv"
        )

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print("❌ Prediction error:", err)
        print(traceback.format_exc())
        return render_template_string(
            UPLOAD_PAGE,
            model_path=MODEL_PATH,
            out_dir=OUT_DIR,
            error=err,
            message=None
        ), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
