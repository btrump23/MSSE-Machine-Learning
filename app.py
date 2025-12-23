import os
import io
<<<<<<< HEAD
import time
import traceback
from pathlib import Path

from flask import Flask, request, jsonify, send_file, render_template_string
import pandas as pd
import joblib
=======
import joblib
import pandas as pd

from flask import Flask, request, jsonify, send_file, render_template_string
>>>>>>> 14f855b (Add training pipeline, Flask prediction service, and UI)

# =====================================================
# APP SETUP
# =====================================================
app = Flask(__name__)

<<<<<<< HEAD
# ---- Paths (Render-safe) ----
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "lgbm_pipeline.joblib"
OUT_DIR = Path(os.environ.get("OUT_DIR", "/tmp"))  # Render allows /tmp

# ---- Load model once at startup ----
model = None
model_error = None
=======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

MODEL = None
MODEL_BUNDLE = None
LOAD_ERROR = None


# =====================================================
# LOAD MODEL
# =====================================================
def load_model(path):
    obj = joblib.load(path)

    if isinstance(obj, dict):
        if "pipeline" in obj:
            return obj["pipeline"], obj
        if "model" in obj:
            return obj["model"], obj
        raise ValueError(f"Model bundle keys not recognised: {list(obj.keys())}")

    return obj, None


try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

    MODEL, MODEL_BUNDLE = load_model(MODEL_PATH)
except Exception as e:
    LOAD_ERROR = str(e)
    MODEL = None
    MODEL_BUNDLE = None


# =====================================================
# GUI PAGE (UPLOAD CSV)
# =====================================================
UPLOAD_PAGE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Malware Detector - CSV Prediction</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 40px; }
    .card { max-width: 760px; padding: 24px; border: 1px solid #ddd; border-radius: 10px; }
    button { padding: 10px 16px; font-size: 16px; cursor: pointer; }
    input[type=file] { margin: 12px 0; }
    .hint { color: #555; font-size: 14px; }
  </style>
</head>
<body>
  <div class="card">
    <h2>Malware Detector — CSV Prediction</h2>
    <p class="hint">
      Upload a CSV with the same feature columns as training.
      If a <b>Label</b> column exists, it will be ignored.
    </p>
    <form action="/predict-csv" method="post" enctype="multipart/form-data">
      <input type="file" name="file" accept=".csv" required />
      <br/>
      <button type="submit">Predict</button>
    </form>
  </div>
</body>
</html>
"""


# =====================================================
# ROUTES
# =====================================================
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "Flask ML service running",
        "model_loaded": MODEL is not None,
        "model_path": MODEL_PATH,
        "error": LOAD_ERROR
    })
>>>>>>> 14f855b (Add training pipeline, Flask prediction service, and UI)

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

<<<<<<< HEAD
@app.get("/")
def index():
    # GUI page
    return render_template_string(
        INDEX_HTML,
        model_loaded=(model is not None),
        model_error=model_error,
        model_path=str(MODEL_PATH),
    )
=======
@app.route("/ui", methods=["GET"])
def ui():
    return render_template_string(UPLOAD_PAGE)
>>>>>>> 14f855b (Add training pipeline, Flask prediction service, and UI)

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

@app.route("/predict-csv", methods=["POST"])
def predict_csv():
<<<<<<< HEAD
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
=======
    if MODEL is None:
        return jsonify({"error": "Model not loaded", "details": LOAD_ERROR}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Failed to read CSV: {e}"}), 400

    # Drop target column if present (Label)
    if MODEL_BUNDLE and "target_col" in MODEL_BUNDLE:
        target_col = MODEL_BUNDLE["target_col"]
        if target_col in df.columns:
            df = df.drop(columns=[target_col])

    # Predict
    try:
        preds = MODEL.predict(df)
    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "details": str(e),
            "columns_received": list(df.columns)
        }), 400

    df_out = df.copy()
    df_out["prediction"] = preds

    # Return as CSV download
    output = io.StringIO()
    df_out.to_csv(output, index=False)
    output.seek(0)

    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="predictions.csv"
    )
>>>>>>> 14f855b (Add training pipeline, Flask prediction service, and UI)


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
<<<<<<< HEAD
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
=======
    app.run(debug=True)
>>>>>>> 14f855b (Add training pipeline, Flask prediction service, and UI)
