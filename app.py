import os
import io
import joblib
import pandas as pd

from flask import Flask, request, jsonify, send_file, render_template_string

# =====================================================
# APP SETUP
# =====================================================
app = Flask(__name__)

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


@app.route("/ui", methods=["GET"])
def ui():
    return render_template_string(UPLOAD_PAGE)


@app.route("/predict-csv", methods=["POST"])
def predict_csv():
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


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    app.run(debug=True)
