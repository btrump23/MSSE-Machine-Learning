import os
import traceback
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify

# --------------------------------------------------
# Flask app
# --------------------------------------------------
app = Flask(__name__)

# --------------------------------------------------
# Globals
# --------------------------------------------------
MODEL = None
MODEL_ERROR = None

# --------------------------------------------------
# Model path resolution (local vs Render)
# --------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))

LOCAL_MODEL = os.path.join(HERE, "model.pkl")
RENDER_MODEL = "/opt/render/project/src/.cache/model/model.pkl"

MODEL_PATH = RENDER_MODEL if os.path.exists(RENDER_MODEL) else LOCAL_MODEL


# --------------------------------------------------
# Model loader
# --------------------------------------------------
def load_model():
    global MODEL, MODEL_ERROR

    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}\n"
                f"HERE={HERE}\n"
                f"Files in HERE: {os.listdir(HERE)}\n"
                f"Cache dir exists? {os.path.exists('/opt/render/project/src/.cache')}\n"
                f"Cache/model exists? {os.path.exists('/opt/render/project/src/.cache/model')}"
            )

        obj = joblib.load(MODEL_PATH)

        # Expecting a bundle dict
        if isinstance(obj, dict) and "pipeline" in obj:
            MODEL = obj["pipeline"]
        else:
            MODEL = obj  # fallback (not expected, but safe)

        if not hasattr(MODEL, "predict"):
            raise TypeError(
                f"Loaded object has no .predict().\n"
                f"Top-level type: {type(obj)}\n"
                f"Model type: {type(MODEL)}"
            )

        MODEL_ERROR = None
        print("[startup] MODEL LOADED OK:", type(MODEL))
        return MODEL

    except Exception:
        MODEL = None
        MODEL_ERROR = traceback.format_exc()
        print("[startup] MODEL LOAD FAILED")
        print(MODEL_ERROR)
        return None


# --------------------------------------------------
# Load model at startup
# --------------------------------------------------
load_model()


# --------------------------------------------------
# Health check
# --------------------------------------------------
@app.route("/", methods=["GET"])
def health():
    if MODEL is None:
        return jsonify({
            "status": "error",
            "message": "Model not loaded",
            "trace": MODEL_ERROR
        }), 500

    return jsonify({
        "status": "ok",
        "model_type": str(type(MODEL))
    })


# --------------------------------------------------
# CSV prediction endpoint
# --------------------------------------------------
@app.route("/predict_csv", methods=["POST", "GET"])
def predict_csv():
    if MODEL is None:
        return jsonify({
            "status": "error",
            "message": "Model not loaded",
            "trace": MODEL_ERROR
        }), 500

    try:
        # Accept file upload or query-based CSV
        if request.method == "POST":
            if "file" not in request.files:
                raise ValueError("No file uploaded")

            file = request.files["file"]
            df = pd.read_csv(file)

        else:
            raise ValueError("POST a CSV file using multipart/form-data")

        if df.empty:
            raise ValueError("Uploaded CSV is empty")

        # Predict
        preds = MODEL.predict(df)

        # Attach predictions
        df["prediction"] = preds

        return jsonify({
            "status": "ok",
            "rows": len(df),
            "predictions": df["prediction"].tolist()
        })

    except Exception:
        return jsonify({
            "status": "error",
            "message": "Prediction failed",
            "trace": traceback.format_exc()
        }), 500


# --------------------------------------------------
# Local dev entrypoint (Render ignores this)
# --------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
