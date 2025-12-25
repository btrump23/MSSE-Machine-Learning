import os
import pickle
import traceback
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.pkl")

MODEL = None
MODEL_ERROR = None


# -----------------------------
# Model loading (SAFE)
# -----------------------------
def load_model():
    global MODEL, MODEL_ERROR
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        with open(MODEL_PATH, "rb") as f:
            obj = pickle.load(f)

        # CASE 1: sklearn-style model
        if hasattr(obj, "predict"):
            MODEL = obj
            print("[startup] Loaded sklearn model")

        # CASE 2: numpy array (THIS IS YOUR CASE)
        elif isinstance(obj, np.ndarray):
            MODEL = obj
            print("[startup] Loaded numpy array model")

        else:
            raise TypeError(f"Unsupported model type: {type(obj)}")

        MODEL_ERROR = None
        return MODEL

    except Exception as e:
        MODEL = None
        MODEL_ERROR = traceback.format_exc()
        print("[startup] MODEL LOAD FAILED")
        print(MODEL_ERROR)
        return None


# Load at startup
load_model()


# -----------------------------
# Health check
# -----------------------------
@app.route("/")
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


# -----------------------------
# CSV Prediction Endpoint
# -----------------------------
@app.route("/predict_csv", methods=["POST", "GET"])
def predict_csv():
    try:
        if MODEL is None:
            raise RuntimeError("Model not loaded")

        if "file" not in request.files:
            raise ValueError("No CSV file uploaded")

        file = request.files["file"]
        df = pd.read_csv(file)

        X = df.values

        # --------- PREDICTION LOGIC ----------
        if hasattr(MODEL, "predict"):
            preds = MODEL.predict(X)

        elif isinstance(MODEL, np.ndarray):
            # Simple dot-product style prediction
            if X.shape[1] != MODEL.shape[0]:
                raise ValueError(
                    f"Shape mismatch: X has {X.shape[1]} columns, "
                    f"model expects {MODEL.shape[0]}"
                )
            preds = X @ MODEL

        else:
            raise TypeError("Unsupported model type at prediction time")

        df["prediction"] = preds

        return jsonify({
            "status": "success",
            "rows": len(df),
            "preview": df.head(5).to_dict(orient="records")
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Prediction failed",
            "trace": traceback.format_exc()
        }), 500


# -----------------------------
# Render / Gunicorn entrypoint
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
