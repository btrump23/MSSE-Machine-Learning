import os
import io
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
import joblib
APP_VERSION = "27d7dc0-msse-prod-check"

app = Flask(__name__)
@app.get("/version")
def version():
    return {"version": APP_VERSION}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
TARGET_COLUMN = None

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    obj = joblib.load(MODEL_PATH)
    if isinstance(obj, dict):
        for k in ("pipeline", "model", "estimator", "clf"):
            if k in obj:
                obj = obj[k]
                break
    if not hasattr(obj, "predict"):
        raise TypeError("Loaded object has no predict()")
    return obj

MODEL = load_model()

HTML = """
<!doctype html>
<html>
<head><title>MSSE ML Predictor</title></head>
<body>
<h2>Upload CSV</h2>
<form method="post" enctype="multipart/form-data" action="/predict_csv">
<input type="file" name="file" required />
<button type="submit">Predict</button>
</form>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    df = pd.read_csv(io.BytesIO(request.files["file"].read()))
    preds = (MODEL.get("pipeline") or MODEL.get("model") or MODEL.get("estimator") or MODEL.get("clf") or MODEL).predict(df) if isinstance(MODEL, dict) else MODEL.predict(df)
    preds = [p.item() if hasattr(p, "item") else p for p in preds]
    return jsonify(predictions=preds[:20])

if __name__ == "__main__":
    app.run(debug=True)
