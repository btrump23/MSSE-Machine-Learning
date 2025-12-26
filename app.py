import os
import io
import traceback
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

# -------------------------
# App setup
# -------------------------
app = Flask(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.pkl")
DOWNLOADS_DIR = os.path.join(HERE, "downloads")
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

MODEL = None
MODEL_ERROR = None
EXPECTED_FEATURES = 29


# -------------------------
# Load model
# -------------------------
def load_model():
    global MODEL, MODEL_ERROR
    try:
        with open(MODEL_PATH, "rb") as f:
            MODEL = pickle.load(f)
    except Exception as e:
        MODEL = None
        MODEL_ERROR = str(e)


load_model()


# -------------------------
# Helpers
# -------------------------
def clean_dataframe(df: pd.DataFrame) -> np.ndarray:
    """
    - Drop label/target columns
    - Coerce everything to numeric
    - Fill NaNs with 0
    - Pad or truncate to EXPECTED_FEATURES
    """
    df = df.copy()

    # Drop common label columns
    for col in df.columns:
        if col.lower() in {"label", "target", "y", "class"}:
            df = df.drop(columns=[col])

    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    X = df.values.astype(float)

    # Pad / truncate
    if X.shape[1] < EXPECTED_FEATURES:
        pad = EXPECTED_FEATURES - X.shape[1]
        X = np.hstack([X, np.zeros((X.shape[0], pad))])
    elif X.shape[1] > EXPECTED_FEATURES:
        X = X[:, :EXPECTED_FEATURES]

    return X


def predict_array(X):
    if MODEL is None:
        raise RuntimeError(f"Model not loaded. {MODEL_ERROR or ''}".strip())

    # HARD FORCE float to prevent sequence × float error
    X = np.asarray(X, dtype=float)

    return MODEL.predict(X)


# -------------------------
# HTML Template
# -------------------------
PAGE = """
<!doctype html>
<title>Malware Detector</title>
<h1>Malware Detector</h1>

<p>
<a href="/predict">Manual prediction (single instance)</a> |
<a href="/predict-csv">CSV Prediction</a> |
<a href="/health">Health</a>
</p>
<hr>

{{ body|safe }}
"""


# -------------------------
# Routes
# -------------------------
@app.get("/")
def home():
    return predict_csv_page()


@app.get("/health")
def health():
    return jsonify({
        "status": "ok" if MODEL is not None else "error",
        "model_loaded": MODEL is not None,
        "expected_features": EXPECTED_FEATURES,
        "downloads_dir": DOWNLOADS_DIR
    })


# -------------------------
# Manual prediction
# -------------------------
@app.get("/predict")
def manual_page():
    body = """
    <h2>Manual prediction</h2>
    <form method="post">
        <input style="width:400px" name="values"
               placeholder="Comma-separated numeric values">
        <button type="submit">Predict</button>
    </form>
    """
    return render_template_string(PAGE, body=body)


@app.post("/predict")
def manual_predict():
    try:
        raw = request.form.get("values", "")
        values = [float(v.strip()) for v in raw.split(",") if v.strip()]

        X = np.array(values, dtype=float).reshape(1, -1)

        # Pad / truncate
        if X.shape[1] < EXPECTED_FEATURES:
            pad = EXPECTED_FEATURES - X.shape[1]
            X = np.hstack([X, np.zeros((1, pad))])
        elif X.shape[1] > EXPECTED_FEATURES:
            X = X[:, :EXPECTED_FEATURES]

        pred = int(predict_array(X)[0])

        body = f"<p><b>Prediction:</b> {pred}</p>"
        return render_template_string(PAGE, body=body)

    except Exception as e:
        body = f"<pre>Error: {e}\n\n{traceback.format_exc()}</pre>"
        return render_template_string(PAGE, body=body)


# -------------------------
# CSV prediction
# -------------------------
@app.get("/predict-csv")
def predict_csv_page():
    body = f"""
    <h2>Upload CSV</h2>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="file">
        <button type="submit">Run Predictions</button>
    </form>
    <p>Expected features: {EXPECTED_FEATURES}.
       Non-numeric values are coerced to 0.
       Label columns are dropped.</p>
    """
    return render_template_string(PAGE, body=body)


@app.post("/predict-csv")
def predict_csv():
    try:
        if "file" not in request.files:
            raise RuntimeError("No file uploaded")

        file = request.files["file"]
        df = pd.read_csv(io.StringIO(file.stream.read().decode("utf-8")))

        X = clean_dataframe(df)
        preds = predict_array(X)

        out = df.copy()
        out["prediction"] = preds

        out_path = os.path.join(DOWNLOADS_DIR, "predictions.csv")
        out.to_csv(out_path, index=False)

        body = f"""
        <p><b>Predictions complete.</b></p>
        <p>Saved to <code>{out_path}</code></p>
        <p>Rows: {len(out)}</p>
        """

        return render_template_string(PAGE, body=body)

    except Exception as e:
        body = f"<pre>Error: {e}\n\n{traceback.format_exc()}</pre>"
        return render_template_string(PAGE, body=body)


# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
