import os
import io
import pickle
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string, send_from_directory

app = Flask(__name__)

# -------------------------------------------------
# Paths / Model load
# -------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.pkl")

MODEL = None
MODEL_ERROR = None


def load_model():
    global MODEL, MODEL_ERROR
    try:
        with open(MODEL_PATH, "rb") as f:
            MODEL = pickle.load(f)
    except Exception as e:
        MODEL = None
        MODEL_ERROR = f"{type(e).__name__}: {e}"


load_model()

# -------------------------------------------------
# Downloads dir (Render-safe)
# -------------------------------------------------
def get_download_dir():
    preferred = os.path.join(HERE, "downloads")
    fallback = os.path.join("/tmp", "downloads")

    try:
        os.makedirs(preferred, exist_ok=True)
        with open(os.path.join(preferred, ".test"), "w") as f:
            f.write("ok")
        return preferred
    except Exception:
        os.makedirs(fallback, exist_ok=True)
        return fallback


DOWNLOAD_DIR = get_download_dir()

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def expected_feature_count():
    if MODEL is None:
        return 0
    return len(getattr(MODEL, "weights", []))


def force_feature_matrix(df: pd.DataFrame, n_features: int) -> np.ndarray:
    """
    🔥 CRITICAL FUNCTION 🔥

    - Drops ALL non-numeric columns
    - Forces float dtype
    - Selects EXACTLY n_features columns
    - Pads or truncates as needed
    """

    # Convert everything to numeric, coercing garbage to NaN
    df_num = df.apply(pd.to_numeric, errors="coerce")

    # Drop columns that are entirely NaN
    df_num = df_num.dropna(axis=1, how="all")

    # Fill remaining NaN with 0
    df_num = df_num.fillna(0.0)

    # Convert to numpy float matrix
    X = df_num.to_numpy(dtype=float)

    # Enforce feature count
    if X.shape[1] < n_features:
        pad = np.zeros((X.shape[0], n_features - X.shape[1]), dtype=float)
        X = np.hstack([X, pad])
    elif X.shape[1] > n_features:
        X = X[:, :n_features]

    return X


def predict_array(X):
    if MODEL is None:
        raise RuntimeError(f"Model not loaded. {MODEL_ERROR}")
    X = np.asarray(X, dtype=float)
    return MODEL.predict(X)


# -------------------------------------------------
# Templates
# -------------------------------------------------
BASE_HTML = """
<!doctype html>
<html>
<head>
  <title>Malware Detector</title>
  <style>
    body { font-family: Arial; background:#f6f7fb; padding:30px; }
    .card { background:#fff; padding:25px; max-width:900px; margin:auto;
            border-radius:14px; box-shadow:0 10px 30px rgba(0,0,0,.1); }
    h1 { font-size:42px; }
    h2 { font-size:28px; }
    .row { display:flex; gap:14px; align-items:center; flex-wrap:wrap; }
    input, button { padding:10px; border-radius:10px; }
    button { background:#111827; color:white; border:none; cursor:pointer; }
    .err { color:#b91c1c; font-weight:700; }
    .ok { color:#166534; font-weight:700; }
    .muted { color:#6b7280; }
    pre { font-size:12px; white-space:pre-wrap; }
  </style>
</head>
<body>
<div class="card">
<h1>Malware Detector</h1>
<div class="row">
<a href="/manual">Manual prediction</a> |
<a href="/predict-csv">CSV Prediction</a> |
<a href="/health">Health</a>
</div>
<hr/>
{{ body|safe }}
</div>
</body>
</html>
"""

# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.get("/")
def home():
    return predict_csv_page()


@app.get("/health")
def health():
    if MODEL is None:
        return {
            "status": "error",
            "model_loaded": False,
            "model_error": MODEL_ERROR,
            "expected_features": 0,
        }, 500

    return {
        "status": "ok",
        "model_loaded": True,
        "expected_features": expected_feature_count(),
        "downloads_dir": DOWNLOAD_DIR,
    }


@app.get("/manual")
def manual_page():
    body = f"""
    <h2>Manual prediction</h2>
    <p class="muted">Model expects {expected_feature_count()} numeric values.</p>
    <form method="post" action="/predict" class="row">
      <input name="features" placeholder="comma separated numbers" size="60"/>
      <button>Predict</button>
    </form>
    """
    return render_template_string(BASE_HTML, body=body)


@app.post("/predict")
def manual_predict():
    try:
        raw = request.form.get("features", "")
        values = [float(x.strip()) for x in raw.split(",") if x.strip()]

        n = expected_feature_count()
        if len(values) < n:
            values += [0.0] * (n - len(values))
        elif len(values) > n:
            values = values[:n]

        X = np.array([values], dtype=float)
        pred = predict_array(X)[0]

        body = f"<div class='ok'>Prediction: {pred}</div>"
        return render_template_string(BASE_HTML, body=body)

    except Exception as e:
        body = f"<div class='err'>{e}</div><pre>{traceback.format_exc()}</pre>"
        return render_template_string(BASE_HTML, body=body), 400


@app.get("/predict-csv")
def predict_csv_page():
    body = f"""
    <h2>Upload CSV</h2>
    <p class="muted">Expected features: {expected_feature_count()}</p>
    <form method="post" enctype="multipart/form-data" class="row">
      <input type="file" name="csv_file"/>
      <button>Run Predictions</button>
    </form>
    """
    return render_template_string(BASE_HTML, body=body)


@app.post("/predict-csv")
def predict_csv():
    try:
        file = request.files.get("csv_file")
        if not file:
            raise ValueError("No CSV uploaded")

        df = pd.read_csv(io.BytesIO(file.read()))
        if df.empty:
            raise ValueError("CSV is empty")

        X = force_feature_matrix(df, expected_feature_count())
        preds = predict_array(X)

        out = df.copy()
        out["prediction"] = preds

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"predictions_{ts}.csv"
        path = os.path.join(DOWNLOAD_DIR, fname)
        out.to_csv(path, index=False)

        body = f"""
        <div class='ok'>Predictions complete ({len(preds)} rows)</div>
        <a href="/downloads/{fname}">Download CSV</a>
        """

        return render_template_string(BASE_HTML, body=body)

    except Exception as e:
        body = f"<div class='err'>{e}</div><pre>{traceback.format_exc()}</pre>"
        return render_template_string(BASE_HTML, body=body), 400


@app.get("/downloads/<path:filename>")
def download(filename):
    return send_from_directory(DOWNLOAD_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
