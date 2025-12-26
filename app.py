import os
import io
import pickle
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string, send_from_directory

# -------------------------------------------------
# App setup
# -------------------------------------------------
app = Flask(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.pkl")

MODEL = None
MODEL_ERROR = None

# -------------------------------------------------
# Load model
# -------------------------------------------------
def load_model():
    global MODEL, MODEL_ERROR
    try:
        with open(MODEL_PATH, "rb") as f:
            MODEL = pickle.load(f)
        MODEL_ERROR = None
    except Exception as e:
        MODEL = None
        MODEL_ERROR = f"{type(e).__name__}: {e}"

load_model()

# -------------------------------------------------
# Downloads directory (Render-safe)
# -------------------------------------------------
def get_download_dir():
    preferred = os.path.join(HERE, "downloads")
    fallback = os.path.join("/tmp", "downloads")

    try:
        os.makedirs(preferred, exist_ok=True)
        test = os.path.join(preferred, ".test")
        with open(test, "w") as f:
            f.write("ok")
        os.remove(test)
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

def coerce_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Drop common label columns
    for c in list(df.columns):
        if c.lower() in {"label", "target", "class", "y", "malware", "is_malware"}:
            df.drop(columns=[c], inplace=True)

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.fillna(0.0)

def predict_array(X):
    if MODEL is None:
        raise RuntimeError(f"Model not loaded: {MODEL_ERROR}")
    return MODEL.predict(X)

# -------------------------------------------------
# HTML template
# -------------------------------------------------
PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Malware Detector</title>
<style>
body { font-family: Arial; background:#f6f7fb; padding:30px; }
.card { background:white; max-width:900px; margin:auto; padding:30px;
        border-radius:14px; box-shadow:0 10px 30px rgba(0,0,0,.08); }
h1 { font-size:42px; margin-bottom:8px; }
h2 { font-size:28px; margin-top:20px; }
a { color:#1a73e8; text-decoration:none; }
a:hover { text-decoration:underline; }
.row { display:flex; gap:14px; align-items:center; flex-wrap:wrap; }
button { padding:10px 16px; border:0; border-radius:10px; cursor:pointer;
         background:#111827; color:white; font-weight:600; }
input[type=text], input[type=file] {
    padding:10px; border-radius:10px; border:1px solid #e5e7eb;
    min-width:280px;
}
.err { color:#b91c1c; font-weight:700; margin-top:12px; }
.ok { color:#166534; font-weight:700; margin-top:12px; }
.muted { color:#6b7280; }
.mono { font-family: monospace; }
.small { font-size:13px; }
hr { border:0; height:1px; background:#e5e7eb; margin:16px 0; }
</style>
</head>
<body>
<div class="card">
<h1>Malware Detector</h1>
<div class="row">
<a href="/manual">Manual prediction (single instance)</a> |
<a href="/predict-csv">CSV Prediction</a> |
<a href="/health">Health</a>
</div>
<hr>
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
    return manual_page()

@app.get("/health")
def health():
    if MODEL is None:
        return {"status": "error", "model_loaded": False, "error": MODEL_ERROR}, 500
    return {
        "status": "ok",
        "model_loaded": True,
        "expected_features": expected_feature_count(),
        "downloads_dir": DOWNLOAD_DIR
    }

# ---------------- Manual prediction ----------------
@app.get("/manual")
def manual_page():
    body = """
    <h2>Manual prediction</h2>
    <p class="muted">
      Paste a single row of numeric feature values (comma-separated).
      Model expects <span class="mono">{{ n }}</span> values.
      Missing values are padded with 0.
    </p>
    <form method="post" action="/predict" class="row">
      <input type="text" name="features" placeholder="e.g. 0,1,0,4,5" />
      <button type="submit">Predict</button>
    </form>
    """
    return render_template_string(PAGE, body=render_template_string(body, n=expected_feature_count()))

@app.post("/predict")
def manual_predict():
    try:
        raw = (request.form.get("features") or "").strip()
        if not raw:
            raise ValueError("No features provided")

        vals = []
        for p in raw.split(","):
            p = p.strip()
            vals.append(float(p) if p else 0.0)

        n = expected_feature_count()

        if len(vals) < n:
            vals += [0.0] * (n - len(vals))
        elif len(vals) > n:
            vals = vals[:n]

        X = np.array([vals], dtype=float)
        pred = predict_array(X)[0]

        body = f"""
        <h2>Manual prediction</h2>
        <p class="ok">Prediction: <span class="mono">{pred}</span></p>
        <p class="small muted">Expected features: {n}</p>
        <a href="/manual">Run another</a>
        """
        return render_template_string(PAGE, body=body)

    except Exception as e:
        body = f"""
        <h2>Manual prediction</h2>
        <div class="err">Error: {e}</div>
