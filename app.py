import os
import io
import pickle
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string, send_from_directory

app = Flask(__name__)

# -----------------------------
# Model loading
# -----------------------------
MODEL = None
MODEL_ERROR = None

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.pkl")

def load_model():
    global MODEL, MODEL_ERROR
    MODEL = None
    MODEL_ERROR = None
    try:
        with open(MODEL_PATH, "rb") as f:
            MODEL = pickle.load(f)
    except Exception as e:
        MODEL_ERROR = f"{type(e).__name__}: {e}"

load_model()

# -----------------------------
# Downloads directory (Render-safe)
# -----------------------------
def get_download_dir():
    # Prefer project /downloads if possible, else fall back to /tmp/downloads (writable on Render)
    preferred = os.path.join(HERE, "downloads")
    fallback = os.path.join("/tmp", "downloads")

    try:
        os.makedirs(preferred, exist_ok=True)
        # quick write test
        test_path = os.path.join(preferred, ".write_test")
        with open(test_path, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(test_path)
        return preferred
    except Exception:
        os.makedirs(fallback, exist_ok=True)
        return fallback

DOWNLOAD_DIR = get_download_dir()

# -----------------------------
# Helpers
# -----------------------------
def coerce_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all columns to numeric (float), coercing errors to NaN, then fill NaN with 0.
    This prevents: "can't multiply sequence by non-int of type 'float'"
    """
    # Strip whitespace in column names
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Drop obvious non-feature columns if present (common in ML datasets)
    drop_candidates = {"label", "target", "class", "y", "malware", "is_malware"}
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in drop_candidates:
        if cand in cols_lower:
            df = df.drop(columns=[cols_lower[cand]])

    # Convert everything to numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Fill missing with 0 (safe fallback)
    df = df.fillna(0.0)

    return df

def predict_array(X: np.ndarray):
    if MODEL is None:
        raise RuntimeError(f"Model not loaded. {MODEL_ERROR or ''}".strip())
    return MODEL.predict(X)

# -----------------------------
# Pages
# -----------------------------
PAGE_BASE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Malware Detector</title>
  <style>
    body { font-family: Arial, sans-serif; background:#f6f7fb; margin:0; padding:30px; }
    .card { background:white; max-width:760px; margin:0 auto; padding:26px; border-radius:14px; box-shadow: 0 10px 30px rgba(0,0,0,.08); }
    h1 { margin:0 0 10px 0; font-size:42px; }
    h2 { margin:18px 0 10px 0; font-size:30px; }
    a { color:#1a73e8; text-decoration:none; }
    a:hover { text-decoration:underline; }
    .row { display:flex; gap:14px; align-items:center; flex-wrap:wrap; }
    input[type=file] { padding:6px; }
    button { padding:10px 14px; border:0; border-radius:10px; cursor:pointer; font-weight:600; }
    button.primary { background:#111827; color:white; }
    .err { color:#b91c1c; font-weight:700; margin-top:12px; }
    .ok { color:#166534; font-weight:700; margin-top:12px; }
    .muted { color:#6b7280; }
    .small { font-size: 13px; }
    .divider { height:1px; background:#e5e7eb; margin:16px 0; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Malware Detector</h1>
    <div class="row">
      <a href="/manual">Manual prediction (single instance)</a>
      <span class="muted">|</span>
      <a href="/predict-csv">CSV Prediction</a>
      <span class="muted">|</span>
      <a href="/health">Health</a>
    </div>
    <div class="divider"></div>
    {{ body|safe }}
  </div>
</body>
</html>
"""

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
            "downloads_dir": DOWNLOAD_DIR,
        }, 500
    return {
        "status": "ok",
        "model_loaded": True,
        "downloads_dir": DOWNLOAD_DIR,
    }

@app.get("/manual")
def manual_page():
    body = """
    <h2>Manual prediction</h2>
    <p class="muted">Paste a single row of numeric feature values (comma-separated). Example: <span class="mono">0,1,0,3.14</span></p>
    <form method="post" action="/predict" class="row">
      <input type="text" name="features" style="flex:1; min-width:260px; padding:10px; border-radius:10px; border:1px solid #e5e7eb;" placeholder="e.g. 0,1,0,3.14" />
      <button class="primary" type="submit">Predict</button>
    </form>
    {% if error %}<div class="err">Error: {{ error }}</div>{% endif %}
    {% if result %}<div class="ok">Prediction: {{ result }}</div>{% endif %}
    """
    return render_template_string(PAGE_BASE, body=render_template_string(body, error=None, result=None))

@app.post("/predict")
def manual_predict():
    try:
        raw = (request.form.get("features") or "").strip()
        if not raw:
            raise ValueError("No features provided.")

        # Parse comma-separated numbers
        parts = [p.strip() for p in raw.split(",")]
        vals = []
        for p in parts:
            if p == "":
                vals.append(0.0)
            else:
                vals.append(float(p))
        X = np.array([vals], dtype=float)

        pred = predict_array(X)[0]
        result = str(pred)

        body = """
        <h2>Manual prediction</h2>
        <p class="muted">Paste a single row of numeric feature values (comma-separated).</p>
        <form method="post" action="/predict" class="row">
          <input type="text" name="features" style="flex:1; min-width:260px; padding:10px; border-radius:10px; border:1px solid #e5e7eb;" value="{{ raw|e }}" />
          <button class="primary" type="submit">Predict</button>
        </form>
        <div class="ok">Prediction: {{ result }}</div>
        """
        return render_template_string(PAGE_BASE, body=render_template_string(body, raw=raw, result=result))

    except Exception as e:
        body = """
        <h2>Manual prediction</h2>
        <p class="muted">Paste a single row of numeric feature values (comma-separated).</p>
        <form method="post" action="/predict" class="row">
          <input type="text" name="features" style="flex:1; min-width:260px; padding:10px; border-radius:10px; border:1px solid #e5e7eb;" value="{{ raw|e }}" />
          <button class="primary" type="submit">Predict</button>
        </form>
        <div class="err">Error: {{ err }}</div>
        <div class="small muted mono" style="white-space:pre-wrap; margin-top:8px;">{{ tb }}</div>
        """
        return render_template_string(
            PAGE_BASE,
            body=render_template_string(body, raw=request.form.get("features") or "", err=str(e), tb=traceback.format_exc()),
        ), 400

@app.get("/predict-csv")
def predict_csv_page(error=None, ok=None, download_link=None):
    body = """
    <h2>Upload CSV</h2>
    <form method="post" action="/predict-csv" enctype="multipart/form-data" class="row">
      <input type="file" name="csv_file" accept=".csv" />
      <button class="primary" type="submit">Run Predictions</button>
    </form>

    {% if error %}
      <div class="err">Error: {{ error }}</div>
      <div class="small muted mono" style="white-space:pre-wrap; margin-top:8px;">{{ tb }}</div>
    {% endif %}

    {% if ok %}
      <div class="ok">{{ ok }}</div>
      {% if download_link %}
        <p class="small"><a href="{{ download_link }}">Download output CSV</a></p>
      {% endif %}
      <p class="muted small">Saved to: <span class="mono">{{ downloads_dir }}</span></p>
    {% endif %}

    <div class="divider"></div>
    <p class="muted small">
      Notes: non-numeric cells will be coerced to NaN then filled with 0 to avoid type errors.
    </p>
    """
    return render_template_string(
        PAGE_BASE,
        body=render_template_string(
            body,
            error=error,
            ok=ok,
            download_link=download_link,
            downloads_dir=DOWNLOAD_DIR,
            tb=traceback.format_exc() if error else "",
        ),
    )

@app.post("/predict-csv")
def predict_csv():
    try:
        if MODEL is None:
            raise RuntimeError(f"Model not loaded. {MODEL_ERROR or ''}".strip())

        file = request.files.get("csv_file")
        if file is None or file.filename == "":
            raise ValueError("No CSV file uploaded.")

        # Read CSV robustly
        # (BytesIO so pandas can handle it reliably)
        content = file.read()
        if not content:
            raise ValueError("Uploaded file is empty.")

        df = pd.read_csv(io.BytesIO(content))

        if df.shape[0] == 0:
            raise ValueError("CSV has no rows.")
        if df.shape[1] == 0:
            raise ValueError("CSV has no columns.")

        # Coerce to numeric to prevent: can't multiply sequence by non-int of type 'float'
        features_df = coerce_numeric_df(df)

        # Predict
        X = features_df.to_numpy(dtype=float)
        preds = MODEL.predict(X)

        # Build output
        out = df.copy()
        out["prediction"] = preds

        # Save output
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"predictions_output_{ts}.csv"
        out_path = os.path.join(DOWNLOAD_DIR, out_name)
        out.to_csv(out_path, index=False)

        return predict_csv_page(
            ok=f"Done — generated {len(out)} predictions.",
            download_link=f"/downloads/{out_name}",
        )

    except Exception as e:
        return predict_csv_page(error=str(e)), 400

@app.get("/downloads/<path:filename>")
def downloads(filename):
    return send_from_directory(DOWNLOAD_DIR, filename, as_attachment=True)

if __name__ == "__main__":
    # Render sets PORT; locally defaults to 5000
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
