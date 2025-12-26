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
# Paths / Model load
# -----------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.pkl")

MODEL = None
MODEL_ERROR = None


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
# Downloads dir (Render-safe)
# -----------------------------
def get_download_dir():
    preferred = os.path.join(HERE, "downloads")
    fallback = os.path.join("/tmp", "downloads")

    try:
        os.makedirs(preferred, exist_ok=True)
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
def expected_feature_count() -> int:
    if MODEL is None:
        return 0
    w = getattr(MODEL, "weights", None)
    if w is None:
        return 0
    try:
        return int(len(w))
    except Exception:
        return 0


def coerce_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all columns to numeric floats, coercing errors to NaN, then fill NaN with 0.
    Also drops common label/target columns if present.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    drop_names = {"label", "target", "class", "y", "malware", "is_malware"}
    for c in list(df.columns):
        if str(c).strip().lower() in drop_names:
            df.drop(columns=[c], inplace=True)

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.fillna(0.0)


def predict_array(X: np.ndarray):
    if MODEL is None:
        raise RuntimeError(f"Model not loaded. {MODEL_ERROR or ''}".strip())
    return MODEL.predict(X)


# -----------------------------
# Templates
# -----------------------------
PAGE_BASE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Malware Detector</title>
  <style>
    body { font-family: Arial, sans-serif; background:#f6f7fb; margin:0; padding:30px; }
    .card { background:white; max-width:900px; margin:0 auto; padding:26px; border-radius:14px; box-shadow: 0 10px 30px rgba(0,0,0,.08); }
    h1 { margin:0 0 10px 0; font-size:42px; }
    h2 { margin:18px 0 10px 0; font-size:30px; }
    a { color:#1a73e8; text-decoration:none; }
    a:hover { text-decoration:underline; }
    .row { display:flex; gap:14px; align-items:center; flex-wrap:wrap; }
    input[type=file], input[type=text] { padding:10px; border-radius:10px; border:1px solid #e5e7eb; min-width:280px; }
    button { padding:10px 14px; border:0; border-radius:10px; cursor:pointer; font-weight:600; background:#111827; color:white; }
    .err { color:#b91c1c; font-weight:700; margin-top:12px; }
    .ok { color:#166534; font-weight:700; margin-top:12px; }
    .muted { color:#6b7280; }
    .small { font-size: 13px; }
    .divider { height:1px; background:#e5e7eb; margin:16px 0; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
    pre { white-space: pre-wrap; }
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

MANUAL_BODY = """
<h2>Manual prediction</h2>
<p class="muted">
  Paste a single row of numeric feature values (comma-separated).
  Model expects <span class="mono">{{ expected_n }}</span> values.
  If you provide fewer, the app pads with zeros; if more, it truncates.
</p>

<form method="post" action="/predict" class="row">
  <input type="text" name="features" value="{{ raw|e }}" placeholder="e.g. 0,1,0,4,5" />
  <button type="submit">Predict</button>
</form>

{% if result is not none %}
  <div class="ok">Prediction: <span class="mono">{{ result }}</span></div>
{% endif %}

{% if error %}
  <div class="err">Error: {{ error }}</div>
  <pre class="small muted mono">{{ tb }}</pre>
{% endif %}
"""

CSV_BODY = """
<h2>Upload CSV</h2>

<form method="post" action="/predict-csv" enctype="multipart/form-data" class="row">
  <input type="file" name="csv_file" accept=".csv" />
  <button type="submit">Run Predictions</button>
</form>

<p class="small muted">
  Expected features: <span class="mono">{{ expected_n }}</span>.
  Non-numeric cells are coerced then filled with 0.
</p>

{% if ok %}
  <div class="ok">{{ ok }}</div>
  {% if download_link %}
    <p class="small"><a href="{{ download_link }}">Download output CSV</a></p>
  {% endif %}
  <p class="muted small">Saved to: <span class="mono">{{ downloads_dir }}</span></p>
{% endif %}

{% if error %}
  <div class="err">Error: {{ error }}</div>
  <pre class="small muted mono">{{ tb }}</pre>
{% endif %}
"""

# -----------------------------
# Routes
# -----------------------------
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
            "expected_features": expected_feature_count(),
            "downloads_dir": DOWNLOAD_DIR,
        }, 500
    return {
        "status": "ok",
        "model_loaded": True,
        "expected_features": expected_feature_count(),
        "downloads_dir": DOWNLOAD_DIR,
    }


@app.get("/manual")
def manual_page():
    body = render_template_string(
        MANUAL_BODY,
        expected_n=expected_feature_count(),
        raw="",
        result=None,
        error="Model not loaded: " + MODEL_ERROR if MODEL is None else "",
        tb="",
    )
    return render_template_string(PAGE_BASE, body=body)


@app.post("/predict")
def manual_predict():
    try:
        if MODEL is None:
            raise RuntimeError(f"Model not loaded. {MODEL_ERROR or ''}".strip())

        raw = (request.form.get("features") or "").strip()
        if not raw:
            raise ValueError("No features provided.")

        parts = [p.strip() for p in raw.split(",")]
        vals = []
        for p in parts:
            if p == "":
                vals.append(0.0)
            else:
                vals.append(float(p))

        expected_n = expected_feature_count()
        if expected_n > 0:
            if len(vals) < expected_n:
                vals = vals + [0.0] * (expected_n - len(vals))
            elif len(vals) > expected_n:
                vals = vals[:expected_n]

        X = np.array([vals], dtype=float)
        pred = predict_array(X)[0]

        body = render_template_string(
            MANUAL_BODY,
            expected_n=expected_n,
            raw=raw,
            result=str(pred),
            error="",
            tb="",
        )
        return render_template_string(PAGE_BASE, body=body)

    except Exception as e:
        body = render_template_string(
            MANUAL_BODY,
            expected_n=expected_feature_count(),
            raw=request.form.get("features") or "",
            result=None,
            error=str(e),
            tb=traceback.format_exc(),
        )
        return render_template_string(PAGE_BASE, body=body), 400


@app.get("/predict-csv")
def predict_csv_page():
    body = render_template_string(
        CSV_BODY,
        expected_n=expected_feature_count(),
        ok="",
        error="Model not loaded: " + MODEL_ERROR if MODEL is None else "",
        tb="",
        downloads_dir=DOWNLOAD_DIR,
        download_link="",
    )
    return render_template_string(PAGE_BASE, body=body)


@app.post("/predict-csv")
def predict_csv():
    try:
        if MODEL is None:
            raise RuntimeError(f"Model not loaded. {MODEL_ERROR or ''}".strip())

        file = request.files.get("csv_file")
        if file is None or file.filename == "":
            raise ValueError("No CSV file uploaded.")

        content = file.read()
        if not content:
            raise ValueError("Uploaded file is empty.")

        df = pd.read_csv(io.BytesIO(content))
        if df.shape[0] == 0:
            raise ValueError("CSV has no rows.")
        if df.shape[1] == 0:
            raise ValueError("CSV has no columns.")

        features_df = coerce_numeric_df(df)
        X = features_df.to_numpy(dtype=float)
        preds = predict_array(X)

        out = df.copy()
        out["prediction"] = preds

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"predictions_output_{ts}.csv"
        out_path = os.path.join(DOWNLOAD_DIR, out_name)
        out.to_csv(out_path, index=False)

        body = render_template_string(
            CSV_BODY,
            expected_n=expected_feature_count(),
            ok=f"Done — generated {len(out)} predictions.",
            error="",
            tb="",
            downloads_dir=DOWNLOAD_DIR,
            download_link=f"/downloads/{out_name}",
        )
        return render_template_string(PAGE_BASE, body=body)

    except Exception as e:
        body = render_template_string(
            CSV_BODY,
            expected_n=expected_feature_count(),
            ok="",
            error=str(e),
            tb=traceback.format_exc(),
            downloads_dir=DOWNLOAD_DIR,
            download_link="",
        )
        return render_template_string(PAGE_BASE, body=body), 400


@app.get("/downloads/<path:filename>")
def downloads(filename):
    return send_from_directory(DOWNLOAD_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
