import os
import io
import traceback
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory

# Prefer joblib for sklearn artifacts (train_save_pipeline.py uses joblib.dump)
try:
    import joblib
except Exception:
    joblib = None

import pickle

app = Flask(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.pkl")
DOWNLOADS_DIR = os.path.join(HERE, "downloads")
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

MODEL = None
MODEL_ERROR = None

# Columns to drop if present (your dataset has lots of non-numeric identifiers)
DROP_COLS = {
    "Label", "label", "target", "Target", "y",
    "MD5", "SHA1", "Name", "Identify", "PE_TYPE", "Magic", "Machine",
    "FormatedTimeDateStamp", "TimeDateStamp"
}

DEFAULT_EXPECTED_FEATURES = 29


def _infer_expected_features(model) -> int:
    # sklearn estimators often expose n_features_in_
    n = getattr(model, "n_features_in_", None)
    if isinstance(n, (int, np.integer)) and n > 0:
        return int(n)

    # your custom wrapper likely has weights
    w = getattr(model, "weights", None)
    if w is not None:
        try:
            return int(np.asarray(w).reshape(-1).shape[0])
        except Exception:
            pass

    return DEFAULT_EXPECTED_FEATURES


def _load_model():
    global MODEL, MODEL_ERROR
    MODEL = None
    MODEL_ERROR = None

    if not os.path.exists(MODEL_PATH):
        MODEL_ERROR = f"model.pkl not found at {MODEL_PATH}"
        return

    # 1) Try joblib.load first (correct for sklearn pipelines)
    if joblib is not None:
        try:
            obj = joblib.load(MODEL_PATH)
            # Guard: you accidentally saved feature_names array (dtype object of strings)
            if isinstance(obj, np.ndarray) and obj.dtype == object:
                first = list(obj[:5]) if obj.size else []
                if any(isinstance(x, str) for x in first):
                    MODEL_ERROR = (
                        "model.pkl is not a trained model. It looks like feature_names were saved "
                        f"(e.g. {first}). Rebuild model.pkl by running: python train_save_pipeline.py"
                    )
                    return
            MODEL = obj
            return
        except Exception as e:
            # keep error, fall back to pickle
            joblib_err = f"joblib.load failed: {e}"
    else:
        joblib_err = None

    # 2) Fallback to pickle.load
    try:
        with open(MODEL_PATH, "rb") as f:
            obj = pickle.load(f)

        # Guard: header array saved by mistake
        if isinstance(obj, np.ndarray) and obj.dtype == object:
            first = list(obj[:5]) if obj.size else []
            if any(isinstance(x, str) for x in first):
                MODEL_ERROR = (
                    "model.pkl is not a trained model. It looks like feature_names were saved "
                    f"(e.g. {first}). Rebuild model.pkl by running: python train_save_pipeline.py"
                )
                return

        MODEL = obj
        return
    except Exception as e:
        MODEL_ERROR = f"pickle.load failed: {e}"
        if joblib_err:
            MODEL_ERROR += f" | {joblib_err}"


def _coerce_numeric_matrix(df: pd.DataFrame, expected_features: int) -> np.ndarray:
    # drop obvious non-feature columns
    cols_to_drop = [c for c in df.columns if c in DROP_COLS]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop, errors="ignore")

    # keep only numeric-ish columns; coerce everything else to NaN then fill with 0
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    X = df.to_numpy(dtype=float)

    # Ensure 2D
    if X.ndim == 1:
        X = X.reshape(1, -1)

    # Pad/truncate to expected feature count
    if X.shape[1] < expected_features:
        pad = expected_features - X.shape[1]
        X = np.hstack([X, np.zeros((X.shape[0], pad), dtype=float)])
    elif X.shape[1] > expected_features:
        X = X[:, :expected_features]

    return X


def _predict_proba_or_label(model, X: np.ndarray):
    # returns (labels, probs or None)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        probs = np.asarray(probs)
        if probs.ndim == 2 and probs.shape[1] >= 2:
            p1 = probs[:, 1]
        else:
            p1 = probs.reshape(-1)
        labels = (p1 >= 0.5).astype(int)
        return labels, p1

    # if custom wrapper has decision_function
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = np.asarray(scores).reshape(-1)
        # logistic
        p1 = 1.0 / (1.0 + np.exp(-scores))
        labels = (p1 >= 0.5).astype(int)
        return labels, p1

    # last resort
    if hasattr(model, "predict"):
        labels = model.predict(X)
        labels = np.asarray(labels).reshape(-1)
        return labels, None

    raise RuntimeError("Loaded model has no predict/predict_proba/decision_function")


@app.route("/")
def home():
    # Make homepage GET-only to avoid 405 confusion
    return (
        "<h2>Malware Detector</h2>"
        "<ul>"
        "<li><a href='/predict'>Manual prediction (single instance)</a></li>"
        "<li><a href='/predict-csv'>CSV Prediction</a></li>"
        "<li><a href='/health'>Health</a></li>"
        "</ul>"
    )


@app.route("/health")
def health():
    if MODEL is None and MODEL_ERROR is None:
        _load_model()
    expected = _infer_expected_features(MODEL) if MODEL is not None else 0
    return jsonify({
        "status": "ok" if MODEL is not None else "error",
        "model_loaded": MODEL is not None,
        "model_error": MODEL_ERROR,
        "expected_features": expected,
        "downloads_dir": DOWNLOADS_DIR
    })


@app.route("/predict", methods=["GET", "POST"])
def manual_predict():
    if MODEL is None and MODEL_ERROR is None:
        _load_model()

    expected = _infer_expected_features(MODEL) if MODEL is not None else DEFAULT_EXPECTED_FEATURES

    if request.method == "GET":
        return f"""
        <html><body style="font-family: Arial; max-width: 900px; margin: 40px auto;">
          <h1>Malware Detector</h1>
          <div style="margin-bottom: 12px;">
            <a href="/predict">Manual prediction (single instance)</a> |
            <a href="/predict-csv">CSV Prediction</a> |
            <a href="/health">Health</a>
          </div>
          <h2>Manual prediction</h2>
          <p>Paste a single row of numeric feature values (comma-separated). Model expects {expected} values.
             If fewer, we pad with zeros; if more, we truncate.</p>

          <form method="POST">
            <input name="row" style="width: 70%; padding: 10px;" placeholder="e.g. 0,1,2,3,..."/>
            <button type="submit" style="padding: 10px 18px;">Predict</button>
          </form>
        </body></html>
        """

    # POST
    if MODEL is None:
        return f"""
        <html><body style="font-family: Arial; max-width: 900px; margin: 40px auto;">
          <h1>Malware Detector</h1>
          <div style="color: red; font-weight: bold;">Error: Model not loaded. {MODEL_ERROR}</div>
          <p>Fix: Run <code>python train_save_pipeline.py</code> to rebuild a valid <code>model.pkl</code>.</p>
          <a href="/health">Health</a>
        </body></html>
        """

    try:
        row = request.form.get("row", "").strip()
        vals = [v.strip() for v in row.split(",") if v.strip() != ""]
        df = pd.DataFrame([vals])

        X = _coerce_numeric_matrix(df, expected)
        labels, probs = _predict_proba_or_label(MODEL, X)

        label = int(labels[0])
        prob_txt = f"{float(probs[0]):.6f}" if probs is not None else "n/a"

        return f"""
        <html><body style="font-family: Arial; max-width: 900px; margin: 40px auto;">
          <h1>Malware Detector</h1>
          <div style="margin-bottom: 12px;">
            <a href="/predict">Manual prediction (single instance)</a> |
            <a href="/predict-csv">CSV Prediction</a> |
            <a href="/health">Health</a>
          </div>
          <h2>Result</h2>
          <p><b>Prediction:</b> {label} &nbsp;&nbsp; <b>Prob(malware):</b> {prob_txt}</p>
          <p><a href="/predict">Try another</a></p>
        </body></html>
        """
    except Exception as e:
        tb = traceback.format_exc()
        return f"""
        <html><body style="font-family: Arial; max-width: 900px; margin: 40px auto;">
          <h1>Malware Detector</h1>
          <div style="color: red; font-weight: bold;">Error: {e}</div>
          <pre>{tb}</pre>
          <p><a href="/predict">Back</a></p>
        </body></html>
        """


@app.route("/predict-csv", methods=["GET", "POST"])
def predict_csv():
    if MODEL is None and MODEL_ERROR is None:
        _load_model()

    expected = _infer_expected_features(MODEL) if MODEL is not None else DEFAULT_EXPECTED_FEATURES

    if request.method == "GET":
        note = "Model is loaded." if MODEL is not None else f"Model not loaded: {MODEL_ERROR}"
        return f"""
        <html><body style="font-family: Arial; max-width: 900px; margin: 40px auto;">
          <h1>Malware Detector</h1>
          <div style="margin-bottom: 12px;">
            <a href="/predict">Manual prediction (single instance)</a> |
            <a href="/predict-csv">CSV Prediction</a> |
            <a href="/health">Health</a>
          </div>
          <h2>Upload CSV</h2>
          <p>Expected features: {expected}. We drop common label columns and coerce non-numeric values to 0.
             Then we pad/truncate to exactly {expected} features.</p>
          <p><i>{note}</i></p>
          <form method="POST" enctype="multipart/form-data">
            <input type="file" name="file" />
            <button type="submit" style="padding: 10px 18px;">Run Predictions</button>
          </form>
        </body></html>
        """

    # POST
    if MODEL is None:
        return f"""
        <html><body style="font-family: Arial; max-width: 900px; margin: 40px auto;">
          <h1>Malware Detector</h1>
          <div style="color: red; font-weight: bold;">Error: Model not loaded. {MODEL_ERROR}</div>
          <p>Fix: Run <code>python train_save_pipeline.py</code> to rebuild a valid <code>model.pkl</code>.</p>
          <a href="/health">Health</a>
        </body></html>
        """

    try:
        f = request.files.get("file")
        if not f or f.filename == "":
            raise ValueError("No file selected")

        # Read CSV
        content = f.read()
        df = pd.read_csv(io.BytesIO(content))

        X = _coerce_numeric_matrix(df, expected)
        labels, probs = _predict_proba_or_label(MODEL, X)

        out = df.copy()
        out["Prediction"] = labels
        if probs is not None:
            out["Prob_malware"] = probs

        out_name = os.path.splitext(os.path.basename(f.filename))[0] + "_predictions.csv"
        out_path = os.path.join(DOWNLOADS_DIR, out_name)
        out.to_csv(out_path, index=False)

        return f"""
        <html><body style="font-family: Arial; max-width: 900px; margin: 40px auto;">
          <h1>Malware Detector</h1>
          <div style="margin-bottom: 12px;">
            <a href="/predict">Manual prediction (single instance)</a> |
            <a href="/predict-csv">CSV Prediction</a> |
            <a href="/health">Health</a>
          </div>
          <h2>Done</h2>
          <p>Wrote output to <code>downloads/{out_name}</code></p>
          <p><a href="/downloads/{out_name}">Download predictions</a></p>
          <p><a href="/predict-csv">Run another</a></p>
        </body></html>
        """
    except Exception as e:
        tb = traceback.format_exc()
        return f"""
        <html><body style="font-family: Arial; max-width: 900px; margin: 40px auto;">
          <h1>Malware Detector</h1>
          <div style="color: red; font-weight: bold;">Error: {e}</div>
          <pre>{tb}</pre>
          <p><a href="/predict-csv">Back</a></p>
        </body></html>
        """


@app.route("/downloads/<path:filename>")
def downloads(filename):
    return send_from_directory(DOWNLOADS_DIR, filename, as_attachment=True)


# Load model at import time
_load_model()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
