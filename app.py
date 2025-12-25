import os
import uuid
import pickle
import datetime as dt

import numpy as np
import pandas as pd

from flask import (
    Flask,
    request,
    render_template,
    jsonify,
    send_file,
    redirect,
    url_for,
)

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


app = Flask(__name__)

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
DOWNLOADS_DIR = os.path.join(BASE_DIR, "downloads")
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

# ---------------------------
# Load model once
# ---------------------------
def load_model(model_path: str):
    with open(model_path, "rb") as f:
        obj = pickle.load(f)

    # Case 1: it's already a model/pipeline
    if hasattr(obj, "predict"):
        return obj

    # Case 2: common pattern: pickle contains a dict bundle
    if isinstance(obj, dict):
        for k in ("model", "pipeline", "estimator", "clf", "classifier"):
            if k in obj and hasattr(obj[k], "predict"):
                return obj[k]

    # Case 3: common pattern: tuple/list bundle (model first)
    if isinstance(obj, (list, tuple)) and len(obj) > 0 and hasattr(obj[0], "predict"):
        return obj[0]

    # If we get here, it's genuinely not a usable model
    raise TypeError(
        f"Loaded object from {model_path} is {type(obj)} and has no predict(). "
        f"If it's a bundle, ensure it contains a key like 'model'/'pipeline' with a sklearn estimator."
    )

model = load_model(MODEL_PATH)



# ---------------------------
# Helpers
# ---------------------------
def _model_feature_names():
    """
    If the model/pipeline was fit on a DataFrame, scikit-learn may store feature_names_in_.
    We use it to:
      - show a correct manual-entry form
      - align CSV columns at inference time
    """
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        return None
    return list(names)


def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    """Convert a pandas series to numeric where possible; keep NaN for bad values."""
    return pd.to_numeric(s, errors="coerce")


def _align_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Align incoming df to the model's expected feature list (if known).
    - Drops Label if present
    - Reorders columns
    - Adds missing columns with 0
    - Coerces to numeric
    """
    df = df.copy()

    if "Label" in df.columns:
        df = df.drop(columns=["Label"])

    expected = _model_feature_names()
    if expected:
        # add missing cols as 0
        for c in expected:
            if c not in df.columns:
                df[c] = 0

        # drop extra columns not used by model
        extra = [c for c in df.columns if c not in expected]
        if extra:
            df = df.drop(columns=extra)

        # reorder
        df = df[expected]

    # coerce all columns numeric (safe for sklearn)
    for c in df.columns:
        df[c] = _coerce_numeric_series(df[c])

    # fill remaining NaNs (if user left blanks) with 0
    df = df.fillna(0)

    return df


def _safe_predict_proba_or_score(X: pd.DataFrame):
    """
    Returns a float score per row used for ROC AUC.
    Prefer predict_proba (class 1 prob), else decision_function.
    If neither exists, returns None.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.ravel()

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return np.asarray(scores).ravel()

    return None


def _make_confusion_matrix_html(cm: np.ndarray, labels=("0", "1")) -> str:
    """Return a simple HTML table string for confusion matrix."""
    return f"""
    <table border="1" cellpadding="6" cellspacing="0">
      <tr>
        <th></th><th>Pred {labels[0]}</th><th>Pred {labels[1]}</th>
      </tr>
      <tr>
        <th>True {labels[0]}</th><td>{cm[0,0]}</td><td>{cm[0,1]}</td>
      </tr>
      <tr>
        <th>True {labels[1]}</th><td>{cm[1,0]}</td><td>{cm[1,1]}</td>
      </tr>
    </table>
    """


def _predict_labels(X: pd.DataFrame) -> pd.Series:
    preds = model.predict(X)
    return pd.Series(np.asarray(preds).ravel(), name="Prediction")


def _label_to_text(pred_value: int) -> str:
    try:
        return "malware" if int(pred_value) == 1 else "goodware"
    except Exception:
        return str(pred_value)


# ---------------------------
# Routes
# ---------------------------
@app.get("/health")
def health():
    return jsonify(
        status="ok",
        model_path=MODEL_PATH,
        model_type=str(type(model)),
        downloads_dir=DOWNLOADS_DIR,
        expected_features=_model_feature_names(),
    )


@app.get("/")
def home():
    # If you have an index.html, you can swap this to render_template("index.html")
    return redirect(url_for("predict_csv"))


# ---------------------------
# Manual single-row prediction
# ---------------------------
@app.route("/predict-manual", methods=["GET", "POST"])
def predict_manual():
    # Build demo dict based on model expected features (best for marking)
    feature_names = _model_feature_names()

    if feature_names:
        demo = {name: 0 for name in feature_names}
    else:
        # fallback demo (only used if feature_names_in_ not available)
        demo = {
            "BaseOfCode": 4096,
            "BaseOfData": 8192,
            "Characteristics": 258,
            "DllCharacteristics": 0,
            "Entropy": 6.5,
            "FileAlignment": 512,
            "ImageBase": 4194304,
            "ImportedDlls": 12,
            "ImportedSymbols": 900,
            "Machine": 332,
            "Magic": 267,
            "NumberOfRvaAndSizes": 16,
            "NumberOfSections": 5,
            "NumberOfSymbols": 0,
            "PE_TYPE": 0,
            "PointerToSymbolTable": 0,
            "Size": 123456,
            "SizeOfCode": 40960,
            "SizeOfHeaders": 1024,
            "SizeOfImage": 65536,
            "SizeOfInitializedData": 8192,
            "SizeOfOptionalHeader": 224,
            "SizeOfUninitializedData": 0,
            "TimeDateStamp": 1234567890,
        }

    if request.method == "GET":
        return render_template(
            "predict_manual.html",
            demo=demo,
            result=None,
            error=None,
        )

    # POST
    try:
        row = {}

        # Only accept expected feature names (prevents random form fields breaking things)
        allowed = set(feature_names) if feature_names else None

        for k, v in request.form.items():
            if allowed is not None and k not in allowed:
                continue

            if v is None or str(v).strip() == "":
                row[k] = 0
                continue

            # try int then float
            try:
                row[k] = int(v)
            except ValueError:
                row[k] = float(v)

        X = pd.DataFrame([row])
        X = _align_features(X)

        pred = _predict_labels(X).iloc[0]
        label_text = _label_to_text(pred)

        # show what user submitted in the form (keep fields stable)
        show_demo = {**demo, **row}

        return render_template(
            "predict_manual.html",
            demo=show_demo,
            result=label_text,
            error=None,
        )

    except Exception as e:
        # show the posted fields if possible
        show_demo = {**demo, **dict(request.form)}
        return render_template(
            "predict_manual.html",
            demo=show_demo,
            result=None,
            error=str(e),
        )


# ---------------------------
# CSV prediction + evaluation
# ---------------------------
@app.route("/predict-csv", methods=["GET", "POST"])
def predict_csv():
    if request.method == "GET":
        return render_template(
            "predict_csv.html",
            metrics=None,
            preview=None,
            download_url=None,
            error=None,
        )

    # POST
    try:
        if "file" not in request.files:
            return render_template(
                "predict_csv.html",
                metrics=None,
                preview=None,
                download_url=None,
                error="No file uploaded (missing form field named 'file').",
            )

        f = request.files["file"]
        if not f or f.filename == "":
            return render_template(
                "predict_csv.html",
                metrics=None,
                preview=None,
                download_url=None,
                error="No file selected.",
            )

        df = pd.read_csv(f)

        has_labels = "Label" in df.columns
        y_true = df["Label"].copy() if has_labels else None

        X = _align_features(df)
        preds = _predict_labels(X)

        df_out = df.copy()
        df_out["Prediction"] = preds
        df_out["PredictionLabel"] = df_out["Prediction"].apply(_label_to_text)

        metrics = None
        if has_labels:
            # Ensure y_true numeric 0/1
            y_true_num = pd.to_numeric(y_true, errors="coerce").fillna(0).astype(int)

            acc = float(accuracy_score(y_true_num, preds.astype(int)))

            scores = _safe_predict_proba_or_score(X)
            auc = None
            if scores is not None:
                try:
                    auc = float(roc_auc_score(y_true_num, scores))
                except Exception:
                    auc = None

            cm = confusion_matrix(y_true_num, preds.astype(int), labels=[0, 1])
            cm_html = _make_confusion_matrix_html(cm, labels=("0", "1"))

            metrics = {
                "accuracy": acc,
                "auc": auc,
                "confusion_matrix_html": cm_html,
            }

        # Save output CSV to downloads dir
        token = uuid.uuid4().hex[:10]
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"predictions_{ts}_{token}.csv"
        out_path = os.path.join(DOWNLOADS_DIR, out_name)
        df_out.to_csv(out_path, index=False)

        # Preview for marking (first 10 rows)
        preview = df_out.head(10).to_dict(orient="records")
        download_url = url_for("download_file", filename=out_name)

        return render_template(
            "predict_csv.html",
            metrics=metrics,
            preview=preview,
            download_url=download_url,
            error=None,
        )

    except Exception as e:
        return render_template(
            "predict_csv.html",
            metrics=None,
            preview=None,
            download_url=None,
            error=str(e),
        )


@app.get("/download/<path:filename>")
def download_file(filename):
    full_path = os.path.join(DOWNLOADS_DIR, filename)
    if not os.path.isfile(full_path):
        return f"File not found: {filename}", 404

    # This returns the file in the HTTP response so the browser downloads it to the user's PC
    return send_file(full_path, as_attachment=True, download_name=filename, mimetype="text/csv")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
