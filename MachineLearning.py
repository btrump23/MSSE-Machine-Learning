# MachineLearning.py
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Optional

# ✅ Import the URL from ONE place only (never redefine it here)
from config import MODEL_ZIP_URL


# -----------------------------------------------------------------------------
# Configuration: change these to match what your model.zip actually contains
# -----------------------------------------------------------------------------
DEFAULT_MODEL_FILENAME = "model.pkl"        # e.g. "model.pkl"
DEFAULT_PREPROC_FILENAME = "preprocess.pkl" # optional; set to None if not used
DEFAULT_FEATURES_FILENAME = "features.txt"  # optional; set to None if not used

# If you do NOT use these extra artifacts, set them to None:
# DEFAULT_PREPROC_FILENAME = None
# DEFAULT_FEATURES_FILENAME = None


@dataclass
class LoadedArtifacts:
    model: Any
    preprocessor: Optional[Any] = None
    feature_names: Optional[list[str]] = None


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _read_pickle(path: str) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def _read_features(path: str) -> list[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f.readlines() if ln.strip()]


# -----------------------------------------------------------------------------
# Public API used by app.py
# -----------------------------------------------------------------------------
def load_artifacts(model_dir: str) -> LoadedArtifacts:
    """
    Loads model + optional preprocessing artifacts from the extracted model directory.

    model_dir should be the directory returned by ensure_model_ready() in app.py,
    e.g. ".cache/model"
    """
    if not isinstance(model_dir, str) or not model_dir:
        raise RuntimeError(f"model_dir is invalid: {model_dir!r}")

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    model_path = os.path.join(model_dir, DEFAULT_MODEL_FILENAME)
    model = _read_pickle(model_path)

    preprocessor = None
    if DEFAULT_PREPROC_FILENAME:
        preproc_path = os.path.join(model_dir, DEFAULT_PREPROC_FILENAME)
        preprocessor = _read_pickle(preproc_path)

    feature_names = None
    if DEFAULT_FEATURES_FILENAME:
        features_path = os.path.join(model_dir, DEFAULT_FEATURES_FILENAME)
        feature_names = _read_features(features_path)

    return LoadedArtifacts(model=model, preprocessor=preprocessor, feature_names=feature_names)


def predict_from_rows(artifacts: LoadedArtifacts, rows: Any) -> Any:
    """
    Predict on rows (typically a pandas DataFrame).
    - If a preprocessor exists, applies it first.
    - Returns raw model predictions.
    """
    if artifacts is None or artifacts.model is None:
        raise RuntimeError("Artifacts not loaded (artifacts.model is None)")

    X = rows
    if artifacts.preprocessor is not None:
        # Common: sklearn ColumnTransformer / Pipeline, etc.
        X = artifacts.preprocessor.transform(rows)

    model = artifacts.model

    # Prefer predict_proba when available (classification)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)

    # Otherwise fall back to predict (regression or classifiers without proba)
    if hasattr(model, "predict"):
        return model.predict(X)

    raise RuntimeError("Loaded model has neither predict_proba nor predict")


def get_model_info(artifacts: LoadedArtifacts) -> Dict[str, Any]:
    """
    Useful for debugging and showing status in the UI.
    """
    info: Dict[str, Any] = {
        "model_type": type(artifacts.model).__name__ if artifacts and artifacts.model else None,
        "has_preprocessor": artifacts.preprocessor is not None if artifacts else False,
        "feature_count": len(artifacts.feature_names) if artifacts and artifacts.feature_names else None,
        "model_zip_url": MODEL_ZIP_URL,  # ✅ string from config.py
    }
    return info
