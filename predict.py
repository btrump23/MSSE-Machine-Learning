# predict.py
import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

MODEL_PATH = Path("artifacts") / "lgbm_pipeline.joblib"
LABEL_COL = "Label"


def load_bundle(model_path: Path):
    bundle = joblib.load(model_path)
    if isinstance(bundle, dict):
        if "pipeline" not in bundle:
            raise KeyError(f"Bundle missing 'pipeline'. Keys: {list(bundle.keys())}")
        if "feature_names" not in bundle:
            raise KeyError(f"Bundle missing 'feature_names'. Keys: {list(bundle.keys())}")
        return bundle
    # If you ever saved only the pipeline, we can't auto-align names
    raise TypeError(
        "Expected a dict bundle containing {'pipeline', 'feature_names'} "
        f"but got: {type(bundle)}"
    )


def align_to_training_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    # Drop label if present
    if LABEL_COL in df.columns:
        df = df.drop(columns=[LABEL_COL])

    # Keep ONLY training columns (drop extras like Identify, MD5, etc.)
    X = df.reindex(columns=feature_names, fill_value=pd.NA)

    # Coerce all to numeric (non-numeric becomes NaN for imputer)
    X = X.apply(pd.to_numeric, errors="coerce")
    return X


def main():
    parser = argparse.ArgumentParser(description="Predict using saved LightGBM pipeline (with feature alignment).")
    parser.add_argument("input_csv", type=str, help="Input CSV path (may include Label column).")
    parser.add_argument("--out", type=str, default="predictions.csv", help="Output CSV path.")
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    out_path = Path(args.out)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run: python train_save_pipeline.py")

    df = pd.read_csv(input_path)

    y_true = df[LABEL_COL].astype(int) if LABEL_COL in df.columns else None

    bundle = load_bundle(MODEL_PATH)
    pipe = bundle["pipeline"]
    feature_names = bundle["feature_names"]

    X = align_to_training_features(df, feature_names)

    y_pred = pipe.predict(X).astype(int)

    y_proba = None
    if hasattr(pipe, "predict_proba"):
        y_proba = pipe.predict_proba(X)[:, 1]

    out_df = df.copy()
    out_df["pred_label"] = y_pred
    if y_proba is not None:
        out_df["pred_proba_malware"] = y_proba

    out_df.to_csv(out_path, index=False)
    print(f"✅ Saved predictions to: {out_path.resolve()}")

    if y_true is not None:
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print("\n=== Metrics (Label present in input) ===")
        print(f"accuracy : {acc:.4f}")
        print(f"precision: {prec:.4f}")
        print(f"recall   : {rec:.4f}")
        print(f"f1       : {f1:.4f}")

        print("\nConfusion Matrix (rows=true, cols=pred):")
        print(confusion_matrix(y_true, y_pred))

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, digits=4))


if __name__ == "__main__":
    main()
