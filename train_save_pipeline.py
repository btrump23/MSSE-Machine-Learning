"""
train_save_pipeline.py

Trains a scikit-learn pipeline on data/train.csv and saves it as a bundle dict:
  - "pipeline": trained pipeline
  - "feature_names": list of feature column names used for training
  - "dropped_all_missing": list of columns dropped because they were all missing
  - "target_col": name of target column

✅ Saves model.pkl to the SAME folder as this script (guaranteed).
"""

import os
import sys
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# =====================================================
# DEBUG: PROVE SCRIPT IS RUNNING
# =====================================================
print("🔥 train_save_pipeline.py STARTED")
print("🔥 Python executable:", sys.executable)


# =====================================================
# CONFIG — EDIT ONLY TARGET_COL IF NEEDED
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Your dataset location (as you said: data/train.csv)
TRAIN_CSV = os.path.join(BASE_DIR, "data", "processed", "train.csv")

# Your label/target column name (change this if your CSV uses a different name)
TARGET_COL = "Label"

# Model output path (saved right next to this script)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")


DROP_ALL_MISSING = True
TEST_SIZE = 0.2
RANDOM_STATE = 42


# =====================================================
# HELPERS
# =====================================================
def drop_all_missing_columns(df: pd.DataFrame):
    all_missing = [c for c in df.columns if df[c].isna().all()]
    if all_missing:
        df = df.drop(columns=all_missing)
    return df, all_missing


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
    )

    model = LogisticRegression(max_iter=1000)

    return Pipeline(steps=[("preprocess", pre), ("model", model)])


# =====================================================
# MAIN
# =====================================================
def main():
    print("🔥 ENTERED main()")
    print("📄 Training CSV:", TRAIN_CSV)
    print("🎯 Target column:", TARGET_COL)
    print("💾 Model output path:", MODEL_PATH)

    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(f"Training CSV not found: {TRAIN_CSV}")

    df = pd.read_csv(TRAIN_CSV)
    df.columns = [str(c).strip() for c in df.columns]

    print("📊 Columns found:", df.columns.tolist())

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found in CSV. "
            f"Found: {df.columns.tolist()}"
        )

    dropped_cols = []
    if DROP_ALL_MISSING:
        df, dropped_cols = drop_all_missing_columns(df)
        print("🧹 Dropped all-missing columns:", dropped_cols)

        if TARGET_COL not in df.columns:
            raise ValueError("Target column disappeared after dropping all-missing columns.")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    feature_names = list(X.columns)
    print("🧠 Feature count:", len(feature_names))
    # Uncomment if you want full feature list printed:
    # print("🧠 Features:", feature_names)

    pipeline = build_pipeline(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y if y.nunique() > 1 else None,
    )

    pipeline.fit(X_train, y_train)

    # Optional evaluation
    try:
        preds = pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"📈 Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds))
    except Exception as e:
        print("ℹ️ Skipped evaluation:", e)

    # Save pipeline + metadata together (matches your bundle structure)
    bundle = {
        "pipeline": pipeline,
        "feature_names": feature_names,
        "dropped_all_missing": dropped_cols,  # list of dropped columns
        "target_col": TARGET_COL,
    }

    print("🔥 ABOUT TO SAVE MODEL TO:", MODEL_PATH)
    joblib.dump(pipeline, MODEL_PATH)
    #joblib.dump(bundle, MODEL_PATH)
    #print("🔥 MODEL SAVED")
    #print(f"\n✅ Saved packaged pipeline to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
