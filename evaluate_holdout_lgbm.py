import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from lightgbm import LGBMClassifier


# -----------------------
# Config
# -----------------------
TRAIN_PATH = "Data/processed/train.csv"
TEST_PATH  = "Data/processed/test.csv"
TARGET_COL = "Label"
RANDOM_STATE = 42

# -----------------------
# Load data
# -----------------------
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

if TARGET_COL not in train_df.columns or TARGET_COL not in test_df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' must exist in both train and test.")

y_train = train_df[TARGET_COL].astype(int)
X_train = train_df.drop(columns=[TARGET_COL])

y_test = test_df[TARGET_COL].astype(int)
X_test = test_df.drop(columns=[TARGET_COL])

# Keep numeric only (safe)
X_train = X_train.select_dtypes(include=[np.number])
X_test = X_test.select_dtypes(include=[np.number])

# Drop all-missing columns based on TRAIN only
all_missing_cols = X_train.columns[X_train.isna().all()].tolist()
if all_missing_cols:
    print("Dropping all-missing columns (train):", all_missing_cols)
    X_train = X_train.drop(columns=all_missing_cols)
    X_test = X_test.drop(columns=all_missing_cols, errors="ignore")

# Align columns (if any mismatch)
X_test = X_test.reindex(columns=X_train.columns, fill_value=np.nan)

print("Train shape:", X_train.shape, "| Test shape:", X_test.shape)

# -----------------------
# Model pipeline (no leakage)
# -----------------------
model = LGBMClassifier(
    n_estimators=800,
    learning_rate=0.05,
    num_leaves=63,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("model", model)
])

# -----------------------
# Fit on FULL train, evaluate ONCE on test
# -----------------------
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)
cm   = confusion_matrix(y_test, y_pred)

print("\n=== Final Hold-out Test Results (LightGBM) ===")
print("accuracy :", round(acc, 4))
print("precision:", round(prec, 4))
print("recall   :", round(rec, 4))
print("f1       :", round(f1, 4))

print("\nConfusion Matrix (rows=true, cols=pred):")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

