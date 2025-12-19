import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# =========================
# Configuration
# =========================
TRAIN_PATH = "Data/processed/train.csv"
TARGET_COL = "Label"
DROP_COLS = ["Identify", "MD5"]   # drop if present

RANDOM_STATE = 42
N_SPLITS = 10

# =========================
# Load data
# =========================
df = pd.read_csv(TRAIN_PATH)

# Drop known ID-like columns if present
df = df.drop(columns=DROP_COLS, errors="ignore")

# Separate X / y
X = df.drop(columns=[TARGET_COL]).copy()
y = df[TARGET_COL].copy()

# Force numeric where possible (handles object columns)
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = pd.to_numeric(X[col], errors="coerce")

# Replace inf with NaN
X = X.replace([np.inf, -np.inf], np.nan)

# Drop columns that are entirely missing
all_missing_cols = X.columns[X.isna().all()].tolist()
if all_missing_cols:
    print("Dropping all-missing columns:", all_missing_cols)
    X = X.drop(columns=all_missing_cols)

# Keep numeric features only (dataset should be numeric features)
X = X.select_dtypes(include="number")

print(f"Features used: {X.shape[1]}")

numeric_features = X.columns.tolist()

# =========================
# CV + scoring
# =========================
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

scoring = {
    "accuracy": make_scorer(accuracy_score),
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
}

def print_cv_results(name, results):
    print(f"\n{name} (Stratified {N_SPLITS}-fold CV, train only):")
    for metric in ["test_accuracy", "test_precision", "test_recall", "test_f1"]:
        vals = results[metric]
        m = np.mean(vals)
        s = np.std(vals)
        label = metric.replace("test_", "")
        print(f"  {label:9s}: mean={m:.4f}  std={s:.4f}")

# =========================
# Preprocessing blocks
# =========================
# Logistic Regression benefits from scaling
preprocess_scaled = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_features)
    ],
    remainder="drop"
)

# Tree models don't need scaling (but still need imputation)
preprocess_tree = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]), numeric_features)
    ],
    remainder="drop"
)

# =========================
# Models
# =========================
models = [
    (
        "Logistic Regression",
        Pipeline(steps=[
            ("preprocess", preprocess_scaled),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ])
    ),
    (
        "Decision Tree",
        Pipeline(steps=[
            ("preprocess", preprocess_tree),
            ("clf", DecisionTreeClassifier(
                max_depth=None,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ])
    ),
    (
        "Random Forest",
        Pipeline(steps=[
            ("preprocess", preprocess_tree),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1
            ))
        ])
    ),
]

# =========================
# Run CV
# =========================
for name, pipe in models:
    results = cross_validate(
        pipe,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        error_score="raise"
    )
    print_cv_results(name, results)

print("\n✅ Completed Logistic Regression + Decision Tree + Random Forest (no leakage).")

