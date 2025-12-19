import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import joblib
from pathlib import Path

# =========================
# Configuration
# =========================
TRAIN_PATH = "Data/processed/train.csv"
TARGET_COL = "Label"
DROP_COLS = ["Identify", "MD5"]

BATCH_SIZE = 256
EPOCHS = 15
LEARNING_RATE = 0.001
RANDOM_STATE = 42

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# =========================
# Load & clean data
# =========================
df = pd.read_csv(TRAIN_PATH)
df = df.drop(columns=DROP_COLS, errors="ignore")

X = df.drop(columns=[TARGET_COL]).copy()
y = df[TARGET_COL].copy()

# Force numeric
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = pd.to_numeric(X[col], errors="coerce")

X = X.replace([np.inf, -np.inf], np.nan)

# Drop columns that are entirely missing
all_missing = X.columns[X.isna().all()].tolist()
if all_missing:
    print("Dropping all-missing columns:", all_missing)
    X = X.drop(columns=all_missing)

X = X.select_dtypes(include="number")
feature_names = X.columns.tolist()

print("Features used:", len(feature_names))

# =========================
# Train / validation split
# =========================
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)

# =========================
# Preprocessing (NO LEAKAGE)
# =========================
imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_train = imputer.fit_transform(X_train)
X_train = scaler.fit_transform(X_train)

X_val = imputer.transform(X_val)
X_val = scaler.transform(X_val)

joblib.dump(imputer, ARTIFACT_DIR / "imputer.joblib")
joblib.dump(scaler, ARTIFACT_DIR / "scaler.joblib")
joblib.dump(feature_names, ARTIFACT_DIR / "feature_names.joblib")

# =========================
# Torch datasets
# =========================
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=BATCH_SIZE,
    shuffle=True
)

# =========================
# Model definition
# =========================
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.out(x)

model = MLP(X_train.shape[1])
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =========================
# Training loop
# =========================
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    with torch.no_grad():
        logits = model(X_val_t).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_val, preds)
    prec = precision_score(y_val, preds, zero_division=0)
    rec = recall_score(y_val, preds, zero_division=0)
    f1 = f1_score(y_val, preds, zero_division=0)

    print(
        "Epoch", epoch,
        "| loss =", round(total_loss / len(train_loader), 4),
        "| acc =", round(acc, 4),
        "| prec =", round(prec, 4),
        "| rec =", round(rec, 4),
        "| f1 =", round(f1, 4)
    )

# =========================
# Save model
# =========================
torch.save(model.state_dict(), ARTIFACT_DIR / "mlp_state_dict.pt")

print("\nFinal validation metrics")
print("accuracy :", round(acc, 4))
print("precision:", round(prec, 4))
print("recall   :", round(rec, 4))
print("f1       :", round(f1, 4))

print("\n✅ PyTorch MLP training COMPLETE (no data leakage)")
