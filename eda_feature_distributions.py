import pandas as pd
import matplotlib.pyplot as plt

TRAIN_PATH = "Data/processed/train.csv"
TARGET_COL = "Label"

df = pd.read_csv(TRAIN_PATH)

# Separate features and label
X = df.drop(columns=[TARGET_COL])

# Summary stats
print("Feature summary statistics:")
print(X.describe().T.head(10))

# Pick a few example features
sample_features = X.columns[:3]

for col in sample_features:
    plt.figure()
    df[df[TARGET_COL] == 0][col].hist(alpha=0.5, bins=50, label="Goodware")
    df[df[TARGET_COL] == 1][col].hist(alpha=0.5, bins=50, label="Malware")
    plt.title(f"Distribution of {col}")
    plt.legend()
    plt.show()

