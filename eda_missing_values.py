
import pandas as pd

TRAIN_PATH = "Data/processed/train.csv"

df = pd.read_csv(TRAIN_PATH)

missing = df.isna().sum()
missing_pct = (missing / len(df)) * 100

summary = pd.DataFrame({
    "missing_count": missing,
    "missing_pct": missing_pct
}).sort_values(by="missing_count", ascending=False)

print("Missing values summary:")
print(summary.head(15))

print("\nAny missing values at all?")
print((missing > 0).any())
