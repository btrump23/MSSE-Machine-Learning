from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

BASE = Path("Data/raw/brazilian-malware-dataset-master/brazilian-malware-dataset-master/goodware-malware")
GOODWARE_PATH = BASE / "goodware.csv"
MALWARE_DIR = BASE / "malware-by-day"

OUT_DIR = Path("Data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = OUT_DIR / "train.csv"
TEST_PATH = OUT_DIR / "test.csv"

TARGET_COL = "Label"
TEST_SIZE = 0.20
RANDOM_STATE = 42

# ---- Load goodware ----
good_df = pd.read_csv(GOODWARE_PATH)
good_df[TARGET_COL] = 0  # 0 = goodware

# ---- Load malware (skip empty files) ----
malware_files = sorted(MALWARE_DIR.glob("*.csv"))
if not malware_files:
    raise FileNotFoundError(f"No malware CSV files found in: {MALWARE_DIR}")

non_empty_files = [f for f in malware_files if f.stat().st_size > 0]
if not non_empty_files:
    raise FileNotFoundError(f"All malware CSV files are empty in: {MALWARE_DIR}")

mal_df_list = []
skipped = 0
for f in non_empty_files:
    try:
        mal_df_list.append(pd.read_csv(f))
    except Exception:
        skipped += 1

if not mal_df_list:
    raise RuntimeError("Could not read any malware CSV files (all failed).")

mal_df = pd.concat(mal_df_list, ignore_index=True)
mal_df[TARGET_COL] = 1  # 1 = malware

print(f"Malware files found: {len(malware_files)}")
print(f"Malware files non-empty: {len(non_empty_files)}")
print(f"Malware files skipped (read errors): {skipped}")

# ---- Align columns (keep common columns) ----
common_cols = sorted(set(good_df.columns).intersection(set(mal_df.columns)))
if TARGET_COL not in common_cols:
    common_cols.append(TARGET_COL)

good_df = good_df[common_cols]
mal_df = mal_df[common_cols]

# ---- Combine + stratified split ----
df = pd.concat([good_df, mal_df], ignore_index=True)

train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    stratify=df[TARGET_COL],
    random_state=RANDOM_STATE
)

train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)

print("\n✅ Split complete")
print("Total rows:", len(df))
print("Train rows:", len(train_df))
print("Test rows:", len(test_df))
print("\nClass balance (full):")
print(df[TARGET_COL].value_counts())
print("\nClass balance (train):")
print(train_df[TARGET_COL].value_counts())
print("\nClass balance (test):")
print(test_df[TARGET_COL].value_counts())
