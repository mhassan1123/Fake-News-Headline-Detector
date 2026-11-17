import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_PATH = "data/processed/final_headlines.csv"
OUTPUT_DIR = "data/splits"
SUMMARY_PATH = "outputs/results/final_split_summary.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("outputs/results", exist_ok=True)

print("FINAL DATASET SPLITTING")
print("=" * 65)

# 1. Load the approved dataset
df = pd.read_csv(INPUT_PATH)

required_columns = ["title", "real", "source_dataset"]

if not all(col in df.columns for col in required_columns):
    raise ValueError("Required dataset columns are missing.")

if df[required_columns].isnull().any().any():
    raise ValueError("Missing values found in the dataset.")

df["real"] = df["real"].astype(int)

if not df["real"].isin([0, 1]).all():
    raise ValueError("Invalid labels found.")

print(f"Total records: {len(df)}")

# 2. Create the training and testing split
train_val, test = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["real"]
)

# 3. Divide the remaining data into training and validation
train, validation = train_test_split(
    train_val,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=train_val["real"]
)

# 4. Save all three datasets
train.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
validation.to_csv(os.path.join(OUTPUT_DIR, "validation.csv"), index=False)
test.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

splits = {
    "Training": train,
    "Validation": validation,
    "Testing": test
}

# 5. Verify split sizes, labels, and headline separation
def normalize_title(title):
    return re.sub(r"\s+", " ", str(title).lower()).strip()

all_titles = set()
summary_rows = []

print("\nSPLIT DISTRIBUTION")
print("-" * 65)

for split_name, split_df in splits.items():
    normalized_titles = split_df["title"].apply(normalize_title)

    if normalized_titles.duplicated().any():
        raise ValueError(f"Duplicate headlines found in {split_name}.")

    overlap = all_titles.intersection(set(normalized_titles))

    if overlap:
        raise ValueError(
            f"Headline leakage detected in {split_name}: {len(overlap)} overlaps."
        )

    all_titles.update(normalized_titles)

    fake_count = int((split_df["real"] == 0).sum())
    real_count = int((split_df["real"] == 1).sum())

    summary_rows.append({
        "split": split_name,
        "total": len(split_df),
        "fake": fake_count,
        "real": real_count,
        "fake_percentage": round(fake_count / len(split_df) * 100, 2),
        "real_percentage": round(real_count / len(split_df) * 100, 2)
    })

    print(f"\n{split_name}")
    print(f"Records: {len(split_df)}")
    print(f"Fake: {fake_count} ({fake_count / len(split_df) * 100:.2f}%)")
    print(f"Real: {real_count} ({real_count / len(split_df) * 100:.2f}%)")

# 6. Verify that every original record was assigned exactly once
total_split_records = len(train) + len(validation) + len(test)

if total_split_records != len(df):
    raise ValueError("Some records are missing or have been assigned more than once.")

if len(all_titles) != len(df):
    raise ValueError("Final headline separation check failed.")

# 7. Verify source representation in each split
print("\nSOURCE DISTRIBUTION")
print("-" * 65)

for split_name, split_df in splits.items():
    print(f"\n{split_name}")
    print(split_df["source_dataset"].value_counts().to_string())

# 8. Save the split summary
summary = pd.DataFrame(summary_rows)
summary.to_csv(SUMMARY_PATH, index=False)

print("\nFINAL VALIDATION")
print("-" * 65)
print(f"Original records: {len(df)}")
print(f"Training records: {len(train)}")
print(f"Validation records: {len(validation)}")
print(f"Testing records: {len(test)}")
print(f"Total assigned records: {total_split_records}")
print(f"Unique headlines across splits: {len(all_titles)}")
print(f"Random state: {RANDOM_STATE}")
print("Stratification: Enabled")
print("Headline overlap: None")

print("\nSaved files:")
print(os.path.join(OUTPUT_DIR, "train.csv"))
print(os.path.join(OUTPUT_DIR, "validation.csv"))
print(os.path.join(OUTPUT_DIR, "test.csv"))
print(SUMMARY_PATH)

print("\nFINAL DATA SPLITTING COMPLETED SUCCESSFULLY")