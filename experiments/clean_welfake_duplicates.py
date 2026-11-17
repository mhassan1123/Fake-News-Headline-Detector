import pandas as pd
from pathlib import Path

input_path = Path("data/raw/WELFake_Dataset.csv")
output_path = Path("data/processed/welfake_deduplicated.csv")

df = pd.read_csv(input_path)

print("WELFAKE DUPLICATE CLEANING")
print("=" * 55)

original_count = len(df)

# Keep records with a valid headline and label.
df = df.dropna(subset=["title", "label"]).copy()
df["title"] = df["title"].astype(str).str.strip()

df = df[df["title"] != ""].copy()

# Check that labels contain only 0 and 1.
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df.dropna(subset=["label"]).copy()

if not df["label"].isin([0, 1]).all():
    raise ValueError("Unexpected label values found in WELFake.")

df["label"] = df["label"].astype(int)

valid_count = len(df)

# Normalize headlines for duplicate comparison.
df["normalized_title"] = (
    df["title"]
    .str.lower()
    .str.split()
    .str.join(" ")
)

# Find headlines that have conflicting labels.
label_counts = df.groupby("normalized_title")["label"].nunique()

conflicting_titles = label_counts[label_counts > 1].index

conflicting_rows = df[
    df["normalized_title"].isin(conflicting_titles)
]

conflict_group_count = len(conflicting_titles)
conflict_record_count = len(conflicting_rows)

# Remove every record belonging to a conflicting group.
df = df[
    ~df["normalized_title"].isin(conflicting_titles)
].copy()

after_conflict_count = len(df)

# Keep one record for each remaining unique headline.
df = df.drop_duplicates(
    subset="normalized_title",
    keep="first"
).copy()

unique_count = len(df)

# WELFake mapping for this downloaded copy:
# Raw 0 = Real, Raw 1 = Fake.
# Convert to our project convention:
# Project 0 = Fake, Project 1 = Real.
df["real"] = df["label"].map({0: 1, 1: 0})

# Keep only useful columns in the cleaned dataset.
cleaned_df = df[["title", "real", "label"]].copy()

cleaned_df = cleaned_df.rename(
    columns={"label": "original_label"}
)

output_path.parent.mkdir(parents=True, exist_ok=True)
cleaned_df.to_csv(output_path, index=False)

# Final statistics.
fake_count = (cleaned_df["real"] == 0).sum()
real_count = (cleaned_df["real"] == 1).sum()

print(f"Original records: {original_count}")
print(f"Valid headline records: {valid_count}")
print(f"Missing/invalid records removed: {original_count - valid_count}")
print(f"Conflicting headline groups: {conflict_group_count}")
print(f"Records in conflicting groups: {conflict_record_count}")
print(f"Records after conflict removal: {after_conflict_count}")
print(f"Repeated records removed: {after_conflict_count - unique_count}")
print(f"Final unique headlines: {unique_count}")

print("\nFINAL LABEL DISTRIBUTION")
print("-" * 55)
print(f"Fake headlines: {fake_count}")
print(f"Real headlines: {real_count}")
print(f"Total headlines: {len(cleaned_df)}")

print("\nOUTPUT")
print("-" * 55)
print(f"Saved to: {output_path}")
print("\nCLEANING COMPLETED")