import os
import re
import pandas as pd

INPUT_PATH = "data/processed/merged_headlines_suffix_candidate.csv"
OUTPUT_PATH = "data/processed/final_headlines.csv"

AUDIT_DIR = "outputs/audits"
SUFFIX_LOG = os.path.join(AUDIT_DIR, "final_suffix_removal_log.csv")
CONFLICT_LOG = os.path.join(AUDIT_DIR, "final_conflicting_records_removed.csv")
DUPLICATE_LOG = os.path.join(AUDIT_DIR, "final_duplicate_groups_consolidated.csv")

os.makedirs("data/processed", exist_ok=True)
os.makedirs(AUDIT_DIR, exist_ok=True)

print("FINAL DATASET PREPARATION")
print("=" * 65)

# 1. Load and validate the input
df = pd.read_csv(INPUT_PATH)

required_columns = ["title", "real", "source_dataset"]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

print(f"Input records: {len(df)}")

if df[required_columns].isnull().any().any():
    raise ValueError(
        "Missing titles, labels, or source information found. "
        "Inspect the input before continuing."
    )

df["real"] = pd.to_numeric(df["real"], errors="raise")

if not df["real"].isin([0, 1]).all():
    raise ValueError("Labels must contain only 0 (Fake) and 1 (Real).")

df["real"] = df["real"].astype(int)
df["title"] = df["title"].astype(str).str.strip()

if (df["title"] == "").any():
    raise ValueError("Empty headlines found. Inspect the input before continuing.")

# 2. Remove the one remaining verified publisher attribution
# Only removes Reuters when it appears at the very end after a colon.
suffix_pattern = re.compile(r"\s*:\s*Reuters\s*$", flags=re.IGNORECASE)

suffix_records = []

for index, row in df.iterrows():
    old_title = row["title"]
    new_title = suffix_pattern.sub("", old_title).strip()

    if new_title != old_title:
        suffix_records.append({
            "original_title": old_title,
            "cleaned_title": new_title,
            "source_dataset": row["source_dataset"],
            "real": row["real"]
        })
        df.at[index, "title"] = new_title

pd.DataFrame(
    suffix_records,
    columns=["original_title", "cleaned_title", "source_dataset", "real"]
).to_csv(SUFFIX_LOG, index=False)

print(f"Publisher suffixes removed: {len(suffix_records)}")

# 3. Normalize headlines for duplicate detection
def normalize_title(title):
    title = str(title).lower()
    title = re.sub(r"\s+", " ", title)
    return title.strip()

df["_normalized_title"] = df["title"].apply(normalize_title)

if (df["_normalized_title"] == "").any():
    raise ValueError("A headline became empty after cleaning.")

# 4. Detect conflicts and consolidate consistent duplicates
final_records = []
conflicting_records = []
duplicate_groups = []

for normalized_title, group in df.groupby("_normalized_title", sort=False):
    labels = group["real"].unique()

    # Conflicting labels: remove every record in this group
    if len(labels) > 1:
        for _, row in group.iterrows():
            conflicting_records.append({
                "title": row["title"],
                "real": row["real"],
                "source_dataset": row["source_dataset"],
                "normalized_title": normalized_title
            })
        continue

    # Keep one record for a consistent group
    first_record = group.iloc[0].drop(labels=["_normalized_title"]).copy()

    if len(group) > 1:
        # Preserve every dataset source represented by the duplicate group
        all_sources = []

        for source_value in group["source_dataset"]:
            for source in str(source_value).split("+"):
                source = source.strip()
                if source and source not in all_sources:
                    all_sources.append(source)

        first_record["source_dataset"] = "+".join(all_sources)

        duplicate_groups.append({
            "normalized_title": normalized_title,
            "records_before": len(group),
            "records_after": 1,
            "real": int(labels[0]),
            "sources_combined": first_record["source_dataset"]
        })

    final_records.append(first_record)

final_df = pd.DataFrame(final_records)

pd.DataFrame(
    conflicting_records,
    columns=["title", "real", "source_dataset", "normalized_title"]
).to_csv(CONFLICT_LOG, index=False)

pd.DataFrame(
    duplicate_groups,
    columns=[
        "normalized_title",
        "records_before",
        "records_after",
        "real",
        "sources_combined"
    ]
).to_csv(DUPLICATE_LOG, index=False)

# 5. Final validation
final_df = final_df.reset_index(drop=True)
final_df["real"] = final_df["real"].astype(int)

final_df["_normalized_title"] = final_df["title"].apply(normalize_title)

missing_count = int(final_df[["title", "real", "source_dataset"]].isnull().sum().sum())
duplicate_count = int(final_df["_normalized_title"].duplicated().sum())
invalid_label_count = int((~final_df["real"].isin([0, 1])).sum())
empty_title_count = int((final_df["title"].str.strip() == "").sum())
remaining_reuters = int(
    final_df["title"].str.contains(
        r":\s*Reuters\s*$",
        case=False,
        regex=True
    ).sum()
)

if missing_count != 0:
    raise ValueError("Final validation failed: missing values remain.")

if duplicate_count != 0:
    raise ValueError("Final validation failed: duplicate normalized headlines remain.")

if invalid_label_count != 0:
    raise ValueError("Final validation failed: invalid labels remain.")

if empty_title_count != 0:
    raise ValueError("Final validation failed: empty headlines remain.")

if remaining_reuters != 0:
    raise ValueError("Final validation failed: Reuters suffix remains.")

# Remove the temporary audit column before saving
final_df = final_df.drop(columns=["_normalized_title"])

final_df.to_csv(OUTPUT_PATH, index=False)

# 6. Print the final report
fake_count = int((final_df["real"] == 0).sum())
real_count = int((final_df["real"] == 1).sum())

print("\nCLEANING SUMMARY")
print("-" * 65)
print(f"Input records: {len(df)}")
print(f"Publisher suffixes removed: {len(suffix_records)}")
print(f"Conflicting records removed: {len(conflicting_records)}")
print(f"Consistent duplicate groups consolidated: {len(duplicate_groups)}")
print(f"Final records: {len(final_df)}")

print("\nFINAL CLASS DISTRIBUTION")
print("-" * 65)
print(f"Fake (0): {fake_count} ({fake_count / len(final_df) * 100:.2f}%)")
print(f"Real (1): {real_count} ({real_count / len(final_df) * 100:.2f}%)")

print("\nFINAL VALIDATION")
print("-" * 65)
print(f"Missing values: {missing_count}")
print(f"Duplicate normalized headlines: {duplicate_count}")
print(f"Invalid labels: {invalid_label_count}")
print(f"Empty headlines: {empty_title_count}")
print(f"Remaining Reuters suffixes: {remaining_reuters}")

print("\nSaved final dataset:")
print(OUTPUT_PATH)

print("\nSaved audit files:")
print(SUFFIX_LOG)
print(CONFLICT_LOG)
print(DUPLICATE_LOG)

print("\nFINAL DATASET VALIDATION PASSED")