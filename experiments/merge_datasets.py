import pandas as pd
from pathlib import Path

fakenewsnet_path = Path("data/processed/cleaned_fakenewsnet.csv")
welfake_path = Path("data/processed/cleaned_welfake_english.csv")
output_path = Path("data/processed/merged_headlines.csv")

print("DATASET MERGING")
print("=" * 60)

# 1. Load both cleaned datasets
fakenewsnet = pd.read_csv(fakenewsnet_path)
welfake = pd.read_csv(welfake_path)

# 2. Verify required columns
if "title" not in fakenewsnet.columns or "real" not in fakenewsnet.columns:
    raise ValueError("FakeNewsNet must contain title and real columns.")

if "title" not in welfake.columns or "real" not in welfake.columns:
    raise ValueError("WELFake must contain title and real columns.")

# 3. Keep only the fields needed for headline classification
fakenewsnet = fakenewsnet[["title", "real"]].copy()
welfake = welfake[["title", "real"]].copy()

fakenewsnet["source_dataset"] = "FakeNewsNet"
welfake["source_dataset"] = "WELFake"

# 4. Combine the datasets
combined = pd.concat(
    [fakenewsnet, welfake],
    ignore_index=True
)

combined["real"] = pd.to_numeric(
    combined["real"],
    errors="raise"
).astype(int)

if not combined["real"].isin([0, 1]).all():
    raise ValueError("Unexpected class labels found.")

# 5. Normalize headlines for duplicate detection
combined["normalized_title"] = (
    combined["title"]
    .astype(str)
    .str.lower()
    .str.split()
    .str.join(" ")
)

print("\n1. INITIAL DATASET COUNTS")
print("-" * 60)
print(f"FakeNewsNet: {len(fakenewsnet)}")
print(f"WELFake: {len(welfake)}")
print(f"Combined: {len(combined)}")

# 6. Identify conflicting headline groups
label_counts = combined.groupby(
    "normalized_title"
)["real"].nunique()

conflicting_titles = label_counts[label_counts > 1].index

conflicts = combined[
    combined["normalized_title"].isin(conflicting_titles)
].copy()

print("\n2. CROSS-DATASET LABEL CONFLICTS")
print("-" * 60)
print(f"Conflicting headline groups: {len(conflicting_titles)}")
print(f"Records belonging to conflicts: {len(conflicts)}")

if len(conflicts) > 0:
    conflicts.to_csv(
        "outputs/audits/merge_label_conflicts.csv",
        index=False
    )

# 7. Remove every record belonging to a conflicting group
combined = combined[
    ~combined["normalized_title"].isin(conflicting_titles)
].copy()

# 8. Combine source names for matching headlines
merged_rows = []

for _, group in combined.groupby("normalized_title", sort=False):
    first = group.iloc[0].copy()
    sources = sorted(group["source_dataset"].unique())

    first["source_dataset"] = "+".join(sources)
    merged_rows.append(first)

merged = pd.DataFrame(merged_rows)

# 9. Remove helper column and arrange final columns
merged = merged[["title", "real", "source_dataset"]].reset_index(drop=True)

# 10. Final validation
if merged["title"].isna().any():
    raise ValueError("Missing headlines found in final dataset.")

if merged["title"].astype(str).str.strip().eq("").any():
    raise ValueError("Empty headlines found in final dataset.")

if merged["real"].isna().any():
    raise ValueError("Missing labels found in final dataset.")

if merged.duplicated(
    subset=["title"]
).any():
    raise ValueError("Exact duplicate titles remain.")

# 11. Save the merged dataset
output_path.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(output_path, index=False)

print("\n3. FINAL MERGED DATASET")
print("-" * 60)
print(f"Total records: {len(merged)}")
print(f"Fake headlines: {(merged['real'] == 0).sum()}")
print(f"Real headlines: {(merged['real'] == 1).sum()}")

print("\n4. SOURCE DISTRIBUTION")
print("-" * 60)
print(merged["source_dataset"].value_counts().to_string())

print("\n5. FINAL VALIDATION")
print("-" * 60)
print(f"Missing values:\n{merged.isna().sum().to_string()}")
print(f"Duplicate exact titles: {merged['title'].duplicated().sum()}")
print(f"Valid labels: {merged['real'].isin([0, 1]).all()}")

print(f"\nSaved: {output_path}")
print("\nMERGING COMPLETED")