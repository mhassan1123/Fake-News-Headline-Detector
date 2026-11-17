import pandas as pd
from pathlib import Path

input_path = Path(
    "data/processed/merged_headlines_suffix_candidate.csv"
)

output_path = Path("data/processed")
audit_path = Path("outputs/audits")

output_path.mkdir(parents=True, exist_ok=True)
audit_path.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_path)

df["normalized_title"] = (
    df["title"]
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

resolved_records = []
conflict_records = []
collapsed_records = []

groups = df.groupby("normalized_title", sort=False)

for normalized_title, group in groups:

    labels = group["real"].unique()

    # Exclude every record in a conflicting group.
    if len(labels) > 1:

        for _, row in group.iterrows():
            conflict_records.append({
                "normalized_title": normalized_title,
                "title": row["title"],
                "real": row["real"],
                "source_dataset": row["source_dataset"]
            })

        continue

    # Collect original dataset names without losing provenance.
    sources = set()

    for source_value in group["source_dataset"].dropna():
        for source in str(source_value).split("+"):
            sources.add(source.strip())

    combined_source = "+".join(sorted(sources))

    # Retain one representative headline.
    first_row = group.iloc[0]

    resolved_records.append({
        "title": first_row["title"],
        "real": int(labels[0]),
        "source_dataset": combined_source
    })

    # Record groups where multiple rows were consolidated.
    if len(group) > 1:

        for _, row in group.iterrows():
            collapsed_records.append({
                "normalized_title": normalized_title,
                "title": row["title"],
                "real": row["real"],
                "source_dataset": row["source_dataset"]
            })

resolved_df = pd.DataFrame(resolved_records)

conflict_df = pd.DataFrame(conflict_records)
collapsed_df = pd.DataFrame(collapsed_records)

# Final verification.
normalized_final = (
    resolved_df["title"]
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

duplicate_count = normalized_final.duplicated().sum()

conflict_check = (
    resolved_df.assign(normalized_title=normalized_final)
    .groupby("normalized_title")["real"]
    .nunique()
)

conflict_count = (conflict_check > 1).sum()

# Save the new candidate and audit files.
resolved_df.to_csv(
    output_path / "merged_headlines_suffix_resolved.csv",
    index=False
)

conflict_df.to_csv(
    audit_path / "suffix_conflicting_records_excluded.csv",
    index=False
)

collapsed_df.to_csv(
    audit_path / "suffix_duplicate_consolidation_log.csv",
    index=False
)

print("SUFFIX DUPLICATE RESOLUTION")
print("=" * 65)

print(f"Input records: {len(df)}")
print(f"Output records: {len(resolved_df)}")
print(f"Records excluded from conflicts: {len(conflict_df)}")
print(f"Records in consolidated groups: {len(collapsed_df)}")

print("\nFINAL CLASS DISTRIBUTION")
print("-" * 65)

print(
    resolved_df["real"]
    .map({0: "Fake", 1: "Real"})
    .value_counts()
    .to_string()
)

print("\nFINAL DATASET VALIDATION")
print("-" * 65)
print(f"Missing values:\n{resolved_df.isna().sum().to_string()}")
print(f"Normalized duplicate titles: {duplicate_count}")
print(f"Conflicting-label groups: {conflict_count}")

print("\nSOURCE DISTRIBUTION")
print("-" * 65)
print(resolved_df["source_dataset"].value_counts().to_string())

print("\nFILES SAVED")
print("data/processed/merged_headlines_suffix_resolved.csv")
print("outputs/audits/suffix_conflicting_records_excluded.csv")
print("outputs/audits/suffix_duplicate_consolidation_log.csv")

if duplicate_count == 0 and conflict_count == 0:
    print("\nVALIDATION PASSED")
else:
    print("\nVALIDATION FAILED - INVESTIGATION REQUIRED")

print("\nDUPLICATE RESOLUTION COMPLETED")