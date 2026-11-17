import pandas as pd
from pathlib import Path

candidate_path = Path(
    "data/processed/merged_headlines_suffix_candidate.csv"
)

removal_path = Path(
    "outputs/audits/publisher_suffix_removal_log.csv"
)

df = pd.read_csv(candidate_path)
removed = pd.read_csv(removal_path)

df["normalized_title"] = (
    df["title"]
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

groups = (
    df.groupby("normalized_title")
    .filter(lambda group: len(group) > 1)
    .groupby("normalized_title")
)

print("POST-SUFFIX DUPLICATE INSPECTION")
print("=" * 70)
print(f"Repeated groups: {len(groups)}")

for number, (normalized_title, group) in enumerate(groups, start=1):

    labels = sorted(group["real"].unique())
    has_conflict = len(labels) > 1

    print(f"\nGROUP {number}")
    print("-" * 70)
    print(f"Conflict: {'YES' if has_conflict else 'NO'}")
    print(f"Cleaned headline: {normalized_title}")
    print(f"Records: {len(group)}")

    for _, row in group.iterrows():
        label = "Real" if row["real"] == 1 else "Fake"

        print(f"\nSource: {row['source_dataset']}")
        print(f"Label: {label}")
        print(f"Title: {row['title']}")

    matching_removals = removed[
        removed["cleaned_title"].str.lower().str.replace(
            r"\s+", " ", regex=True
        ).str.strip() == normalized_title
    ]

    if not matching_removals.empty:
        print("\nORIGINAL TITLES BEFORE SUFFIX REMOVAL")

        for _, row in matching_removals.iterrows():
            print(f"Publisher: {row['publisher']}")
            print(f"Original: {row['original_title']}")
            print(f"Removed suffix: {row['removed_suffix']}")

print("\nINSPECTION COMPLETED")