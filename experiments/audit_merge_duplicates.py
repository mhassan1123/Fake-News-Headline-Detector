import pandas as pd
from pathlib import Path

fakenewsnet_path = Path("data/processed/cleaned_fakenewsnet.csv")
welfake_path = Path("data/processed/cleaned_welfake_english.csv")

fakenewsnet = pd.read_csv(fakenewsnet_path)
welfake = pd.read_csv(welfake_path)

fakenewsnet = fakenewsnet[["title", "real"]].copy()
welfake = welfake[["title", "real"]].copy()

fakenewsnet["source_dataset"] = "FakeNewsNet"
welfake["source_dataset"] = "WELFake"

combined = pd.concat(
    [fakenewsnet, welfake],
    ignore_index=True
)

combined["normalized_title"] = (
    combined["title"]
    .astype(str)
    .str.lower()
    .str.split()
    .str.join(" ")
)

groups = combined.groupby("normalized_title", sort=False)

duplicate_groups = groups.filter(
    lambda group: len(group) > 1
)

conflicting_groups = groups.filter(
    lambda group: group["real"].nunique() > 1
)

print("MERGE DUPLICATE AND CONFLICT AUDIT")
print("=" * 65)

print("\n1. SUMMARY")
print("-" * 65)
print(f"Input records: {len(combined)}")
print(f"Repeated normalized headline groups: {len(duplicate_groups['normalized_title'].unique())}")
print(f"Records in repeated groups: {len(duplicate_groups)}")
print(f"Conflicting headline groups: {len(conflicting_groups['normalized_title'].unique())}")
print(f"Records in conflicting groups: {len(conflicting_groups)}")

print("\n2. CONFLICTING HEADLINE GROUPS")
print("-" * 65)

if conflicting_groups.empty:
    print("No conflicting groups found.")
else:
    for number, (title, group) in enumerate(
        conflicting_groups.groupby("normalized_title", sort=False),
        1
    ):
        print(f"\nCONFLICT {number}")
        print(f"Normalized headline: {title}")

        for _, row in group.iterrows():
            print(
                f"Source: {row['source_dataset']} | "
                f"Label: {row['real']} | "
                f"Original title: {row['title']}"
            )

print("\n3. ALL REPEATED NORMALIZED HEADLINES")
print("-" * 65)

if duplicate_groups.empty:
    print("No repeated groups found.")
else:
    for number, (title, group) in enumerate(
        duplicate_groups.groupby("normalized_title", sort=False),
        1
    ):
        print(f"\nGROUP {number}")
        print(f"Normalized headline: {title}")

        for _, row in group.iterrows():
            print(
                f"Source: {row['source_dataset']} | "
                f"Label: {row['real']} | "
                f"Original title: {row['title']}"
            )

print("\n4. EXACT DUPLICATES WITHIN EACH INPUT")
print("-" * 65)
print(
    "FakeNewsNet exact duplicate titles:",
    fakenewsnet["title"].duplicated().sum()
)
print(
    "WELFake exact duplicate titles:",
    welfake["title"].duplicated().sum()
)

print("\nAUDIT COMPLETED - NO FILES MODIFIED")