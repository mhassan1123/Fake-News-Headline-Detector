import pandas as pd
import re
from pathlib import Path

input_path = Path("data/processed/merged_headlines.csv")
output_path = Path("data/processed")
audit_path = Path("outputs/audits")

output_path.mkdir(parents=True, exist_ok=True)
audit_path.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_path)

publishers = [
    "new york times",
    "breitbart",
    "reuters",
    "cnn",
    "fox news",
    "washington post",
    "guardian",
    "bbc",
    "associated press",
    "daily caller",
    "huffington post",
    "buzzfeed",
    "russia news now",
    "new eastern outlook",
    "endingfed news network",
    "vineyard of the saker",
    "guardian liberty voice",
    "liberty writers news",
    "conservative daily post",
    "paul craig roberts"
]

separator_pattern = r"[-–—|·•:*]"

patterns = []

for publisher in publishers:
    pattern = re.compile(
        r"\s*" + separator_pattern +
        r"\s*(?:the\s+)?" +
        re.escape(publisher) +
        r"\s*$",
        re.IGNORECASE
    )

    patterns.append((publisher, pattern))

cleaned_titles = []
removed_records = []

for _, row in df.iterrows():

    original_title = str(row["title"]).strip()
    cleaned_title = original_title
    matched_publisher = None
    matched_suffix = None

    for publisher, pattern in patterns:
        match = pattern.search(original_title)

        if match:
            cleaned_title = original_title[:match.start()].strip()
            matched_publisher = publisher
            matched_suffix = match.group().strip()
            break

    cleaned_titles.append(cleaned_title)

    if matched_publisher:
        removed_records.append({
            "source_dataset": row["source_dataset"],
            "label": "Real" if row["real"] == 1 else "Fake",
            "publisher": matched_publisher,
            "original_title": original_title,
            "cleaned_title": cleaned_title,
            "removed_suffix": matched_suffix
        })

candidate = df.copy()
candidate["title"] = cleaned_titles

removed_df = pd.DataFrame(removed_records)

# Normalize titles for duplicate auditing.
normalized = (
    candidate["title"]
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

candidate["normalized_title"] = normalized

duplicate_groups = (
    candidate.groupby("normalized_title")
    .agg(
        record_count=("real", "size"),
        label_count=("real", "nunique"),
        labels=("real", lambda values: ",".join(
            sorted(values.astype(str).unique())
        )),
        titles=("title", lambda values: " | ".join(
            values.astype(str).unique()
        ))
    )
    .reset_index()
)

duplicate_groups = duplicate_groups[
    duplicate_groups["record_count"] > 1
].copy()

conflicts = duplicate_groups[
    duplicate_groups["label_count"] > 1
].copy()

# Keep the model dataset limited to its intended columns.
candidate = candidate[
    ["title", "real", "source_dataset"]
]

candidate.to_csv(
    output_path / "merged_headlines_suffix_candidate.csv",
    index=False
)

removed_df.to_csv(
    audit_path / "publisher_suffix_removal_log.csv",
    index=False
)

duplicate_groups.to_csv(
    audit_path / "post_suffix_duplicate_groups.csv",
    index=False
)

conflicts.to_csv(
    audit_path / "post_suffix_label_conflicts.csv",
    index=False
)

print("PUBLISHER CLEANING CANDIDATE")
print("=" * 65)

print(f"Original records: {len(df)}")
print(f"Candidate records: {len(candidate)}")
print(f"Suffixes removed: {len(removed_df)}")
print(f"Records retained unchanged: {len(df) - len(removed_df)}")

print("\nCLASS DISTRIBUTION")
print("-" * 65)
print(
    candidate["real"]
    .map({0: "Fake", 1: "Real"})
    .value_counts()
    .to_string()
)

print("\nPOST-CLEANING DUPLICATE AUDIT")
print("-" * 65)
print(f"Repeated normalized groups: {len(duplicate_groups)}")
print(
    "Records in repeated groups: "
    f"{duplicate_groups['record_count'].sum()}"
)
print(f"Conflicting-label groups: {len(conflicts)}")

print("\nPUBLISHER SUFFIX REMOVALS BY PUBLISHER")
print("-" * 65)
print(
    removed_df["publisher"]
    .value_counts()
    .to_string()
)

print("\nFILES SAVED")
print("data/processed/merged_headlines_suffix_candidate.csv")
print("outputs/audits/publisher_suffix_removal_log.csv")
print("outputs/audits/post_suffix_duplicate_groups.csv")
print("outputs/audits/post_suffix_label_conflicts.csv")

print("\nCANDIDATE CREATION COMPLETED")