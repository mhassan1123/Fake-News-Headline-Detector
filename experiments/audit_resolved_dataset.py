import pandas as pd
import re
from pathlib import Path

original_path = Path("data/processed/merged_headlines.csv")
final_path = Path(
    "data/processed/merged_headlines_suffix_resolved.csv"
)

df_original = pd.read_csv(original_path)
df_final = pd.read_csv(final_path)

publishers = [
    "new york times", "breitbart", "reuters", "cnn",
    "fox news", "washington post", "guardian", "bbc",
    "associated press", "daily caller", "huffington post",
    "buzzfeed", "russia news now", "new eastern outlook",
    "endingfed news network", "vineyard of the saker",
    "guardian liberty voice", "liberty writers news",
    "conservative daily post", "paul craig roberts"
]

separator = r"[-–—|·•:*]"

patterns = [
    re.compile(
        r"\s*" + separator +
        r"\s*(?:the\s+)?" + re.escape(publisher) + r"\s*$",
        re.IGNORECASE
    )
    for publisher in publishers
]


def prepare_source_counts(data):

    records = []

    for _, row in data.iterrows():
        sources = str(row["source_dataset"]).split("+")

        for source in sources:
            records.append({
                "source": source.strip(),
                "label": "Real" if row["real"] == 1 else "Fake"
            })

    result = pd.DataFrame(records)

    counts = pd.crosstab(
        result["source"],
        result["label"]
    )

    counts["Total"] = counts.sum(axis=1)

    counts["Fake %"] = (
        counts["Fake"] / counts["Total"] * 100
    ).round(2)

    counts["Real %"] = (
        counts["Real"] / counts["Total"] * 100
    ).round(2)

    return counts


def headline_length_summary(data):

    temp = data.copy()

    temp["label"] = temp["real"].map({
        0: "Fake",
        1: "Real"
    })

    temp["word_count"] = (
        temp["title"].str.split().str.len()
    )

    return temp.groupby("label")["word_count"].agg(
        ["count", "mean", "median", "min", "max"]
    ).round(2)


print("FINAL SOURCE-BIAS AUDIT")
print("=" * 70)

print("\n1. SOURCE CLASS DISTRIBUTION BEFORE CLEANING")
print("-" * 70)
print(prepare_source_counts(df_original).to_string())

print("\n2. SOURCE CLASS DISTRIBUTION AFTER CLEANING")
print("-" * 70)
print(prepare_source_counts(df_final).to_string())

print("\n3. HEADLINE LENGTH BEFORE CLEANING")
print("-" * 70)
print(headline_length_summary(df_original).to_string())

print("\n4. HEADLINE LENGTH AFTER CLEANING")
print("-" * 70)
print(headline_length_summary(df_final).to_string())

# Check whether any explicit publisher suffixes remain.
remaining = []

for _, row in df_final.iterrows():

    title = str(row["title"]).strip()

    for publisher, pattern in zip(publishers, patterns):

        if pattern.search(title):
            remaining.append({
                "publisher": publisher,
                "title": title,
                "source_dataset": row["source_dataset"]
            })
            break

remaining_df = pd.DataFrame(remaining)

print("\n5. REMAINING EXPLICIT PUBLISHER SUFFIXES")
print("-" * 70)
print(f"Remaining matches: {len(remaining_df)}")

if not remaining_df.empty:
    print(
        remaining_df["publisher"]
        .value_counts()
        .to_string()
    )

    remaining_df.to_csv(
        "outputs/audits/remaining_publisher_suffixes.csv",
        index=False
    )

print("\n6. FINAL RECORD CHECK")
print("-" * 70)
print(f"Original records: {len(df_original)}")
print(f"Final records: {len(df_final)}")
print(f"Fake records: {(df_final['real'] == 0).sum()}")
print(f"Real records: {(df_final['real'] == 1).sum()}")
print(f"Missing titles: {df_final['title'].isna().sum()}")
print(f"Missing labels: {df_final['real'].isna().sum()}")

print("\nFINAL SOURCE-BIAS AUDIT COMPLETED")

df = pd.read_csv(
    "outputs/audits/remaining_publisher_suffixes.csv"
)

print("REMAINING PUBLISHER MATCH")
print("=" * 60)

for _, row in df.iterrows():
    print(f"Publisher: {row['publisher']}")
    print(f"Source: {row['source_dataset']}")
    print(f"Title: {row['title']}")