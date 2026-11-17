import pandas as pd
import re
from collections import Counter
from pathlib import Path

data_path = Path("data/processed/merged_headlines.csv")
output_path = Path("outputs/audits")

output_path.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data_path)

print("SOURCE BIAS AND HEADLINE PATTERN AUDIT")
print("=" * 65)

# 1. Dataset source and class distribution
print("\n1. SOURCE AND CLASS DISTRIBUTION")
print("-" * 65)

source_counts = pd.crosstab(
    df["source_dataset"],
    df["real"],
    margins=True
)

source_counts.columns = [
    "Fake" if col == 0 else "Real" if col == 1 else "Total"
    for col in source_counts.columns
]

print(source_counts.to_string())

# 2. Headline length by source and class
df["word_count"] = df["title"].astype(str).apply(
    lambda title: len(title.split())
)

print("\n2. HEADLINE LENGTH BY SOURCE AND CLASS")
print("-" * 65)

length_stats = df.groupby(
    ["source_dataset", "real"]
)["word_count"].agg(["count", "mean", "median"])

print(length_stats.round(2).to_string())

# 3. Detect common publisher-like words
publisher_terms = [
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
    "buzzfeed"
]

print("\n3. PUBLISHER NAME OCCURRENCES")
print("-" * 65)

publisher_results = []

for publisher in publisher_terms:
    matches = df[
        df["title"].str.contains(
            re.escape(publisher),
            case=False,
            regex=True,
            na=False
        )
    ]

    if len(matches) > 0:
        fake_count = (matches["real"] == 0).sum()
        real_count = (matches["real"] == 1).sum()

        publisher_results.append({
            "publisher": publisher,
            "total": len(matches),
            "fake": fake_count,
            "real": real_count,
            "sources": ", ".join(
                sorted(matches["source_dataset"].unique())
            )
        })

if publisher_results:
    publisher_df = pd.DataFrame(publisher_results)
    print(publisher_df.to_string(index=False))
    publisher_df.to_csv(
        output_path / "publisher_occurrences.csv",
        index=False
    )
else:
    print("No listed publisher names found.")

# 4. Inspect common headline endings
print("\n4. COMMON HEADLINE ENDINGS")
print("-" * 65)

def get_ending(title):
    words = str(title).strip().split()
    return " ".join(words[-4:]).lower()

ending_results = []

for source in df["source_dataset"].unique():
    source_df = df[df["source_dataset"] == source]

    endings = Counter(
        source_df["title"].apply(get_ending)
    )

    for ending, count in endings.most_common(15):
        ending_results.append({
            "source": source,
            "ending": ending,
            "frequency": count
        })

ending_df = pd.DataFrame(ending_results)
print(ending_df.to_string(index=False))

ending_df.to_csv(
    output_path / "common_headline_endings.csv",
    index=False
)

# 5. Inspect examples containing publisher names
print("\n5. EXAMPLE HEADLINES WITH PUBLISHER NAMES")
print("-" * 65)

for publisher in publisher_terms:
    matches = df[
        df["title"].str.contains(
            re.escape(publisher),
            case=False,
            regex=True,
            na=False
        )
    ]

    if len(matches) > 0:
        print(f"\nPUBLISHER: {publisher}")

        for _, row in matches.head(3).iterrows():
            print(
                f"Label: {'Real' if row['real'] == 1 else 'Fake'} | "
                f"Source: {row['source_dataset']}"
            )
            print(f"Title: {row['title']}")

# 6. Save source and length summaries
source_counts.to_csv(
    output_path / "source_class_counts.csv"
)

length_stats.to_csv(
    output_path / "source_headline_length.csv"
)

print("\nAUDIT FILES SAVED")
print("outputs/audits/publisher_occurrences.csv")
print("outputs/audits/common_headline_endings.csv")
print("outputs/audits/source_class_counts.csv")
print("outputs/audits/source_headline_length.csv")

print("\nSOURCE BIAS AUDIT COMPLETED")