import pandas as pd
import re
from pathlib import Path

data_path = Path("data/processed/merged_headlines.csv")
output_path = Path("outputs/audits")

output_path.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data_path)

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

# Separators indicate a stronger possibility of appended metadata.
separator_pattern = r"[-–—|·•:*]"

strong_patterns = []
review_patterns = []

for publisher in publishers:
    publisher_pattern = re.escape(publisher)

    strong_patterns.append((
        publisher,
        re.compile(
            r"\s*" + separator_pattern +
            r"\s*(?:the\s+)?" +
            publisher_pattern +
            r"\s*$",
            re.IGNORECASE
        )
    ))

    review_patterns.append((
        publisher,
        re.compile(
            r"\s+(?:the\s+)?" +
            publisher_pattern +
            r"\s*$",
            re.IGNORECASE
        )
    ))

strong_results = []
review_results = []

for _, row in df.iterrows():

    title = str(row["title"]).strip()

    # First check for an explicit separator.
    matched = False

    for publisher, pattern in strong_patterns:

        match = pattern.search(title)

        if match:
            strong_results.append({
                "source_dataset": row["source_dataset"],
                "label": "Real" if row["real"] == 1 else "Fake",
                "publisher": publisher,
                "original_title": title,
                "possible_headline": title[:match.start()].strip(),
                "matched_suffix": match.group().strip()
            })

            matched = True
            break

    if matched:
        continue

    # Without a separator, keep the match for manual review.
    for publisher, pattern in review_patterns:

        match = pattern.search(title)

        if match:
            review_results.append({
                "source_dataset": row["source_dataset"],
                "label": "Real" if row["real"] == 1 else "Fake",
                "publisher": publisher,
                "original_title": title,
                "possible_headline": title[:match.start()].strip(),
                "matched_suffix": match.group().strip()
            })

            break

strong_df = pd.DataFrame(strong_results)
review_df = pd.DataFrame(review_results)

print("PUBLISHER SUFFIX AUDIT V2")
print("=" * 65)

print("\n1. MATCH COUNTS")
print("-" * 65)
print(f"Strong matches: {len(strong_df)}")
print(f"Review matches: {len(review_df)}")

for name, result in [
    ("STRONG MATCHES", strong_df),
    ("REVIEW MATCHES", review_df)
]:

    print(f"\n2. {name} BY PUBLISHER")
    print("-" * 65)

    if result.empty:
        print("No matches.")
        continue

    summary = result.groupby(
        "publisher"
    ).size().sort_values(ascending=False)

    print(summary.to_string())

    print(f"\n{name} BY DATASET")
    print("-" * 65)

    print(
        pd.crosstab(
            result["publisher"],
            result["source_dataset"]
        ).to_string()
    )

    print(f"\n{name} BY LABEL")
    print("-" * 65)

    print(
        pd.crosstab(
            result["publisher"],
            result["label"]
        ).to_string()
    )

    print(f"\n{name} EXAMPLES")
    print("-" * 65)

    for publisher in summary.index:

        examples = result[
            result["publisher"] == publisher
        ].head(3)

        print(f"\nPublisher: {publisher}")

        for _, item in examples.iterrows():
            print(f"Source: {item['source_dataset']}")
            print(f"Label: {item['label']}")
            print(f"Original: {item['original_title']}")
            print(f"Possible headline: {item['possible_headline']}")
            print(f"Matched suffix: {item['matched_suffix']}")

strong_df.to_csv(
    output_path / "publisher_suffix_strong_v2.csv",
    index=False
)

review_df.to_csv(
    output_path / "publisher_suffix_review_v2.csv",
    index=False
)

print("\nAUDIT FILES SAVED")
print("outputs/audits/publisher_suffix_strong_v2.csv")
print("outputs/audits/publisher_suffix_review_v2.csv")

print("\nPUBLISHER SUFFIX AUDIT V2 COMPLETED")