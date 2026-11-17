import pandas as pd
import re
from pathlib import Path

data_path = Path("data/processed/merged_headlines.csv")
output_path = Path("outputs/audits")

output_path.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data_path)

# Publisher names and website names found during the previous audit
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

# Match a publisher name only when it appears at the end
# of a headline, optionally separated by punctuation.
patterns = []

for publisher in publishers:
    pattern = (
        r"(?:\s*[-–—|·•:]\s*|\s+)"
        + re.escape(publisher)
        + r"\s*$"
    )

    patterns.append((publisher, re.compile(pattern, re.IGNORECASE)))

results = []

for _, row in df.iterrows():
    title = str(row["title"]).strip()

    for publisher, pattern in patterns:
        match = pattern.search(title)

        if match:
            original_part = title[:match.start()].strip()
            suffix = title[match.start():].strip()

            results.append({
                "source_dataset": row["source_dataset"],
                "label": "Real" if row["real"] == 1 else "Fake",
                "publisher": publisher,
                "original_title": title,
                "possible_headline": original_part,
                "matched_suffix": suffix,
                "has_separator": bool(
                    re.match(r"^[-–—|·•:]", suffix)
                )
            })

            break

results_df = pd.DataFrame(results)

print("PUBLISHER SUFFIX INVESTIGATION")
print("=" * 65)

print("\n1. TOTAL MATCHES")
print("-" * 65)
print(f"Matching headlines: {len(results_df)}")

if not results_df.empty:

    print("\n2. MATCHES BY PUBLISHER")
    print("-" * 65)

    publisher_summary = results_df.groupby(
        "publisher"
    ).size().sort_values(ascending=False)

    print(publisher_summary.to_string())

    print("\n3. MATCHES BY DATASET")
    print("-" * 65)

    print(
        pd.crosstab(
            results_df["publisher"],
            results_df["source_dataset"]
        ).to_string()
    )

    print("\n4. MATCHES BY LABEL")
    print("-" * 65)

    print(
        pd.crosstab(
            results_df["publisher"],
            results_df["label"]
        ).to_string()
    )

    print("\n5. EXAMPLE HEADLINES")
    print("-" * 65)

    for publisher in publisher_summary.index:
        examples = results_df[
            results_df["publisher"] == publisher
        ].head(5)

        print(f"\nPUBLISHER: {publisher}")

        for _, row in examples.iterrows():
            print(f"Source: {row['source_dataset']}")
            print(f"Label: {row['label']}")
            print(f"Original: {row['original_title']}")
            print(f"Possible headline: {row['possible_headline']}")
            print(f"Matched suffix: {row['matched_suffix']}")

    print("\n6. SEPARATOR ANALYSIS")
    print("-" * 65)

    print(
        results_df.groupby(
            ["publisher", "has_separator"]
        ).size().to_string()
    )

    results_df.to_csv(
        output_path / "publisher_suffix_matches.csv",
        index=False
    )

    publisher_summary.to_csv(
        output_path / "publisher_suffix_summary.csv"
    )

    print("\nAUDIT FILES SAVED")
    print("outputs/audits/publisher_suffix_matches.csv")
    print("outputs/audits/publisher_suffix_summary.csv")

else:
    print("No publisher suffixes matched.")

print("\nPUBLISHER SUFFIX INVESTIGATION COMPLETED")