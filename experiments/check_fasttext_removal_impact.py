import pandas as pd
from pathlib import Path

input_path = Path("outputs/audits/welfake_language_fasttext.csv")

df = pd.read_csv(input_path)

candidates = df[
    (df["fasttext_language"] != "en") &
    (df["fasttext_score"] >= 0.90)
].copy()

print("FASTTEXT REMOVAL IMPACT AUDIT")
print("=" * 60)

print(f"Total cleaned WELFake headlines: {len(df)}")
print(f"Candidate removals: {len(candidates)}")
print(f"Remaining records: {len(df) - len(candidates)}")

print("\n1. ORIGINAL LABEL DISTRIBUTION")
print("-" * 60)
print(df["real"].value_counts().sort_index().to_string())

print("\n2. CANDIDATE REMOVAL LABEL DISTRIBUTION")
print("-" * 60)
print(candidates["real"].value_counts().sort_index().to_string())

print("\n3. CANDIDATE REMOVAL LANGUAGE DISTRIBUTION")
print("-" * 60)
print(candidates["fasttext_language"].value_counts().to_string())

print("\n4. REMOVAL PERCENTAGE BY CLASS")
print("-" * 60)

for label in sorted(df["real"].dropna().unique()):
    original_count = (df["real"] == label).sum()
    removed_count = (candidates["real"] == label).sum()
    percentage = removed_count / original_count * 100

    print(
        f"Label {label}: {removed_count}/{original_count} "
        f"({percentage:.2f}%)"
    )

print("\n5. CANDIDATE RECORDS WITH UNUSABLE HEADLINES")
print("-" * 60)

invalid = candidates[
    candidates["title"].isna() |
    (candidates["title"].astype(str).str.strip().isin(["", ":"]))
]

print(f"Unusable headlines: {len(invalid)}")

print("\nAUDIT COMPLETED")