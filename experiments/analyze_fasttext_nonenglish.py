import pandas as pd
from pathlib import Path

input_path = Path("outputs/audits/welfake_language_fasttext.csv")
df = pd.read_csv(input_path)

print("FASTTEXT NON-ENGLISH REVIEW")
print("=" * 60)

non_english = df[df["fasttext_language"] != "en"].copy()

print(f"Total non-English predictions: {len(non_english)}")

print("\n1. NON-ENGLISH CONFIDENCE GROUPS")
print("-" * 60)

for threshold in [0.50, 0.70, 0.80, 0.90, 0.95, 0.99]:
    selected = non_english[
        non_english["fasttext_score"] >= threshold
    ]
    print(
        f"Score >= {threshold:.2f}: "
        f"{len(selected)} headlines"
    )

print("\n2. HIGH-CONFIDENCE NON-ENGLISH EXAMPLES")
print("-" * 60)

high_confidence = non_english[
    non_english["fasttext_score"] >= 0.90
].sort_values("fasttext_score", ascending=False)

for i, (_, row) in enumerate(high_confidence.head(60).iterrows(), 1):
    print(
        f"{i}. {row['fasttext_language']} "
        f"({row['fasttext_score']:.4f}) | "
        f"Label: {row['real']} | "
        f"{row['title']}"
    )

print("\n3. LOW-CONFIDENCE NON-ENGLISH EXAMPLES")
print("-" * 60)

low_confidence = non_english[
    non_english["fasttext_score"] < 0.70
].sort_values("fasttext_score", ascending=False)

for i, (_, row) in enumerate(low_confidence.head(40).iterrows(), 1):
    print(
        f"{i}. {row['fasttext_language']} "
        f"({row['fasttext_score']:.4f}) | "
        f"{row['title']}"
    )

print("\n4. LANGUAGE COUNTS AMONG HIGH-CONFIDENCE RECORDS")
print("-" * 60)

print(
    high_confidence["fasttext_language"]
    .value_counts()
    .to_string()
)

print("\nAUDIT COMPLETED")