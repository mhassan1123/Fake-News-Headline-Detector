import pandas as pd
from pathlib import Path

input_path = Path("outputs/audits/welfake_language_audit.csv")

df = pd.read_csv(input_path)

print("WELFAKE LANGUAGE MARGIN ANALYSIS")
print("=" * 60)

# Calculate the difference between the first and second language scores.
df["margin"] = (
    df["top_confidence"] - df["second_confidence"]
)

# 1. Distribution of top-ranked languages.
print("\n1. TOP-RANKED LANGUAGE DISTRIBUTION")
print("-" * 60)

print(df["detected_language"].value_counts().to_string())

# 2. Examine different margin ranges.
print("\n2. LANGUAGE MARGIN DISTRIBUTION")
print("-" * 60)

bins = [0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.01]

labels = [
    "0.00 - 0.05",
    "0.05 - 0.10",
    "0.10 - 0.20",
    "0.20 - 0.30",
    "0.30 - 0.50",
    "0.50 - 1.00"
]

df["margin_group"] = pd.cut(
    df["margin"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

print(
    df["margin_group"]
    .value_counts(sort=False)
    .to_string()
)

# 3. English predictions with the smallest margins.
print("\n3. ENGLISH PREDICTIONS WITH LOWEST MARGINS")
print("-" * 60)

english = df[
    df["detected_language"] == "Language.ENGLISH"
].sort_values("margin")

for i, (_, row) in enumerate(english.head(40).iterrows(), 1):
    print(
        f"{i}. Margin: {row['margin']:.4f} | "
        f"Top score: {row['top_confidence']:.4f} | "
        f"Second: {row['second_language']} | "
        f"{row['title']}"
    )

# 4. Non-English predictions with the smallest margins.
print("\n4. NON-ENGLISH PREDICTIONS WITH LOWEST MARGINS")
print("-" * 60)

non_english = df[
    df["detected_language"] != "Language.ENGLISH"
].sort_values("margin")

for i, (_, row) in enumerate(non_english.head(40).iterrows(), 1):
    print(
        f"{i}. Language: {row['detected_language']} | "
        f"Margin: {row['margin']:.4f} | "
        f"Second: {row['second_language']} | "
        f"{row['title']}"
    )

# 5. Non-English predictions with the highest margins.
print("\n5. NON-ENGLISH PREDICTIONS WITH HIGHEST MARGINS")
print("-" * 60)

for i, (_, row) in enumerate(non_english.sort_values(
    "margin", ascending=False
).head(30).iterrows(), 1):
    print(
        f"{i}. Language: {row['detected_language']} | "
        f"Margin: {row['margin']:.4f} | "
        f"{row['title']}"
    )

# 6. Save detailed analysis.
output_path = Path("outputs/audits/welfake_language_margin_analysis.csv")

output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print("\n6. OUTPUT")
print("-" * 60)
print(f"Saved to: {output_path}")
print("\nMARGIN ANALYSIS COMPLETED")