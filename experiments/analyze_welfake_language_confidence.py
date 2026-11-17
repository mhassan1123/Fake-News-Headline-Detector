import pandas as pd
from pathlib import Path

input_path = Path("outputs/audits/welfake_language_audit.csv")

df = pd.read_csv(input_path)

print("WELFAKE LANGUAGE CONFIDENCE ANALYSIS")
print("=" * 60)

# 1. Check the relationship between detected language and classification.
print("\n1. DETECTED LANGUAGE VS CLASSIFICATION")
print("-" * 60)

cross_table = pd.crosstab(
    df["detected_language"],
    df["classification"]
)

print(cross_table.sort_values(
    by="English",
    ascending=False
).head(15).to_string())

# 2. Examine confidence scores for English detections.
english = df[
    df["detected_language"] == "Language.ENGLISH"
].copy()

print("\n2. CONFIDENCE STATISTICS FOR ENGLISH DETECTIONS")
print("-" * 60)

print(english["top_confidence"].describe().to_string())

# 3. Test different confidence thresholds.
print("\n3. ENGLISH DETECTION AT DIFFERENT THRESHOLDS")
print("-" * 60)

for threshold in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
    count = (
        english["top_confidence"] >= threshold
    ).sum()

    percentage = count / len(df) * 100

    print(
        f"Threshold {threshold:.2f}: "
        f"{count} headlines ({percentage:.2f}% of dataset)"
    )

# 4. Inspect English detections with low confidence.
print("\n4. ENGLISH DETECTIONS WITH LOW CONFIDENCE")
print("-" * 60)

low_confidence = english[
    english["top_confidence"] < 0.70
].sort_values("top_confidence")

for i, (_, row) in enumerate(low_confidence.head(30).iterrows(), 1):
    print(
        f"{i}. Confidence: {row['top_confidence']:.4f} | "
        f"{row['title']}"
    )

# 5. Inspect headlines with non-English top predictions.
print("\n5. NON-ENGLISH TOP PREDICTIONS")
print("-" * 60)

non_english = df[
    df["detected_language"] != "Language.ENGLISH"
].sort_values("top_confidence", ascending=False)

for i, (_, row) in enumerate(non_english.head(30).iterrows(), 1):
    print(
        f"{i}. Language: {row['detected_language']} | "
        f"Confidence: {row['top_confidence']:.4f} | "
        f"{row['title']}"
    )

# 6. Check whether the second-ranked language is close to the first.
print("\n6. TOP TWO LANGUAGE CONFIDENCE DIFFERENCE")
print("-" * 60)

df["confidence_difference"] = (
    df["top_confidence"] - df["second_confidence"]
)

print(df["confidence_difference"].describe().to_string())

print("\nLANGUAGE CONFIDENCE ANALYSIS COMPLETED")