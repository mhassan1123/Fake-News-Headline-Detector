import pandas as pd
from pathlib import Path
from ftlangdetect import detect

input_path = Path("data/processed/welfake_deduplicated.csv")
output_path = Path("outputs/audits/welfake_language_fasttext.csv")

df = pd.read_csv(input_path)

print("WELFAKE FASTTEXT LANGUAGE AUDIT")
print("=" * 60)
print(f"Headlines to inspect: {len(df)}")
print("Loading language model...")

languages = []
scores = []
second_languages = []
second_scores = []

print("Detecting languages. Please wait...")

for index, headline in enumerate(df["title"]):

    try:
        results = detect(
            text=str(headline),
            low_memory=False,
            k=2
        )

        top = results[0]

        if len(results) > 1:
            second = results[1]
            second_language = second["lang"]
            second_score = second["score"]
        else:
            second_language = "Unknown"
            second_score = 0.0

        languages.append(top["lang"])
        scores.append(top["score"])
        second_languages.append(second_language)
        second_scores.append(second_score)

    except Exception as error:
        languages.append("unknown")
        scores.append(0.0)
        second_languages.append("unknown")
        second_scores.append(0.0)

    if (index + 1) % 10000 == 0:
        print(f"Processed: {index + 1}/{len(df)}")

# Add language predictions to the dataset.
df["fasttext_language"] = languages
df["fasttext_score"] = scores
df["fasttext_second_language"] = second_languages
df["fasttext_second_score"] = second_scores

# Provisional classification.
# This threshold is for audit grouping only.
def classify_language(language, score):
    if score < 0.50:
        return "Uncertain"

    if language == "en":
        return "English"

    return "Non-English"


df["fasttext_classification"] = [
    classify_language(language, score)
    for language, score in zip(languages, scores)
]

# Compare the new detector with Lingua.
if "detected_language" in df.columns:
    pass

lingua_path = Path("outputs/audits/welfake_language_audit.csv")

if lingua_path.exists():
    lingua_df = pd.read_csv(lingua_path)

    df["lingua_language"] = lingua_df["detected_language"].values

    df["detector_agreement"] = [
        "Agree" if
        (new_language == "en") ==
        (old_language == "Language.ENGLISH")
        else "Disagree"
        for new_language, old_language in zip(
            df["fasttext_language"],
            df["lingua_language"]
        )
    ]
else:
    df["lingua_language"] = "Unavailable"
    df["detector_agreement"] = "Unavailable"

# 1. Language distribution.
print("\n1. FASTTEXT LANGUAGE DISTRIBUTION")
print("-" * 60)

print(df["fasttext_language"].value_counts().head(20).to_string())

# 2. Classification distribution.
print("\n2. PROVISIONAL CLASSIFICATION")
print("-" * 60)

distribution = df["fasttext_classification"].value_counts()

for category in ["English", "Non-English", "Uncertain"]:
    count = distribution.get(category, 0)
    percentage = count / len(df) * 100
    print(f"{category}: {count} ({percentage:.2f}%)")

# 3. Confidence score statistics.
print("\n3. FASTTEXT SCORE STATISTICS")
print("-" * 60)

print(df["fasttext_score"].describe().to_string())

# 4. Detector agreement.
print("\n4. COMPARISON WITH LINGUA")
print("-" * 60)

print(df["detector_agreement"].value_counts().to_string())

# 5. Disagreement examples.
print("\n5. DISAGREEMENT EXAMPLES")
print("-" * 60)

disagreements = df[
    df["detector_agreement"] == "Disagree"
]

for i, (_, row) in enumerate(disagreements.head(40).iterrows(), 1):
    print(
        f"{i}. FastText: {row['fasttext_language']} "
        f"({row['fasttext_score']:.4f}) | "
        f"Lingua: {row['lingua_language']} | "
        f"{row['title']}"
    )

# 6. High-scoring non-English examples.
print("\n6. NON-ENGLISH EXAMPLES")
print("-" * 60)

non_english = df[
    df["fasttext_classification"] == "Non-English"
].sort_values("fasttext_score", ascending=False)

for i, (_, row) in enumerate(non_english.head(30).iterrows(), 1):
    print(
        f"{i}. Language: {row['fasttext_language']} | "
        f"Score: {row['fasttext_score']:.4f} | "
        f"{row['title']}"
    )

# 7. Save complete audit.
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print("\n7. OUTPUT")
print("-" * 60)
print(f"Saved to: {output_path}")
print("\nFASTTEXT LANGUAGE AUDIT COMPLETED")