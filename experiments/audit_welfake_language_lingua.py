import pandas as pd
from pathlib import Path
from lingua import Language, LanguageDetectorBuilder

input_path = Path("data/processed/welfake_deduplicated.csv")
output_path = Path("outputs/audits/welfake_language_audit.csv")

df = pd.read_csv(input_path)

print("WELFAKE LINGUA LANGUAGE AUDIT")
print("=" * 55)
print(f"Unique headlines to inspect: {len(df)}")
print("Building language detector...")

# Build a detector supporting all available languages.
detector = (
    LanguageDetectorBuilder
    .from_all_languages()
    .with_preloaded_language_models()
    .build()
)

detected_languages = []
top_confidences = []
second_languages = []
second_confidences = []
classifications = []

print("Detecting languages. Please wait...")

for index, headline in enumerate(df["title"]):

    results = detector.compute_language_confidence_values(headline)

    if len(results) == 0:
        detected_languages.append("Unknown")
        top_confidences.append(0.0)
        second_languages.append("Unknown")
        second_confidences.append(0.0)
        classifications.append("Uncertain")
        continue

    top = results[0]

    top_language = str(top.language)
    top_confidence = top.value

    if len(results) > 1:
        second = results[1]
        second_language = str(second.language)
        second_confidence = second.value
    else:
        second_language = "Unknown"
        second_confidence = 0.0

    # These are relative confidence scores, not calibrated probabilities.
    if top.language == Language.ENGLISH and top_confidence >= 0.70:
        category = "English"

    elif top.language != Language.ENGLISH and top_confidence >= 0.70:
        category = "Non-English"

    else:
        category = "Uncertain"

    detected_languages.append(top_language)
    top_confidences.append(top_confidence)
    second_languages.append(second_language)
    second_confidences.append(second_confidence)
    classifications.append(category)

    if (index + 1) % 10000 == 0:
        print(f"Processed: {index + 1}/{len(df)}")

# Store results without changing the cleaned dataset.
df["detected_language"] = detected_languages
df["top_confidence"] = top_confidences
df["second_language"] = second_languages
df["second_confidence"] = second_confidences
df["classification"] = classifications

# Summary.
print("\n1. LANGUAGE DISTRIBUTION")
print("-" * 55)

distribution = df["classification"].value_counts()

for category in ["English", "Non-English", "Uncertain"]:
    count = distribution.get(category, 0)
    percentage = count / len(df) * 100 if len(df) else 0
    print(f"{category}: {count} ({percentage:.2f}%)")

print("\n2. DETECTED LANGUAGE COUNTS")
print("-" * 55)
print(df["detected_language"].value_counts().head(15).to_string())

print("\n3. NON-ENGLISH EXAMPLES")
print("-" * 55)

non_english = df[df["classification"] == "Non-English"]

for i, headline in enumerate(non_english["title"].head(25), 1):
    print(f"{i}. {headline}")

print("\n4. UNCERTAIN EXAMPLES")
print("-" * 55)

uncertain = df[df["classification"] == "Uncertain"]

for i, headline in enumerate(uncertain["title"].head(25), 1):
    print(f"{i}. {headline}")

# Save complete audit for inspection.
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print("\n5. AUDIT FILE")
print("-" * 55)
print(f"Saved to: {output_path}")
print("\nLANGUAGE AUDIT COMPLETED")