import pandas as pd
from langdetect import detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0

input_path = "data/raw/WELFake_Dataset.csv"

print("WELFAKE LANGUAGE AUDIT\n" + "=" * 55)

df = pd.read_csv(input_path, low_memory=False)

# Keep only records with a valid headline.
df = df.dropna(subset=["title"]).copy()
df["title"] = df["title"].astype(str).str.strip()
df = df[df["title"] != ""]

def classify_language(headline):
    try:
        results = detect_langs(headline)

        if not results:
            return "uncertain"

        best = results[0]

        if best.lang == "en" and best.prob >= 0.80:
            return "English"

        if best.lang != "en" and best.prob >= 0.80:
            return "Non-English"

        return "Uncertain"

    except LangDetectException:
        return "Uncertain"


print(f"\nValid headlines to inspect: {len(df)}")
print("Detecting languages. Please wait...")

df["language_status"] = df["title"].apply(classify_language)

print("\n1. LANGUAGE DISTRIBUTION")
print("-" * 55)

counts = df["language_status"].value_counts()

for language, count in counts.items():
    percentage = count / len(df) * 100
    print(f"{language}: {count} ({percentage:.2f}%)")

print("\n2. NON-ENGLISH HEADLINE EXAMPLES")
print("-" * 55)

non_english = df[df["language_status"] == "Non-English"]

for index, headline in enumerate(non_english["title"].head(20), start=1):
    print(f"{index}. {headline[:250]}")

print("\n3. UNCERTAIN HEADLINE EXAMPLES")
print("-" * 55)

uncertain = df[df["language_status"] == "Uncertain"]

for index, headline in enumerate(uncertain["title"].head(20), start=1):
    print(f"{index}. {headline[:250]}")

print("\nLANGUAGE AUDIT COMPLETED")
print("Original CSV was not modified.")