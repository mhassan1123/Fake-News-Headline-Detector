import pandas as pd
from pathlib import Path

raw_path = Path("data/raw/WELFake_Dataset.csv")
deduplicated_path = Path("data/processed/welfake_deduplicated.csv")
cleaned_path = Path("data/processed/cleaned_welfake_english.csv")

raw = pd.read_csv(raw_path)
deduplicated = pd.read_csv(deduplicated_path)
cleaned = pd.read_csv(cleaned_path)

print("WELFAKE ORIGINAL LABEL INVESTIGATION")
print("=" * 65)

print("\n1. RAW FILE COLUMNS")
print("-" * 65)
print(raw.columns.tolist())

if "label" not in raw.columns:
    raise ValueError("Original label column not found in raw dataset.")

print("\n2. RAW ORIGINAL LABEL DISTRIBUTION")
print("-" * 65)
print(raw["label"].value_counts(dropna=False).sort_index().to_string())

print("\n3. DEDUPLICATED ORIGINAL LABEL DISTRIBUTION")
print("-" * 65)
print(
    deduplicated["original_label"]
    .value_counts(dropna=False)
    .sort_index()
    .to_string()
)

print("\n4. CLEANED ORIGINAL LABEL DISTRIBUTION")
print("-" * 65)
print(
    cleaned["original_label"]
    .value_counts(dropna=False)
    .sort_index()
    .to_string()
)

print("\n5. TEN RANDOM RECORDS FROM EACH ORIGINAL LABEL")
print("-" * 65)

for label in sorted(raw["label"].dropna().unique()):
    sample = raw[raw["label"] == label].sample(
        n=min(10, (raw["label"] == label).sum()),
        random_state=42
    )

    print(f"\nORIGINAL LABEL: {label}")
    print("-" * 65)

    for i, (_, row) in enumerate(sample.iterrows(), 1):
        title = str(row["title"])
        text = str(row["text"]).replace("\n", " ")

        print(f"{i}. TITLE: {title}")
        print(f"   ARTICLE TEXT: {text[:250]}")
        print()

print("\n6. CURRENT CONVERSION CHECK")
print("-" * 65)

expected = cleaned["original_label"].map({0: 1, 1: 0})

print("Current conversion: original 0 -> project 1")
print("                    original 1 -> project 0")

print(
    "Conversion matches cleaned file:",
    expected.equals(cleaned["real"])
)

print("\nINVESTIGATION COMPLETED")