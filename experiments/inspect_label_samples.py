import pandas as pd
from pathlib import Path

welfake_path = Path("data/processed/cleaned_welfake_english.csv")
fakenewsnet_path = Path("data/processed/cleaned_fakenewsnet.csv")

welfake = pd.read_csv(welfake_path)
fakenewsnet = pd.read_csv(fakenewsnet_path)

print("LABEL SAMPLE INSPECTION")
print("=" * 65)

# Verify WELFake original-to-project label conversion.
if "original_label" in welfake.columns:
    raw_column = "original_label"
elif "label" in welfake.columns:
    raw_column = "label"
else:
    raise ValueError("Original WELFake label column not found.")

expected_labels = welfake[raw_column].map({0: 1, 1: 0})

if expected_labels.isna().any():
    raise ValueError("Unexpected original WELFake label values.")

if not expected_labels.equals(welfake["real"]):
    raise ValueError("WELFake label conversion is incorrect.")

print("\n1. WELFAKE LABEL CONVERSION")
print("-" * 65)
print("Original label 0 -> Project label 1 (Real)")
print("Original label 1 -> Project label 0 (Fake)")
print("All WELFake records passed conversion verification.")

# Verify FakeNewsNet project labels.
if not set(fakenewsnet["real"].dropna().unique()).issubset({0, 1}):
    raise ValueError("Unexpected FakeNewsNet label values.")

print("\n2. FAKENEWSNET LABEL VERIFICATION")
print("-" * 65)
print("FakeNewsNet labels already follow the project convention.")

# Extract 10 examples from each class.
for dataset_name, dataset in [
    ("WELFAKE", welfake),
    ("FAKENEWSNET", fakenewsnet)
]:
    print(f"\n3. {dataset_name} SAMPLE RECORDS")
    print("=" * 65)

    for label, class_name in [(0, "FAKE"), (1, "REAL")]:
        sample = dataset[dataset["real"] == label].sample(
            n=min(10, (dataset["real"] == label).sum()),
            random_state=42
        )

        print(f"\n{class_name} HEADLINES (Project label = {label})")
        print("-" * 65)

        for index, (_, row) in enumerate(sample.iterrows(), 1):
            title = str(row["title"])

            if dataset_name == "WELFAKE":
                original = row[raw_column]
                print(
                    f"{index}. Original label: {original} | "
                    f"Project label: {row['real']} | {title}"
                )
            else:
                print(
                    f"{index}. Project label: {row['real']} | {title}"
                )

print("\n4. FINAL VERIFICATION")
print("=" * 65)
print(f"WELFake records: {len(welfake)}")
print(f"FakeNewsNet records: {len(fakenewsnet)}")
print("WELFake mapping: PASSED")
print("FakeNewsNet label values: PASSED")
print("Sample extraction: COMPLETED")