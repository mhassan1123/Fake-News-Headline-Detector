import pandas as pd
from pathlib import Path

welfake_path = Path("data/processed/cleaned_welfake_english.csv")
audit_path = Path("outputs/audits/welfake_language_fasttext.csv")
fakenewsnet_path = Path("data/processed/cleaned_fakenewsnet.csv")

welfake = pd.read_csv(welfake_path)
audit = pd.read_csv(audit_path)
fakenewsnet = pd.read_csv(fakenewsnet_path)

print("LABEL CONSISTENCY VERIFICATION")
print("=" * 60)

# Verify WELFake's original-to-project mapping.
expected_labels = audit["original_label"].map({
    0: 1,
    1: 0
})

if not expected_labels.equals(audit["real"]):
    raise ValueError("WELFake audit mapping is incorrect.")

# Verify the final cleaned WELFake labels.
if not set(welfake["real"].dropna().unique()).issubset({0, 1}):
    raise ValueError("Invalid WELFake label values.")

# Verify FakeNewsNet labels.
if not set(fakenewsnet["real"].dropna().unique()).issubset({0, 1}):
    raise ValueError("Invalid FakeNewsNet label values.")

print("WELFake label distribution:")
print(welfake["real"].value_counts().sort_index().to_string())

print("\nFakeNewsNet label distribution:")
print(fakenewsnet["real"].value_counts().sort_index().to_string())

print("\nProject label convention:")
print("0 = Fake")
print("1 = Real")

print("\nLABEL CONSISTENCY CHECK COMPLETED")