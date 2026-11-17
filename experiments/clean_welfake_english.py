import pandas as pd
from pathlib import Path

source_path = Path("data/processed/welfake_deduplicated.csv")
audit_path = Path("outputs/audits/welfake_language_fasttext.csv")

clean_path = Path("data/processed/cleaned_welfake_english.csv")
removed_path = Path("outputs/audits/welfake_removed_nonenglish.csv")

df = pd.read_csv(source_path)
audit = pd.read_csv(audit_path)

print("FINAL WELFAKE LANGUAGE CLEANING")
print("=" * 60)

if len(df) != len(audit):
    raise ValueError("Source and audit record counts do not match.")

if not df["title"].equals(audit["title"]):
    raise ValueError("Headline order or content does not match.")

if not df["real"].equals(audit["real"]):
    raise ValueError("Label values do not match.")

remove_mask = (
    (audit["fasttext_language"] != "en") &
    (audit["fasttext_score"] >= 0.90)
)

cleaned = df.loc[~remove_mask].copy()
removed = df.loc[remove_mask].copy()

clean_path.parent.mkdir(parents=True, exist_ok=True)
removed_path.parent.mkdir(parents=True, exist_ok=True)

cleaned.to_csv(clean_path, index=False)
removed.to_csv(removed_path, index=False)

print(f"Original cleaned records: {len(df)}")
print(f"Removed records: {len(removed)}")
print(f"Remaining records: {len(cleaned)}")

print("\n1. REMAINING CLASS DISTRIBUTION")
print("-" * 60)

print(cleaned["real"].value_counts().sort_index().to_string())

print("\n2. REMOVED CLASS DISTRIBUTION")
print("-" * 60)

print(removed["real"].value_counts().sort_index().to_string())

print("\n3. SAVED FILES")
print("-" * 60)
print(f"Clean dataset: {clean_path}")
print(f"Removed records: {removed_path}")

if len(cleaned) != 61853 or len(removed) != 382:
    raise ValueError("Unexpected final record counts.")

print("\nFINAL CLEANING VERIFIED")