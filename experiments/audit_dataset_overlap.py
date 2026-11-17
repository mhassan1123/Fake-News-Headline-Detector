import pandas as pd

welfake_path = "data/raw/WELFake_Dataset.csv"
fakenewsnet_path = "data/processed/cleaned_fakenewsnet.csv"

print("DATASET OVERLAP AUDIT\n" + "=" * 55)

# Load both datasets.
welfake = pd.read_csv(welfake_path, low_memory=False)
fakenewsnet = pd.read_csv(fakenewsnet_path)

# Standardize column names.
welfake = welfake.rename(columns={
    "title": "headline",
    "label": "original_label"
})

fakenewsnet = fakenewsnet.rename(columns={
    "title": "headline"
})

# Select only the columns needed for comparison.
welfake = welfake[["headline", "original_label"]].copy()
fakenewsnet = fakenewsnet[["headline", "real"]].copy()

# Normalize labels into one common convention:
# 0 = Fake, 1 = Real.
welfake["real"] = welfake["original_label"].map({
    0: 1,
    1: 0
})

fakenewsnet["real"] = fakenewsnet["real"].astype(int)

# Normalize headline text for conservative matching.
# Lowercase and standardize whitespace only.
def normalize_headline(value):
    if pd.isna(value):
        return None

    return " ".join(str(value).lower().split())


welfake["headline_key"] = welfake["headline"].apply(normalize_headline)
fakenewsnet["headline_key"] = fakenewsnet["headline"].apply(normalize_headline)

welfake = welfake.dropna(subset=["headline_key", "real"])
fakenewsnet = fakenewsnet.dropna(subset=["headline_key", "real"])

print("\n1. VALID HEADLINES")
print("-" * 55)
print(f"WELFake records: {len(welfake)}")
print(f"FakeNewsNet records: {len(fakenewsnet)}")
print(f"WELFake unique headlines: {welfake['headline_key'].nunique()}")
print(f"FakeNewsNet unique headlines: {fakenewsnet['headline_key'].nunique()}")

# Check repeated headline groups within each dataset.
def inspect_duplicates(data, dataset_name):
    label_counts = data.groupby("headline_key")["real"].nunique()

    duplicate_groups = label_counts[label_counts > 1]
    repeated_groups = label_counts[label_counts == 1]

    print(f"\n{dataset_name}")
    print(f"Repeated headline groups: {(data['headline_key'].value_counts() > 1).sum()}")
    print(f"Groups with conflicting labels: {len(duplicate_groups)}")
    print(f"Groups with consistent labels: {len(repeated_groups)}")


print("\n2. DUPLICATES WITHIN EACH DATASET")
print("-" * 55)
inspect_duplicates(welfake, "WELFake")
inspect_duplicates(fakenewsnet, "FakeNewsNet")

# Compare unique headline-label combinations.
w_pairs = welfake[["headline_key", "real"]].drop_duplicates()
f_pairs = fakenewsnet[["headline_key", "real"]].drop_duplicates()

overlap = w_pairs.merge(
    f_pairs,
    on="headline_key",
    how="inner",
    suffixes=("_welfake", "_fakenewsnet")
)

shared_headlines = overlap["headline_key"].nunique()

conflicts = overlap[
    overlap["real_welfake"] != overlap["real_fakenewsnet"]
]

conflicting_headlines = conflicts["headline_key"].nunique()

print("\n3. CROSS-DATASET OVERLAP")
print("-" * 55)
print(f"Shared unique headlines: {shared_headlines}")
print(f"Shared headlines with matching labels: {shared_headlines - conflicting_headlines}")
print(f"Shared headlines with conflicting labels: {conflicting_headlines}")

print("\n4. EXAMPLES OF CROSS-DATASET LABEL CONFLICTS")
print("-" * 55)

if conflicting_headlines > 0:
    examples = conflicts.drop_duplicates("headline_key").head(15)

    for _, row in examples.iterrows():
        print(f"\nHeadline: {row['headline_key']}")
        print(f"WELFake label: {row['real_welfake']}")
        print(f"FakeNewsNet label: {row['real_fakenewsnet']}")
else:
    print("No cross-dataset label conflicts found.")

# Count WELFake headlines not found in FakeNewsNet.
w_keys = set(w_pairs["headline_key"])
f_keys = set(f_pairs["headline_key"])

new_welfake_headlines = w_keys - f_keys

print("\n5. POTENTIALLY NEW WELFAKE HEADLINES")
print("-" * 55)
print(f"Unique WELFake headlines absent from FakeNewsNet: {len(new_welfake_headlines)}")

print("\nAUDIT COMPLETED")
print("Neither original dataset was modified.")