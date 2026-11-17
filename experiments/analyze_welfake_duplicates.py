import pandas as pd

welfake_path = "data/raw/WELFake_Dataset.csv"
fakenewsnet_path = "data/processed/cleaned_fakenewsnet.csv"

print("WELFAKE CLEANING ANALYSIS\n" + "=" * 55)

welfake = pd.read_csv(welfake_path, low_memory=False)
fakenewsnet = pd.read_csv(fakenewsnet_path)

# Convert WELFake labels to our project convention:
# Project: 0 = Fake, 1 = Real
# Downloaded WELFake: 0 = Real, 1 = Fake
welfake["real"] = welfake["label"].map({
    0: 1,
    1: 0
})

# Normalize headline text for comparison.
def normalize_headline(value):
    if pd.isna(value):
        return None

    return " ".join(str(value).lower().split())


welfake["headline_key"] = welfake["title"].apply(normalize_headline)
fakenewsnet["headline_key"] = fakenewsnet["title"].apply(normalize_headline)

# Remove records without usable headlines or labels.
valid = welfake.dropna(subset=["headline_key", "real"]).copy()

print("\n1. INITIAL COUNTS")
print("-" * 55)
print(f"Original WELFake rows: {len(welfake)}")
print(f"Rows with valid headlines: {len(valid)}")
print(f"Rows removed due to missing headlines: {len(welfake) - len(valid)}")

# Find headline groups containing conflicting labels.
label_counts = valid.groupby("headline_key")["real"].nunique()
conflict_keys = set(label_counts[label_counts > 1].index)

conflict_rows = valid[
    valid["headline_key"].isin(conflict_keys)
]

print("\n2. CONFLICTING HEADLINE GROUPS")
print("-" * 55)
print(f"Conflicting groups: {len(conflict_keys)}")
print(f"Records belonging to conflicting groups: {len(conflict_rows)}")

if len(conflict_keys) > 0:
    for key in sorted(conflict_keys):
        group = conflict_rows[
            conflict_rows["headline_key"] == key
        ]

        print(f"\nHeadline: {key}")
        print(f"Number of records: {len(group)}")
        print(f"Project labels: {sorted(group['real'].unique().tolist())}")
        print(f"Original WELFake labels: {sorted(group['label'].unique().tolist())}")

# Remove all records from conflicting groups.
consistent = valid[
    ~valid["headline_key"].isin(conflict_keys)
].copy()

# Keep one record per consistent headline.
unique_welfake = consistent.drop_duplicates(
    subset=["headline_key"],
    keep="first"
).copy()

print("\n3. AFTER INTERNAL DUPLICATE CLEANING")
print("-" * 55)
print(f"Records after removing conflicts: {len(consistent)}")
print(f"Unique WELFake headlines: {len(unique_welfake)}")
print(f"Repeated records removed: {len(consistent) - len(unique_welfake)}")

print("\nLabel distribution:")
print(
    unique_welfake["real"]
    .value_counts()
    .sort_index()
    .rename(index={0: "Fake", 1: "Real"})
    .to_string()
)

# Find WELFake headlines already present in FakeNewsNet.
fakenewsnet_keys = set(
    fakenewsnet["headline_key"].dropna()
)

overlapping = unique_welfake[
    unique_welfake["headline_key"].isin(fakenewsnet_keys)
]

print("\n4. CROSS-DATASET DUPLICATES")
print("-" * 55)
print(f"Shared headlines: {len(overlapping)}")

# Remove shared headlines from WELFake.
new_welfake = unique_welfake[
    ~unique_welfake["headline_key"].isin(fakenewsnet_keys)
].copy()

print("\n5. FINAL WELFAKE CONTRIBUTION")
print("-" * 55)
print(f"New unique WELFake headlines: {len(new_welfake)}")

print("\nLabel distribution:")
print(
    new_welfake["real"]
    .value_counts()
    .sort_index()
    .rename(index={0: "Fake", 1: "Real"})
    .to_string()
)

print("\n6. SUMMARY")
print("-" * 55)
print(f"FakeNewsNet existing records: {len(fakenewsnet)}")
print(f"New WELFake records: {len(new_welfake)}")
print(f"Potential combined records: {len(fakenewsnet) + len(new_welfake)}")

print("\nANALYSIS COMPLETED")
print("No files were modified or created.")