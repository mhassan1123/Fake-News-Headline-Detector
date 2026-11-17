import pandas as pd

input_path = "data/raw/WELFake_Dataset.csv"

df = pd.read_csv(input_path, low_memory=False)

print("WELFAKE LABEL INSPECTION\n" + "-" * 55)

for label in sorted(df["label"].dropna().unique()):
    group = df[
        (df["label"] == label) &
        (df["title"].notna())
    ]

    print(f"\nLABEL {label}")
    print("-" * 55)
    print(f"Available headlines: {len(group)}")

    samples = group.sample(
        n=min(15, len(group)),
        random_state=42
    )

    for index, (_, row) in enumerate(samples.iterrows(), start=1):
        title = str(row["title"]).replace("\n", " ").strip()
        text = str(row["text"]).replace("\n", " ").strip()

        print(f"\n{index}. HEADLINE: {title[:250]}")
        print(f"   ARTICLE: {text[:180]}")

print("\nINSPECTION COMPLETED")