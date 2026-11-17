import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter
from pathlib import Path

data_path = Path("data/processed/merged_headlines.csv")
figure_path = Path("outputs/figures")
result_path = Path("outputs/results")

figure_path.mkdir(parents=True, exist_ok=True)
result_path.mkdir(parents=True, exist_ok=True)

print("MERGED DATASET EDA")
print("=" * 65)

# 1. Load dataset
df = pd.read_csv(data_path)

required_columns = ["title", "real", "source_dataset"]

if not all(column in df.columns for column in required_columns):
    raise ValueError("Required columns are missing.")

if df[required_columns].isna().any().any():
    raise ValueError("Missing values found in the dataset.")

if not df["real"].isin([0, 1]).all():
    raise ValueError("Invalid class labels found.")

df["word_count"] = df["title"].astype(str).apply(
    lambda title: len(title.split())
)

df["normalized_title"] = (
    df["title"]
    .astype(str)
    .str.lower()
    .str.split()
    .str.join(" ")
)

print("\n1. DATASET OVERVIEW")
print("-" * 65)
print(f"Total headlines: {len(df)}")
print(f"Fake headlines: {(df['real'] == 0).sum()}")
print(f"Real headlines: {(df['real'] == 1).sum()}")
print(f"Fake percentage: {(df['real'] == 0).mean() * 100:.2f}%")
print(f"Real percentage: {(df['real'] == 1).mean() * 100:.2f}%")

print("\n2. DATA QUALITY")
print("-" * 65)
print(f"Missing values:\n{df[required_columns].isna().sum().to_string()}")
print(f"Exact duplicate titles: {df['title'].duplicated().sum()}")
print(f"Normalized duplicate titles: {df['normalized_title'].duplicated().sum()}")
print(f"Conflicting normalized titles: {df.groupby('normalized_title')['real'].nunique().gt(1).sum()}")
print(f"Invalid labels: {(~df['real'].isin([0, 1])).sum()}")

if df["normalized_title"].duplicated().any():
    raise ValueError("Normalized duplicate headlines remain.")

# 3. Headline length analysis
print("\n3. HEADLINE LENGTH")
print("-" * 65)

length_summary = df.groupby("real")["word_count"].agg(
    ["count", "mean", "median", "min", "max"]
)

length_summary.index = ["Fake", "Real"]
print(length_summary.round(2).to_string())

df["class_name"] = df["real"].map({
    0: "Fake",
    1: "Real"
})

# 4. Source distribution
print("\n4. SOURCE DISTRIBUTION")
print("-" * 65)

print(df["source_dataset"].value_counts().to_string())

# 5. Class distribution by source
print("\n5. CLASS DISTRIBUTION BY SOURCE")
print("-" * 65)

source_class = pd.crosstab(
    df["source_dataset"],
    df["class_name"],
    normalize="index"
) * 100

print(source_class.round(2).to_string())

# 6. Word frequency analysis
stop_words = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in",
    "on", "at", "for", "from", "with", "by", "as", "is",
    "are", "was", "were", "be", "been", "being", "this",
    "that", "these", "those", "it", "its", "he", "she",
    "they", "them", "his", "her", "their", "you", "your",
    "we", "our", "i", "my", "me", "have", "has", "had",
    "will", "would", "can", "could", "should", "do", "does",
    "did", "what", "who", "when", "where", "why", "how",
    "after", "before", "about", "into", "over", "new"
}

def get_top_words(headlines, limit=15):
    words = []

    for headline in headlines:
        tokens = re.findall(r"[a-zA-Z']+", str(headline).lower())

        for word in tokens:
            if word not in stop_words and len(word) > 1:
                words.append(word)

    return Counter(words).most_common(limit)


fake_words = get_top_words(
    df.loc[df["real"] == 0, "title"]
)

real_words = get_top_words(
    df.loc[df["real"] == 1, "title"]
)

print("\n6. TOP FAKE HEADLINE WORDS")
print("-" * 65)
print(pd.DataFrame(fake_words, columns=["Word", "Frequency"]).to_string(index=False))

print("\n7. TOP REAL HEADLINE WORDS")
print("-" * 65)
print(pd.DataFrame(real_words, columns=["Word", "Frequency"]).to_string(index=False))

# 8. Chart: class distribution
class_counts = df["class_name"].value_counts().reindex(["Fake", "Real"])

plt.figure(figsize=(8, 5))
class_counts.plot(kind="bar")
plt.title("Fake vs Real Headline Distribution")
plt.xlabel("Headline Class")
plt.ylabel("Number of Headlines")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(figure_path / "merged_label_distribution.png")
plt.close()

# 9. Chart: headline length
plt.figure(figsize=(9, 5))

for label in ["Fake", "Real"]:
    df.loc[df["class_name"] == label, "word_count"].hist(
        bins=25,
        alpha=0.5,
        label=label
    )

plt.title("Headline Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Number of Headlines")
plt.legend()
plt.tight_layout()
plt.savefig(figure_path / "merged_headline_length.png")
plt.close()

# 10. Chart: most frequent words
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for axis, words, label in [
    (axes[0], fake_words, "Fake"),
    (axes[1], real_words, "Real")
]:
    word_df = pd.DataFrame(words, columns=["Word", "Frequency"])
    axis.barh(word_df["Word"], word_df["Frequency"])
    axis.set_title(f"Top Words in {label} Headlines")
    axis.set_xlabel("Frequency")
    axis.invert_yaxis()

plt.tight_layout()
plt.savefig(figure_path / "merged_top_words.png")
plt.close()

# 11. Chart: source distribution
plt.figure(figsize=(9, 5))
df["source_dataset"].value_counts().plot(kind="bar")
plt.title("Headline Distribution by Source Dataset")
plt.xlabel("Source Dataset")
plt.ylabel("Number of Headlines")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(figure_path / "merged_source_distribution.png")
plt.close()

# 12. Chart: class distribution by source
source_class.plot(
    kind="bar",
    stacked=True,
    figsize=(9, 5)
)

plt.title("Fake and Real Distribution by Source")
plt.xlabel("Source Dataset")
plt.ylabel("Percentage")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(figure_path / "merged_source_class_distribution.png")
plt.close()

# 13. Save summaries
length_summary.to_csv(result_path / "merged_headline_length_summary.csv")
source_class.to_csv(result_path / "merged_source_class_distribution.csv")

print("\n8. GENERATED CHARTS")
print("-" * 65)
print("merged_label_distribution.png")
print("merged_headline_length.png")
print("merged_top_words.png")
print("merged_source_distribution.png")
print("merged_source_class_distribution.png")

print("\nEDA COMPLETED")