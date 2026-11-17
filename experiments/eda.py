import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter
from pathlib import Path


# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_fakenewsnet.csv")

# Create output folder
output_dir = Path("outputs/figures")
output_dir.mkdir(parents=True, exist_ok=True)

# Basic dataset information
print(
    f"DATASET OVERVIEW\n"
    f"----------------\n"
    f"Total headlines: {len(df)}\n"
    f"Fake headlines: {(df['real'] == 0).sum()}\n"
    f"Real headlines: {(df['real'] == 1).sum()}\n"
    f"Missing headlines: {df['title'].isnull().sum()}"
)

# Calculate headline word counts
df["word_count"] = df["title"].apply(lambda title: len(str(title).split()))

print(
    f"\nHEADLINE LENGTH\n"
    f"---------------\n"
    f"Average words: {df['word_count'].mean():.2f}\n"
    f"Shortest headline: {df['word_count'].min()} words\n"
    f"Longest headline: {df['word_count'].max()} words"
)

print("\nAVERAGE LENGTH BY CLASS")
print(df.groupby("real")["word_count"].mean().rename(index={0: "Fake", 1: "Real"}))


# Chart 1: Fake vs Real distribution
label_counts = df["real"].value_counts().reindex([0, 1])

plt.figure(figsize=(7, 5))
bars = plt.bar(["Fake", "Real"], label_counts.values)
plt.title("Distribution of Fake and Real Headlines")
plt.xlabel("Headline Class")
plt.ylabel("Number of Headlines")

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        str(int(bar.get_height())),
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(output_dir / "label_distribution.png", dpi=300)
plt.close()


# Chart 2: Headline length distribution
plt.figure(figsize=(8, 5))

plt.hist(
    df[df["real"] == 0]["word_count"],
    bins=30,
    alpha=0.6,
    label="Fake"
)

plt.hist(
    df[df["real"] == 1]["word_count"],
    bins=30,
    alpha=0.6,
    label="Real"
)

plt.title("Headline Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Number of Headlines")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "headline_length.png", dpi=300)
plt.close()


# Function to find common words
stop_words = {
    "the", "and", "for", "are", "was", "with", "that",
    "this", "from", "his", "her", "has", "have", "had",
    "will", "you", "your", "about", "after", "before",
    "what", "when", "where", "who", "why", "how"
}


def get_top_words(headlines):
    words = re.findall(r"\b[a-zA-Z]{3,}\b", " ".join(headlines).lower())
    words = [word for word in words if word not in stop_words]
    return Counter(words).most_common(10)


# Chart 3: Common words in Fake headlines
fake_words = get_top_words(df[df["real"] == 0]["title"].astype(str))

plt.figure(figsize=(9, 5))
plt.bar([word for word, count in fake_words],
        [count for word, count in fake_words])

plt.title("Top 10 Words in Fake Headlines")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_dir / "fake_top_words.png", dpi=300)
plt.close()


# Chart 4: Common words in Real headlines
real_words = get_top_words(df[df["real"] == 1]["title"].astype(str))

plt.figure(figsize=(9, 5))
plt.bar([word for word, count in real_words],
        [count for word, count in real_words])

plt.title("Top 10 Words in Real Headlines")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_dir / "real_top_words.png", dpi=300)
plt.close()


# Chart 5: News source distribution
if "source_domain" in df.columns:
    source_counts = df["source_domain"].fillna("Unknown").value_counts().head(10)

    plt.figure(figsize=(10, 5))
    plt.bar(source_counts.index, source_counts.values)
    plt.title("Top 10 News Sources")
    plt.xlabel("Source Domain")
    plt.ylabel("Number of Headlines")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "source_distribution.png", dpi=300)
    plt.close()


# Display most common words
print("\nTOP WORDS IN FAKE HEADLINES")
print(fake_words)

print("\nTOP WORDS IN REAL HEADLINES")
print(real_words)

print("\nEDA completed. Charts saved in outputs/figures/")