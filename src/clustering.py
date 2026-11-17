from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data/splits_cleaned_v2/train.csv"

OUTPUT_DIR = ROOT / "outputs/cleaned_v2/clustering"

MODEL_PATH = OUTPUT_DIR / "topic_kmeans.joblib"
VECTORIZER_PATH = OUTPUT_DIR / "topic_tfidf_vectorizer.joblib"

N_CLUSTERS = 12
SAMPLE_SIZE = 10000
RANDOM_STATE = 42
TOP_WORDS = 10
EXAMPLES_PER_CLUSTER = 5


def load_clustering_data():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned-v2 training dataset not found: {TRAIN_PATH}"
        )

    data = pd.read_csv(TRAIN_PATH)

    if "title" not in data.columns:
        raise ValueError(
            "Training dataset must contain a title column."
        )

    data = data.dropna(subset=["title"]).copy()
    data["title"] = data["title"].astype(str).str.strip()
    data = data[data["title"] != ""]
    data = data.reset_index(drop=True)

    if len(data) < N_CLUSTERS:
        raise ValueError(
            "Not enough valid headlines to create the requested clusters."
        )

    return data


def fit_topic_model():
    data = load_clustering_data()

    # Fit a separate clustering vectorizer on training headlines only.
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.5,
        sublinear_tf=True
    )

    all_features = vectorizer.fit_transform(data["title"])

    if all_features.shape[1] == 0:
        raise ValueError(
            "No usable TF-IDF features were found."
        )

    # Identify headlines with at least one usable TF-IDF feature.
    row_feature_counts = np.asarray(
        all_features.getnnz(axis=1)
    ).ravel()

    valid_positions = np.flatnonzero(row_feature_counts > 0)

    if len(valid_positions) < N_CLUSTERS:
        raise ValueError(
            "Not enough headlines have usable TF-IDF features."
        )

    # Select a reproducible sample from usable headlines.
    sample_size = min(SAMPLE_SIZE, len(valid_positions))

    rng = np.random.RandomState(RANDOM_STATE)

    sample_positions = rng.choice(
        valid_positions,
        size=sample_size,
        replace=False
    )

    sample_features = all_features[sample_positions]

    # Fit K-means on the selected sample.
    model = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=10,
        max_iter=100
    )

    model.fit(sample_features)

    # Assign all usable headlines to clusters.
    # Zero-feature headlines remain -1.
    cluster_labels = np.full(len(data), -1, dtype=int)

    valid_features = all_features[valid_positions]

    cluster_labels[valid_positions] = model.predict(
        valid_features
    )

    data["cluster"] = cluster_labels

    # Save the model and its matching vectorizer.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "vectorizer": vectorizer,
            "n_clusters": N_CLUSTERS,
            "random_state": RANDOM_STATE,
            "sample_size": sample_size
        },
        MODEL_PATH
    )

    joblib.dump(
        vectorizer,
        VECTORIZER_PATH
    )

    return data, model, vectorizer, all_features, sample_size


def create_cluster_summary(data, model, vectorizer, all_features):
    feature_names = vectorizer.get_feature_names_out()

    summaries = []

    for cluster_id in range(N_CLUSTERS):
        center = model.cluster_centers_[cluster_id]

        # Get the highest-weight terms in this cluster.
        top_indices = np.argsort(center)[::-1][:TOP_WORDS]

        top_words = [
            feature_names[index]
            for index in top_indices
        ]

        # Find headlines assigned to this cluster.
        member_indices = np.flatnonzero(
            data["cluster"].to_numpy() == cluster_id
        )

        if len(member_indices) == 0:
            summaries.append({
                "cluster": cluster_id,
                "headline_count": 0,
                "top_words": ", ".join(top_words),
                "example_headlines": "No headlines assigned"
            })
            continue

        # Calculate distances for this cluster's members.
        member_features = all_features[member_indices]

        member_distances = model.transform(
            member_features
        )[:, cluster_id]

        # Select headlines closest to the cluster centre.
        nearest_order = np.argsort(member_distances)

        nearest_indices = member_indices[
            nearest_order[:EXAMPLES_PER_CLUSTER]
        ]

        examples = data.iloc[nearest_indices]["title"].tolist()

        summaries.append({
            "cluster": cluster_id,
            "headline_count": len(member_indices),
            "top_words": ", ".join(top_words),
            "example_headlines": " | ".join(examples)
        })

    return pd.DataFrame(summaries)


def audit_clustering_features(data, all_features):
    # Count headlines with no usable TF-IDF features.
    zero_mask = np.asarray(
        all_features.getnnz(axis=1) == 0
    ).ravel()

    zero_count = int(zero_mask.sum())
    total_count = len(data)
    usable_count = total_count - zero_count
    zero_percentage = (zero_count / total_count) * 100

    print(
        f"\nTF-IDF FEATURE QUALITY AUDIT\n"
        f"----------------------------\n"
        f"Total headlines: {total_count}\n"
        f"Headlines with zero features: {zero_count}\n"
        f"Headlines with usable features: {usable_count}\n"
        f"Zero-feature percentage: {zero_percentage:.2f}%"
    )

    if zero_count > 0:
        print("\nEXAMPLES WITH ZERO TF-IDF FEATURES")

        examples = data.loc[zero_mask, "title"].head(15)

        for headline in examples:
            print(f"- {headline}")

    print("\nZERO-FEATURE HEADLINES BY CLUSTER")
    print("--------------------------------")

    zero_data = data.loc[zero_mask]

    if len(zero_data) > 0:
        print(
            zero_data["cluster"]
            .value_counts()
            .sort_index()
            .to_string()
        )
    else:
        print("No zero-feature headlines found.")


def run_clustering():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data, model, vectorizer, all_features, sample_size = fit_topic_model()

    audit_clustering_features(data, all_features)

    summary = create_cluster_summary(
        data,
        model,
        vectorizer,
        all_features
    )

    # Save headline-to-cluster assignments.
    assignments = data[["title", "cluster"]].copy()

    if "source_dataset" in data.columns:
        assignments["source_dataset"] = data["source_dataset"]

    assignments.to_csv(
        OUTPUT_DIR / "cluster_assignments.csv",
        index=False
    )

    # Save cluster summaries.
    summary.to_csv(
        OUTPUT_DIR / "cluster_summary.csv",
        index=False
    )

    unclustered_count = int((data["cluster"] == -1).sum())

    print(
        f"\nCLEANED-V2 K-MEANS TOPIC CLUSTERING\n"
        f"==================================\n"
        f"Training headlines: {len(data)}\n"
        f"Headlines used to fit K-means: {sample_size}\n"
        f"Number of clusters: {N_CLUSTERS}\n"
        f"TF-IDF features: {all_features.shape[1]}\n"
        f"English stop-word removal: Enabled\n"
        f"Zero-feature headlines: Excluded from clustering\n"
        f"Representative examples: Nearest to cluster centre\n"
        f"Random state: {RANDOM_STATE}"
    )

    print("\nCLUSTER SUMMARY")
    print("---------------")

    for _, row in summary.iterrows():
        print(f"\nCluster {row['cluster']}")
        print(f"Headlines: {row['headline_count']}")
        print(f"Top words: {row['top_words']}")

        if row["headline_count"] == 0:
            print("No headlines assigned")
            continue

        for example in row["example_headlines"].split(" | "):
            print(f"  - {example}")

    print("\nCLUSTER DISTRIBUTION")
    print("--------------------")

    for _, row in summary.iterrows():
        percentage = row["headline_count"] / len(data) * 100

        print(
            f"Cluster {row['cluster']}: "
            f"{row['headline_count']} headlines ({percentage:.2f}%)"
        )

    unclustered_percentage = unclustered_count / len(data) * 100

    print(
        f"Unclustered: {unclustered_count} "
        f"headlines ({unclustered_percentage:.2f}%)"
    )

    print(
        f"\nSAVED RESULTS\n"
        f"-------------\n"
        f"{OUTPUT_DIR / 'cluster_assignments.csv'}\n"
        f"{OUTPUT_DIR / 'cluster_summary.csv'}\n"
        f"{MODEL_PATH}\n"
        f"{VECTORIZER_PATH}\n"
        f"\nCLEANED-V2 CLUSTERING COMPLETED"
    )