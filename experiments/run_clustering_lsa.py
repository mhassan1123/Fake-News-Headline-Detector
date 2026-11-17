from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer


ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "splits_cleaned_v2" / "train.csv"
OUTPUT_DIR = ROOT / "outputs" / "cleaned_v2" / "clustering_lsa"

RANDOM_STATE = 42
N_CLUSTERS = 12
SAMPLE_SIZE = 10000
N_COMPONENTS = 100
TOP_WORDS = 10
EXAMPLES_PER_CLUSTER = 5


def run_clustering_lsa():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLSA + NORMALIZED K-MEANS CLUSTERING")
    print("=" * 60)

    # Load the same cleaned-v2 training data.
    data = pd.read_csv(TRAIN_PATH)

    label_column = next(
        (column for column in ["label", "real", "target"]
            if column in data.columns),
        None
    )

    required_columns = ["title", "source_dataset"]

    for column in required_columns:
        if column not in data.columns:
            raise ValueError(f"Missing required column: {column}")

    if label_column is None:
        raise ValueError(
            f"Could not find label column. Available columns: {list(data.columns)}"
        )

    # Standardize the label column name for the rest of the script.
    if label_column != "label":
        data = data.rename(columns={label_column: "label"})

    headlines = data["title"].fillna("").astype(str).tolist()

    print(f"Training headlines: {len(headlines)}")

    # Build the same clustering-specific TF-IDF representation.
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.5,
        sublinear_tf=True
    )

    X = vectorizer.fit_transform(headlines)

    usable_mask = np.asarray(X.getnnz(axis=1)).ravel() > 0
    usable_indices = np.flatnonzero(usable_mask)

    X_usable = X[usable_mask]

    print(f"TF-IDF features: {X.shape[1]}")
    print(f"Usable headlines: {X_usable.shape[0]}")
    print(f"Zero-feature headlines: {(~usable_mask).sum()}")

    if len(usable_indices) < N_COMPONENTS:
        raise ValueError(
            "Not enough usable headlines for the requested SVD components."
        )

    # Reduce sparse TF-IDF features to 100 components.
    print("\nREDUCING FEATURES WITH TRUNCATED SVD")
    print("-" * 60)

    svd = TruncatedSVD(
        n_components=N_COMPONENTS,
        n_iter=7,
        random_state=RANDOM_STATE
    )

    X_reduced = svd.fit_transform(X_usable)

    # Normalize the reduced vectors.
    normalizer = Normalizer(norm="l2")
    X_normalized = normalizer.fit_transform(X_reduced)

    explained_variance = svd.explained_variance_ratio_.sum()

    print(f"Original features: {X.shape[1]}")
    print(f"Reduced components: {N_COMPONENTS}")
    print(f"Explained variance ratio sum: {explained_variance:.4f}")

    # Select a reproducible sample for K-means fitting.
    rng = np.random.RandomState(RANDOM_STATE)

    sample_size = min(SAMPLE_SIZE, len(X_normalized))

    sample_indices = rng.choice(
        len(X_normalized),
        size=sample_size,
        replace=False
    )

    X_sample = X_normalized[sample_indices]

    print("\nFITTING K-MEANS")
    print("-" * 60)
    print(f"Headlines used for fitting: {sample_size}")
    print(f"Number of clusters: {N_CLUSTERS}")

    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=10
    )

    kmeans.fit(X_sample)

    # Assign every usable headline to its nearest cluster.
    usable_clusters = kmeans.predict(X_normalized)

    # Preserve zero-feature headlines as unclustered (-1).
    all_clusters = np.full(len(data), -1, dtype=int)
    all_clusters[usable_mask] = usable_clusters

    data["cluster"] = all_clusters

    # Save all headline assignments, including source and label.
    assignments = data[
        ["title", "label", "source_dataset", "cluster"]
    ].copy()

    assignments.to_csv(
        OUTPUT_DIR / "cluster_assignments_lsa.csv",
        index=False
    )

    # Find the most important original TF-IDF words in each cluster.
    feature_names = vectorizer.get_feature_names_out()

    summary_rows = []

    for cluster_id in range(N_CLUSTERS):
        cluster_mask = usable_clusters == cluster_id

        cluster_positions = np.flatnonzero(cluster_mask)
        cluster_size = len(cluster_positions)

        if cluster_size == 0:
            continue

        cluster_matrix = X_usable[cluster_mask]

        mean_scores = np.asarray(
            cluster_matrix.mean(axis=0)
        ).ravel()

        top_indices = np.argsort(mean_scores)[-TOP_WORDS:][::-1]

        top_words = [
            feature_names[index]
            for index in top_indices
        ]

        # Select representative headlines nearest the cluster center.
        distances = np.linalg.norm(
            X_normalized[cluster_positions]
            - kmeans.cluster_centers_[cluster_id],
            axis=1
        )

        closest_positions = np.argsort(distances)[
            :EXAMPLES_PER_CLUSTER
        ]

        representative_headlines = [
            headlines[usable_indices[cluster_positions[position]]]
            for position in closest_positions
        ]

        summary_rows.append({
            "cluster": cluster_id,
            "size": cluster_size,
            "percentage": cluster_size / len(data) * 100,
            "top_words": ", ".join(top_words),
            "representative_headlines": " | ".join(
                representative_headlines
            )
        })

    summary = pd.DataFrame(summary_rows).sort_values(
        "size", ascending=False
    )

    summary.to_csv(
        OUTPUT_DIR / "cluster_summary_lsa.csv",
        index=False
    )

    # Post-clustering source and label audits.
    source_counts = pd.crosstab(
        data["cluster"],
        data["source_dataset"]
    )

    source_percentages = source_counts.div(
        source_counts.sum(axis=1), axis=0
    ) * 100

    label_names = data["label"].map({
        0: "Fake",
        1: "Real"
    })

    if label_names.isna().any():
        raise ValueError("Unexpected or missing labels found.")

    label_counts = pd.crosstab(
        data["cluster"],
        label_names
    ).reindex(columns=["Fake", "Real"], fill_value=0)

    label_percentages = label_counts.div(
        label_counts.sum(axis=1), axis=0
    ) * 100

    source_counts.to_csv(
        OUTPUT_DIR / "cluster_source_counts_lsa.csv"
    )

    source_percentages.to_csv(
        OUTPUT_DIR / "cluster_source_percentages_lsa.csv"
    )

    label_counts.to_csv(
        OUTPUT_DIR / "cluster_label_counts_lsa.csv"
    )

    label_percentages.to_csv(
        OUTPUT_DIR / "cluster_label_percentages_lsa.csv"
    )

    # Save fitted components separately.
    joblib.dump(
        vectorizer,
        OUTPUT_DIR / "topic_tfidf_vectorizer_lsa.joblib"
    )

    joblib.dump(
        svd,
        OUTPUT_DIR / "topic_svd.joblib"
    )

    joblib.dump(
        normalizer,
        OUTPUT_DIR / "topic_normalizer.joblib"
    )

    joblib.dump(
        kmeans,
        OUTPUT_DIR / "topic_kmeans_lsa.joblib"
    )

    print("\nCLUSTERING RESULTS")
    print("=" * 60)
    print(f"Total assignments: {len(data)}")
    print(f"Number of clusters: {N_CLUSTERS}")
    print(f"Unclustered headlines: {(~usable_mask).sum()}")

    print("\nCLUSTER SUMMARY")
    print("-" * 60)
    print(summary.round(2).to_string(index=False))

    print("\nSOURCE DISTRIBUTION (%)")
    print("-" * 60)
    print(source_percentages.round(2).to_string())

    print("\nLABEL DISTRIBUTION (%)")
    print("-" * 60)
    print(label_percentages.round(2).to_string())

    print("\nAll LSA clustering outputs saved in:")
    print(OUTPUT_DIR)


run_clustering_lsa()