from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def run_comparison():
    train_path = Path("data/splits/train.csv")
    vectorizer_path = Path("outputs/models/final/topic_tfidf_vectorizer.joblib")
    output_dir = Path("outputs/clustering/k_comparison")

    output_dir.mkdir(parents=True, exist_ok=True)

    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")

    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Vectorizer not found: {vectorizer_path}")

    data = pd.read_csv(train_path).reset_index(drop=True)

    if "title" not in data.columns:
        raise ValueError("Training data must contain a 'title' column.")

    data["title"] = data["title"].fillna("").astype(str)

    vectorizer = joblib.load(vectorizer_path)
    all_features = vectorizer.transform(data["title"])

    zero_mask = np.asarray(all_features.getnnz(axis=1) == 0).ravel()
    usable_positions = np.flatnonzero(~zero_mask)

    if len(usable_positions) == 0:
        raise ValueError("No headlines have usable TF-IDF features.")

    usable_features = all_features[usable_positions]

    sample_size = min(10000, len(usable_positions))

    sample_positions = (
        pd.Series(usable_positions)
        .sample(n=sample_size, random_state=42)
        .to_numpy()
    )

    sample_features = all_features[sample_positions]
    feature_names = vectorizer.get_feature_names_out()

    print("\nK-MEANS CLUSTER COUNT COMPARISON")
    print("=" * 70)
    print(f"Training headlines: {len(data)}")
    print(f"Usable headlines: {len(usable_positions)}")
    print(f"Zero-feature headlines: {int(zero_mask.sum())}")
    print(f"Same fitting sample for every K: {sample_size}")
    print("K values: 8, 10, 12")
    print("Random state: 42")

    all_summaries = []

    for k in [8, 10, 12]:
        print(f"\n\n{'=' * 70}")
        print(f"EXPERIMENT: K = {k}")
        print("=" * 70)

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
            max_iter=100
        )

        model.fit(sample_features)

        usable_labels = model.predict(usable_features)

        assignments = np.full(len(data), -1, dtype=int)
        assignments[usable_positions] = usable_labels

        data_with_clusters = data.copy()
        data_with_clusters["cluster"] = assignments

        assignment_path = output_dir / f"cluster_assignments_k{k}.csv"
        data_with_clusters.to_csv(assignment_path, index=False)

        distances = model.transform(usable_features)
        cluster_counts = np.bincount(usable_labels, minlength=k)

        print(f"Fitting sample: {sample_size}")
        print(f"Inertia: {model.inertia_:.2f}")
        print("\nCLUSTER DISTRIBUTION")
        print("-" * 60)

        for cluster_id in range(k):
            count = int(cluster_counts[cluster_id])
            percentage = (count / len(data)) * 100

            print(
                f"Cluster {cluster_id}: "
                f"{count} headlines ({percentage:.2f}%)"
            )

        print(
            f"Unclustered: {int(zero_mask.sum())} "
            f"headlines ({zero_mask.mean() * 100:.2f}%)"
        )

        cluster_summaries = []

        for cluster_id in range(k):
            center = model.cluster_centers_[cluster_id]
            top_indices = np.argsort(center)[-10:][::-1]
            top_words = feature_names[top_indices].tolist()

            member_positions = np.flatnonzero(usable_labels == cluster_id)

            if len(member_positions) > 0:
                nearest_order = np.argsort(
                    distances[member_positions, cluster_id]
                )[:5]

                representative_headlines = (
                    data.iloc[
                        usable_positions[
                            member_positions[nearest_order]
                        ]
                    ]["title"].tolist()
                )
            else:
                representative_headlines = []

            count = int(cluster_counts[cluster_id])
            percentage = (count / len(data)) * 100

            summary = {
                "k": k,
                "cluster": cluster_id,
                "headline_count": count,
                "percentage_of_training_data": round(percentage, 2),
                "top_words": ", ".join(top_words),
                "representative_headlines": " || ".join(
                    representative_headlines
                )
            }

            cluster_summaries.append(summary)
            all_summaries.append(summary)

            print(f"\nCluster {cluster_id}")
            print(f"Headlines: {count} ({percentage:.2f}%)")
            print(f"Top words: {', '.join(top_words)}")

            for headline in representative_headlines:
                print(f"  - {headline}")

        summary_path = output_dir / f"cluster_summary_k{k}.csv"
        pd.DataFrame(cluster_summaries).to_csv(
            summary_path,
            index=False
        )

        model_path = output_dir / f"topic_kmeans_k{k}.joblib"
        joblib.dump(model, model_path)

    comparison_path = output_dir / "all_k_cluster_summaries.csv"

    pd.DataFrame(all_summaries).to_csv(
        comparison_path,
        index=False
    )

    print("\n\nCOMPARISON COMPLETED")
    print("=" * 70)
    print(f"Results folder: {output_dir}")
    print(f"Combined summaries: {comparison_path}")
    print("Original K=8 clustering files were not changed.")


run_comparison()