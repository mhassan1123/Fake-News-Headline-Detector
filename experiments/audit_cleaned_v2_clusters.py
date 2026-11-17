from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "splits_cleaned_v2" / "train.csv"
ASSIGNMENTS_PATH = ROOT / "outputs" / "cleaned_v2" / "clustering" / "cluster_assignments.csv"

OUTPUT_DIR = ROOT / "outputs" / "cleaned_v2" / "clustering" / "audit"


def find_column(data, possible_names):
    columns = {column.lower(): column for column in data.columns}

    for name in possible_names:
        if name.lower() in columns:
            return columns[name.lower()]

    return None


def normalize_headline(value):
    return " ".join(str(value).casefold().split())


def load_audit_data():
    assignments = pd.read_csv(ASSIGNMENTS_PATH)
    training = pd.read_csv(TRAIN_PATH)

    assignment_title = find_column(
        assignments, ["title", "headline", "text"]
    )
    assignment_cluster = find_column(
        assignments, ["cluster", "cluster_id", "cluster_label"]
    )

    train_title = find_column(
        training, ["title", "headline", "text"]
    )
    train_source = find_column(
        training, ["source_dataset", "source", "dataset"]
    )
    train_label = find_column(
        training, ["label", "real", "target"]
    )

    if assignment_title is None or assignment_cluster is None:
        raise ValueError(
            "Could not identify headline and cluster columns in "
            f"{ASSIGNMENTS_PATH}. Found columns: {list(assignments.columns)}"
        )

    if train_title is None:
        raise ValueError(
            f"Could not identify the headline column in {TRAIN_PATH}."
        )

    assignments = assignments.copy()
    assignments["_audit_key"] = assignments[assignment_title].map(
        normalize_headline
    )

    training = training.copy()
    training["_audit_key"] = training[train_title].map(
        normalize_headline
    )

    # Use metadata already present in the assignments file when available.
    assignment_source = find_column(
        assignments, ["source_dataset", "source", "dataset"]
    )
    assignment_label = find_column(
        assignments, ["label", "real", "target"]
    )

    required_metadata = {}

    if assignment_source is not None:
        assignments["source_dataset"] = assignments[assignment_source]
    elif train_source is not None:
        required_metadata["source_dataset"] = train_source

    if assignment_label is not None:
        assignments["label"] = assignments[assignment_label]
    elif train_label is not None:
        required_metadata["label"] = train_label

    if required_metadata:
        metadata_columns = ["_audit_key"] + list(required_metadata.values())
        metadata = training[metadata_columns].copy()

        metadata = metadata.rename(
            columns={
                original: new
                for new, original in required_metadata.items()
            }
        )

        if metadata["_audit_key"].duplicated().any():
            raise ValueError(
                "Training headlines are not unique after normalization. "
                "Cannot safely attach source/label metadata."
            )

        assignments = assignments.merge(
            metadata,
            on="_audit_key",
            how="left",
            validate="one_to_one"
        )

    if "source_dataset" not in assignments.columns:
        raise ValueError(
            "Source information is missing from both CSV files."
        )

    if "label" not in assignments.columns:
        raise ValueError(
            "Label information is missing from both CSV files."
        )

    if assignments["source_dataset"].isna().any():
        raise ValueError(
            "Some cluster assignments could not be matched to their source."
        )

    if assignments["label"].isna().any():
        raise ValueError(
            "Some cluster assignments could not be matched to their label."
        )

    assignments = assignments.rename(
        columns={assignment_cluster: "cluster"}
    )

    return assignments


def run_audit():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_audit_data()

    data["label"] = pd.to_numeric(data["label"], errors="coerce")

    if data["label"].isna().any():
        raise ValueError("Some labels are not numeric.")

    if not data["label"].isin([0, 1]).all():
        raise ValueError("Unexpected labels found. Expected 0 and 1.")

    data["label_name"] = data["label"].map({
        0: "Fake",
        1: "Real"
    })

    data["cluster"] = data["cluster"].astype(str)

    # Count and percentage of each source within each cluster.
    source_counts = pd.crosstab(
        data["cluster"],
        data["source_dataset"]
    )

    source_percentages = source_counts.div(
        source_counts.sum(axis=1), axis=0
    ) * 100

    # Count and percentage of Fake/Real labels within each cluster.
    label_counts = pd.crosstab(
        data["cluster"],
        data["label_name"]
    )

    label_counts = label_counts.reindex(
        columns=["Fake", "Real"], fill_value=0
    )

    label_percentages = label_counts.div(
        label_counts.sum(axis=1), axis=0
    ) * 100

    # Combine the main cluster statistics.
    summary = pd.DataFrame({
        "cluster": source_counts.index,
        "cluster_size": source_counts.sum(axis=1).values,
        "fake_count": label_counts["Fake"].values,
        "real_count": label_counts["Real"].values,
        "fake_percentage": label_percentages["Fake"].values,
        "real_percentage": label_percentages["Real"].values
    })

    summary["cluster_percentage"] = (
        summary["cluster_size"] / len(data) * 100
    )

    summary = summary.sort_values(
        "cluster_size", ascending=False
    )

    # Save all audit tables separately.
    source_counts.to_csv(OUTPUT_DIR / "cluster_source_counts.csv")
    source_percentages.to_csv(
        OUTPUT_DIR / "cluster_source_percentages.csv"
    )

    label_counts.to_csv(OUTPUT_DIR / "cluster_label_counts.csv")
    label_percentages.to_csv(
        OUTPUT_DIR / "cluster_label_percentages.csv"
    )

    summary.to_csv(
        OUTPUT_DIR / "cluster_quality_summary.csv",
        index=False
    )

    print("\nCLEANED-V2 CLUSTER QUALITY AUDIT")
    print("=" * 55)
    print(f"Total assignments: {len(data)}")
    print(f"Number of clusters, including -1: {data['cluster'].nunique()}")
    print(f"Missing source values: {data['source_dataset'].isna().sum()}")
    print(f"Missing labels: {data['label'].isna().sum()}")

    print("\nCLUSTER SIZE AND LABEL DISTRIBUTION")
    print("-" * 55)
    print(summary.round(2).to_string(index=False))

    print("\nSOURCE DISTRIBUTION (%)")
    print("-" * 55)
    print(source_percentages.round(2).to_string())

    print("\nLABEL DISTRIBUTION (%)")
    print("-" * 55)
    print(label_percentages.round(2).to_string())

    print("\nAudit files saved in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    run_audit()