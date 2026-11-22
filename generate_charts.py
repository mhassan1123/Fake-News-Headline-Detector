
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

MODEL_RESULTS_PATH = (
    ROOT
    / "outputs/cleaned_v2/reproducible_experiment/results/"
    / "model_comparison_metrics.csv"
)

CLUSTER_ASSIGNMENTS_PATH = (
    ROOT
    / "outputs/cleaned_v2/clustering/cluster_assignments.csv"
)

CHARTS_FOLDER = ROOT / "docs" / "charts"


# --------------------------------------------------
# 2. Find the final test confusion matrix
# --------------------------------------------------

def find_confusion_matrix():

    outputs_folder = ROOT / "outputs" / "cleaned_v2"

    files = list(
        outputs_folder.rglob("*confusion*matrix*.csv")
    )

    # Prefer a file whose name identifies both final and test.
    preferred = [
        file for file in files
        if "final" in file.name.lower()
        and "test" in file.name.lower()
    ]

    if len(preferred) == 1:
        return preferred[0]

    # Otherwise, look for a uniquely identifiable test matrix.
    test_files = [
        file for file in files
        if "test" in file.name.lower()
    ]

    if len(test_files) == 1:
        return test_files[0]

    if len(files) == 1:
        return files[0]

    if not files:
        raise FileNotFoundError(
            "Could not find a confusion matrix CSV file "
            "inside outputs/cleaned_v2."
        )

    raise FileNotFoundError(
        "Multiple confusion matrix files were found. "
        "Please identify the final test confusion matrix:\n"
        + "\n".join(str(file) for file in files)
    )


# --------------------------------------------------
# 3. Create output folder
# --------------------------------------------------

CHARTS_FOLDER.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# CHART 1: Model Performance Comparison
# --------------------------------------------------

def create_model_comparison_chart():

    print("\nCreating Model Performance Comparison...")

    if not MODEL_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Model results not found: {MODEL_RESULTS_PATH}"
        )

    data = pd.read_csv(MODEL_RESULTS_PATH)

    required_columns = {
        "model",
        "threshold",
        "accuracy",
        "macro_f1",
        "roc_auc"
    }

    if not required_columns.issubset(data.columns):
        raise ValueError(
            "Model comparison CSV is missing required columns."
        )

    # Compare both models at the same threshold: 0.500.
    data["threshold"] = pd.to_numeric(
        data["threshold"],
        errors="coerce"
    )

    data = data[
        (data["threshold"] - 0.500).abs() < 0.000001
    ].copy()

    data = data[
        data["model"].isin(
            ["naive_bayes", "logistic_regression"]
        )
    ]

    if len(data) != 2:
        raise ValueError(
            "Expected exactly two model results at threshold 0.500."
        )

    metrics = ["accuracy", "macro_f1", "roc_auc"]

    for metric in metrics:
        data[metric] = pd.to_numeric(
            data[metric],
            errors="raise"
        )

    models = data["model"].tolist()

    model_names = [
        "Naive Bayes" if name == "naive_bayes"
        else "Logistic Regression"
        for name in models
    ]

    x = range(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    values_1 = data.iloc[0][metrics].tolist()
    values_2 = data.iloc[1][metrics].tolist()

    bars_1 = ax.bar(
        [i - width / 2 for i in x],
        values_1,
        width,
        label=model_names[0]
    )

    bars_2 = ax.bar(
        [i + width / 2 for i in x],
        values_2,
        width,
        label=model_names[1]
    )

    ax.set_title(
        "Model Performance Comparison on Validation Data",
        fontsize=14
    )

    ax.set_ylabel("Score")
    ax.set_xlabel("Evaluation Metrics")

    ax.set_xticks(list(x))
    ax.set_xticklabels(
        ["Accuracy", "Macro F1", "ROC-AUC"]
    )

    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Display metric values above each bar.
    ax.bar_label(
        bars_1,
        fmt="%.3f",
        padding=3
    )

    ax.bar_label(
        bars_2,
        fmt="%.3f",
        padding=3
    )

    fig.tight_layout()

    output_path = (
        CHARTS_FOLDER / "model_comparison_validation.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


# --------------------------------------------------
# CHART 2: Final Test Confusion Matrix
# --------------------------------------------------

def create_confusion_matrix_chart():

    print("\nCreating Final Test Confusion Matrix...")

    confusion_path = find_confusion_matrix()

    print(f"Using confusion matrix: {confusion_path}")

    matrix = pd.read_csv(
        confusion_path,
        index_col=0
    )

    if matrix.shape != (2, 2):
        raise ValueError(
            "Expected a 2x2 confusion matrix."
        )

    matrix = matrix.apply(
        pd.to_numeric,
        errors="raise"
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    image = ax.imshow(matrix.values)

    fig.colorbar(
        image,
        ax=ax,
        label="Number of Headlines"
    )

    ax.set_title(
        "Final Test Confusion Matrix (Threshold = 0.477)",
        fontsize=14
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)

    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)

    # Display the actual number in every cell.
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):

            ax.text(
                col,
                row,
                f"{int(matrix.iloc[row, col]):,}",
                ha="center",
                va="center",
                fontsize=13
            )

    fig.tight_layout()

    output_path = (
        CHARTS_FOLDER / "final_test_confusion_matrix.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


# --------------------------------------------------
# CHART 3: Cluster Distribution
# --------------------------------------------------

def create_cluster_distribution_chart():

    print("\nCreating Cluster Distribution...")

    if not CLUSTER_ASSIGNMENTS_PATH.exists():
        raise FileNotFoundError(
            f"Cluster assignments not found: "
            f"{CLUSTER_ASSIGNMENTS_PATH}"
        )

    data = pd.read_csv(CLUSTER_ASSIGNMENTS_PATH)

    if "cluster" not in data.columns:
        raise ValueError(
            "The cluster assignments file has no cluster column."
        )

    # Count all headlines, including cluster -1.
    cluster_counts = (
        data.groupby("cluster")
        .size()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    bars = ax.bar(
        cluster_counts.index.astype(str),
        cluster_counts.values
    )

    ax.set_title(
        "Cluster Distribution of Training Headlines",
        fontsize=14
    )

    ax.set_xlabel("Cluster Number")
    ax.set_ylabel("Number of Headlines")

    ax.grid(axis="y", alpha=0.3)

    # Display headline counts above each bar.
    ax.bar_label(
        bars,
        fmt="%.0f",
        padding=3
    )

    total_headlines = int(cluster_counts.sum())

    ax.text(
        0.98,
        0.95,
        f"Total headlines: {total_headlines:,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11
    )

    fig.tight_layout()

    output_path = (
        CHARTS_FOLDER / "cluster_distribution.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {output_path}")

    print(
        f"Total headlines included: {total_headlines:,}"
    )


# --------------------------------------------------
# 4. Run all three charts
# --------------------------------------------------

if __name__ == "__main__":

    print("=" * 55)
    print("FAKE NEWS HEADLINE DETECTOR - CHART GENERATION")
    print("=" * 55)

    create_model_comparison_chart()

    create_confusion_matrix_chart()

    create_cluster_distribution_chart()

    print("\nAll three charts generated successfully!")

    print(f"\nChart location: {CHARTS_FOLDER}")
