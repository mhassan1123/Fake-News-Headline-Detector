import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.sparse import load_npz

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# Load the same test data used by both models
X_test = load_npz("outputs/features/X_test.npz")
y_test = np.load("outputs/features/y_test.npy")


# Load both trained models
naive_bayes = joblib.load(
    "outputs/models/naive_bayes_model.joblib"
)

logistic_regression = joblib.load(
    "outputs/models/logistic_regression_model.joblib"
)


# Generate predictions
nb_predictions = naive_bayes.predict(X_test)
lr_predictions = logistic_regression.predict(X_test)


# Calculate evaluation metrics
def evaluate_model(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro Precision": precision_score(
            y_true, y_pred, average="macro"
        ),
        "Macro Recall": recall_score(
            y_true, y_pred, average="macro"
        ),
        "Macro F1": f1_score(
            y_true, y_pred, average="macro"
        ),
        "Fake Recall": recall_score(
            y_true, y_pred, pos_label=0
        )
    }


nb_results = evaluate_model(y_test, nb_predictions)
lr_results = evaluate_model(y_test, lr_predictions)


# Create comparison table
comparison = pd.DataFrame(
    [nb_results, lr_results],
    index=["Naive Bayes", "Logistic Regression"]
)

print("\nBASELINE MODEL COMPARISON")
print("-------------------------")
print(comparison.round(4).to_string())


# Save comparison results
output_path = Path("outputs/results")
output_path.mkdir(parents=True, exist_ok=True)

comparison.to_csv(
    output_path / "model_comparison.csv"
)


# Create comparison chart
ax = comparison.plot(
    kind="bar",
    figsize=(11, 6),
    ylim=(0, 1),
    width=0.75
)

ax.set_title("Baseline Model Performance Comparison")
ax.set_xlabel("Machine Learning Models")
ax.set_ylabel("Score")
ax.set_xticklabels(
    comparison.index,
    rotation=0
)

ax.legend(
    title="Evaluation Metrics",
    loc="lower right"
)

ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(
    "outputs/figures/model_comparison.png",
    dpi=300
)
plt.close()

print("\nComparison table saved in outputs/results/")
print("Comparison chart saved in outputs/figures/")