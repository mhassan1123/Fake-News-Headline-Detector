import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Load training and validation features
feature_dir = Path("outputs/validation_features")

X_train = load_npz(feature_dir / "X_train.npz")
X_val = load_npz(feature_dir / "X_val.npz")

y_train = np.load(feature_dir / "y_train.npy")
y_val = np.load(feature_dir / "y_val.npy")

# Create two models for a fair comparison
models = {
    "Standard Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    "Balanced Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    )
}

results = []

# Train and evaluate both models on the same data
for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)
    macro_precision = precision_score(
        y_val, predictions, average="macro", zero_division=0
    )
    macro_recall = recall_score(
        y_val, predictions, average="macro", zero_division=0
    )
    macro_f1 = f1_score(
        y_val, predictions, average="macro", zero_division=0
    )

    fake_precision, real_precision = precision_score(
        y_val, predictions, labels=[0, 1],
        average=None, zero_division=0
    )

    fake_recall, real_recall = recall_score(
        y_val, predictions, labels=[0, 1],
        average=None, zero_division=0
    )

    fake_f1, real_f1 = f1_score(
        y_val, predictions, labels=[0, 1],
        average=None, zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1": macro_f1,
        "Fake Precision": fake_precision,
        "Fake Recall": fake_recall,
        "Fake F1": fake_f1,
        "Real Precision": real_precision,
        "Real Recall": real_recall,
        "Real F1": real_f1
    })

    print(
        f"\n{name.upper()}\n"
        f"{'-' * len(name)}\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Macro Precision: {macro_precision:.4f}\n"
        f"Macro Recall: {macro_recall:.4f}\n"
        f"Macro F1: {macro_f1:.4f}\n"
        f"\nCONFUSION MATRIX\n"
        f"{confusion_matrix(y_val, predictions, labels=[0, 1])}\n"
        f"\nCLASSIFICATION REPORT\n"
        f"{classification_report(y_val, predictions, labels=[0, 1], target_names=['Fake', 'Real'], zero_division=0)}"
    )

    # Save both experiment models separately
    model_dir = Path("outputs/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    if name == "Standard Logistic Regression":
        filename = "logistic_regression_validation_standard.joblib"
    else:
        filename = "logistic_regression_balanced_validation.joblib"

    joblib.dump(model, model_dir / filename)

# Save comparison results
results_dir = Path("outputs/results")
results_dir.mkdir(parents=True, exist_ok=True)

results_df = pd.DataFrame(results)
results_df.to_csv(
    results_dir / "logistic_regression_imbalance_comparison.csv",
    index=False
)

print(
    "\nVALIDATION EXPERIMENT SUMMARY\n"
    "----------------------------\n"
    f"{results_df.to_string(index=False)}\n"
    f"\nSaved comparison in: {results_dir}"
)