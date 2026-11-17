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
    f1_score
)

# Load training and validation data
feature_dir = Path("outputs/validation_features")

X_train = load_npz(feature_dir / "X_train.npz")
X_val = load_npz(feature_dir / "X_val.npz")

y_train = np.load(feature_dir / "y_train.npy")
y_val = np.load(feature_dir / "y_val.npy")

# Parameter combinations
c_values = [0.1, 0.5, 1.0, 2.0, 5.0]
class_weights = [None, "balanced"]

results = []

model_dir = Path("outputs/models")
model_dir.mkdir(parents=True, exist_ok=True)

# Train and evaluate each configuration
for weight in class_weights:

    for c in c_values:

        model = LogisticRegression(
            C=c,
            max_iter=1000,
            random_state=42,
            class_weight=weight
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        accuracy = accuracy_score(y_val, predictions)

        fake_precision, real_precision = precision_score(
            y_val, predictions,
            labels=[0, 1],
            average=None,
            zero_division=0
        )

        fake_recall, real_recall = recall_score(
            y_val, predictions,
            labels=[0, 1],
            average=None,
            zero_division=0
        )

        fake_f1, real_f1 = f1_score(
            y_val, predictions,
            labels=[0, 1],
            average=None,
            zero_division=0
        )

        macro_f1 = f1_score(
            y_val, predictions,
            average="macro",
            zero_division=0
        )

        weight_name = "balanced" if weight else "standard"

        results.append({
            "Class Weight": weight_name,
            "C": c,
            "Accuracy": accuracy,
            "Fake Precision": fake_precision,
            "Fake Recall": fake_recall,
            "Fake F1": fake_f1,
            "Real Precision": real_precision,
            "Real Recall": real_recall,
            "Real F1": real_f1,
            "Macro F1": macro_f1
        })

        filename = f"logistic_regression_{weight_name}_C_{c}.joblib"
        joblib.dump(model, model_dir / filename)

# Sort by macro F1
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="Macro F1",
    ascending=False
)

# Save results
results_dir = Path("outputs/results")
results_dir.mkdir(parents=True, exist_ok=True)

output_path = results_dir / "lr_parameter_comparison.csv"
results_df.to_csv(output_path, index=False)

print(
    f"LOGISTIC REGRESSION PARAMETER RESULTS\n"
    f"-------------------------------------\n"
    f"{results_df.to_string(index=False)}\n"
    f"\nSaved comparison: {output_path}"
)