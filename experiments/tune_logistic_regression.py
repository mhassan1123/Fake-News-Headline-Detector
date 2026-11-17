import os
import numpy as np
import pandas as pd
import joblib

from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

FEATURE_DIR = "outputs/final_features"
MODEL_DIR = "outputs/models/parameter_experiments"
RESULT_DIR = "outputs/parameter_experiments"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

print("LOGISTIC REGRESSION PARAMETER TUNING")
print("=" * 70)

# 1. Load prepared features
X_train = load_npz(
    os.path.join(FEATURE_DIR, "X_train_tfidf.npz")
)

X_validation = load_npz(
    os.path.join(FEATURE_DIR, "X_validation_tfidf.npz")
)

y_train = np.load(
    os.path.join(FEATURE_DIR, "y_train.npy")
)

y_validation = np.load(
    os.path.join(FEATURE_DIR, "y_validation.npy")
)

if X_train.shape[0] != len(y_train):
    raise ValueError("Training dimensions do not match.")

if X_validation.shape[0] != len(y_validation):
    raise ValueError("Validation dimensions do not match.")

if X_train.shape[1] != X_validation.shape[1]:
    raise ValueError("Feature dimensions do not match.")

# 2. Experiment settings
c_values = [0.01, 0.1, 0.5, 1, 2, 5, 10]

class_weights = {
    "Standard": None,
    "Balanced": "balanced"
}

thresholds = np.round(
    np.arange(0.20, 0.801, 0.05),
    2
)

results = []

# 3. Train and evaluate all configurations
for strategy, class_weight in class_weights.items():

    for c_value in c_values:

        print(
            f"\nTraining {strategy} model with C={c_value}"
        )

        model = LogisticRegression(
            C=c_value,
            class_weight=class_weight,
            max_iter=2000,
            random_state=42
        )

        model.fit(X_train, y_train)

        # Save every trained model
        model_name = (
            f"lr_{strategy.lower()}_C_{str(c_value).replace('.', '_')}.joblib"
        )

        joblib.dump(
            model,
            os.path.join(MODEL_DIR, model_name)
        )

        # Confirm class order before using probabilities
        fake_index = np.where(model.classes_ == 0)[0]

        if len(fake_index) != 1:
            raise ValueError("Fake class not found correctly.")

        fake_probability = model.predict_proba(
            X_validation
        )[:, fake_index[0]]

        # Evaluate every threshold
        for threshold in thresholds:

            predictions = np.where(
                fake_probability >= threshold,
                0,
                1
            )

            fake_precision = precision_score(
                y_validation,
                predictions,
                pos_label=0,
                zero_division=0
            )

            fake_recall = recall_score(
                y_validation,
                predictions,
                pos_label=0,
                zero_division=0
            )

            fake_f1 = f1_score(
                y_validation,
                predictions,
                pos_label=0,
                zero_division=0
            )

            real_precision = precision_score(
                y_validation,
                predictions,
                pos_label=1,
                zero_division=0
            )

            real_recall = recall_score(
                y_validation,
                predictions,
                pos_label=1,
                zero_division=0
            )

            real_f1 = f1_score(
                y_validation,
                predictions,
                pos_label=1,
                zero_division=0
            )

            accuracy = accuracy_score(
                y_validation,
                predictions
            )

            macro_f1 = f1_score(
                y_validation,
                predictions,
                average="macro",
                zero_division=0
            )

            results.append({
                "strategy": strategy,
                "C": c_value,
                "threshold": threshold,
                "accuracy": accuracy,
                "fake_precision": fake_precision,
                "fake_recall": fake_recall,
                "fake_f1": fake_f1,
                "real_precision": real_precision,
                "real_recall": real_recall,
                "real_f1": real_f1,
                "macro_f1": macro_f1
            })

# 4. Save all results
results_df = pd.DataFrame(results)

output_path = os.path.join(
    RESULT_DIR,
    "logistic_regression_parameter_results.csv"
)

results_df.to_csv(output_path, index=False)

# 5. Find configurations satisfying both targets
eligible = results_df[
    (results_df["fake_recall"] >= 0.80) &
    (results_df["real_recall"] >= 0.97)
]

print("\nEXPERIMENT SUMMARY")
print("=" * 70)
print(f"Models trained: {len(class_weights) * len(c_values)}")
print(f"Configurations tested: {len(results_df)}")

# 6. Show best configuration for each C and strategy
best_per_model = results_df.loc[
    results_df.groupby(["strategy", "C"])["macro_f1"].idxmax()
].sort_values("macro_f1", ascending=False)

print("\nBEST THRESHOLD FOR EACH MODEL")
print("-" * 70)

print(
    best_per_model[
        [
            "strategy",
            "C",
            "threshold",
            "accuracy",
            "fake_recall",
            "real_recall",
            "macro_f1"
        ]
    ].to_string(index=False)
)

# 7. Check target requirements
print("\nTARGET CHECK")
print("=" * 70)

if not eligible.empty:

    best = eligible.loc[
        eligible["macro_f1"].idxmax()
    ]

    print("Configurations meeting BOTH recall targets:")
    print(eligible.to_string(index=False))

    print("\nBEST ELIGIBLE CONFIGURATION")
    print(f"Strategy: {best['strategy']}")
    print(f"C: {best['C']}")
    print(f"Threshold: {best['threshold']:.2f}")
    print(f"Accuracy: {best['accuracy']:.4f}")
    print(f"Fake Recall: {best['fake_recall']:.4f}")
    print(f"Real Recall: {best['real_recall']:.4f}")
    print(f"Macro F1: {best['macro_f1']:.4f}")

else:

    results_df["fake_shortfall"] = (
        0.80 - results_df["fake_recall"]
    ).clip(lower=0)

    results_df["real_shortfall"] = (
        0.97 - results_df["real_recall"]
    ).clip(lower=0)

    results_df["total_shortfall"] = (
        results_df["fake_shortfall"] +
        results_df["real_shortfall"]
    )

    closest = results_df.sort_values(
        ["total_shortfall", "macro_f1"],
        ascending=[True, False]
    ).head(10)

    print("No configuration meets both targets.")

    print("\nTEN CLOSEST CONFIGURATIONS")
    print(
        closest[
            [
                "strategy",
                "C",
                "threshold",
                "accuracy",
                "fake_recall",
                "real_recall",
                "macro_f1",
                "total_shortfall"
            ]
        ].to_string(index=False)
    )

print("\nSAVED RESULTS")
print(output_path)

print("\nPARAMETER TUNING COMPLETED")