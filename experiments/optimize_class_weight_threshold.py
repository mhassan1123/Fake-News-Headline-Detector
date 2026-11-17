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
MODEL_DIR = "outputs/models"
RESULT_DIR = "outputs/results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

print("CLASS WEIGHT AND THRESHOLD OPTIMIZATION")
print("=" * 70)

# 1. Load training and validation features
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

# 2. Define the two class-weight strategies
class_weights = {
    "Standard": None,
    "Balanced": "balanced"
}

thresholds = np.round(
    np.arange(0.20, 0.801, 0.05),
    2
)

results = []

# 3. Train each model
for strategy, class_weight in class_weights.items():

    print(f"\nTRAINING: {strategy.upper()}")
    print("-" * 70)

    model = LogisticRegression(
        C=1.0,
        class_weight=class_weight,
        max_iter=2000,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Save each trained model
    model_filename = (
        "lr_standard_class_weight.joblib"
        if strategy == "Standard"
        else "lr_balanced_class_weight.joblib"
    )

    joblib.dump(
        model,
        os.path.join(MODEL_DIR, model_filename)
    )

    # 4. Get Fake probabilities
    probabilities = model.predict_proba(X_validation)

    fake_probability = probabilities[:, 0]

    # 5. Evaluate all thresholds
    for threshold in thresholds:

        # Class 0 = Fake, Class 1 = Real
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

# 6. Save every experiment
results_df = pd.DataFrame(results)

output_path = os.path.join(
    RESULT_DIR,
    "class_weight_threshold_results.csv"
)

results_df.to_csv(output_path, index=False)

# 7. Identify configurations meeting both recall targets
eligible = results_df[
    (results_df["fake_recall"] >= 0.80) &
    (results_df["real_recall"] >= 0.97)
]

print("\nALL EXPERIMENT RESULTS")
print("=" * 70)

print(
    results_df[
        [
            "strategy",
            "threshold",
            "accuracy",
            "fake_recall",
            "real_recall",
            "macro_f1"
        ]
    ].to_string(index=False)
)

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
    print(f"Threshold: {best['threshold']:.2f}")
    print(f"Accuracy: {best['accuracy']:.4f}")
    print(f"Fake Recall: {best['fake_recall']:.4f}")
    print(f"Real Recall: {best['real_recall']:.4f}")
    print(f"Macro F1: {best['macro_f1']:.4f}")

else:

    # Rank configurations by how close they are to both targets
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
    ).head(5)

    print("No configuration meets both recall targets yet.")
    print("\nFIVE CLOSEST CONFIGURATIONS")
    print(
        closest[
            [
                "strategy",
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

print("\nCLASS WEIGHT AND THRESHOLD EXPERIMENT COMPLETED")