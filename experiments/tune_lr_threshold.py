from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from scipy.sparse import load_npz
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

ROOT = Path(__file__).resolve().parents[1]

VALIDATION_PATH = ROOT / "data/splits_cleaned_v2/validation.csv"
TFIDF_DIR = ROOT / "outputs/cleaned_v2/tfidf"

MODEL_PATH = (
    ROOT
    / "outputs/cleaned_v2/reproducible_experiment/models/logistic_regression.joblib"
)

OUTPUT_DIR = ROOT / "outputs/cleaned_v2/threshold_tuning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load validation data and precomputed features
validation = pd.read_csv(VALIDATION_PATH)
X_validation = load_npz(TFIDF_DIR / "X_validation.npz")
y_true = validation["real"].astype(int).to_numpy()

# Load the cleaned-v2 Logistic Regression model
model = joblib.load(MODEL_PATH)

fake_index = list(model.classes_).index(0)
fake_probabilities = model.predict_proba(X_validation)[:, fake_index]

# Evaluate a fine threshold range, including our previous threshold
thresholds = sorted(
    set(np.round(np.arange(0.0, 1.001, 0.001), 3).tolist())
    | {0.496, 0.500}
)

results = []

for threshold in thresholds:
    # Label 0 = Fake, label 1 = Real
    y_pred = np.where(fake_probabilities >= threshold, 0, 1)

    results.append({
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "fake_precision": precision_score(
            y_true, y_pred, pos_label=0, zero_division=0
        ),
        "fake_recall": recall_score(
            y_true, y_pred, pos_label=0, zero_division=0
        ),
        "real_precision": precision_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "real_recall": recall_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        )
    })

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "lr_threshold_comparison.csv",
    index=False
)

# Check the original recall targets
fake_target_candidates = results_df[
    results_df["real_recall"] >= 0.97
]

real_target_candidates = results_df[
    results_df["fake_recall"] >= 0.80
]

both_target_candidates = results_df[
    (results_df["fake_recall"] >= 0.80) &
    (results_df["real_recall"] >= 0.97)
]

print("\nRECALL TARGET AUDIT")
print("=" * 60)

if not fake_target_candidates.empty:
    best_fake = fake_target_candidates.sort_values(
        ["fake_recall", "macro_f1"],
        ascending=False
    ).iloc[0]

    print("\nBest Fake Recall while Real Recall >= 97%")
    print(f"Threshold: {best_fake['threshold']:.3f}")
    print(f"Fake Recall: {best_fake['fake_recall']:.4f}")
    print(f"Real Recall: {best_fake['real_recall']:.4f}")
    print(f"Macro F1: {best_fake['macro_f1']:.4f}")
else:
    print("\nNo threshold achieved Real Recall >= 97%.")

if not real_target_candidates.empty:
    best_real = real_target_candidates.sort_values(
        ["real_recall", "macro_f1"],
        ascending=False
    ).iloc[0]

    print("\nBest Real Recall while Fake Recall >= 80%")
    print(f"Threshold: {best_real['threshold']:.3f}")
    print(f"Fake Recall: {best_real['fake_recall']:.4f}")
    print(f"Real Recall: {best_real['real_recall']:.4f}")
    print(f"Macro F1: {best_real['macro_f1']:.4f}")
else:
    print("\nNo threshold achieved Fake Recall >= 80%.")

if not both_target_candidates.empty:
    print("\nBoth recall targets are achievable.")
else:
    print("\nNo threshold achieved both recall targets simultaneously.")

# Identify the threshold with the highest validation Macro F1
best_index = results_df["macro_f1"].idxmax()
best_result = results_df.loc[best_index]

print(
    "\nCLEANED-V2 LOGISTIC REGRESSION THRESHOLD AUDIT\n"
    "============================================================\n"
    f"Validation records: {len(y_true)}\n"
    f"Thresholds evaluated: {len(results_df)}\n"
    f"Previous threshold (0.496) Macro F1: "
    f"{results_df.loc[results_df['threshold'] == 0.496, 'macro_f1'].iloc[0]:.4f}\n"
    f"Standard threshold (0.500) Macro F1: "
    f"{results_df.loc[results_df['threshold'] == 0.500, 'macro_f1'].iloc[0]:.4f}\n"
    "\nBEST VALIDATION THRESHOLD\n"
    "------------------------------------------------------------\n"
    f"Threshold: {best_result['threshold']:.3f}\n"
    f"Accuracy: {best_result['accuracy']:.4f}\n"
    f"Fake Precision: {best_result['fake_precision']:.4f}\n"
    f"Fake Recall: {best_result['fake_recall']:.4f}\n"
    f"Real Precision: {best_result['real_precision']:.4f}\n"
    f"Real Recall: {best_result['real_recall']:.4f}\n"
    f"Macro F1: {best_result['macro_f1']:.4f}\n"
    "\nFull threshold results saved."
)

print(f"Output: {OUTPUT_DIR / 'lr_threshold_comparison.csv'}")