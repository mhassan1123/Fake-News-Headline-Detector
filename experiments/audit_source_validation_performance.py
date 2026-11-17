import os
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

MODEL_DIR = "outputs/models/tfidf_experiments"
AUDIT_DIR = "outputs/audits"

os.makedirs(AUDIT_DIR, exist_ok=True)

print("SOURCE-WISE VALIDATION PERFORMANCE AUDIT")
print("=" * 70)

# 1. Load the validation dataset and selected model
validation_df = pd.read_csv(
    "data/splits/validation.csv"
)

vectorizer = joblib.load(
    os.path.join(MODEL_DIR, "tfidf_baseline.joblib")
)

model = joblib.load(
    os.path.join(MODEL_DIR, "lr_baseline_standard.joblib")
)

required_columns = ["title", "real", "source_dataset"]

for column in required_columns:
    if column not in validation_df.columns:
        raise ValueError(f"Missing required column: {column}")

if validation_df[required_columns].isnull().any().any():
    raise ValueError("Missing values found in required columns.")

# 2. Generate predictions
X_validation = vectorizer.transform(
    validation_df["title"].astype(str)
)

y_true = validation_df["real"].to_numpy()

fake_index = np.where(model.classes_ == 0)[0]

if len(fake_index) != 1:
    raise ValueError("Fake class not found correctly.")

fake_probability = model.predict_proba(
    X_validation
)[:, fake_index[0]]

THRESHOLD = 0.45

y_pred = np.where(
    fake_probability >= THRESHOLD,
    0,
    1
)

# 3. Prepare prediction records
audit_df = validation_df.copy()

audit_df["fake_probability"] = fake_probability
audit_df["predicted_label"] = y_pred

audit_df["prediction_result"] = np.where(
    y_true == y_pred,
    "Correct",
    "Incorrect"
)

# 4. Calculate overall performance
print("\nOVERALL VALIDATION PERFORMANCE")
print("-" * 70)

print(f"Total headlines: {len(audit_df)}")
print(
    f"Accuracy: {accuracy_score(y_true, y_pred):.4f}"
)

print(
    f"Fake Recall: {recall_score(y_true, y_pred, pos_label=0):.4f}"
)

print(
    f"Real Recall: {recall_score(y_true, y_pred, pos_label=1):.4f}"
)

print(
    f"Macro F1: {f1_score(y_true, y_pred, average='macro'):.4f}"
)

# 5. Evaluate each source separately
source_results = []

print("\nSOURCE-WISE PERFORMANCE")
print("=" * 70)

for source, group in audit_df.groupby("source_dataset"):

    actual = group["real"].to_numpy()
    predicted = group["predicted_label"].to_numpy()

    fake_support = int(np.sum(actual == 0))
    real_support = int(np.sum(actual == 1))

    fake_recall = recall_score(
        actual,
        predicted,
        pos_label=0,
        zero_division=0
    )

    real_recall = recall_score(
        actual,
        predicted,
        pos_label=1,
        zero_division=0
    )

    fake_precision = precision_score(
        actual,
        predicted,
        pos_label=0,
        zero_division=0
    )

    real_precision = precision_score(
        actual,
        predicted,
        pos_label=1,
        zero_division=0
    )

    macro_f1 = f1_score(
        actual,
        predicted,
        average="macro",
        zero_division=0
    )

    accuracy = accuracy_score(actual, predicted)

    missed_fake = int(
        np.sum((actual == 0) & (predicted == 1))
    )

    real_marked_fake = int(
        np.sum((actual == 1) & (predicted == 0))
    )

    source_results.append({
        "source": source,
        "total": len(group),
        "fake_support": fake_support,
        "real_support": real_support,
        "accuracy": accuracy,
        "fake_precision": fake_precision,
        "fake_recall": fake_recall,
        "real_precision": real_precision,
        "real_recall": real_recall,
        "macro_f1": macro_f1,
        "missed_fake": missed_fake,
        "real_marked_fake": real_marked_fake
    })

    print(f"\nSource: {source}")
    print(f"Total headlines: {len(group)}")
    print(f"Fake headlines: {fake_support}")
    print(f"Real headlines: {real_support}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Fake Precision: {fake_precision:.4f}")
    print(f"Fake Recall: {fake_recall:.4f}")
    print(f"Real Precision: {real_precision:.4f}")
    print(f"Real Recall: {real_recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Missed Fake headlines: {missed_fake}")
    print(f"Real headlines predicted Fake: {real_marked_fake}")

# 6. Save source-wise metrics
source_results_df = pd.DataFrame(source_results)

metrics_path = os.path.join(
    AUDIT_DIR,
    "source_validation_performance.csv"
)

source_results_df.to_csv(
    metrics_path,
    index=False
)

# 7. Save incorrect predictions for manual inspection
errors_df = audit_df[
    audit_df["prediction_result"] == "Incorrect"
].copy()

errors_path = os.path.join(
    AUDIT_DIR,
    "validation_misclassified_headlines.csv"
)

errors_df.to_csv(errors_path, index=False)

# 8. Separate the two types of mistakes
missed_fake_df = errors_df[
    (errors_df["real"] == 0) &
    (errors_df["predicted_label"] == 1)
]

real_marked_fake_df = errors_df[
    (errors_df["real"] == 1) &
    (errors_df["predicted_label"] == 0)
]

missed_fake_path = os.path.join(
    AUDIT_DIR,
    "validation_missed_fake_headlines.csv"
)

real_marked_fake_path = os.path.join(
    AUDIT_DIR,
    "validation_real_predicted_fake.csv"
)

missed_fake_df.to_csv(
    missed_fake_path,
    index=False
)

real_marked_fake_df.to_csv(
    real_marked_fake_path,
    index=False
)

print("\nERROR SUMMARY")
print("=" * 70)
print(f"Total incorrect predictions: {len(errors_df)}")
print(f"Fake predicted as Real: {len(missed_fake_df)}")
print(f"Real predicted as Fake: {len(real_marked_fake_df)}")

print("\nSAVED FILES")
print(metrics_path)
print(errors_path)
print(missed_fake_path)
print(real_marked_fake_path)

print("\nSOURCE-WISE VALIDATION AUDIT COMPLETED")