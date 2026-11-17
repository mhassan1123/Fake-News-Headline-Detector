import os
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import roc_curve, roc_auc_score

MODEL_DIR = "outputs/models/tfidf_experiments"
AUDIT_DIR = "outputs/audits"

os.makedirs(AUDIT_DIR, exist_ok=True)

print("ROC CURVE AND OPERATING POINT AUDIT")
print("=" * 70)

# 1. Load validation data and the selected model
validation_df = pd.read_csv(
    "data/splits/validation.csv"
)

vectorizer = joblib.load(
    os.path.join(MODEL_DIR, "tfidf_baseline.joblib")
)

model = joblib.load(
    os.path.join(MODEL_DIR, "lr_baseline_standard.joblib")
)

# 2. Generate Fake probabilities
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

# Treat Fake as the positive class for ROC analysis
y_fake = (y_true == 0).astype(int)

# 3. Calculate exact ROC operating points
fpr, tpr, thresholds = roc_curve(
    y_fake,
    fake_probability
)

roc_auc = roc_auc_score(
    y_fake,
    fake_probability
)

# Fake Recall = TPR
# Real Recall = 1 - FPR
fake_recall = tpr
real_recall = 1 - fpr

roc_df = pd.DataFrame({
    "threshold": thresholds,
    "fake_recall": fake_recall,
    "real_recall": real_recall,
    "false_fake_rate": fpr
})

# Save all ROC operating points
roc_path = os.path.join(
    AUDIT_DIR,
    "roc_operating_points.csv"
)

roc_df.to_csv(roc_path, index=False)

print(f"\nOverall ROC-AUC: {roc_auc:.4f}")
print(f"ROC operating points: {len(roc_df)}")

# 4. Find best Real Recall while Fake Recall >= 80%
fake_condition = roc_df[
    roc_df["fake_recall"] >= 0.80
]

print("\nFAKE RECALL AT LEAST 80%")
print("-" * 70)

if not fake_condition.empty:

    best_real = fake_condition.loc[
        fake_condition["real_recall"].idxmax()
    ]

    print(f"Best Fake Recall: {best_real['fake_recall']:.4f}")
    print(f"Best Real Recall: {best_real['real_recall']:.4f}")
    print(f"Threshold: {best_real['threshold']:.6f}")

else:
    print("No operating point reaches 80% Fake Recall.")

# 5. Find best Fake Recall while Real Recall >= 97%
real_condition = roc_df[
    roc_df["real_recall"] >= 0.97
]

print("\nREAL RECALL AT LEAST 97%")
print("-" * 70)

if not real_condition.empty:

    best_fake = real_condition.loc[
        real_condition["fake_recall"].idxmax()
    ]

    print(f"Best Fake Recall: {best_fake['fake_recall']:.4f}")
    print(f"Best Real Recall: {best_fake['real_recall']:.4f}")
    print(f"Threshold: {best_fake['threshold']:.6f}")

else:
    print("No operating point reaches 97% Real Recall.")

# 6. Check whether both targets can be achieved
eligible = roc_df[
    (roc_df["fake_recall"] >= 0.80) &
    (roc_df["real_recall"] >= 0.97)
]

print("\nBOTH TARGETS CHECK")
print("=" * 70)

if not eligible.empty:

    best = eligible.loc[
        eligible["fake_recall"].idxmax()
    ]

    print("BOTH TARGETS ARE ACHIEVABLE ON VALIDATION.")
    print(f"Threshold: {best['threshold']:.6f}")
    print(f"Fake Recall: {best['fake_recall']:.4f}")
    print(f"Real Recall: {best['real_recall']:.4f}")

    eligible_path = os.path.join(
        AUDIT_DIR,
        "roc_eligible_operating_points.csv"
    )

    eligible.to_csv(eligible_path, index=False)
    print(f"Eligible points saved: {eligible_path}")

else:

    print("No single threshold satisfies both targets.")
    print("We need to improve the model's class separation.")

# 7. Source-wise ROC-AUC and operating points
print("\nSOURCE-WISE ROC ANALYSIS")
print("=" * 70)

source_results = []

for source, group in validation_df.groupby("source_dataset"):

    indices = group.index.to_numpy()

    actual = (group["real"].to_numpy() == 0).astype(int)
    probabilities = fake_probability[indices]

    fake_count = int(np.sum(actual == 1))
    real_count = int(np.sum(actual == 0))

    print(f"\nSource: {source}")
    print(f"Fake records: {fake_count}")
    print(f"Real records: {real_count}")

    if fake_count == 0 or real_count == 0:

        print("ROC-AUC unavailable: only one class is present.")
        continue

    source_fpr, source_tpr, source_thresholds = roc_curve(
        actual,
        probabilities
    )

    source_auc = roc_auc_score(
        actual,
        probabilities
    )

    source_real_recall = 1 - source_fpr

    source_eligible = np.where(
        (source_tpr >= 0.80) &
        (source_real_recall >= 0.97)
    )[0]

    print(f"ROC-AUC: {source_auc:.4f}")

    if len(source_eligible) > 0:

        best_index = source_eligible[
            np.argmax(source_tpr[source_eligible])
        ]

        print("Both targets achieved for this source.")
        print(f"Threshold: {source_thresholds[best_index]:.6f}")
        print(f"Fake Recall: {source_tpr[best_index]:.4f}")
        print(f"Real Recall: {source_real_recall[best_index]:.4f}")

    else:

        print("Both targets NOT achieved for this source.")

    source_results.append({
        "source": source,
        "fake_count": fake_count,
        "real_count": real_count,
        "roc_auc": source_auc
    })

# 8. Save source-wise results
source_results_df = pd.DataFrame(source_results)

source_path = os.path.join(
    AUDIT_DIR,
    "source_roc_auc.csv"
)

source_results_df.to_csv(
    source_path,
    index=False
)

print("\nSAVED FILES")
print(roc_path)
print(source_path)

print("\nROC OPERATING POINT AUDIT COMPLETED")