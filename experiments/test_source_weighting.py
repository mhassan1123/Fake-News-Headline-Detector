import os
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

os.makedirs("outputs/audits/source_weighting", exist_ok=True)
os.makedirs("outputs/models/source_weighting", exist_ok=True)

TRAIN_PATH = "data/splits/train.csv"
VALIDATION_PATH = "data/splits/validation.csv"

VECTORIZER_PATH = "outputs/models/tfidf_experiments/tfidf_baseline.joblib"
BASELINE_MODEL_PATH = "outputs/models/tfidf_experiments/lr_baseline_standard.joblib"

TARGET_FAKE_RECALL = 0.80
TARGET_REAL_RECALL = 0.97

train_df = pd.read_csv(TRAIN_PATH)
validation_df = pd.read_csv(VALIDATION_PATH)

required = {"title", "real", "source_dataset"}

if not required.issubset(train_df.columns):
    raise ValueError("Training data is missing required columns.")

if not required.issubset(validation_df.columns):
    raise ValueError("Validation data is missing required columns.")

train_df = train_df.dropna(subset=["title", "real", "source_dataset"]).copy()
validation_df = validation_df.dropna(subset=["title", "real", "source_dataset"]).copy()

train_df["title"] = train_df["title"].astype(str)
validation_df["title"] = validation_df["title"].astype(str)

train_df["real"] = train_df["real"].astype(int)
validation_df["real"] = validation_df["real"].astype(int)

train_df["source_dataset"] = train_df["source_dataset"].astype(str).str.strip()
validation_df["source_dataset"] = validation_df["source_dataset"].astype(str).str.strip()

vectorizer = joblib.load(VECTORIZER_PATH)
baseline_model = joblib.load(BASELINE_MODEL_PATH)

X_validation = vectorizer.transform(validation_df["title"])
y_validation = validation_df["real"].to_numpy()

fake_index = list(baseline_model.classes_).index(0)

# Use the existing vectorizer, fitted only on the original training data.
# All weighted models therefore use exactly the same vocabulary.
X_train_all = vectorizer.transform(train_df["title"])
y_train_all = train_df["real"].to_numpy()

# Exclude the one mixed-source record from the weighted experiments.
main_train = train_df[
    train_df["source_dataset"].isin(["FakeNewsNet", "WELFake"])
].copy()

def normalize_weights(weights):
    return weights / np.mean(weights)

# Approach 1: Equal total influence for both sources.
source_counts = main_train["source_dataset"].value_counts()

source_weights = main_train["source_dataset"].map(
    lambda source: len(main_train) / (
        2 * source_counts[source]
    )
).to_numpy()

source_weights = normalize_weights(source_weights)

# Approach 2: Equal total influence for each source/class group.
group_counts = main_train.groupby(
    ["source_dataset", "real"]
)["real"].transform("count")

group_weights = (
    len(main_train) / (4 * group_counts)
).to_numpy()

group_weights = normalize_weights(group_weights)

X_weighted = vectorizer.transform(main_train["title"])
y_weighted = main_train["real"].to_numpy()

# Evaluate every threshold for a model.
def evaluate_thresholds(model_name, model, probabilities):
    rows = []

    for threshold in np.arange(0.000, 1.001, 0.001):
        predictions = np.where(
            probabilities >= threshold, 0, 1
        )

        fake_recall = recall_score(
            y_validation == 0,
            predictions == 0,
            zero_division=0
        )

        real_recall = recall_score(
            y_validation == 1,
            predictions == 1,
            zero_division=0
        )

        fake_precision = precision_score(
            y_validation == 0,
            predictions == 0,
            zero_division=0
        )

        real_precision = precision_score(
            y_validation == 1,
            predictions == 1,
            zero_division=0
        )

        macro_f1 = f1_score(
            y_validation,
            predictions,
            average="macro",
            zero_division=0
        )

        score = min(
            fake_recall / TARGET_FAKE_RECALL,
            real_recall / TARGET_REAL_RECALL
        )

        rows.append({
            "Model": model_name,
            "Threshold": threshold,
            "Accuracy": accuracy_score(y_validation, predictions),
            "Fake Precision": fake_precision,
            "Fake Recall": fake_recall,
            "Real Precision": real_precision,
            "Real Recall": real_recall,
            "Macro F1": macro_f1,
            "ROC-AUC": roc_auc_score(
                y_validation == 0, probabilities
            ),
            "Target Score": score,
            "Both Targets Met": (
                fake_recall >= TARGET_FAKE_RECALL
                and real_recall >= TARGET_REAL_RECALL
            )
        })

    return pd.DataFrame(rows)


def train_and_evaluate(name, sample_weights):
    print(f"\nTRAINING: {name}")
    print("-" * 65)

    model = LogisticRegression(
        C=2,
        max_iter=1000,
        random_state=42
    )

    model.fit(
        X_weighted,
        y_weighted,
        sample_weight=sample_weights
    )

    model_path = (
        f"outputs/models/source_weighting/{name}.joblib"
    )

    joblib.dump(model, model_path)

    model_fake_index = list(model.classes_).index(0)

    probabilities = model.predict_proba(
        X_validation
    )[:, model_fake_index]

    results = evaluate_thresholds(
        name, model, probabilities
    )

    results.to_csv(
        f"outputs/audits/source_weighting/{name}_thresholds.csv",
        index=False
    )

    best = results.sort_values(
        ["Target Score", "Macro F1"],
        ascending=False
    ).iloc[0]

    eligible = results[
        results["Both Targets Met"]
    ]

    print(f"ROC-AUC: {best['ROC-AUC']:.4f}")
    print(f"Best threshold: {best['Threshold']:.3f}")
    print(f"Fake Recall: {best['Fake Recall']:.4f}")
    print(f"Real Recall: {best['Real Recall']:.4f}")
    print(f"Macro F1: {best['Macro F1']:.4f}")
    print(f"Target Score: {best['Target Score']:.4f}")
    print(f"Both targets achieved: {not eligible.empty}")

    return results


print("SOURCE WEIGHTING EXPERIMENT")
print("=" * 70)

print(f"Original training records: {len(train_df)}")
print(f"Weighted training records: {len(main_train)}")
print(f"Validation records: {len(validation_df)}")

print("\nTRAINING DISTRIBUTION")
print(
    pd.crosstab(
        main_train["source_dataset"],
        main_train["real"]
    ).to_string()
)

all_results = []

# Existing baseline: evaluate its full threshold range.
baseline_fake_index = list(
    baseline_model.classes_
).index(0)

baseline_probabilities = baseline_model.predict_proba(
    X_validation
)[:, baseline_fake_index]

baseline_results = evaluate_thresholds(
    "Combined Baseline",
    baseline_model,
    baseline_probabilities
)

baseline_results.to_csv(
    "outputs/audits/source_weighting/baseline_thresholds.csv",
    index=False
)

all_results.append(baseline_results)

# Two new weighted approaches.
source_results = train_and_evaluate(
    "source_balanced",
    source_weights
)

all_results.append(source_results)

group_results = train_and_evaluate(
    "source_class_balanced",
    group_weights
)

all_results.append(group_results)

combined_results = pd.concat(
    all_results,
    ignore_index=True
)

combined_results.to_csv(
    "outputs/audits/source_weighting/all_threshold_results.csv",
    index=False
)

summary = combined_results.sort_values(
    ["Target Score", "Macro F1"],
    ascending=False
).groupby("Model").first().reset_index()

summary = summary.sort_values(
    ["Target Score", "Macro F1"],
    ascending=False
)

summary.to_csv(
    "outputs/audits/source_weighting/model_summary.csv",
    index=False
)

print("\nFINAL MODEL COMPARISON")
print("=" * 70)

columns = [
    "Model",
    "Threshold",
    "Fake Recall",
    "Real Recall",
    "Macro F1",
    "ROC-AUC",
    "Target Score",
    "Both Targets Met"
]

print(summary[columns].round(4).to_string(index=False))

print("\nSAVED RESULTS")
print("outputs/audits/source_weighting/")
print("outputs/models/source_weighting/")
print("SOURCE WEIGHTING EXPERIMENT COMPLETED")