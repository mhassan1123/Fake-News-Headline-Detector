import os
import joblib
import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

os.makedirs("outputs/audits/alternative_algorithms", exist_ok=True)
os.makedirs("outputs/models/alternative_algorithms", exist_ok=True)

TRAIN_PATH = "data/splits/train.csv"
VALIDATION_PATH = "data/splits/validation.csv"

VECTORIZER_PATH = "outputs/models/tfidf_experiments/tfidf_baseline.joblib"
BASELINE_MODEL_PATH = "outputs/models/tfidf_experiments/lr_baseline_standard.joblib"

TARGET_FAKE_RECALL = 0.80
TARGET_REAL_RECALL = 0.97

train_df = pd.read_csv(TRAIN_PATH)
validation_df = pd.read_csv(VALIDATION_PATH)

train_df = train_df.dropna(subset=["title", "real"]).copy()
validation_df = validation_df.dropna(subset=["title", "real", "source_dataset"]).copy()

train_df["title"] = train_df["title"].astype(str)
validation_df["title"] = validation_df["title"].astype(str)

train_df["real"] = train_df["real"].astype(int)
validation_df["real"] = validation_df["real"].astype(int)

vectorizer = joblib.load(VECTORIZER_PATH)
baseline_model = joblib.load(BASELINE_MODEL_PATH)

X_train = vectorizer.transform(train_df["title"])
y_train = train_df["real"].to_numpy()

X_validation = vectorizer.transform(validation_df["title"])
y_validation = validation_df["real"].to_numpy()

def get_fake_probability(model, X):
    classes = list(model.classes_)

    if 0 not in classes or 1 not in classes:
        raise ValueError("Model must contain Fake (0) and Real (1).")

    fake_index = classes.index(0)
    return model.predict_proba(X)[:, fake_index]


def evaluate_thresholds(model_name, probabilities):
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

        target_score = min(
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
            "Target Score": target_score,
            "Both Targets Met": (
                fake_recall >= TARGET_FAKE_RECALL
                and real_recall >= TARGET_REAL_RECALL
            )
        })

    return pd.DataFrame(rows)


def evaluate_source_performance(model_name, probabilities, threshold):
    predictions = np.where(
        probabilities >= threshold, 0, 1
    )

    rows = []

    for source in ["FakeNewsNet", "WELFake"]:
        mask = (
            validation_df["source_dataset"].to_numpy() == source
        )

        y_source = y_validation[mask]
        pred_source = predictions[mask]
        prob_source = probabilities[mask]

        if len(y_source) == 0 or len(np.unique(y_source)) < 2:
            continue

        rows.append({
            "Model": model_name,
            "Source": source,
            "Records": len(y_source),
            "Accuracy": accuracy_score(y_source, pred_source),
            "Fake Recall": recall_score(
                y_source == 0, pred_source == 0, zero_division=0
            ),
            "Real Recall": recall_score(
                y_source == 1, pred_source == 1, zero_division=0
            ),
            "Macro F1": f1_score(
                y_source, pred_source,
                average="macro", zero_division=0
            ),
            "ROC-AUC": roc_auc_score(
                y_source == 0, prob_source
            )
        })

    return rows


print("ALTERNATIVE ALGORITHM EXPERIMENT")
print("=" * 70)
print(f"Training records: {len(train_df)}")
print(f"Validation records: {len(validation_df)}")
print(f"TF-IDF features: {X_train.shape[1]}")

models = {
    "calibrated_linear_svm": CalibratedClassifierCV(
        estimator=LinearSVC(
            C=1,
            max_iter=5000,
            random_state=42
        ),
        method="sigmoid",
        cv=3,
        n_jobs=-1
    ),

    "sgd_logistic": SGDClassifier(
        loss="log_loss",
        alpha=0.00001,
        max_iter=2000,
        tol=0.0001,
        random_state=42
    ),

    "complement_naive_bayes": ComplementNB(
        alpha=0.1
    )
}

all_threshold_results = []
source_results = []

# Evaluate the existing baseline first.
baseline_probabilities = get_fake_probability(
    baseline_model, X_validation
)

baseline_results = evaluate_thresholds(
    "logistic_regression_baseline",
    baseline_probabilities
)

baseline_results.to_csv(
    "outputs/audits/alternative_algorithms/logistic_regression_thresholds.csv",
    index=False
)

baseline_best = baseline_results.sort_values(
    ["Target Score", "Macro F1"],
    ascending=False
).iloc[0]

all_threshold_results.append(baseline_results)

source_results.extend(
    evaluate_source_performance(
        "logistic_regression_baseline",
        baseline_probabilities,
        baseline_best["Threshold"]
    )
)

print("\nBASELINE LOGISTIC REGRESSION")
print("-" * 65)
print(f"ROC-AUC: {baseline_best['ROC-AUC']:.4f}")
print(f"Best threshold: {baseline_best['Threshold']:.3f}")
print(f"Fake Recall: {baseline_best['Fake Recall']:.4f}")
print(f"Real Recall: {baseline_best['Real Recall']:.4f}")
print(f"Macro F1: {baseline_best['Macro F1']:.4f}")
print(f"Target Score: {baseline_best['Target Score']:.4f}")

# Train and evaluate alternative algorithms.
for name, model in models.items():
    print(f"\nTRAINING: {name}")
    print("-" * 65)

    model.fit(X_train, y_train)

    model_path = (
        f"outputs/models/alternative_algorithms/{name}.joblib"
    )

    joblib.dump(model, model_path)

    probabilities = get_fake_probability(
        model, X_validation
    )

    results = evaluate_thresholds(
        name, probabilities
    )

    results.to_csv(
        f"outputs/audits/alternative_algorithms/{name}_thresholds.csv",
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

    all_threshold_results.append(results)

    source_results.extend(
        evaluate_source_performance(
            name, probabilities, best["Threshold"]
        )
    )

# Combine all threshold results.
all_results = pd.concat(
    all_threshold_results,
    ignore_index=True
)

all_results.to_csv(
    "outputs/audits/alternative_algorithms/all_threshold_results.csv",
    index=False
)

# Select each model's best balanced operating point.
summary = all_results.sort_values(
    ["Target Score", "Macro F1"],
    ascending=False
).groupby("Model").first().reset_index()

summary = summary.sort_values(
    ["Target Score", "Macro F1"],
    ascending=False
)

summary.to_csv(
    "outputs/audits/alternative_algorithms/model_summary.csv",
    index=False
)

source_df = pd.DataFrame(source_results)

source_df.to_csv(
    "outputs/audits/alternative_algorithms/source_performance.csv",
    index=False
)

print("\nFINAL ALGORITHM COMPARISON")
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

print("\nSOURCE-WISE PERFORMANCE AT SELECTED THRESHOLDS")
print("=" * 70)
print(source_df.round(4).to_string(index=False))

print("\nSAVED RESULTS")
print("outputs/audits/alternative_algorithms/")
print("outputs/models/alternative_algorithms/")
print("ALTERNATIVE ALGORITHM EXPERIMENT COMPLETED")