import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

os.makedirs("outputs/audits/ensembles", exist_ok=True)

VALIDATION_PATH = "data/splits/validation.csv"

VECTORIZER_PATH = "outputs/models/tfidf_experiments/tfidf_baseline.joblib"

MODEL_PATHS = {
    "logistic_regression": (
        "outputs/models/tfidf_experiments/lr_baseline_standard.joblib"
    ),
    "linear_svm": (
        "outputs/models/alternative_algorithms/calibrated_linear_svm.joblib"
    ),
    "sgd_logistic": (
        "outputs/models/alternative_algorithms/sgd_logistic.joblib"
    ),
    "complement_nb": (
        "outputs/models/alternative_algorithms/complement_naive_bayes.joblib"
    )
}

TARGET_FAKE_RECALL = 0.80
TARGET_REAL_RECALL = 0.97

validation_df = pd.read_csv(VALIDATION_PATH)
validation_df = validation_df.dropna(
    subset=["title", "real", "source_dataset"]
).copy()

validation_df["title"] = validation_df["title"].astype(str)
validation_df["real"] = validation_df["real"].astype(int)

vectorizer = joblib.load(VECTORIZER_PATH)
X_validation = vectorizer.transform(validation_df["title"])
y_validation = validation_df["real"].to_numpy()

models = {
    name: joblib.load(path)
    for name, path in MODEL_PATHS.items()
}

def get_fake_probability(model):
    classes = list(model.classes_)

    if 0 not in classes:
        raise ValueError("Model does not contain Fake class (0).")

    fake_index = classes.index(0)

    return model.predict_proba(X_validation)[:, fake_index]


probabilities = {
    name: get_fake_probability(model)
    for name, model in models.items()
}

# Each ensemble uses the average of its members' Fake probabilities.
ensemble_definitions = {
    "logistic_svm_ensemble": [
        "logistic_regression",
        "linear_svm"
    ],
    "logistic_sgd_ensemble": [
        "logistic_regression",
        "sgd_logistic"
    ],
    "logistic_svm_nb_ensemble": [
        "logistic_regression",
        "linear_svm",
        "complement_nb"
    ]
}

# Include individual models as comparison candidates.
candidates = {
    name: probabilities[name]
    for name in probabilities
}

for ensemble_name, members in ensemble_definitions.items():
    candidates[ensemble_name] = np.mean(
        [probabilities[name] for name in members],
        axis=0
    )

def evaluate_thresholds(name, fake_probability):
    rows = []

    for threshold in np.arange(0.000, 1.001, 0.001):
        predictions = np.where(
            fake_probability >= threshold, 0, 1
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
            "Model": name,
            "Threshold": threshold,
            "Accuracy": accuracy_score(y_validation, predictions),
            "Fake Precision": fake_precision,
            "Fake Recall": fake_recall,
            "Real Precision": real_precision,
            "Real Recall": real_recall,
            "Macro F1": macro_f1,
            "ROC-AUC": roc_auc_score(
                y_validation == 0,
                fake_probability
            ),
            "Target Score": target_score,
            "Both Targets Met": (
                fake_recall >= TARGET_FAKE_RECALL
                and real_recall >= TARGET_REAL_RECALL
            )
        })

    return pd.DataFrame(rows)


all_results = []
source_results = []

print("MODEL ENSEMBLE EXPERIMENT")
print("=" * 70)
print(f"Validation records: {len(validation_df)}")
print(f"Ensemble methods: {len(ensemble_definitions)}")

for name, fake_probability in candidates.items():
    results = evaluate_thresholds(name, fake_probability)

    results.to_csv(
        f"outputs/audits/ensembles/{name}_thresholds.csv",
        index=False
    )

    best = results.sort_values(
        ["Target Score", "Macro F1"],
        ascending=False
    ).iloc[0]

    eligible = results[
        results["Both Targets Met"]
    ]

    print(f"\n{name}")
    print("-" * 60)
    print(f"ROC-AUC: {best['ROC-AUC']:.4f}")
    print(f"Best threshold: {best['Threshold']:.3f}")
    print(f"Fake Recall: {best['Fake Recall']:.4f}")
    print(f"Real Recall: {best['Real Recall']:.4f}")
    print(f"Macro F1: {best['Macro F1']:.4f}")
    print(f"Target Score: {best['Target Score']:.4f}")
    print(f"Both targets achieved: {not eligible.empty}")

    all_results.append(results)

    predictions = np.where(
        fake_probability >= best["Threshold"], 0, 1
    )

    for source in ["FakeNewsNet", "WELFake"]:
        mask = (
            validation_df["source_dataset"].to_numpy() == source
        )

        y_source = y_validation[mask]
        pred_source = predictions[mask]
        prob_source = fake_probability[mask]

        if len(y_source) == 0 or len(np.unique(y_source)) < 2:
            continue

        source_results.append({
            "Model": name,
            "Source": source,
            "Records": len(y_source),
            "Fake Recall": recall_score(
                y_source == 0, pred_source == 0,
                zero_division=0
            ),
            "Real Recall": recall_score(
                y_source == 1, pred_source == 1,
                zero_division=0
            ),
            "Macro F1": f1_score(
                y_source, pred_source,
                average="macro", zero_division=0
            ),
            "ROC-AUC": roc_auc_score(
                y_source == 0, prob_source
            )
        })

combined_results = pd.concat(
    all_results,
    ignore_index=True
)

combined_results.to_csv(
    "outputs/audits/ensembles/all_threshold_results.csv",
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
    "outputs/audits/ensembles/model_summary.csv",
    index=False
)

source_df = pd.DataFrame(source_results)

source_df.to_csv(
    "outputs/audits/ensembles/source_performance.csv",
    index=False
)

print("\nFINAL ENSEMBLE COMPARISON")
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

print("\nSOURCE-WISE PERFORMANCE")
print("=" * 70)
print(source_df.round(4).to_string(index=False))

print("\nSAVED RESULTS")
print("outputs/audits/ensembles/")
print("MODEL ENSEMBLE EXPERIMENT COMPLETED")