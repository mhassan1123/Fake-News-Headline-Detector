import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)


input_path = "data/processed/cleaned_fakenewsnet.csv"
output_dir = Path("outputs/results")
output_dir.mkdir(parents=True, exist_ok=True)

# Load and validate the dataset
df = pd.read_csv(input_path)

if "title" not in df.columns or "real" not in df.columns:
    raise ValueError("Dataset must contain title and real columns.")

if df["title"].isnull().any() or df["real"].isnull().any():
    raise ValueError("Missing headlines or labels found.")

if not set(df["real"].unique()).issubset({0, 1}):
    raise ValueError("Unexpected label values found.")

X = df["title"].astype(str)
y = df["real"].astype(int)

# Recreate the exact existing 64/16/20 split
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    random_state=42,
    stratify=y_train_full
)

print(
    f"LINEAR SVM EXPERIMENT\n"
    f"---------------------\n"
    f"Training records: {len(X_train)}\n"
    f"Validation records: {len(X_val)}\n"
    f"Test records reserved: {len(X_test)}\n"
    f"Test set will not be evaluated or used for selection."
)

# Two word-feature configurations
feature_configs = [
    {
        "name": "Word_10000",
        "max_features": 10000,
        "min_df": 1
    },
    {
        "name": "Word_15000_MinDF2",
        "max_features": 15000,
        "min_df": 2
    }
]

c_values = [0.5, 1.0, 2.0, 5.0]
class_weights = [None, "balanced"]

all_results = []

for feature_config in feature_configs:
    print(
        f"\nPREPARING FEATURES: {feature_config['name']}\n"
        f"--------------------------------------------"
    )

    vectorizer = TfidfVectorizer(
        max_features=feature_config["max_features"],
        ngram_range=(1, 2),
        min_df=feature_config["min_df"]
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    print(f"Actual features: {X_train_tfidf.shape[1]}")

    for class_weight in class_weights:
        for c_value in c_values:
            weight_name = (
                "Balanced" if class_weight == "balanced"
                else "Standard"
            )

            config_name = (
                f"{feature_config['name']}_"
                f"{weight_name}_C{c_value}"
            )

            print(f"\nTraining: {config_name}")

            model = LinearSVC(
                C=c_value,
                class_weight=class_weight,
                max_iter=5000,
                random_state=42
            )

            model.fit(X_train_tfidf, y_train)

            # LinearSVC's positive score belongs to classes_[1].
            # Since labels are [0, 1], negate it to obtain a Fake score.
            if list(model.classes_) != [0, 1]:
                raise ValueError("Unexpected class order in SVM model.")

            fake_scores = -model.decision_function(X_val_tfidf)

            # Candidate thresholds from validation score distribution
            thresholds = np.unique(
                np.concatenate((
                    [0.0],
                    np.quantile(
                        fake_scores,
                        np.linspace(0.01, 0.99, 99)
                    )
                ))
            )

            config_results = []

            for threshold in thresholds:
                predictions = np.ones(len(y_val), dtype=int)
                predictions[fake_scores >= threshold] = 0

                accuracy = accuracy_score(y_val, predictions)

                precision, recall, f1, _ = (
                    precision_recall_fscore_support(
                        y_val,
                        predictions,
                        labels=[0, 1],
                        zero_division=0
                    )
                )

                macro_f1 = (f1[0] + f1[1]) / 2

                result = {
                    "configuration": config_name,
                    "features": X_train_tfidf.shape[1],
                    "C": c_value,
                    "class_weight": weight_name,
                    "threshold": float(threshold),
                    "accuracy": accuracy,
                    "fake_precision": precision[0],
                    "fake_recall": recall[0],
                    "fake_f1": f1[0],
                    "real_precision": precision[1],
                    "real_recall": recall[1],
                    "real_f1": f1[1],
                    "macro_f1": macro_f1
                }

                config_results.append(result)
                all_results.append(result)

            best_unrestricted = max(
                config_results,
                key=lambda item: item["macro_f1"]
            )

            eligible_results = [
                item for item in config_results
                if item["real_recall"] >= 0.90
            ]

            if eligible_results:
                best_selected = max(
                    eligible_results,
                    key=lambda item: item["macro_f1"]
                )

                print(
                    f"Best with Real recall >= 90%:\n"
                    f"Threshold: {best_selected['threshold']:.5f}\n"
                    f"Accuracy: {best_selected['accuracy']:.4f}\n"
                    f"Fake Recall: {best_selected['fake_recall']:.4f}\n"
                    f"Real Recall: {best_selected['real_recall']:.4f}\n"
                    f"Macro F1: {best_selected['macro_f1']:.4f}"
                )
            else:
                print("No threshold achieved Real recall >= 90%.")

            print(
                f"Best unrestricted:\n"
                f"Threshold: {best_unrestricted['threshold']:.5f}\n"
                f"Accuracy: {best_unrestricted['accuracy']:.4f}\n"
                f"Fake Recall: {best_unrestricted['fake_recall']:.4f}\n"
                f"Real Recall: {best_unrestricted['real_recall']:.4f}\n"
                f"Macro F1: {best_unrestricted['macro_f1']:.4f}"
            )

# Save every threshold result
results_df = pd.DataFrame(all_results)

results_path = output_dir / "linear_svm_threshold_results.csv"
results_df.to_csv(results_path, index=False)

# Select the best result for each model configuration
eligible_df = results_df[
    results_df["real_recall"] >= 0.90
].copy()

if not eligible_df.empty:
    summary_df = (
        eligible_df.sort_values(
            "macro_f1",
            ascending=False
        )
        .groupby("configuration", as_index=False)
        .first()
        .sort_values("macro_f1", ascending=False)
    )

    summary_path = output_dir / "linear_svm_comparison.csv"
    summary_df.to_csv(summary_path, index=False)

    print(
        f"\nFINAL LINEAR SVM COMPARISON\n"
        f"---------------------------\n"
        f"{summary_df[[
            'configuration',
            'features',
            'C',
            'class_weight',
            'threshold',
            'accuracy',
            'fake_recall',
            'real_recall',
            'macro_f1'
        ]].to_string(index=False)}"
    )

    best = summary_df.iloc[0]

    print(
        f"\nBEST SVM CONFIGURATION: {best['configuration']}\n"
        f"Threshold: {best['threshold']:.5f}\n"
        f"Accuracy: {best['accuracy']:.4f}\n"
        f"Fake Recall: {best['fake_recall']:.4f}\n"
        f"Real Recall: {best['real_recall']:.4f}\n"
        f"Macro F1: {best['macro_f1']:.4f}"
    )
else:
    print("\nNo SVM configuration satisfied Real recall >= 90%.")

print(
    f"\nSaved threshold results: {results_path}\n"
    f"Test set remained unused."
)