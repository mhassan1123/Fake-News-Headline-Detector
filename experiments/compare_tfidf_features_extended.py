import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)


input_path = "data/processed/cleaned_fakenewsnet.csv"
output_dir = Path("outputs/results")

output_dir.mkdir(parents=True, exist_ok=True)

# Load the same cleaned dataset
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

# The test set is deliberately not used in this experiment.
print(
    f"FEATURE EXPERIMENT\n"
    f"------------------\n"
    f"Training records: {len(X_train)}\n"
    f"Validation records: {len(X_val)}\n"
    f"Test records reserved: {len(X_test)}\n"
    f"Test set will not be evaluated or used for selection."
)

# Define the feature configurations
configurations = [
    {
        "name": "Baseline_10000_Word",
        "analyzer": "word",
        "max_features": 10000,
        "ngram_range": (1, 2),
        "min_df": 1
    },
    {
        "name": "Word_15000_MinDF2",
        "analyzer": "word",
        "max_features": 15000,
        "ngram_range": (1, 2),
        "min_df": 2
    },
    {
        "name": "Word_20000_MinDF2",
        "analyzer": "word",
        "max_features": 20000,
        "ngram_range": (1, 2),
        "min_df": 2
    },
    {
        "name": "Word_15000_MinDF3",
        "analyzer": "word",
        "max_features": 15000,
        "ngram_range": (1, 2),
        "min_df": 3
    },
    {
        "name": "Character_20000",
        "analyzer": "char_wb",
        "max_features": 20000,
        "ngram_range": (3, 5),
        "min_df": 2
    }
]

thresholds = np.arange(0.20, 0.801, 0.025)

all_results = []
all_threshold_results = []

for config in configurations:
    print(
        f"\nRUNNING: {config['name']}\n"
        f"------------------------------"
    )

    # Create a fresh vectorizer for each configuration
    vectorizer = TfidfVectorizer(
        analyzer=config["analyzer"],
        max_features=config["max_features"],
        ngram_range=config["ngram_range"],
        min_df=config["min_df"]
    )

    # Fit only on training headlines
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # Transform validation headlines
    X_val_tfidf = vectorizer.transform(X_val)

    # Train the same model for a fair comparison
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train_tfidf, y_train)

    # Find the probability column for Fake (label 0)
    fake_index = list(model.classes_).index(0)
    fake_probabilities = model.predict_proba(X_val_tfidf)[:, fake_index]

    config_results = []

    for threshold in thresholds:
        # Predict Fake when its probability reaches the threshold
        predictions = np.ones(len(y_val), dtype=int)
        predictions[fake_probabilities >= threshold] = 0

        accuracy = accuracy_score(y_val, predictions)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val,
            predictions,
            labels=[0, 1],
            zero_division=0
        )

        macro_f1 = (f1[0] + f1[1]) / 2

        result = {
            "configuration": config["name"],
            "features": X_train_tfidf.shape[1],
            "threshold": round(float(threshold), 3),
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
        all_threshold_results.append(result)

    # Best result without a Real recall restriction
    best_unrestricted = max(
        config_results,
        key=lambda item: item["macro_f1"]
    )

    # Best result while maintaining at least 90% Real recall
    eligible_results = [
        item for item in config_results
        if item["real_recall"] >= 0.90
    ]

    if eligible_results:
        best_selected = max(
            eligible_results,
            key=lambda item: item["macro_f1"]
        )
    else:
        best_selected = None

    if best_selected is not None:
        all_results.append(best_selected)

    print(
        f"Actual TF-IDF features: {X_train_tfidf.shape[1]}\n"
        f"Training matrix: {X_train_tfidf.shape}\n"
        f"Validation matrix: {X_val_tfidf.shape}\n"
        f"\nBEST UNRESTRICTED RESULT\n"
        f"Threshold: {best_unrestricted['threshold']:.3f}\n"
        f"Accuracy: {best_unrestricted['accuracy']:.4f}\n"
        f"Fake Recall: {best_unrestricted['fake_recall']:.4f}\n"
        f"Real Recall: {best_unrestricted['real_recall']:.4f}\n"
        f"Macro F1: {best_unrestricted['macro_f1']:.4f}"
    )

    if best_selected is not None:
        print(
            f"\nBEST RESULT WITH REAL RECALL >= 90%\n"
            f"Threshold: {best_selected['threshold']:.3f}\n"
            f"Accuracy: {best_selected['accuracy']:.4f}\n"
            f"Fake Precision: {best_selected['fake_precision']:.4f}\n"
            f"Fake Recall: {best_selected['fake_recall']:.4f}\n"
            f"Fake F1: {best_selected['fake_f1']:.4f}\n"
            f"Real Precision: {best_selected['real_precision']:.4f}\n"
            f"Real Recall: {best_selected['real_recall']:.4f}\n"
            f"Real F1: {best_selected['real_f1']:.4f}\n"
            f"Macro F1: {best_selected['macro_f1']:.4f}"
        )
    else:
        print("\nNo threshold achieved the required 90% Real recall.")

# Save threshold-level results
threshold_df = pd.DataFrame(all_threshold_results)
threshold_path = output_dir / "extended_tfidf_threshold_results.csv"
threshold_df.to_csv(threshold_path, index=False)

# Save the best eligible result for each configuration
summary_df = pd.DataFrame(all_results)

if not summary_df.empty:
    summary_df = summary_df.sort_values(
        by="macro_f1",
        ascending=False
    )

summary_path = output_dir / "extended_tfidf_comparison.csv"
summary_df.to_csv(summary_path, index=False)

print(
    f"\nFINAL FEATURE COMPARISON\n"
    f"------------------------"
)

if not summary_df.empty:
    display_columns = [
        "configuration",
        "features",
        "threshold",
        "accuracy",
        "fake_recall",
        "real_recall",
        "macro_f1"
    ]

    print(summary_df[display_columns].to_string(index=False))

    best = summary_df.iloc[0]

    print(
        f"\nBEST CONFIGURATION: {best['configuration']}\n"
        f"Threshold: {best['threshold']:.3f}\n"
        f"Accuracy: {best['accuracy']:.4f}\n"
        f"Fake Recall: {best['fake_recall']:.4f}\n"
        f"Real Recall: {best['real_recall']:.4f}\n"
        f"Macro F1: {best['macro_f1']:.4f}"
    )
else:
    print("No configuration satisfied the Real recall requirement.")

print(
    f"\nSaved threshold results: {threshold_path}\n"
    f"Saved comparison results: {summary_path}\n"
    f"Test set remained unused."
)