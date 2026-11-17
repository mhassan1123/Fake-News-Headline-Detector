import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load the cleaned dataset
df = pd.read_csv("data/processed/cleaned_fakenewsnet.csv")

X = df["title"].astype(str)
y = df["real"].astype(int)

# Create the same training, validation and test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)

# Define combinations to test
feature_sizes = [5000, 10000]
class_weights = [None, "balanced"]
c_values = [0.5, 1.0, 5.0]
thresholds = np.arange(0.20, 0.801, 0.025)

all_results = []

# Train and evaluate every combination
for max_features in feature_sizes:

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2)
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    for class_weight in class_weights:

        for c_value in c_values:

            model = LogisticRegression(
                C=c_value,
                class_weight=class_weight,
                max_iter=1000,
                random_state=42
            )

            model.fit(X_train_tfidf, y_train)

            probabilities = model.predict_proba(X_val_tfidf)
            fake_index = list(model.classes_).index(0)
            fake_probabilities = probabilities[:, fake_index]

            weight_name = (
                "Balanced" if class_weight == "balanced"
                else "Standard"
            )

            configuration = (
                f"{max_features}_Features_"
                f"{weight_name}_C{c_value}"
            )

            # Evaluate different Fake probability thresholds
            for threshold in thresholds:

                predictions = np.where(
                    fake_probabilities >= threshold,
                    0,
                    1
                )

                fake_precision, real_precision = precision_score(
                    y_val,
                    predictions,
                    labels=[0, 1],
                    average=None,
                    zero_division=0
                )

                fake_recall, real_recall = recall_score(
                    y_val,
                    predictions,
                    labels=[0, 1],
                    average=None,
                    zero_division=0
                )

                fake_f1, real_f1 = f1_score(
                    y_val,
                    predictions,
                    labels=[0, 1],
                    average=None,
                    zero_division=0
                )

                all_results.append({
                    "Configuration": configuration,
                    "Features": max_features,
                    "Class Weight": weight_name,
                    "C": c_value,
                    "Threshold": round(float(threshold), 3),
                    "Accuracy": accuracy_score(y_val, predictions),
                    "Fake Precision": fake_precision,
                    "Fake Recall": fake_recall,
                    "Fake F1": fake_f1,
                    "Real Precision": real_precision,
                    "Real Recall": real_recall,
                    "Real F1": real_f1,
                    "Macro F1": f1_score(
                        y_val,
                        predictions,
                        average="macro",
                        zero_division=0
                    )
                })

            print(
                f"Completed: {configuration}"
            )

# Convert results into a DataFrame
results_df = pd.DataFrame(all_results)

# Keep only configurations with at least 90% Real recall
eligible_results = results_df[
    results_df["Real Recall"] >= 0.90
].copy()

# Sort eligible configurations by Macro F1
eligible_results = eligible_results.sort_values(
    by="Macro F1",
    ascending=False
)

# Find the best threshold for each training configuration
best_per_configuration = (
    eligible_results
    .sort_values("Macro F1", ascending=False)
    .groupby("Configuration")
    .head(1)
    .sort_values("Macro F1", ascending=False)
)

# Save complete results and summaries
results_dir = Path("outputs/results")
results_dir.mkdir(parents=True, exist_ok=True)

all_path = results_dir / "combined_strategy_all_results.csv"
best_path = results_dir / "combined_strategy_best_results.csv"

results_df.to_csv(all_path, index=False)
best_per_configuration.to_csv(best_path, index=False)

# Display the best configurations
print(
    f"\nBEST CONFIGURATION FOR EACH MODEL SETUP\n"
    f"---------------------------------------\n"
    f"{best_per_configuration.to_string(index=False)}"
)

print(
    f"\nTOP 10 OVERALL CANDIDATES "
    f"(REAL RECALL >= 90%)\n"
    f"---------------------------------------\n"
    f"{eligible_results.head(10).to_string(index=False)}"
)

print(
    f"\nTotal training configurations: "
    f"{len(feature_sizes) * len(class_weights) * len(c_values)}\n"
    f"Total threshold evaluations: {len(results_df)}\n"
    f"Saved all results: {all_path}\n"
    f"Saved best results: {best_path}"
)