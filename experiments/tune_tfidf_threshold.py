import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load the dataset
df = pd.read_csv("data/processed/cleaned_fakenewsnet.csv")

X = df["title"].astype(str)
y = df["real"].astype(int)

# Recreate the same training and validation split
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

# Load the selected TF-IDF vectorizer and model
vectorizer = joblib.load(
    "outputs/tfidf_experiments/larger_vocabulary_vectorizer.joblib"
)

model = joblib.load(
    "outputs/models/tfidf_larger_vocabulary_lr_C5.joblib"
)

# Transform validation headlines
X_val_tfidf = vectorizer.transform(X_val)

# Get probabilities for the Fake class (label 0)
probabilities = model.predict_proba(X_val_tfidf)
fake_index = list(model.classes_).index(0)
fake_probabilities = probabilities[:, fake_index]

results = []

# Test different Fake probability thresholds
for threshold in np.arange(0.20, 0.801, 0.025):

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

    results.append({
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

# Sort by Macro F1
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Macro F1",
    ascending=False
)

# Show all results
print(
    f"10,000-FEATURE MODEL: THRESHOLD RESULTS\n"
    f"---------------------------------------\n"
    f"{results_df.to_string(index=False)}"
)

# Show configurations maintaining at least 90% Real recall
balanced_results = results_df[
    results_df["Real Recall"] >= 0.90
]

print(
    f"\nCONFIGURATIONS WITH AT LEAST 90% REAL RECALL\n"
    f"--------------------------------------------\n"
    f"{balanced_results.to_string(index=False)}"
)

# Save results
output_path = Path("outputs/results/tfidf_threshold_comparison.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

results_df.to_csv(output_path, index=False)

print(f"\nSaved comparison: {output_path}")