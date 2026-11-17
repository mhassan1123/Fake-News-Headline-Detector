import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load original cleaned headlines
df = pd.read_csv("data/processed/cleaned_fakenewsnet.csv")

X = df["title"].astype(str)
y = df["real"].astype(int)

# Recreate the same training and validation split
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(
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

# Define TF-IDF experiments
configurations = {
    "Baseline": {
        "max_features": 5000,
        "ngram_range": (1, 2)
    },
    "Larger Vocabulary": {
        "max_features": 10000,
        "ngram_range": (1, 2)
    },
    "Unigrams Only": {
        "max_features": 5000,
        "ngram_range": (1, 1)
    },
    "Sublinear TF": {
        "max_features": 5000,
        "ngram_range": (1, 2),
        "sublinear_tf": True
    }
}

results = []

feature_dir = Path("outputs/tfidf_experiments")
feature_dir.mkdir(parents=True, exist_ok=True)

model_dir = Path("outputs/models")
model_dir.mkdir(parents=True, exist_ok=True)

# Train and evaluate each feature configuration
for name, settings in configurations.items():

    vectorizer = TfidfVectorizer(**settings)

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    model = LogisticRegression(
        C=5.0,
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_val_tfidf)

    fake_precision, real_precision = precision_score(
        y_val, predictions,
        labels=[0, 1],
        average=None,
        zero_division=0
    )

    fake_recall, real_recall = recall_score(
        y_val, predictions,
        labels=[0, 1],
        average=None,
        zero_division=0
    )

    fake_f1, real_f1 = f1_score(
        y_val, predictions,
        labels=[0, 1],
        average=None,
        zero_division=0
    )

    accuracy = accuracy_score(y_val, predictions)

    macro_f1 = f1_score(
        y_val, predictions,
        average="macro",
        zero_division=0
    )

    results.append({
        "Configuration": name,
        "Features": X_train_tfidf.shape[1],
        "Accuracy": accuracy,
        "Fake Precision": fake_precision,
        "Fake Recall": fake_recall,
        "Fake F1": fake_f1,
        "Real Precision": real_precision,
        "Real Recall": real_recall,
        "Real F1": real_f1,
        "Macro F1": macro_f1
    })

    # Save experiment-specific files
    safe_name = name.lower().replace(" ", "_")

    joblib.dump(
        vectorizer,
        feature_dir / f"{safe_name}_vectorizer.joblib"
    )

    joblib.dump(
        model,
        model_dir / f"tfidf_{safe_name}_lr_C5.joblib"
    )

# Save comparison
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="Macro F1",
    ascending=False
)

results_dir = Path("outputs/results")
results_dir.mkdir(parents=True, exist_ok=True)

output_path = results_dir / "tfidf_feature_comparison.csv"
results_df.to_csv(output_path, index=False)

print(
    f"TF-IDF FEATURE EXPERIMENT RESULTS\n"
    f"--------------------------------\n"
    f"{results_df.to_string(index=False)}\n"
    f"\nSaved comparison: {output_path}"
)