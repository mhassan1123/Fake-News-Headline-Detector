import os
import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

TRAIN_PATH = "data/splits/train.csv"
VALIDATION_PATH = "data/splits/validation.csv"

MODEL_DIR = "outputs/models/tfidf_experiments"
RESULT_DIR = "outputs/tfidf_experiments"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

print("TF-IDF FEATURE EXPERIMENT")
print("=" * 70)

# 1. Load the existing training and validation splits
train_df = pd.read_csv(TRAIN_PATH)
validation_df = pd.read_csv(VALIDATION_PATH)

X_train_text = train_df["title"].fillna("").astype(str)
y_train = train_df["real"].to_numpy()

X_validation_text = (
    validation_df["title"].fillna("").astype(str)
)

y_validation = validation_df["real"].to_numpy()

if len(X_train_text) != len(y_train):
    raise ValueError("Training dimensions do not match.")

if len(X_validation_text) != len(y_validation):
    raise ValueError("Validation dimensions do not match.")

# 2. Define TF-IDF configurations
feature_configs = {
    "baseline": {
        "max_features": 10000,
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.95
    },

    "unigram": {
        "max_features": 10000,
        "ngram_range": (1, 1),
        "min_df": 2,
        "max_df": 0.95
    },

    "trigram": {
        "max_features": 10000,
        "ngram_range": (1, 3),
        "min_df": 2,
        "max_df": 0.95
    },

    "sublinear": {
        "max_features": 10000,
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.95,
        "sublinear_tf": True
    },

    "min_df_3": {
        "max_features": 10000,
        "ngram_range": (1, 2),
        "min_df": 3,
        "max_df": 0.95
    },

    "max_df_90": {
        "max_features": 10000,
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.90
    },

    "character": {
        "analyzer": "char_wb",
        "max_features": 10000,
        "ngram_range": (3, 5),
        "min_df": 2,
        "max_df": 0.95
    }
}

class_weights = {
    "Standard": None,
    "Balanced": "balanced"
}

thresholds = np.round(
    np.arange(0.20, 0.801, 0.05),
    2
)

C_VALUE = 2.0

results = []

# 3. Train and evaluate every feature configuration
for feature_name, settings in feature_configs.items():

    print(f"\nFEATURE CONFIGURATION: {feature_name.upper()}")
    print("-" * 70)

    vectorizer = TfidfVectorizer(**settings)

    X_train = vectorizer.fit_transform(X_train_text)

    # Transform validation data without fitting again
    X_validation = vectorizer.transform(
        X_validation_text
    )

    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Training matrix: {X_train.shape}")
    print(f"Validation matrix: {X_validation.shape}")

    vectorizer_path = os.path.join(
        MODEL_DIR,
        f"tfidf_{feature_name}.joblib"
    )

    joblib.dump(vectorizer, vectorizer_path)

    # 4. Compare standard and balanced models
    for strategy, class_weight in class_weights.items():

        model = LogisticRegression(
            C=C_VALUE,
            class_weight=class_weight,
            max_iter=2000,
            random_state=42
        )

        model.fit(X_train, y_train)

        model_name = (
            f"lr_{feature_name}_{strategy.lower()}.joblib"
        )

        joblib.dump(
            model,
            os.path.join(MODEL_DIR, model_name)
        )

        fake_index = np.where(model.classes_ == 0)[0]

        if len(fake_index) != 1:
            raise ValueError("Fake class not found correctly.")

        fake_probability = model.predict_proba(
            X_validation
        )[:, fake_index[0]]

        # 5. Evaluate all thresholds
        for threshold in thresholds:

            predictions = np.where(
                fake_probability >= threshold,
                0,
                1
            )

            fake_precision = precision_score(
                y_validation,
                predictions,
                pos_label=0,
                zero_division=0
            )

            fake_recall = recall_score(
                y_validation,
                predictions,
                pos_label=0,
                zero_division=0
            )

            fake_f1 = f1_score(
                y_validation,
                predictions,
                pos_label=0,
                zero_division=0
            )

            real_precision = precision_score(
                y_validation,
                predictions,
                pos_label=1,
                zero_division=0
            )

            real_recall = recall_score(
                y_validation,
                predictions,
                pos_label=1,
                zero_division=0
            )

            real_f1 = f1_score(
                y_validation,
                predictions,
                pos_label=1,
                zero_division=0
            )

            accuracy = accuracy_score(
                y_validation,
                predictions
            )

            macro_f1 = f1_score(
                y_validation,
                predictions,
                average="macro",
                zero_division=0
            )

            results.append({
                "feature": feature_name,
                "strategy": strategy,
                "C": C_VALUE,
                "threshold": threshold,
                "vocabulary_size": len(vectorizer.vocabulary_),
                "accuracy": accuracy,
                "fake_precision": fake_precision,
                "fake_recall": fake_recall,
                "fake_f1": fake_f1,
                "real_precision": real_precision,
                "real_recall": real_recall,
                "real_f1": real_f1,
                "macro_f1": macro_f1
            })

# 6. Save experiment results
results_df = pd.DataFrame(results)

output_path = os.path.join(
    RESULT_DIR,
    "tfidf_feature_comparison.csv"
)

results_df.to_csv(output_path, index=False)

# 7. Find the best threshold for each feature/model
best_per_model = results_df.loc[
    results_df.groupby(
        ["feature", "strategy"]
    )["macro_f1"].idxmax()
].sort_values(
    "macro_f1",
    ascending=False
)

print("\nBEST THRESHOLD FOR EACH FEATURE CONFIGURATION")
print("=" * 70)

print(
    best_per_model[
        [
            "feature",
            "strategy",
            "threshold",
            "vocabulary_size",
            "accuracy",
            "fake_recall",
            "real_recall",
            "macro_f1"
        ]
    ].to_string(index=False)
)

# 8. Check whether both recall targets are achieved
eligible = results_df[
    (results_df["fake_recall"] >= 0.80) &
    (results_df["real_recall"] >= 0.97)
]

print("\nTARGET CHECK")
print("=" * 70)

if not eligible.empty:

    best = eligible.loc[
        eligible["macro_f1"].idxmax()
    ]

    print("Configurations meeting BOTH targets:")
    print(eligible.to_string(index=False))

    print("\nBEST ELIGIBLE CONFIGURATION")
    print(f"Feature: {best['feature']}")
    print(f"Strategy: {best['strategy']}")
    print(f"Threshold: {best['threshold']:.2f}")
    print(f"Accuracy: {best['accuracy']:.4f}")
    print(f"Fake Recall: {best['fake_recall']:.4f}")
    print(f"Real Recall: {best['real_recall']:.4f}")
    print(f"Macro F1: {best['macro_f1']:.4f}")

else:

    results_df["fake_shortfall"] = (
        0.80 - results_df["fake_recall"]
    ).clip(lower=0)

    results_df["real_shortfall"] = (
        0.97 - results_df["real_recall"]
    ).clip(lower=0)

    results_df["total_shortfall"] = (
        results_df["fake_shortfall"] +
        results_df["real_shortfall"]
    )

    closest = results_df.sort_values(
        ["total_shortfall", "macro_f1"],
        ascending=[True, False]
    ).head(10)

    print("No configuration meets both targets.")

    print("\nTEN CLOSEST CONFIGURATIONS")
    print(
        closest[
            [
                "feature",
                "strategy",
                "threshold",
                "accuracy",
                "fake_recall",
                "real_recall",
                "macro_f1",
                "total_shortfall"
            ]
        ].to_string(index=False)
    )

print("\nSAVED RESULTS")
print(output_path)

print("\nTF-IDF FEATURE EXPERIMENT COMPLETED")