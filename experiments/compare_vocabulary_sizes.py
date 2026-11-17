import os
import time
import numpy as np
import pandas as pd
import joblib

from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

TRAIN_PATH = "data/splits/train.csv"
VALIDATION_PATH = "data/splits/validation.csv"

OUTPUT_DIR = "outputs/vocabulary_experiments"
MODEL_DIR = "outputs/models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print("TF-IDF VOCABULARY SIZE COMPARISON")
print("=" * 70)

# 1. Load the same training and validation splits
train = pd.read_csv(TRAIN_PATH)
validation = pd.read_csv(VALIDATION_PATH)

X_train = train["title"].astype(str)
X_validation = validation["title"].astype(str)

y_train = train["real"].to_numpy(dtype=np.int64)
y_validation = validation["real"].to_numpy(dtype=np.int64)

if len(X_train) != len(y_train):
    raise ValueError("Training data dimensions do not match.")

if len(X_validation) != len(y_validation):
    raise ValueError("Validation data dimensions do not match.")

# 2. Define vocabulary sizes and model configurations
vocabulary_sizes = [10000, 15000, 20000]

model_configs = {
    "Naive Bayes": lambda: MultinomialNB(alpha=1.0),
    "Logistic Regression": lambda: LogisticRegression(
        C=1.0,
        max_iter=2000,
        random_state=42
    )
}

results = []
class_results = []

# 3. Compare all vocabulary sizes
for vocabulary_size in vocabulary_sizes:
    print(f"\nVOCABULARY SIZE: {vocabulary_size}")
    print("-" * 70)

    vectorizer = TfidfVectorizer(
        max_features=vocabulary_size,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

    # Fit only on training data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_validation_tfidf = vectorizer.transform(X_validation)

    actual_features = len(vectorizer.get_feature_names_out())

    if X_train_tfidf.shape[1] != X_validation_tfidf.shape[1]:
        raise ValueError("Feature dimensions do not match.")

    print(f"Requested features: {vocabulary_size}")
    print(f"Actual vocabulary size: {actual_features}")

    # Save vectorizer and feature matrices separately
    vectorizer_path = os.path.join(
        MODEL_DIR,
        f"tfidf_vectorizer_{vocabulary_size}.joblib"
    )

    joblib.dump(vectorizer, vectorizer_path)

    feature_dir = os.path.join(
        OUTPUT_DIR,
        f"{vocabulary_size}_features"
    )

    os.makedirs(feature_dir, exist_ok=True)

    save_npz(
        os.path.join(feature_dir, "X_train.npz"),
        X_train_tfidf
    )

    save_npz(
        os.path.join(feature_dir, "X_validation.npz"),
        X_validation_tfidf
    )

    # 4. Train and evaluate both models
    for model_name, model_factory in model_configs.items():
        model = model_factory()

        start_time = time.perf_counter()
        model.fit(X_train_tfidf, y_train)
        training_time = time.perf_counter() - start_time

        predictions = model.predict(X_validation_tfidf)

        accuracy = accuracy_score(y_validation, predictions)

        macro_precision = precision_score(
            y_validation,
            predictions,
            average="macro",
            zero_division=0
        )

        macro_recall = recall_score(
            y_validation,
            predictions,
            average="macro",
            zero_division=0
        )

        macro_f1 = f1_score(
            y_validation,
            predictions,
            average="macro",
            zero_division=0
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

        results.append({
            "vocabulary_size": vocabulary_size,
            "actual_features": actual_features,
            "model": model_name,
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "fake_precision": fake_precision,
            "fake_recall": fake_recall,
            "fake_f1": fake_f1,
            "real_precision": real_precision,
            "real_recall": real_recall,
            "real_f1": real_f1,
            "training_time_seconds": training_time
        })

        # Save the model without overwriting previous baseline files
        model_name_file = (
            "naive_bayes"
            if model_name == "Naive Bayes"
            else "logistic_regression"
        )

        model_path = os.path.join(
            MODEL_DIR,
            f"{model_name_file}_{vocabulary_size}.joblib"
        )

        joblib.dump(model, model_path)

        class_results.append({
            "vocabulary_size": vocabulary_size,
            "model": model_name,
            "class": "Fake",
            "precision": fake_precision,
            "recall": fake_recall,
            "f1": fake_f1
        })

        class_results.append({
            "vocabulary_size": vocabulary_size,
            "model": model_name,
            "class": "Real",
            "precision": real_precision,
            "recall": real_recall,
            "f1": real_f1
        })

        print(
            f"{model_name}: "
            f"Accuracy={accuracy:.4f}, "
            f"Macro F1={macro_f1:.4f}, "
            f"Fake Recall={fake_recall:.4f}, "
            f"Real Recall={real_recall:.4f}"
        )

# 5. Save comparison results
results_df = pd.DataFrame(results)
class_df = pd.DataFrame(class_results)

results_path = os.path.join(
    OUTPUT_DIR,
    "vocabulary_comparison.csv"
)

class_path = os.path.join(
    OUTPUT_DIR,
    "vocabulary_class_metrics.csv"
)

results_df.to_csv(results_path, index=False)
class_df.to_csv(class_path, index=False)

# 6. Identify the best configuration by validation Macro F1
best_result = results_df.loc[
    results_df["macro_f1"].idxmax()
]

print("\nCOMPLETE VOCABULARY COMPARISON")
print("=" * 70)

print(
    results_df[
        [
            "vocabulary_size",
            "model",
            "accuracy",
            "macro_f1",
            "fake_recall",
            "real_recall",
            "training_time_seconds"
        ]
    ].to_string(index=False)
)

print("\nBEST CONFIGURATION BY VALIDATION MACRO F1")
print("-" * 70)
print(f"Vocabulary size: {best_result['vocabulary_size']}")
print(f"Model: {best_result['model']}")
print(f"Macro F1: {best_result['macro_f1']:.4f}")
print(f"Fake Recall: {best_result['fake_recall']:.4f}")
print(f"Real Recall: {best_result['real_recall']:.4f}")

print("\nSAVED RESULTS")
print("-" * 70)
print(results_path)
print(class_path)

print("\nVOCABULARY COMPARISON COMPLETED")