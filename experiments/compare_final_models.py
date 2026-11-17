import os
import time
import numpy as np
import pandas as pd
import joblib

from scipy.sparse import load_npz
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

FEATURE_DIR = "outputs/final_features"
MODEL_DIR = "outputs/models"
RESULT_DIR = "outputs/results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

print("FINAL BASELINE MODEL COMPARISON")
print("=" * 65)

# 1. Load prepared features and labels
X_train = load_npz(os.path.join(FEATURE_DIR, "X_train_tfidf.npz"))
X_validation = load_npz(
    os.path.join(FEATURE_DIR, "X_validation_tfidf.npz")
)

y_train = np.load(os.path.join(FEATURE_DIR, "y_train.npy"))
y_validation = np.load(
    os.path.join(FEATURE_DIR, "y_validation.npy")
)

print(f"Training records: {X_train.shape[0]}")
print(f"Validation records: {X_validation.shape[0]}")
print(f"TF-IDF features: {X_train.shape[1]}")

# 2. Verify feature and label dimensions
if X_train.shape[0] != len(y_train):
    raise ValueError("Training feature and label counts differ.")

if X_validation.shape[0] != len(y_validation):
    raise ValueError("Validation feature and label counts differ.")

if X_train.shape[1] != X_validation.shape[1]:
    raise ValueError("Training and validation feature counts differ.")

if not np.isin(y_train, [0, 1]).all():
    raise ValueError("Invalid training labels.")

if not np.isin(y_validation, [0, 1]).all():
    raise ValueError("Invalid validation labels.")

# 3. Define both baseline models
models = {
    "Naive Bayes": MultinomialNB(alpha=1.0),
    "Logistic Regression": LogisticRegression(
        C=1.0,
        max_iter=2000,
        random_state=42
    )
}

comparison_results = []
class_results = []

# 4. Train and evaluate each model
for model_name, model in models.items():
    print(f"\n{model_name.upper()}")
    print("-" * 65)

    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start_time

    predictions = model.predict(X_validation)

    accuracy = accuracy_score(y_validation, predictions)
    macro_precision = precision_score(
        y_validation, predictions, average="macro", zero_division=0
    )
    macro_recall = recall_score(
        y_validation, predictions, average="macro", zero_division=0
    )
    macro_f1 = f1_score(
        y_validation, predictions, average="macro", zero_division=0
    )

    # Class 0 = Fake, Class 1 = Real
    report = classification_report(
        y_validation,
        predictions,
        labels=[0, 1],
        target_names=["Fake", "Real"],
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_validation,
        predictions,
        labels=[0, 1]
    )

    comparison_results.append({
        "model": model_name,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "training_time_seconds": training_time
    })

    for class_name in ["Fake", "Real"]:
        class_results.append({
            "model": model_name,
            "class": class_name,
            "precision": report[class_name]["precision"],
            "recall": report[class_name]["recall"],
            "f1_score": report[class_name]["f1-score"],
            "support": report[class_name]["support"]
        })

    # Save each trained model separately
    model_filename = (
        "merged_naive_bayes.joblib"
        if model_name == "Naive Bayes"
        else "merged_logistic_regression.joblib"
    )

    joblib.dump(
        model,
        os.path.join(MODEL_DIR, model_filename)
    )

    # Save predictions for later analysis
    prediction_filename = (
        "merged_naive_bayes_validation_predictions.csv"
        if model_name == "Naive Bayes"
        else "merged_logistic_regression_validation_predictions.csv"
    )

    pd.DataFrame({
        "actual": y_validation,
        "predicted": predictions
    }).to_csv(
        os.path.join(RESULT_DIR, prediction_filename),
        index=False
    )

    # Save confusion matrix
    matrix_filename = (
        "merged_naive_bayes_confusion_matrix.csv"
        if model_name == "Naive Bayes"
        else "merged_logistic_regression_confusion_matrix.csv"
    )

    pd.DataFrame(
        matrix,
        index=["Actual Fake", "Actual Real"],
        columns=["Predicted Fake", "Predicted Real"]
    ).to_csv(os.path.join(RESULT_DIR, matrix_filename))

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Training time: {training_time:.2f} seconds")

    print("\nCLASSIFICATION REPORT")
    print(
        classification_report(
            y_validation,
            predictions,
            labels=[0, 1],
            target_names=["Fake", "Real"],
            zero_division=0
        )
    )

    print("CONFUSION MATRIX")
    print("Rows = Actual, Columns = Predicted")
    print(matrix)

# 5. Save combined comparison results
comparison_df = pd.DataFrame(comparison_results)
class_df = pd.DataFrame(class_results)

comparison_df.to_csv(
    os.path.join(RESULT_DIR, "merged_model_comparison.csv"),
    index=False
)

class_df.to_csv(
    os.path.join(RESULT_DIR, "merged_class_metrics.csv"),
    index=False
)

# 6. Identify the better baseline using Macro F1
best_model = comparison_df.loc[
    comparison_df["macro_f1"].idxmax()
]

print("\nFINAL BASELINE COMPARISON")
print("=" * 65)
print(comparison_df.to_string(index=False))

print("\nBEST BASELINE BY VALIDATION MACRO F1")
print("-" * 65)
print(f"Model: {best_model['model']}")
print(f"Macro F1: {best_model['macro_f1']:.4f}")

print("\nSAVED RESULTS")
print("-" * 65)
print(os.path.join(RESULT_DIR, "merged_model_comparison.csv"))
print(os.path.join(RESULT_DIR, "merged_class_metrics.csv"))

print("\nBASELINE MODEL COMPARISON COMPLETED")