import json
import joblib
import pandas as pd

from pathlib import Path
from scipy.sparse import load_npz
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# Existing cleaned-v2 final model and untouched test data
bundle_path = Path("outputs/cleaned_v2/models/final/final_model_bundle.joblib")
test_csv_path = Path("data/splits_cleaned_v2/test.csv")
test_features_path = Path("outputs/cleaned_v2/tfidf/X_test.npz")

# Load the selected model bundle
bundle = joblib.load(bundle_path)

model = bundle["model"]
vectorizer = bundle["vectorizer"]
threshold = bundle["threshold"]

# Load test data and precomputed TF-IDF features
test_df = pd.read_csv(test_csv_path)
X_test = load_npz(test_features_path)

y_test = test_df["real"].astype(int)

# Validate test data and feature dimensions
if len(y_test) != X_test.shape[0]:
    raise ValueError("Test labels and TF-IDF rows do not match.")

if X_test.shape[1] != len(vectorizer.get_feature_names_out()):
    raise ValueError("Test TF-IDF features do not match the saved vectorizer.")

if X_test.shape[1] != model.n_features_in_:
    raise ValueError("Test feature count does not match the final model.")

if set(y_test.unique()) != {0, 1}:
    raise ValueError("Test data must contain both Fake (0) and Real (1) labels.")

# Predict Fake probabilities using the existing model
fake_index = list(model.classes_).index(0)
fake_probabilities = model.predict_proba(X_test)[:, fake_index]

# Apply the previously selected threshold
# 0 = Fake, 1 = Real
predictions = (fake_probabilities < threshold).astype(int)

# Calculate final test metrics
accuracy = accuracy_score(y_test, predictions)

fake_precision, real_precision = precision_score(
    y_test, predictions,
    labels=[0, 1],
    average=None,
    zero_division=0
)

fake_recall, real_recall = recall_score(
    y_test, predictions,
    labels=[0, 1],
    average=None,
    zero_division=0
)

fake_f1, real_f1 = f1_score(
    y_test, predictions,
    labels=[0, 1],
    average=None,
    zero_division=0
)

macro_precision = precision_score(
    y_test, predictions,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    y_test, predictions,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_test, predictions,
    average="macro",
    zero_division=0
)

roc_auc = roc_auc_score(
    (y_test == 0).astype(int),
    fake_probabilities
)

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
)

# Save test results separately; do not overwrite the model
result_dir = Path("outputs/cleaned_v2/audits/test")
result_dir.mkdir(parents=True, exist_ok=True)

metrics = {
    "Model": "Logistic Regression",
    "C": 2.0,
    "Threshold": threshold,
    "Test Records": len(y_test),
    "Accuracy": accuracy,
    "Fake Precision": fake_precision,
    "Fake Recall": fake_recall,
    "Fake F1": fake_f1,
    "Real Precision": real_precision,
    "Real Recall": real_recall,
    "Real F1": real_f1,
    "Macro Precision": macro_precision,
    "Macro Recall": macro_recall,
    "Macro F1": macro_f1,
    "ROC-AUC": roc_auc
}

pd.DataFrame([metrics]).to_csv(
    result_dir / "final_test_metrics.csv",
    index=False
)

pd.DataFrame(
    matrix,
    index=["Actual Fake", "Actual Real"],
    columns=["Predicted Fake", "Predicted Real"]
).to_csv(result_dir / "final_test_confusion_matrix.csv")

with open(result_dir / "final_test_metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

report = classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Fake", "Real"],
    zero_division=0
)

with open(result_dir / "final_test_classification_report.txt", "w") as file:
    file.write(report)

print(
    f"CLEANED-V2 FINAL TEST RESULTS\n"
    f"=============================\n"
    f"Model: Logistic Regression (C=2.0)\n"
    f"Test records: {len(y_test)}\n"
    f"TF-IDF features: {X_test.shape[1]}\n"
    f"Threshold: {threshold}\n"
    f"Training or tuning performed: No\n"
    f"\nAccuracy: {accuracy:.4f}\n"
    f"Fake Precision: {fake_precision:.4f}\n"
    f"Fake Recall: {fake_recall:.4f}\n"
    f"Fake F1: {fake_f1:.4f}\n"
    f"Real Precision: {real_precision:.4f}\n"
    f"Real Recall: {real_recall:.4f}\n"
    f"Real F1: {real_f1:.4f}\n"
    f"Macro Precision: {macro_precision:.4f}\n"
    f"Macro Recall: {macro_recall:.4f}\n"
    f"Macro F1: {macro_f1:.4f}\n"
    f"ROC-AUC: {roc_auc:.4f}\n"
    f"\nCONFUSION MATRIX\n"
    f"{matrix}\n"
    f"\nCLASSIFICATION REPORT\n"
    f"{report}\n"
    f"Results saved in: {result_dir}"
)