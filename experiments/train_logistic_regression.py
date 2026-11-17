import numpy as np
import joblib
from scipy.sparse import load_npz

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from pathlib import Path


# Load the same training and testing data used by Naive Bayes
X_train = load_npz("outputs/features/X_train.npz")
X_test = load_npz("outputs/features/X_test.npz")

y_train = np.load("outputs/features/y_train.npy")
y_test = np.load("outputs/features/y_test.npy")


# Create and train the Logistic Regression model
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)


# Predict labels for testing data
y_pred = model.predict(X_test)


# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")


# Display results
print("\nLOGISTIC REGRESSION MODEL RESULTS")
print("---------------------------------")
print(f"Training records: {X_train.shape[0]}")
print(f"Testing records: {X_test.shape[0]}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro Precision: {precision:.4f}")
print(f"Macro Recall: {recall:.4f}")
print(f"Macro F1-score: {f1:.4f}")

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Fake", "Real"]
))

print("CONFUSION MATRIX")
print("Rows = Actual, Columns = Predicted")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))


# Save the trained model
output_path = Path("outputs/models")
output_path.mkdir(parents=True, exist_ok=True)

joblib.dump(
    model,
    output_path / "logistic_regression_model.joblib"
)

print("\nTrained model saved in outputs/models/")