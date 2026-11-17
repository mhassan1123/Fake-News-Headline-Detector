import numpy as np
import joblib

from pathlib import Path
from scipy.sparse import load_npz
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# Load training and testing features
feature_dir = Path("outputs/features")

X_train = load_npz(feature_dir / "X_train.npz")
X_test = load_npz(feature_dir / "X_test.npz")

y_train = np.load(feature_dir / "y_train.npy")
y_test = np.load(feature_dir / "y_test.npy")


# Create and train Naive Bayes model
model = MultinomialNB(alpha=1.0)

model.fit(X_train, y_train)


# Make predictions on unseen testing data
y_pred = model.predict(X_test)


# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")

conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])


# Display results
print(
    f"NAIVE BAYES MODEL RESULTS\n"
    f"-------------------------\n"
    f"Training records: {len(y_train)}\n"
    f"Testing records: {len(y_test)}\n"
    f"Accuracy: {accuracy:.4f}\n"
    f"Macro Precision: {precision:.4f}\n"
    f"Macro Recall: {recall:.4f}\n"
    f"Macro F1-score: {f1:.4f}"
)

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Fake", "Real"],
        zero_division=0
    )
)

print("CONFUSION MATRIX")
print("Rows = Actual, Columns = Predicted")
print(conf_matrix)


# Save trained model
model_dir = Path("outputs/models")
model_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(model, model_dir / "naive_bayes_model.joblib")

print("\nTrained model saved in outputs/models/")