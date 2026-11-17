from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import load_npz

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


ROOT = Path(__file__).resolve().parents[1]

VALIDATION_PATH = ROOT / "data/splits_cleaned_v2/validation.csv"
TFIDF_DIR = ROOT / "outputs/cleaned_v2/tfidf"

MODEL_PATH = (
    ROOT
    / "outputs/cleaned_v2/reproducible_experiment/models/logistic_regression.joblib"
)

VECTORIZER_PATH = (
    ROOT
    / "outputs/cleaned_v2/reproducible_experiment/models/tfidf_vectorizer.joblib"
)

OUTPUT_DIR = ROOT / "outputs/cleaned_v2/models/final"
AUDIT_DIR = ROOT / "outputs/cleaned_v2/audits/final"

THRESHOLD = 0.477

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD VALIDATION DATA AND TRAINED MODEL
# --------------------------------------------------

validation = pd.read_csv(VALIDATION_PATH)

y_true = validation["real"].astype(int).to_numpy()

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

X_validation = load_npz(
    TFIDF_DIR / "X_validation.npz"
)

if X_validation.shape[0] != len(y_true):
    raise ValueError(
        "Validation features and labels have different record counts."
    )

if X_validation.shape[1] != len(vectorizer.get_feature_names_out()):
    raise ValueError(
        "Validation features do not match the saved TF-IDF vectorizer."
    )

if not np.array_equal(model.classes_, np.array([0, 1])):
    raise ValueError(
        "The model must contain both labels: Fake (0) and Real (1)."
    )


# --------------------------------------------------
# GENERATE VALIDATION PREDICTIONS
# --------------------------------------------------

fake_index = list(model.classes_).index(0)

fake_probability = model.predict_proba(
    X_validation
)[:, fake_index]

# Label 0 = Fake, Label 1 = Real
y_pred = (fake_probability < THRESHOLD).astype(int)


# --------------------------------------------------
# CALCULATE OVERALL METRICS
# --------------------------------------------------

accuracy = accuracy_score(y_true, y_pred)

fake_precision = precision_score(
    y_true, y_pred, pos_label=0, zero_division=0
)

fake_recall = recall_score(
    y_true, y_pred, pos_label=0, zero_division=0
)

real_precision = precision_score(
    y_true, y_pred, pos_label=1, zero_division=0
)

real_recall = recall_score(
    y_true, y_pred, pos_label=1, zero_division=0
)

macro_precision = precision_score(
    y_true, y_pred, average="macro", zero_division=0
)

macro_recall = recall_score(
    y_true, y_pred, average="macro", zero_division=0
)

macro_f1 = f1_score(
    y_true, y_pred, average="macro", zero_division=0
)

roc_auc = roc_auc_score(
    (y_true == 0).astype(int),
    fake_probability
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)


# --------------------------------------------------
# SAVE FINAL MODEL METADATA AND BUNDLE
# --------------------------------------------------

summary = {
    "dataset": "final_headlines_cleaned_v2",
    "model": "Logistic Regression",
    "C": 2.0,
    "class_weight": "None",
    "threshold": THRESHOLD,
    "tfidf_features": int(X_validation.shape[1]),
    "validation_records": int(len(validation)),
    "accuracy": float(accuracy),
    "fake_precision": float(fake_precision),
    "fake_recall": float(fake_recall),
    "real_precision": float(real_precision),
    "real_recall": float(real_recall),
    "macro_precision": float(macro_precision),
    "macro_recall": float(macro_recall),
    "macro_f1": float(macro_f1),
    "roc_auc": float(roc_auc),
    "fake_recall_target": 0.80,
    "real_recall_target": 0.97,
    "fake_recall_target_met": bool(fake_recall >= 0.80),
    "real_recall_target_met": bool(real_recall >= 0.97),
    "both_targets_met": bool(
        fake_recall >= 0.80 and real_recall >= 0.97
    ),
    "test_data_used": False,
    "label_mapping": {
        "0": "Fake",
        "1": "Real"
    }
}

joblib.dump(
    {
        "model": model,
        "vectorizer": vectorizer,
        "threshold": THRESHOLD,
        "fake_label": 0,
        "real_label": 1
    },
    OUTPUT_DIR / "final_model_bundle.joblib"
)

with open(
    OUTPUT_DIR / "final_model_metadata.json",
    "w"
) as file:
    json.dump(summary, file, indent=4)


# --------------------------------------------------
# SAVE OVERALL VALIDATION AUDIT
# --------------------------------------------------

pd.DataFrame([summary]).to_csv(
    AUDIT_DIR / "final_validation_summary.csv",
    index=False
)

pd.DataFrame(
    cm,
    index=["Actual Fake", "Actual Real"],
    columns=["Predicted Fake", "Predicted Real"]
).to_csv(
    AUDIT_DIR / "final_validation_confusion_matrix.csv"
)


# --------------------------------------------------
# SOURCE-WISE VALIDATION AUDIT
# --------------------------------------------------

source_audit = validation[["source_dataset"]].copy()

source_audit["actual"] = y_true
source_audit["predicted"] = y_pred
source_audit["fake_probability"] = fake_probability

source_results = []

for source, group in source_audit.groupby(
    "source_dataset",
    dropna=False
):
    source_y = group["actual"].to_numpy()
    source_pred = group["predicted"].to_numpy()
    source_probability = group["fake_probability"].to_numpy()

    source_auc = None

    if len(np.unique(source_y)) == 2:
        source_auc = float(
            roc_auc_score(
                (source_y == 0).astype(int),
                source_probability
            )
        )

    source_results.append({
        "source": str(source),
        "records": int(len(group)),
        "fake_recall": float(
            recall_score(
                source_y, source_pred,
                pos_label=0, zero_division=0
            )
        ),
        "real_recall": float(
            recall_score(
                source_y, source_pred,
                pos_label=1, zero_division=0
            )
        ),
        "macro_f1": float(
            f1_score(
                source_y, source_pred,
                average="macro", zero_division=0
            )
        ),
        "roc_auc": source_auc
    })

pd.DataFrame(source_results).to_csv(
    AUDIT_DIR / "final_source_validation_summary.csv",
    index=False
)


# --------------------------------------------------
# DISPLAY FINAL RESULTS
# --------------------------------------------------

print(
    "\nCLEANED-V2 FINAL MODEL SELECTION\n"
    "============================================================\n"
    "Selected model: Logistic Regression (C=2)\n"
    f"Threshold: {THRESHOLD}\n"
    f"Validation records: {len(validation)}\n"
    f"Accuracy: {accuracy:.4f}\n"
    f"Fake Precision: {fake_precision:.4f}\n"
    f"Fake Recall: {fake_recall:.4f}\n"
    f"Real Precision: {real_precision:.4f}\n"
    f"Real Recall: {real_recall:.4f}\n"
    f"Macro F1: {macro_f1:.4f}\n"
    f"ROC-AUC: {roc_auc:.4f}\n"
    f"Fake Recall target met: {summary['fake_recall_target_met']}\n"
    f"Real Recall target met: {summary['real_recall_target_met']}\n"
    f"Both targets met: {summary['both_targets_met']}\n"
    "Test data used: No\n"
    "\nFinal model bundle and validation audits saved."
)