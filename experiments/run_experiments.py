from pathlib import Path
import json
import joblib
import pandas as pd

from scipy.sparse import load_npz

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data/splits_cleaned_v2/train.csv"
VALIDATION_PATH = ROOT / "data/splits_cleaned_v2/validation.csv"

TFIDF_DIR = ROOT / "outputs/cleaned_v2/tfidf"
OUTPUT_DIR = ROOT / "outputs/cleaned_v2/reproducible_experiment"

MODEL_DIR = OUTPUT_DIR / "models"
RESULT_DIR = OUTPUT_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
LR_THRESHOLD = 0.496
DEFAULT_THRESHOLD = 0.500


# --------------------------------------------------
# LOAD TRAINING AND VALIDATION DATA
# --------------------------------------------------

train = pd.read_csv(TRAIN_PATH)
validation = pd.read_csv(VALIDATION_PATH)

y_train = train["real"].astype(int)
y_validation = validation["real"].astype(int)

if set(y_train.unique()) != {0, 1}:
    raise ValueError("Training data must contain both labels: 0 and 1.")

if set(y_validation.unique()) != {0, 1}:
    raise ValueError("Validation data must contain both labels: 0 and 1.")


# --------------------------------------------------
# LOAD PREPARED TF-IDF FEATURES
# --------------------------------------------------

vectorizer = joblib.load(
    TFIDF_DIR / "tfidf_vectorizer.joblib"
)

X_train = load_npz(
    TFIDF_DIR / "X_train.npz"
)

X_validation = load_npz(
    TFIDF_DIR / "X_validation.npz"
)

if X_train.shape[0] != len(y_train):
    raise ValueError("Training features and labels have different record counts.")

if X_validation.shape[0] != len(y_validation):
    raise ValueError("Validation features and labels have different record counts.")

if X_train.shape[1] != X_validation.shape[1]:
    raise ValueError("Training and validation feature counts do not match.")

if X_train.shape[1] != len(vectorizer.get_feature_names_out()):
    raise ValueError("TF-IDF features do not match the loaded vectorizer.")


# --------------------------------------------------
# CREATE MODELS
# --------------------------------------------------

models = {
    "naive_bayes": MultinomialNB(alpha=1.0),

    "logistic_regression": LogisticRegression(
        C=2.0,
        max_iter=2000,
        random_state=RANDOM_STATE
    )
}


# --------------------------------------------------
# TRAIN AND EVALUATE MODELS
# --------------------------------------------------

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    fake_index = list(model.classes_).index(0)

    fake_probability = model.predict_proba(X_validation)[:, fake_index]

    thresholds = [DEFAULT_THRESHOLD]

    if name == "logistic_regression":
        thresholds.append(LR_THRESHOLD)

    for threshold in thresholds:

        # Label 0 = Fake, Label 1 = Real
        y_pred = (fake_probability < threshold).astype(int)

        metrics = {
            "model": name,
            "threshold": threshold,

            "accuracy": accuracy_score(
                y_validation, y_pred
            ),

            "fake_precision": precision_score(
                y_validation,
                y_pred,
                pos_label=0,
                zero_division=0
            ),

            "fake_recall": recall_score(
                y_validation,
                y_pred,
                pos_label=0,
                zero_division=0
            ),

            "real_precision": precision_score(
                y_validation,
                y_pred,
                pos_label=1,
                zero_division=0
            ),

            "real_recall": recall_score(
                y_validation,
                y_pred,
                pos_label=1,
                zero_division=0
            ),

            "macro_precision": precision_score(
                y_validation,
                y_pred,
                average="macro",
                zero_division=0
            ),

            "macro_recall": recall_score(
                y_validation,
                y_pred,
                average="macro",
                zero_division=0
            ),

            "macro_f1": f1_score(
                y_validation,
                y_pred,
                average="macro",
                zero_division=0
            ),

            "roc_auc": roc_auc_score(
                (y_validation == 0).astype(int),
                fake_probability
            )
        }

        results.append(metrics)

        matrix = confusion_matrix(
            y_validation,
            y_pred,
            labels=[0, 1]
        )

        matrix_df = pd.DataFrame(
            matrix,
            index=["Actual Fake", "Actual Real"],
            columns=["Predicted Fake", "Predicted Real"]
        )

        matrix_df.to_csv(
            RESULT_DIR /
            f"{name}_threshold_{threshold:.3f}_confusion_matrix.csv"
        )

    # Save the trained model
    joblib.dump(
        model,
        MODEL_DIR / f"{name}.joblib"
    )


# --------------------------------------------------
# SAVE VECTORIZER
# --------------------------------------------------

joblib.dump(
    vectorizer,
    MODEL_DIR / "tfidf_vectorizer.joblib"
)


# --------------------------------------------------
# SAVE EXPERIMENT RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    RESULT_DIR / "model_comparison_metrics.csv",
    index=False
)


# --------------------------------------------------
# SAVE EXPERIMENT METADATA
# --------------------------------------------------

metadata = {
    "random_state": RANDOM_STATE,
    "training_records": len(train),
    "validation_records": len(validation),
    "tfidf_features": X_train.shape[1],
    "test_data_used": False,

    "label_mapping": {
        "0": "Fake",
        "1": "Real"
    },

    "models": {
        "naive_bayes": {
            "alpha": 1.0
        },

        "logistic_regression": {
            "C": 2.0,
            "max_iter": 2000,
            "random_state": RANDOM_STATE
        }
    },

    "thresholds": {
        "naive_bayes": [DEFAULT_THRESHOLD],
        "logistic_regression": [
            DEFAULT_THRESHOLD,
            LR_THRESHOLD
        ]
    }
}

with open(
    OUTPUT_DIR / "experiment_metadata.json",
    "w"
) as file:
    json.dump(metadata, file, indent=4)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print(
    "\nFAIR MODEL COMPARISON\n"
    "=================================================================\n"
    f"Training records: {len(train)}\n"
    f"Validation records: {len(validation)}\n"
    f"TF-IDF features: {X_train.shape[1]}\n"
    "Test data used: No\n"
    "\nMODEL RESULTS\n"
    "-----------------------------------------------------------------"
)

print(
    results_df[
        [
            "model",
            "threshold",
            "accuracy",
            "fake_recall",
            "real_recall",
            "macro_f1",
            "roc_auc"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print(
    "\nResults saved in outputs/cleaned_v2/reproducible_experiment/\n"
    "EXPERIMENT COMPLETED"
)