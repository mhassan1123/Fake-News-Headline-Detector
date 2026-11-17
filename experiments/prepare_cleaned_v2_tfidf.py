from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[1]

SPLIT_DIR = ROOT / "data/splits_cleaned_v2"
OUTPUT_DIR = ROOT / "outputs/cleaned_v2/tfidf"

TRAIN_PATH = SPLIT_DIR / "train.csv"
VALIDATION_PATH = SPLIT_DIR / "validation.csv"
TEST_PATH = SPLIT_DIR / "test.csv"

VECTOR_PATH = OUTPUT_DIR / "tfidf_vectorizer.joblib"

TRAIN_FEATURES_PATH = OUTPUT_DIR / "X_train.npz"
VALIDATION_FEATURES_PATH = OUTPUT_DIR / "X_validation.npz"
TEST_FEATURES_PATH = OUTPUT_DIR / "X_test.npz"

SUMMARY_PATH = OUTPUT_DIR / "tfidf_summary.txt"
METADATA_PATH = OUTPUT_DIR / "tfidf_metadata.json"

MAX_FEATURES = 10000
NGRAM_RANGE = (1, 2)
MIN_DF = 2
MAX_DF = 0.95


def load_split(path, split_name):
    """Load a split and validate its required columns."""

    if not path.exists():
        raise FileNotFoundError(
            f"{split_name} file not found: {path}"
        )

    data = pd.read_csv(path)

    required_columns = {"title", "real"}

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"{split_name} is missing columns: {missing_columns}"
        )

    if data["title"].isna().any():
        raise ValueError(
            f"{split_name} contains missing headlines."
        )

    if data["real"].isna().any():
        raise ValueError(
            f"{split_name} contains missing labels."
        )

    return data


def count_zero_rows(matrix):
    """Count rows containing no non-zero TF-IDF features."""

    row_counts = np.asarray(
        matrix.getnnz(axis=1)
    ).ravel()

    return int((row_counts == 0).sum())


def run_tfidf():

    train = load_split(TRAIN_PATH, "Training")
    validation = load_split(VALIDATION_PATH, "Validation")
    test = load_split(TEST_PATH, "Testing")

    # Create the TF-IDF vectorizer using the previous configuration.
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_df=MAX_DF
    )

    # Learn vocabulary and IDF weights from training data only.
    X_train = vectorizer.fit_transform(train["title"])

    # Transform validation and test data using the fitted vectorizer.
    X_validation = vectorizer.transform(validation["title"])
    X_test = vectorizer.transform(test["title"])

    # Verify that all matrices use the same feature count.
    feature_count = len(vectorizer.get_feature_names_out())

    if (
        X_train.shape[1] != feature_count
        or X_validation.shape[1] != feature_count
        or X_test.shape[1] != feature_count
    ):
        raise ValueError(
            "TF-IDF feature dimensions do not match."
        )

    # Count headlines that produce no TF-IDF features.
    zero_train = count_zero_rows(X_train)
    zero_validation = count_zero_rows(X_validation)
    zero_test = count_zero_rows(X_test)

    # Create a separate output directory.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save the fitted vectorizer and sparse feature matrices.
    joblib.dump(vectorizer, VECTOR_PATH)

    save_npz(TRAIN_FEATURES_PATH, X_train)
    save_npz(VALIDATION_FEATURES_PATH, X_validation)
    save_npz(TEST_FEATURES_PATH, X_test)

    # Save experiment metadata.
    metadata = {
        "dataset": "final_headlines_cleaned_v2.csv",
        "training_records": len(train),
        "validation_records": len(validation),
        "test_records": len(test),
        "max_features": MAX_FEATURES,
        "actual_features": feature_count,
        "ngram_range": list(NGRAM_RANGE),
        "min_df": MIN_DF,
        "max_df": MAX_DF,
        "vectorizer_fitted_on": "training_only",
        "zero_feature_rows": {
            "training": zero_train,
            "validation": zero_validation,
            "test": zero_test
        }
    }

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8"
    )

    # Prepare a readable summary.
    summary_lines = [
        "CLEANED V2 TF-IDF FEATURE PREPARATION",
        "=" * 65,
        f"Training records: {len(train)}",
        f"Validation records: {len(validation)}",
        f"Testing records: {len(test)}",
        "",
        "TF-IDF CONFIGURATION",
        "-" * 65,
        f"Maximum features: {MAX_FEATURES}",
        f"Actual features: {feature_count}",
        f"N-gram range: {NGRAM_RANGE}",
        f"Minimum document frequency: {MIN_DF}",
        f"Maximum document frequency: {MAX_DF}",
        "Vectorizer fitted on: Training data only",
        "",
        "FEATURE MATRIX SHAPES",
        "-" * 65,
        f"Training: {X_train.shape}",
        f"Validation: {X_validation.shape}",
        f"Testing: {X_test.shape}",
        "",
        "ZERO-FEATURE HEADLINES",
        "-" * 65,
        f"Training: {zero_train}",
        f"Validation: {zero_validation}",
        f"Testing: {zero_test}",
        "",
        "OUTPUT FILES",
        "-" * 65,
        str(VECTOR_PATH),
        str(TRAIN_FEATURES_PATH),
        str(VALIDATION_FEATURES_PATH),
        str(TEST_FEATURES_PATH),
        str(METADATA_PATH),
        str(SUMMARY_PATH)
    ]

    summary_text = "\n".join(summary_lines)

    SUMMARY_PATH.write_text(
        summary_text,
        encoding="utf-8"
    )

    print(summary_text)


run_tfidf()