import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz


def prepare_validation_features(input_path, output_dir):
    # Load cleaned dataset
    df = pd.read_csv(input_path)

    # Validate required columns
    if "title" not in df.columns or "real" not in df.columns:
        raise ValueError("Dataset must contain title and real columns.")

    if df["title"].isnull().any() or df["real"].isnull().any():
        raise ValueError("Missing headlines or labels found.")

    if not set(df["real"].unique()).issubset({0, 1}):
        raise ValueError("Unexpected label values found.")

    # Separate headlines and labels
    X = df["title"].astype(str)
    y = df["real"].astype(int)

    # First split: preserve the original 80/20 split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Second split: divide original training data into 80/20
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.20,
        random_state=42,
        stratify=y_train_full
    )

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    # Learn vocabulary and weights from training data only
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # Transform validation and test data
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save feature matrices
    save_npz(output_dir / "X_train.npz", X_train_tfidf)
    save_npz(output_dir / "X_val.npz", X_val_tfidf)
    save_npz(output_dir / "X_test.npz", X_test_tfidf)

    # Save labels
    np.save(output_dir / "y_train.npy", y_train.to_numpy())
    np.save(output_dir / "y_val.npy", y_val.to_numpy())
    np.save(output_dir / "y_test.npy", y_test.to_numpy())

    # Save fitted vectorizer
    joblib.dump(vectorizer, output_dir / "tfidf_vectorizer.joblib")

    # Calculate class distributions
    train_counts = y_train.value_counts()
    val_counts = y_val.value_counts()
    test_counts = y_test.value_counts()

    print(
        f"VALIDATION DATA SPLIT\n"
        f"---------------------\n"
        f"Total records: {len(df)}\n"
        f"Training records: {len(X_train)}\n"
        f"Validation records: {len(X_val)}\n"
        f"Testing records: {len(X_test)}\n"
        f"\nTRAINING LABELS\n"
        f"Fake: {train_counts.get(0, 0)}\n"
        f"Real: {train_counts.get(1, 0)}\n"
        f"\nVALIDATION LABELS\n"
        f"Fake: {val_counts.get(0, 0)}\n"
        f"Real: {val_counts.get(1, 0)}\n"
        f"\nTESTING LABELS\n"
        f"Fake: {test_counts.get(0, 0)}\n"
        f"Real: {test_counts.get(1, 0)}\n"
        f"\nTF-IDF RESULTS\n"
        f"Maximum features: 5000\n"
        f"Actual features: {X_train_tfidf.shape[1]}\n"
        f"Training matrix: {X_train_tfidf.shape}\n"
        f"Validation matrix: {X_val_tfidf.shape}\n"
        f"Testing matrix: {X_test_tfidf.shape}\n"
        f"\nSaved files in: {output_dir}"
    )