import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz


def prepare_features(input_path, output_dir):
    # Load cleaned dataset
    df = pd.read_csv(input_path)

    # Validate required columns
    if "title" not in df.columns or "real" not in df.columns:
        raise ValueError("Dataset must contain title and real columns.")

    if df["title"].isnull().any() or df["real"].isnull().any():
        raise ValueError("Missing headlines or labels found.")

    if not set(df["real"].unique()).issubset({0, 1}):
        raise ValueError("Unexpected label values found.")

    # Separate input headlines and target labels
    X = df["title"].astype(str)
    y = df["real"].astype(int)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Convert headlines into numerical TF-IDF features
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    # Learn vocabulary and IDF weights from training data only
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # Transform test data using the same learned vocabulary
    X_test_tfidf = vectorizer.transform(X_test)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save feature matrices
    save_npz(output_dir / "X_train.npz", X_train_tfidf)
    save_npz(output_dir / "X_test.npz", X_test_tfidf)

    # Save target labels
    np.save(output_dir / "y_train.npy", y_train.to_numpy())
    np.save(output_dir / "y_test.npy", y_test.to_numpy())

    # Save fitted vectorizer
    joblib.dump(vectorizer, output_dir / "tfidf_vectorizer.joblib")

    # Display results
    train_counts = y_train.value_counts()
    test_counts = y_test.value_counts()

    print(
        f"DATA SPLIT RESULTS\n"
        f"------------------\n"
        f"Total records: {len(df)}\n"
        f"Training records: {len(X_train)}\n"
        f"Testing records: {len(X_test)}\n"
        f"Training Fake: {train_counts.get(0, 0)}\n"
        f"Training Real: {train_counts.get(1, 0)}\n"
        f"Testing Fake: {test_counts.get(0, 0)}\n"
        f"Testing Real: {test_counts.get(1, 0)}\n"
        f"\nTF-IDF RESULTS\n"
        f"-------------\n"
        f"Maximum features: 5000\n"
        f"Actual features: {X_train_tfidf.shape[1]}\n"
        f"Training matrix shape: {X_train_tfidf.shape}\n"
        f"Testing matrix shape: {X_test_tfidf.shape}\n"
        f"\nSaved feature files in: {output_dir}"
    )