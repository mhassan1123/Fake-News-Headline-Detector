import os
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

TRAIN_PATH = "data/splits/train.csv"
VALIDATION_PATH = "data/splits/validation.csv"
TEST_PATH = "data/splits/test.csv"

FEATURE_DIR = "outputs/final_features"
MODEL_DIR = "outputs/models"

os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print("FINAL TF-IDF FEATURE PREPARATION")
print("=" * 65)

# 1. Load the three prepared splits
train = pd.read_csv(TRAIN_PATH)
validation = pd.read_csv(VALIDATION_PATH)
test = pd.read_csv(TEST_PATH)

required_columns = ["title", "real"]

for name, dataset in [
    ("Training", train),
    ("Validation", validation),
    ("Testing", test)
]:
    if not all(col in dataset.columns for col in required_columns):
        raise ValueError(f"Required columns missing from {name} data.")

    if dataset[required_columns].isnull().any().any():
        raise ValueError(f"Missing headline or label in {name} data.")

    if not dataset["real"].isin([0, 1]).all():
        raise ValueError(f"Invalid labels found in {name} data.")

print(f"Training headlines: {len(train)}")
print(f"Validation headlines: {len(validation)}")
print(f"Testing headlines: {len(test)}")

# 2. Separate headline text and labels
X_train = train["title"].astype(str)
X_validation = validation["title"].astype(str)
X_test = test["title"].astype(str)

y_train = train["real"].to_numpy(dtype=np.int64)
y_validation = validation["real"].to_numpy(dtype=np.int64)
y_test = test["real"].to_numpy(dtype=np.int64)

# 3. Create the TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

# 4. Fit ONLY on training headlines
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform validation and test using the same vocabulary
X_validation_tfidf = vectorizer.transform(X_validation)
X_test_tfidf = vectorizer.transform(X_test)

# 5. Save sparse feature matrices
save_npz(
    os.path.join(FEATURE_DIR, "X_train_tfidf.npz"),
    X_train_tfidf
)

save_npz(
    os.path.join(FEATURE_DIR, "X_validation_tfidf.npz"),
    X_validation_tfidf
)

save_npz(
    os.path.join(FEATURE_DIR, "X_test_tfidf.npz"),
    X_test_tfidf
)

# 6. Save labels separately
np.save(os.path.join(FEATURE_DIR, "y_train.npy"), y_train)
np.save(os.path.join(FEATURE_DIR, "y_validation.npy"), y_validation)
np.save(os.path.join(FEATURE_DIR, "y_test.npy"), y_test)

# 7. Save the fitted vectorizer
joblib.dump(
    vectorizer,
    os.path.join(MODEL_DIR, "merged_tfidf_vectorizer.joblib")
)

# 8. Save vocabulary for inspection
feature_names = vectorizer.get_feature_names_out()

pd.DataFrame({
    "feature": feature_names
}).to_csv(
    os.path.join(FEATURE_DIR, "tfidf_vocabulary.csv"),
    index=False
)

# 9. Verify feature matrices
if X_train_tfidf.shape[1] != X_validation_tfidf.shape[1]:
    raise ValueError("Training and validation feature counts differ.")

if X_train_tfidf.shape[1] != X_test_tfidf.shape[1]:
    raise ValueError("Training and test feature counts differ.")

if X_train_tfidf.shape[1] != len(feature_names):
    raise ValueError("Vocabulary size does not match feature count.")

if X_train_tfidf.shape[0] != len(y_train):
    raise ValueError("Training feature and label counts differ.")

if X_validation_tfidf.shape[0] != len(y_validation):
    raise ValueError("Validation feature and label counts differ.")

if X_test_tfidf.shape[0] != len(y_test):
    raise ValueError("Test feature and label counts differ.")

if X_train_tfidf.nnz == 0:
    raise ValueError("Training feature matrix is empty.")

if X_validation_tfidf.nnz == 0:
    raise ValueError("Validation feature matrix is empty.")

if X_test_tfidf.nnz == 0:
    raise ValueError("Test feature matrix is empty.")

# 10. Print final report
print("\nTF-IDF CONFIGURATION")
print("-" * 65)
print("Maximum features: 10000")
print("N-gram range: (1, 2)")
print("Minimum document frequency: 2")
print("Maximum document frequency: 0.95")
print("Vectorizer fitted on: Training data ONLY")

print("\nFEATURE MATRIX SHAPES")
print("-" * 65)
print(f"Training: {X_train_tfidf.shape}")
print(f"Validation: {X_validation_tfidf.shape}")
print(f"Testing: {X_test_tfidf.shape}")

print("\nNON-ZERO VALUES")
print("-" * 65)
print(f"Training: {X_train_tfidf.nnz}")
print(f"Validation: {X_validation_tfidf.nnz}")
print(f"Testing: {X_test_tfidf.nnz}")

print(f"\nVocabulary size: {len(feature_names)}")

print("\nSAVED FILES")
print("-" * 65)
print(FEATURE_DIR)
print(os.path.join(MODEL_DIR, "merged_tfidf_vectorizer.joblib"))

print("\nFINAL TF-IDF VALIDATION PASSED")