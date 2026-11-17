import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

os.makedirs("outputs/audits", exist_ok=True)
os.makedirs("outputs/models/source_experiments", exist_ok=True)

TRAIN_PATH = "data/splits/train.csv"
VALIDATION_PATH = "data/splits/validation.csv"

COMBINED_VECTORIZER_PATH = "outputs/models/tfidf_experiments/tfidf_baseline.joblib"
COMBINED_MODEL_PATH = "outputs/models/tfidf_experiments/lr_baseline_standard.joblib"

THRESHOLD = 0.45

train_df = pd.read_csv(TRAIN_PATH)
validation_df = pd.read_csv(VALIDATION_PATH)

required_columns = {"title", "real", "source_dataset"}

if not required_columns.issubset(train_df.columns):
    raise ValueError("Training CSV is missing required columns.")

if not required_columns.issubset(validation_df.columns):
    raise ValueError("Validation CSV is missing required columns.")

train_df = train_df.dropna(subset=["title", "real", "source_dataset"])
validation_df = validation_df.dropna(subset=["title", "real", "source_dataset"])

train_df["title"] = train_df["title"].astype(str)
validation_df["title"] = validation_df["title"].astype(str)

train_df["real"] = train_df["real"].astype(int)
validation_df["real"] = validation_df["real"].astype(int)

train_df["source_dataset"] = train_df["source_dataset"].astype(str).str.strip()
validation_df["source_dataset"] = validation_df["source_dataset"].astype(str).str.strip()

if set(train_df["real"].unique()) != {0, 1}:
    raise ValueError("Training labels must contain both 0 (Fake) and 1 (Real).")

# Model A: Existing combined-data model
combined_vectorizer = joblib.load(COMBINED_VECTORIZER_PATH)
combined_model = joblib.load(COMBINED_MODEL_PATH)

# Model B: Train using WELFake records only
welfake_train = train_df[
    train_df["source_dataset"] == "WELFake"
].copy()

if welfake_train.empty or welfake_train["real"].nunique() != 2:
    raise ValueError("WELFake training data must contain both classes.")

X_train = welfake_train["title"]
y_train = welfake_train["real"]

welfake_vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X_train_tfidf = welfake_vectorizer.fit_transform(X_train)

welfake_model = LogisticRegression(
    C=2,
    max_iter=1000,
    random_state=42
)

welfake_model.fit(X_train_tfidf, y_train)

joblib.dump(
    welfake_vectorizer,
    "outputs/models/source_experiments/welfake_vectorizer.joblib"
)

joblib.dump(
    welfake_model,
    "outputs/models/source_experiments/welfake_logistic_regression.joblib"
)

# Evaluate both models on identical validation records
results = []

def evaluate_model(model_name, model, vectorizer, data, source_name):
    X = data["title"]
    y = data["real"]

    X_tfidf = vectorizer.transform(X)

    classes = list(model.classes_)

    if 0 not in classes or 1 not in classes:
        raise ValueError("Model must contain both Fake (0) and Real (1) classes.")

    fake_index = classes.index(0)
    fake_probability = model.predict_proba(X_tfidf)[:, fake_index]

    predictions = (fake_probability < THRESHOLD).astype(int)
    predictions[fake_probability >= THRESHOLD] = 0

    fake_recall = recall_score(
        y == 0, predictions == 0, zero_division=0
    )

    real_recall = recall_score(
        y == 1, predictions == 1, zero_division=0
    )

    fake_precision = precision_score(
        y == 0, predictions == 0, zero_division=0
    )

    real_precision = precision_score(
        y == 1, predictions == 1, zero_division=0
    )

    macro_f1 = f1_score(
        y, predictions, average="macro", zero_division=0
    )

    auc = roc_auc_score(y == 0, fake_probability)

    results.append({
        "Model": model_name,
        "Evaluation Source": source_name,
        "Records": len(data),
        "Accuracy": accuracy_score(y, predictions),
        "Fake Precision": fake_precision,
        "Fake Recall": fake_recall,
        "Real Precision": real_precision,
        "Real Recall": real_recall,
        "Macro F1": macro_f1,
        "ROC-AUC": auc
    })

    matrix = confusion_matrix(y, predictions, labels=[0, 1])

    print(f"\n{model_name} - {source_name}")
    print("-" * 60)
    print(f"Records: {len(data)}")
    print(f"Accuracy: {accuracy_score(y, predictions):.4f}")
    print(f"Fake Precision: {fake_precision:.4f}")
    print(f"Fake Recall: {fake_recall:.4f}")
    print(f"Real Precision: {real_precision:.4f}")
    print(f"Real Recall: {real_recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print("Confusion Matrix:")
    print(matrix)

print("TRAINING SOURCE COMPARISON")
print("=" * 70)
print(f"Combined training records: {len(train_df)}")
print(f"WELFake-only training records: {len(welfake_train)}")
print(f"Validation records: {len(validation_df)}")
print(f"Fake threshold: {THRESHOLD}")

evaluation_groups = {
    "Overall": validation_df,
    "WELFake": validation_df[
        validation_df["source_dataset"] == "WELFake"
    ],
    "FakeNewsNet": validation_df[
        validation_df["source_dataset"] == "FakeNewsNet"
    ]
}

for source_name, data in evaluation_groups.items():
    if len(data) == 0 or data["real"].nunique() < 2:
        print(f"\nSkipping {source_name}: insufficient class data.")
        continue

    evaluate_model(
        "Combined Training",
        combined_model,
        combined_vectorizer,
        data,
        source_name
    )

    evaluate_model(
        "WELFake-Only Training",
        welfake_model,
        welfake_vectorizer,
        data,
        source_name
    )

results_df = pd.DataFrame(results)

results_path = "outputs/audits/training_source_comparison.csv"
results_df.to_csv(results_path, index=False)

print("\nFINAL COMPARISON")
print("=" * 70)
print(results_df.round(4).to_string(index=False))

print(f"\nSaved results: {results_path}")
print("TRAINING SOURCE COMPARISON COMPLETED")