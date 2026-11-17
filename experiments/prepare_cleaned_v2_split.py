from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT / "data/processed/final_headlines_cleaned_v2.csv"
)

OUTPUT_DIR = ROOT / "data/splits_cleaned_v2"

TRAIN_PATH = OUTPUT_DIR / "train.csv"
VALIDATION_PATH = OUTPUT_DIR / "validation.csv"
TEST_PATH = OUTPUT_DIR / "test.csv"
SUMMARY_PATH = OUTPUT_DIR / "split_summary.txt"

RANDOM_STATE = 42


def normalize_headline(title):
    """Normalize a headline for duplicate checking."""

    return " ".join(str(title).split()).casefold()


def check_split_duplicates(train, validation, test):
    """Check that no normalized headline appears in multiple splits."""

    train_titles = set(train["title"].apply(normalize_headline))
    validation_titles = set(
        validation["title"].apply(normalize_headline)
    )
    test_titles = set(test["title"].apply(normalize_headline))

    train_validation = train_titles.intersection(validation_titles)
    train_test = train_titles.intersection(test_titles)
    validation_test = validation_titles.intersection(test_titles)

    total_overlap = (
        len(train_validation)
        + len(train_test)
        + len(validation_test)
    )

    print("\nDUPLICATE CHECK")
    print("-" * 60)
    print(f"Train-validation overlap: {len(train_validation)}")
    print(f"Train-test overlap: {len(train_test)}")
    print(f"Validation-test overlap: {len(validation_test)}")

    if total_overlap > 0:
        raise ValueError(
            "Normalized headline overlap found between splits."
        )

    print("Result: No normalized headline overlap.")


def describe_split(name, data):
    """Return record and class distribution information."""

    total = len(data)
    fake_count = int((data["real"] == 0).sum())
    real_count = int((data["real"] == 1).sum())

    fake_percentage = fake_count / total * 100
    real_percentage = real_count / total * 100

    return [
        name,
        f"Records: {total}",
        f"Fake: {fake_count} ({fake_percentage:.2f}%)",
        f"Real: {real_count} ({real_percentage:.2f}%)"
    ]


def run_split():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {INPUT_PATH}"
        )

    data = pd.read_csv(INPUT_PATH)

    required_columns = {"title", "real", "source_dataset"}
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if data["title"].isna().any():
        raise ValueError("Missing headlines found.")

    if data["real"].isna().any():
        raise ValueError("Missing labels found.")

    if not data["real"].isin([0, 1]).all():
        raise ValueError("Labels must contain only 0 and 1.")

    # Check that the input has no duplicate normalized headlines.
    normalized_titles = data["title"].apply(normalize_headline)

    duplicate_count = int(
        normalized_titles.duplicated().sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"Input dataset contains {duplicate_count} duplicate headlines."
        )

    # First split: 64% training and 36% temporary data.
    train, temporary = train_test_split(
        data,
        test_size=0.36,
        random_state=RANDOM_STATE,
        stratify=data["real"]
    )

    # Second split: divide the temporary data into
    # 16% validation and 20% testing.
    validation, test = train_test_split(
        temporary,
        test_size=20 / 36,
        random_state=RANDOM_STATE,
        stratify=temporary["real"]
    )

    # Reset row indexes.
    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    test = test.reset_index(drop=True)

    # Confirm all records are assigned exactly once.
    total_assigned = len(train) + len(validation) + len(test)

    if total_assigned != len(data):
        raise ValueError("Some records are missing from the splits.")

    # Check duplicate headlines across all splits.
    check_split_duplicates(train, validation, test)

    # Create a separate output directory.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save all three splits.
    train.to_csv(
        TRAIN_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    validation.to_csv(
        VALIDATION_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    test.to_csv(
        TEST_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    # Prepare split summary.
    summary_lines = [
        "CLEANED V2 DATASET SPLITTING",
        "=" * 60,
        f"Total records: {len(data)}",
        f"Random state: {RANDOM_STATE}",
        "Split ratio: 64% training, 16% validation, 20% testing",
        "",
        "SPLIT DISTRIBUTION",
        "-" * 60
    ]

    for name, split_data in [
        ("Training", train),
        ("Validation", validation),
        ("Testing", test)
    ]:
        summary_lines.extend(
            describe_split(name, split_data)
        )
        summary_lines.append("")

    summary_lines.extend([
        "QUALITY CHECKS",
        "-" * 60,
        f"Input duplicate headlines: {duplicate_count}",
        "Missing headlines: 0",
        "Missing labels: 0",
        "Invalid labels: 0",
        "Normalized overlap between splits: 0",
        f"Total assigned records: {total_assigned}",
        "",
        "OUTPUT FILES",
        "-" * 60,
        str(TRAIN_PATH),
        str(VALIDATION_PATH),
        str(TEST_PATH)
    ])

    summary_text = "\n".join(summary_lines)

    SUMMARY_PATH.write_text(
        summary_text,
        encoding="utf-8"
    )

    print("\n" + summary_text)


run_split()