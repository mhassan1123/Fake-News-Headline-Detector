import pandas as pd
from pathlib import Path


def load_and_clean_dataset(input_path, output_path):
    # Load the original dataset
    df = pd.read_csv(input_path)

    # Check required columns
    required_columns = ["title", "real"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Required column is missing: {column}")

    # Check for missing headlines or labels
    if df["title"].isnull().any() or df["real"].isnull().any():
        raise ValueError("Dataset contains missing headlines or labels.")

    # Check that labels are only 0 and 1
    if not set(df["real"].unique()).issubset({0, 1}):
        raise ValueError("Unexpected label values found.")

    original_count = len(df)

    # Find headlines that have conflicting labels
    label_counts = df.groupby("title")["real"].nunique()
    conflicting_titles = label_counts[label_counts > 1].index

    # Remove all rows belonging to conflicting headlines
    safe_df = df[~df["title"].isin(conflicting_titles)]

    conflicting_rows_removed = original_count - len(safe_df)

    # Remove repeated headlines with consistent labels
    clean_df = safe_df.drop_duplicates(subset="title", keep="first")

    duplicate_rows_removed = len(safe_df) - len(clean_df)

    # Create output folder if it does not exist
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save cleaned dataset
    clean_df.to_csv(output_path, index=False)

    # Display cleaning summary
    print(
        f"Original records: {original_count}\n"
        f"Conflicting headline groups: {len(conflicting_titles)}\n"
        f"Rows removed due to conflicting labels: {conflicting_rows_removed}\n"
        f"Repeated rows removed: {duplicate_rows_removed}\n"
        f"Final records: {len(clean_df)}\n"
        f"Cleaned dataset saved to: {output_path}"
    )

    return clean_df