import pandas as pd

input_path = "data/raw/WELFake_Dataset.csv"

print("WELFAKE DATASET AUDIT\n" + "-" * 50)

# Load the original CSV without changing it.
df = pd.read_csv(input_path, low_memory=False)

print("\n1. DATASET STRUCTURE")
print("-" * 50)
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(f"Column names: {df.columns.tolist()}")

print("\n2. COLUMN DATA TYPES")
print("-" * 50)
print(df.dtypes.to_string())

# Identify important columns without assuming their exact capitalization.
column_map = {
    str(col).strip().lower(): col
    for col in df.columns
}

title_col = column_map.get("title") or column_map.get("headline")
text_col = column_map.get("text")
label_col = column_map.get("label")

if title_col is None or label_col is None:
    print("\nERROR: Could not identify the Title or Label column.")
    print("Please share the column names printed above.")
    raise SystemExit

# Treat empty strings and whitespace-only strings as missing.
df[title_col] = df[title_col].replace(r"^\s*$", pd.NA, regex=True)

if text_col is not None:
    df[text_col] = df[text_col].replace(r"^\s*$", pd.NA, regex=True)

print("\n3. LABEL DISTRIBUTION")
print("-" * 50)
print(df[label_col].value_counts(dropna=False).to_string())

print("\n4. MISSING VALUES")
print("-" * 50)
print(df.isna().sum().to_string())

print("\n5. EMPTY HEADLINES AND ARTICLE TEXT")
print("-" * 50)
print(f"Missing headlines: {df[title_col].isna().sum()}")

if text_col is not None:
    print(f"Missing article text: {df[text_col].isna().sum()}")
else:
    print("Article text column not found.")

print("\n6. DUPLICATE HEADLINES")
print("-" * 50)
valid_titles = df.dropna(subset=[title_col])

print(f"Rows with valid headlines: {len(valid_titles)}")
print(f"Duplicate headline rows: {valid_titles.duplicated(subset=[title_col]).sum()}")
print(f"Unique headlines: {valid_titles[title_col].nunique()}")

print("\n7. HEADLINES WITH CONFLICTING LABELS")
print("-" * 50)

label_counts = (
    valid_titles.groupby(title_col)[label_col]
    .nunique(dropna=True)
)

conflicting_titles = label_counts[label_counts > 1]

print(f"Conflicting headline groups: {len(conflicting_titles)}")

if len(conflicting_titles) > 0:
    print("\nExamples:")
    print(
        valid_titles[
            valid_titles[title_col].isin(conflicting_titles.index)
        ][[title_col, label_col]].head(10).to_string(index=False)
    )

print("\n8. COMPLETELY DUPLICATE RECORDS")
print("-" * 50)
print(f"Duplicate complete rows: {df.duplicated().sum()}")

print("\n9. FIRST FIVE RECORDS")
print("-" * 50)

display_columns = [title_col, label_col]

if text_col is not None:
    display_columns.insert(1, text_col)

print(
    df[display_columns]
    .head(5)
    .to_string(index=False, max_colwidth=100)
)

print("\n10. RECORDS WITH MISSING HEADLINES")
print("-" * 50)

missing_headlines = df[df[title_col].isna()]

if len(missing_headlines) > 0:
    print(missing_headlines[display_columns].head(5).to_string(index=False))
else:
    print("No missing headlines.")

print("\n11. RECORDS WITH MISSING ARTICLE TEXT")
print("-" * 50)

if text_col is not None:
    missing_text = df[df[text_col].isna()]

    print(f"Records with missing text: {len(missing_text)}")

    if len(missing_text) > 0:
        print(
            missing_text[[title_col, label_col]]
            .head(10)
            .to_string(index=False)
        )
else:
    print("Article text column not found.")

print("\nAUDIT COMPLETED")
print("Original CSV was not modified.")