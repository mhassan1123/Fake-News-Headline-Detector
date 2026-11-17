from pathlib import Path
from collections import Counter
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

CANDIDATE_PATH = (
    ROOT / "data/processed/final_headlines_cleaning_candidate.csv"
)

AUDIT_PATH = (
    ROOT / "data/processed/additional_cleaning_audit.csv"
)

OUTPUT_PATH = (
    ROOT / "data/processed/final_headlines_cleaned_v2.csv"
)

EXCLUSIONS_PATH = (
    ROOT / "data/processed/stage2_cleaning_exclusions.csv"
)

DUPLICATES_PATH = (
    ROOT / "data/processed/stage2_duplicate_groups.csv"
)

SUMMARY_PATH = (
    ROOT / "data/processed/stage2_cleaning_summary.txt"
)


def normalize_for_comparison(title):
    """
    Normalize case and whitespace for duplicate comparison.

    Punctuation and words are preserved.
    """

    text = str(title)
    text = re.sub(r"\s+", " ", text)
    return text.strip().casefold()


def count_words(title):
    """Count words separated by whitespace."""

    if pd.isna(title):
        return 0

    return len(str(title).split())


def run_cleaning():

    # Check input files.
    if not CANDIDATE_PATH.exists():
        raise FileNotFoundError(
            f"Candidate dataset not found: {CANDIDATE_PATH}"
        )

    if not AUDIT_PATH.exists():
        raise FileNotFoundError(
            f"Cleaning audit not found: {AUDIT_PATH}"
        )

    data = pd.read_csv(CANDIDATE_PATH)
    audit = pd.read_csv(AUDIT_PATH)

    required_columns = {
        "title",
        "real",
        "source_dataset"
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Candidate dataset is missing columns: {missing_columns}"
        )

    if "original_row" not in audit.columns or "flags" not in audit.columns:
        raise ValueError(
            "Audit file must contain original_row and flags columns."
        )

    if data["real"].isna().any():
        raise ValueError(
            "Some records have missing labels. Inspect them before cleaning."
        )

    # Preserve the original CSV row position.
    data = data.copy()
    data["_original_row"] = range(len(data))

    # Store all removal reasons for each record.
    removal_reasons = {
        row: []
        for row in data["_original_row"]
    }

    # --------------------------------------------------
    # 1. Exclude scraper-artifact records.
    # --------------------------------------------------

    scraper_rows = set(
        pd.to_numeric(
            audit.loc[
                audit["flags"].fillna("").str.contains(
                    "possible_scraper_artifact",
                    case=False
                ),
                "original_row"
            ],
            errors="coerce"
        )
        .dropna()
        .astype(int)
    )

    for row in scraper_rows:
        if row in removal_reasons:
            removal_reasons[row].append(
                "possible_scraper_artifact"
            )

    # --------------------------------------------------
    # 2. Exclude headlines with one or two words.
    # --------------------------------------------------

    for _, record in data.iterrows():

        row = int(record["_original_row"])

        if count_words(record["title"]) <= 2:
            removal_reasons[row].append(
                "one_or_two_words"
            )

    # --------------------------------------------------
    # 3. Find conflicting-label headline groups.
    # --------------------------------------------------

    data["_comparison_key"] = data["title"].apply(
        normalize_for_comparison
    )

    label_counts = (
        data.groupby("_comparison_key")["real"]
        .nunique(dropna=False)
    )

    conflicting_keys = set(
        label_counts[label_counts > 1].index
    )

    conflicting_records = data[
        data["_comparison_key"].isin(conflicting_keys)
    ]

    for row in conflicting_records["_original_row"]:
        removal_reasons[int(row)].append(
            "conflicting_labels"
        )

    # --------------------------------------------------
    # 4. Keep one copy of consistent duplicates.
    # --------------------------------------------------

    already_excluded = {
        row
        for row, reasons in removal_reasons.items()
        if reasons
    }

    eligible = data[
        ~data["_original_row"].isin(already_excluded)
    ].copy()

    duplicate_groups = []

    for key, group in eligible.groupby(
        "_comparison_key",
        sort=False
    ):

        if len(group) <= 1:
            continue

        group_rows = group["_original_row"].tolist()

        kept_row = int(group_rows[0])
        removed_rows = [int(row) for row in group_rows[1:]]

        duplicate_groups.append({
            "headline": group.iloc[0]["title"],
            "records_in_group": len(group),
            "labels": ", ".join(
                str(value)
                for value in sorted(group["real"].unique())
            ),
            "kept_original_row": kept_row,
            "removed_original_rows": ", ".join(
                str(row) for row in removed_rows
            )
        })

        for row in removed_rows:
            removal_reasons[row].append(
                "duplicate_same_label"
            )

    # --------------------------------------------------
    # 5. Separate kept and excluded records.
    # --------------------------------------------------

    excluded_rows = {
        row
        for row, reasons in removal_reasons.items()
        if reasons
    }

    final_data = data[
        ~data["_original_row"].isin(excluded_rows)
    ].copy()

    excluded_data = data[
        data["_original_row"].isin(excluded_rows)
    ].copy()

    # Prepare exclusion audit.
    excluded_data["removal_reason"] = excluded_data[
        "_original_row"
    ].apply(
        lambda row: ", ".join(
            removal_reasons[int(row)]
        )
    )

    exclusions = excluded_data[
        [
            "_original_row",
            "title",
            "real",
            "source_dataset",
            "removal_reason"
        ]
    ].rename(
        columns={"_original_row": "original_row"}
    )

    # Remove internal columns from final dataset.
    final_data = final_data.drop(
        columns=["_original_row", "_comparison_key"]
    )

    # --------------------------------------------------
    # 6. Final quality checks.
    # --------------------------------------------------

    final_keys = final_data["title"].apply(
        normalize_for_comparison
    )

    remaining_duplicates = int(
        final_keys.duplicated().sum()
    )

    remaining_conflicts = int(
        final_data.assign(
            _comparison_key=final_keys
        )
        .groupby("_comparison_key")["real"]
        .nunique(dropna=False)
        .gt(1)
        .sum()
    )

    short_headlines_remaining = int(
        final_data["title"].apply(count_words).le(2).sum()
    )

    empty_headlines = int(
        final_data["title"].fillna("").str.strip().eq("").sum()
    )

    if remaining_duplicates != 0:
        raise ValueError(
            "Duplicate headlines remain in the final dataset."
        )

    if remaining_conflicts != 0:
        raise ValueError(
            "Conflicting labels remain in the final dataset."
        )

    if short_headlines_remaining != 0:
        raise ValueError(
            "Headlines with two or fewer words remain."
        )

    if empty_headlines != 0:
        raise ValueError(
            "Empty headlines remain in the final dataset."
        )

    # --------------------------------------------------
    # 7. Save new files without changing old files.
    # --------------------------------------------------

    final_data.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    exclusions.to_csv(
        EXCLUSIONS_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    duplicate_audit = pd.DataFrame(
        duplicate_groups,
        columns=[
            "headline",
            "records_in_group",
            "labels",
            "kept_original_row",
            "removed_original_rows"
        ]
    )

    duplicate_audit.to_csv(
        DUPLICATES_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------
    # 8. Prepare summary.
    # --------------------------------------------------

    reason_counts = Counter()

    for reasons in removal_reasons.values():
        reason_counts.update(reasons)

    summary_lines = [
        "SECOND-STAGE HEADLINE CLEANING",
        "=" * 65,
        f"Input records: {len(data)}",
        f"Final records: {len(final_data)}",
        f"Excluded records: {len(exclusions)}",
        "",
        "REMOVAL REASONS",
        "-" * 65
    ]

    for reason, count in sorted(reason_counts.items()):
        summary_lines.append(
            f"{reason}: {count}"
        )

    summary_lines.extend([
        "",
        "DUPLICATE AND CONFLICT CHECKS",
        "-" * 65,
        f"Conflicting headline groups found: {len(conflicting_keys)}",
        f"Records in conflicting groups: {len(conflicting_records)}",
        f"Consistent duplicate groups found: {len(duplicate_groups)}",
        f"Remaining duplicate headlines: {remaining_duplicates}",
        f"Remaining conflicting-label groups: {remaining_conflicts}",
        "",
        "FINAL DATASET CHECKS",
        "-" * 65,
        f"Headlines with one or two words: {short_headlines_remaining}",
        f"Empty headlines: {empty_headlines}",
        f"Final Fake records: {(final_data['real'] == 0).sum()}",
        f"Final Real records: {(final_data['real'] == 1).sum()}",
        "",
        "OUTPUT FILES",
        "-" * 65,
        str(OUTPUT_PATH),
        str(EXCLUSIONS_PATH),
        str(DUPLICATES_PATH),
        str(SUMMARY_PATH),
        "",
        "Original files were not modified."
    ])

    summary_text = "\n".join(summary_lines)

    SUMMARY_PATH.write_text(
        summary_text,
        encoding="utf-8"
    )

    print(summary_text)


run_cleaning()