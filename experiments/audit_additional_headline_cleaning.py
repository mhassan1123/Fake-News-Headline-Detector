from pathlib import Path
from collections import Counter
import html
import re

import ftfy
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "data/processed/final_headlines.csv"

CANDIDATE_PATH = (
    ROOT / "data/processed/final_headlines_cleaning_candidate.csv"
)

AUDIT_PATH = (
    ROOT / "data/processed/additional_cleaning_audit.csv"
)

SUMMARY_PATH = (
    ROOT / "data/processed/additional_cleaning_summary.txt"
)


# Supported bracket pairs.
OPEN_TO_CLOSE = {
    "(": ")",
    "[": "]",
    "{": "}",
    "<": ">"
}

CLOSE_TO_OPEN = {
    closing: opening
    for opening, closing in OPEN_TO_CLOSE.items()
}


# Obvious scraper messages and boilerplate.
# These are flagged for inspection, not automatically deleted.
SCRAPER_PATTERNS = [
    r"^you are being redirected\b",
    r"^newsticker\b",
    r"^access denied\b",
    r"^page not found\b",
    r"^javascript is required\b",
    r"^enable javascript\b",
    r"^please enable cookies\b"
]


def remove_terminal_bracket_suffix(text):
    """
    Remove a final balanced bracketed section only when:
    - It ends the headline.
    - There are zero, one, or two ordinary spaces after it.
    - No punctuation or other character follows it.

    The text before the opening bracket is preserved.
    """

    if not text:
        return text, False

    # Count ordinary spaces at the end.
    trailing_spaces = len(text) - len(text.rstrip(" "))

    if trailing_spaces > 2:
        return text, False

    closing_position = len(text) - trailing_spaces - 1

    if closing_position < 0:
        return text, False

    final_character = text[closing_position]

    if final_character not in CLOSE_TO_OPEN:
        return text, False

    # Find the matching opening bracket, supporting nested pairs.
    expected_openings = [CLOSE_TO_OPEN[final_character]]

    for position in range(closing_position - 1, -1, -1):
        character = text[position]

        if character in CLOSE_TO_OPEN:
            expected_openings.append(
                CLOSE_TO_OPEN[character]
            )

        elif character in OPEN_TO_CLOSE:
            if not expected_openings:
                return text, False

            if character != expected_openings[-1]:
                return text, False

            expected_openings.pop()

            if not expected_openings:
                remaining_text = text[:position].rstrip()
                return remaining_text, True

    return text, False


def contains_scraper_artifact(text):
    lowered = text.lower()

    return any(
        re.search(pattern, lowered)
        for pattern in SCRAPER_PATTERNS
    )


def contains_html_tags(text):
    return bool(
        re.search(r"<\s*/?\s*[a-zA-Z][^>]*>", text)
    )


def normalize_whitespace(text):
    # Replace repeated spaces and tabs with one space.
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def clean_headline(original):
    """
    Clean one headline and return:
    cleaned headline, change categories, audit flags.
    """

    text = str(original)
    changes = []
    flags = []

    # 1. Remove an eligible final bracketed section.
    text, bracket_removed = remove_terminal_bracket_suffix(text)

    if bracket_removed:
        changes.append("terminal_bracket_suffix_removed")

    # 2. Decode HTML entities.
    decoded_text = html.unescape(text)

    if decoded_text != text:
        changes.append("html_entities_decoded")

    text = decoded_text

    # 3. Repair common encoding problems.
    repaired_text = ftfy.fix_text(text)

    if repaired_text != text:
        changes.append("encoding_repaired")

    text = repaired_text

    # 4. Normalize repeated whitespace.
    normalized_text = normalize_whitespace(text)

    if normalized_text != text:
        changes.append("whitespace_normalized")

    text = normalized_text

    # 5. Identify suspicious records without deleting them.
    if contains_scraper_artifact(text):
        flags.append("possible_scraper_artifact")

    if contains_html_tags(text):
        flags.append("possible_html_markup")

    if not text:
        flags.append("empty_after_cleaning")

    return text, changes, flags


def run_audit():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Original dataset not found: {INPUT_PATH}"
        )

    data = pd.read_csv(INPUT_PATH)

    required_columns = {
        "title",
        "real",
        "source_dataset"
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    candidate = data.copy()

    change_counts = Counter()
    audit_rows = []

    cleaned_titles = []
    total_flags = Counter()

    for index, original in enumerate(data["title"]):

        if pd.isna(original):
            cleaned_titles.append(original)
            continue

        cleaned, changes, flags = clean_headline(original)

        cleaned_titles.append(cleaned)

        for change in changes:
            change_counts[change] += 1

        for flag in flags:
            total_flags[flag] += 1

        if changes or flags:
            audit_rows.append({
                "original_row": index,
                "original_title": original,
                "cleaned_title": cleaned,
                "changes": ", ".join(changes),
                "flags": ", ".join(flags),
                "real": data.iloc[index]["real"],
                "source_dataset": data.iloc[index]["source_dataset"]
            })

    candidate["title"] = cleaned_titles

    # Check for empty headlines.
    empty_count = int(
        candidate["title"].fillna("").str.strip().eq("").sum()
    )

    # Check exact duplicate headlines after cleaning.
    duplicate_mask = candidate["title"].duplicated(keep=False)
    duplicate_rows = int(duplicate_mask.sum())

    # Check whether identical cleaned headlines have conflicting labels.
    label_counts = (
        candidate.dropna(subset=["title"])
        .groupby("title")["real"]
        .nunique()
    )

    conflicting_titles = label_counts[label_counts > 1]

    conflict_row_count = int(
        candidate["title"].isin(conflicting_titles.index).sum()
    )

    # Save the candidate dataset and audit.
    CANDIDATE_PATH.parent.mkdir(parents=True, exist_ok=True)

    candidate.to_csv(
        CANDIDATE_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    audit = pd.DataFrame(audit_rows)

    audit.to_csv(
        AUDIT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    # Prepare a readable summary.
    summary_lines = [
        "ADDITIONAL HEADLINE CLEANING AUDIT",
        "=" * 60,
        f"Original records: {len(data)}",
        f"Records with changes or flags: {len(audit)}",
        "",
        "CLEANING CHANGES",
        "-" * 60
    ]

    if change_counts:
        for name, count in sorted(change_counts.items()):
            summary_lines.append(f"{name}: {count}")
    else:
        summary_lines.append("No cleaning changes were required.")

    summary_lines.extend([
        "",
        "QUALITY FLAGS",
        "-" * 60
    ])

    if total_flags:
        for name, count in sorted(total_flags.items()):
            summary_lines.append(f"{name}: {count}")
    else:
        summary_lines.append("No quality flags found.")

    summary_lines.extend([
        "",
        "POST-CLEANING CHECKS",
        "-" * 60,
        f"Empty headlines: {empty_count}",
        f"Rows involved in exact duplicate headlines: {duplicate_rows}",
        f"Distinct headlines with conflicting labels: {len(conflicting_titles)}",
        f"Rows involved in conflicting-label headlines: {conflict_row_count}",
        "",
        "OUTPUT FILES",
        "-" * 60,
        str(CANDIDATE_PATH),
        str(AUDIT_PATH),
        str(SUMMARY_PATH),
        "",
        "IMPORTANT: The original dataset was not modified."
    ])

    summary_text = "\n".join(summary_lines)

    SUMMARY_PATH.write_text(
        summary_text,
        encoding="utf-8"
    )

    print(summary_text)


run_audit()