"""Load and clean labeled reviews from a CSV file."""

from io import BytesIO
from pathlib import Path

import pandas as pd


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def load_dataset(
    path: str | Path, text_column: str = "text", label_column: str = "label"
) -> tuple[pd.DataFrame, dict[str, int]]:
    return clean_csv(BytesIO(Path(path).read_bytes()), text_column, label_column)


def clean_csv(
    source: BytesIO, text_column: str, label_column: str
) -> tuple[pd.DataFrame, dict[str, int]]:
    try:
        frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"CSV is empty. Expected columns {text_column!r} and {label_column!r}.") from exc

    for column in (text_column, label_column):
        if column not in frame:
            raise ValueError(f"CSV is missing required column {column!r}.")
    if text_column == label_column:
        raise ValueError("The text and label columns must be different.")

    texts = frame[text_column].map(normalize_text)
    labels = frame[label_column]
    populated = texts.ne("") & labels.str.strip().ne("")
    cleaned = frame.loc[populated].copy()
    cleaned[text_column] = texts.loc[populated]

    label_counts = cleaned.groupby(text_column)[label_column].transform("nunique")
    without_conflicts = cleaned.loc[label_counts.eq(1)]
    unique = without_conflicts.drop_duplicates(subset=[text_column]).reset_index(drop=True)

    # Preserve numeric IDs when every label is a canonical integer.
    labels = unique[label_column]
    if not labels.empty and labels.str.fullmatch(r"0|-?[1-9][0-9]*").all():
        numeric = labels.map(int)
        unique[label_column] = numeric

    return unique, {
        "rows_before": len(frame),
        "rows_after": len(unique),
        "removed_empty": int((~populated).sum()),
        "removed_conflicts": len(cleaned) - len(without_conflicts),
        "removed_duplicates": len(without_conflicts) - len(unique),
    }
