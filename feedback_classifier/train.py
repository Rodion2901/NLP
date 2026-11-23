"""Train and evaluate a review classifier."""

import argparse
import hashlib
import json
import platform
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

if __package__:
    from .data import clean_csv
else:
    from data import clean_csv


def _class_distribution(labels: pd.Series) -> dict[str, int]:
    return {str(label): int(count) for label, count in labels.value_counts().items()}


def _cover_classes(
    subset: pd.DataFrame, other: pd.DataFrame, label_column: str, classes: pd.Index
) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = subset[label_column].value_counts()
    for label in classes.difference(counts.index):
        # A split can round a rare class down to zero. Swap one row while
        # retaining the original split sizes and all classes already present.
        donor_label = counts.idxmax()
        outgoing = subset[subset[label_column].eq(donor_label)].iloc[[-1]]
        incoming = other[other[label_column].eq(label)].iloc[[-1]]
        subset = pd.concat([subset.drop(outgoing.index), incoming])
        other = pd.concat([other.drop(incoming.index), outgoing])
        counts.loc[donor_label] -= 1
        counts.loc[label] = 1
    return subset, other


def _atomic_write(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        writer(temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def train(
    dataset: str | Path,
    text_column: str = "text",
    label_column: str = "label",
    test_size: float = 0.2,
    seed: int = 42,
    model_path: str | Path = "model/pipeline.joblib",
) -> dict:
    source = Path(dataset).resolve()
    content = source.read_bytes()
    cleaned, cleaning = clean_csv(BytesIO(content), text_column, label_column)
    labels = cleaned[label_column]
    if labels.nunique() < 2:
        raise ValueError("Training requires at least two classes after cleaning.")
    if not 0 < test_size < 1:
        raise ValueError("--test-size must be strictly between 0 and 1.")

    try:
        train_df, test_df = train_test_split(
            cleaned, test_size=test_size, random_state=seed, stratify=labels
        )
    except ValueError as exc:
        raise ValueError(f"Cannot create a stratified train/test split: {exc}") from exc

    classes = pd.Index(labels.unique())
    train_df, test_df = _cover_classes(train_df, test_df, label_column, classes)
    test_df, train_df = _cover_classes(test_df, train_df, label_column, classes)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(token_pattern=r"(?u)\w+|[^\w\s]", lowercase=False)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=seed)),
    ])
    pipeline.fit(train_df[text_column], train_df[label_column])
    expected = test_df[label_column]
    predicted = pipeline.predict(test_df[text_column])
    classes = pipeline.classes_
    precision, recall, f1, support = precision_recall_fscore_support(
        expected, predicted, labels=classes, zero_division=0
    )
    report = {
        "dataset": {
            "path": str(source),
            "sha256": hashlib.sha256(content).hexdigest(),
            "text_column": text_column,
            "label_column": label_column,
        },
        "cleaning": cleaning,
        "split": {
            "test_size": test_size,
            "seed": seed,
            "stratified": True,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "train_class_distribution": _class_distribution(train_df[label_column]),
            "test_class_distribution": _class_distribution(test_df[label_column]),
        },
        "metrics": {
            "accuracy": float(accuracy_score(expected, predicted)),
            "macro_f1": float(f1_score(expected, predicted, labels=classes, average="macro")),
            "per_class": {
                str(label): {
                    "precision": float(p),
                    "recall": float(r),
                    "f1": float(score),
                    "support": int(count),
                }
                for label, p, r, score, count in zip(classes, precision, recall, f1, support)
            },
        },
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    _atomic_write(Path(model_path), lambda path: joblib.dump(pipeline, path))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a classifier on labeled CSV reviews.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        report = train(args.dataset, args.text_column, args.label_column, args.test_size, args.seed)
        report_json = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        _atomic_write(
            Path("reports/report.json"), lambda path: path.write_text(report_json, encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Training failed: {exc}\n")


if __name__ == "__main__":
    main()
