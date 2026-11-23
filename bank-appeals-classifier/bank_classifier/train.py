"""python -m bank_classifier.train --data data/demo.csv"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import joblib
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from .data import load_data

ROOT = Path(__file__).resolve().parents[1]


def train(data_path: Path, output_dir: Path, report_dir: Path,
          test_size: float = 0.2, seed: int = 42) -> dict:
    if not 0 < test_size < 1:
        raise ValueError("test-size должен быть между 0 и 1")
    frame, cleaning = load_data(data_path)
    try:
        train_df, test_df = train_test_split(
            frame, test_size=test_size, stratify=frame["label"], random_state=seed)
    except ValueError as exc:
        raise ValueError("Не удалось разделить классы на train/test: добавьте данные "
                         "или измените --test-size. " + str(exc)) from exc
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True)),
        ("classifier", LinearSVC(class_weight="balanced", random_state=seed, max_iter=10000)),
    ])
    pipeline.fit(train_df["text"], train_df["label"])
    predicted = pipeline.predict(test_df["text"])
    labels = pipeline.classes_.tolist()
    report = {
        "dataset": str(data_path.resolve()),
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "seed": seed, "test_size": test_size, "cleaning": cleaning,
        "train_rows": len(train_df), "test_rows": len(test_df),
        "train_class_counts": train_df["label"].value_counts().to_dict(),
        "test_class_counts": test_df["label"].value_counts().to_dict(),
        "accuracy": accuracy_score(test_df["label"], predicted),
        "macro_f1": f1_score(test_df["label"], predicted, average="macro"),
        "classification_report": classification_report(
            test_df["label"], predicted, output_dict=True, zero_division=0),
        "labels": labels,
        "confusion_matrix": confusion_matrix(test_df["label"], predicted, labels=labels).tolist(),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "pandas": pd.__version__, "joblib": joblib.__version__},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_dir / "pipeline.joblib")
    (report_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    test_result = test_df.copy()
    test_result["prediction"] = predicted
    test_result.to_csv(report_dir / "test_predictions.csv", index=False)
    return report


def main():
    parser = argparse.ArgumentParser(description="Обучение классификатора банковских обращений")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        report = train(args.data, args.models_dir, args.reports_dir, args.test_size, args.seed)
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        parser.exit(2, f"Ошибка: {exc}\n")
    print(f"Train: {report['train_rows']}, test: {report['test_rows']}")
    print(f"Accuracy: {report['accuracy']:.3f}; macro-F1: {report['macro_f1']:.3f}")
    print(f"Pipeline: {args.models_dir / 'pipeline.joblib'}")


if __name__ == "__main__":
    main()
