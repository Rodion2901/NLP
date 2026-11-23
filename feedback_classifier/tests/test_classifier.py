import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from data import load_dataset, normalize_text
from predict import Classifier
from train import train


PROJECT = Path(__file__).resolve().parents[1]


class ClassifierTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.dataset = self.directory / "reviews.csv"

    def write_csv(self, rows, columns=("text", "label")):
        with self.dataset.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(columns)
            writer.writerows(rows)

    def test_missing_column(self):
        self.write_csv([("Пример", "класс")], ("review", "category"))
        with self.assertRaisesRegex(ValueError, "category_name"):
            load_dataset(self.dataset, "review", "category_name")

    def test_empty_values_and_zero_label(self):
        self.write_csv([
            ("  ХОРОШО\tРАБОТАЕТ  ", 0), ("", 1), ("\u2003", 1), ("Без метки", ""),
            ("Пробельная метка", " \t"),
        ])
        cleaned, stats = load_dataset(self.dataset)
        self.assertEqual(cleaned["text"].tolist(), ["хорошо работает"])
        self.assertEqual(cleaned["label"].tolist(), [0])
        self.assertEqual(stats, {
            "rows_before": 5, "rows_after": 1, "removed_empty": 4,
            "removed_conflicts": 0, "removed_duplicates": 0,
        })

    def test_conflicting_labels_remove_all_matching_rows(self):
        self.write_csv([
            ("  ОДИН\tТЕКСТ", "a"), ("один текст", "b"),
            ("Один текст", "a"), ("Другой текст", "b"),
        ])
        cleaned, stats = load_dataset(self.dataset)
        self.assertEqual(cleaned["text"].tolist(), ["другой текст"])
        self.assertEqual(stats["removed_conflicts"], 3)
        self.assertEqual(stats["removed_duplicates"], 0)

    def test_duplicates_are_removed_after_normalization(self):
        self.write_csv([("  Один  текст\n", "a"), ("ОДИН ТЕКСТ", "a")])
        cleaned, stats = load_dataset(self.dataset)
        self.assertEqual(cleaned["text"].tolist(), ["один текст"])
        self.assertEqual(stats["removed_duplicates"], 1)

    def test_training_cli_and_prediction_from_saved_pipeline(self):
        rows = [
            (f"{phrase} {index}", label)
            for phrase, label in [
                ("отличный сервис", "советую"), ("ужасное обслуживание", "жалоба")
            ]
            for index in range(6)
        ]
        self.write_csv(rows, ("review", "category"))
        command = [
            sys.executable, str(PROJECT / "train.py"), "--dataset", str(self.dataset),
            "--text-column", "review", "--label-column", "category", "--test-size", "0.5",
        ]
        subprocess.run(command, cwd=self.directory, check=True)
        model = self.directory / "model/pipeline.joblib"
        report_path = self.directory / "reports/report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["dataset"]["sha256"], hashlib.sha256(self.dataset.read_bytes()).hexdigest())
        self.assertEqual(report["split"]["test_class_distribution"], {"советую": 3, "жалоба": 3})
        self.assertEqual(set(report["metrics"]["per_class"]), {"советую", "жалоба"})

        classifier = Classifier(model)
        reviews = ["\t ОТЛИЧНЫЙ\nсервис  ", "Ужасное   обслуживание"]
        self.assertEqual(classifier.predict(reviews), ["советую", "жалоба"])
        self.assertEqual(classifier.predict([]), [])
        self.assertEqual(normalize_text(reviews[0]), "отличный сервис")
        output = subprocess.check_output([
            sys.executable, str(PROJECT / "predict.py"), "--text", reviews[1], "--model", str(model)
        ], text=True)
        self.assertEqual(output.strip(), "жалоба")

        self.write_csv([(review, f"новый {label}") for review, label in rows], ("review", "category"))
        subprocess.run(command, cwd=self.directory, check=True)
        updated_report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertNotEqual(updated_report["dataset"]["sha256"], report["dataset"]["sha256"])
        self.assertEqual(Classifier(model).predict(reviews), ["новый советую", "новый жалоба"])

    def test_impossible_stratification(self):
        self.write_csv([("Один", "a"), ("Два", "a"), ("Три", "b")])
        with self.assertRaisesRegex(ValueError, "stratified train/test split"):
            train(self.dataset, test_size=0.5, model_path=self.directory / "pipeline.joblib")

    def test_rare_class_is_present_in_both_subsets(self):
        self.write_csv([(f"Частый {i}", "a") for i in range(20)] + [
            ("Редкий один", "b"), ("Редкий два", "b"),
        ])
        report = train(self.dataset, test_size=0.1, model_path=self.directory / "pipeline.joblib")
        self.assertEqual(report["split"]["train_class_distribution"], {"a": 18, "b": 1})
        self.assertEqual(report["split"]["test_class_distribution"], {"a": 2, "b": 1})


if __name__ == "__main__":
    unittest.main()
