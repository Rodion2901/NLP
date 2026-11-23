"""Загрузка Pipeline один раз, затем predict для новых текстов."""
import argparse
import json
from pathlib import Path
import joblib
from .data import texts_normalize

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "pipeline.joblib"


class Classifier:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL):
        # Загружайте только доверенные артефакты joblib.
        self.pipeline = joblib.load(model_path)

    def predict(self, texts: list[str]) -> list[str]:
        if not texts:
            raise ValueError("Передайте хотя бы один текст")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Каждое обращение должно быть непустой строкой")
        normalized = [texts_normalize(text) for text in texts]
        features = self.pipeline.named_steps["tfidf"].transform(normalized)
        if (features.getnnz(axis=1) == 0).any():
            raise ValueError("В одном из обращений нет известных модели слов")
        return self.pipeline.named_steps["classifier"].predict(features)


def main():
    parser = argparse.ArgumentParser(description="Определить категорию обращения")
    parser.add_argument("texts", nargs="+")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    try:
        labels = Classifier(args.model).predict(args.texts)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Ошибка: {exc}\n")
    print(json.dumps([{"text": text, "label": label}
                      for text, label in zip(args.texts, labels)], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
