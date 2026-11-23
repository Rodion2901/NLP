"""Predict review labels with a trained pipeline."""

import argparse
from pathlib import Path

import joblib

if __package__:
    from .data import normalize_text
else:
    from data import normalize_text


class Classifier:
    def __init__(self, model_path: str | Path):
        self.pipeline = joblib.load(model_path)

    def predict(self, texts: list[str]) -> list[str | int]:
        if not texts:
            return []
        return self.pipeline.predict([normalize_text(text) for text in texts]).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict the label of one review.")
    parser.add_argument("--text", required=True)
    parser.add_argument("--model", type=Path, default=Path("model/pipeline.joblib"))
    args = parser.parse_args()

    try:
        label, = Classifier(args.model).predict([args.text])
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Prediction failed: {exc}\n")
    print(label)


if __name__ == "__main__":
    main()
