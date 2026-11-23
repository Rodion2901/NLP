"""Keep the existing classifier loaded for the lifetime of the Rust service."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bank_classifier.predict import Classifier


def main() -> None:
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    classifier = Classifier()
    print("ready", flush=True)
    for line in sys.stdin:
        text = json.loads(line)
        try:
            reply = {"label": str(classifier.predict([text])[0])}
        except ValueError as error:
            reply = {"invalid": str(error)}
        except Exception as error:
            print(type(error).__name__, file=sys.stderr, flush=True)
            reply = "failed"
        print(json.dumps(reply, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
