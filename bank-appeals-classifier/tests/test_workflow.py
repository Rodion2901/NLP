import tempfile
import unittest
from pathlib import Path
import pandas as pd
from bank_classifier.data import load_data
from bank_classifier.train import ROOT, train
from bank_classifier.predict import Classifier


class WorkflowTests(unittest.TestCase):
    def test_training_serialization_and_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = train(ROOT / 'data/demo.csv', root / 'models', root / 'reports')
            self.assertEqual(report['cleaning']['duplicates_removed'], 1)
            self.assertEqual(report['train_rows'] + report['test_rows'], 48)
            classifier = Classifier(root / 'models/pipeline.joblib')
            texts = ['Хочу погасить кредит досрочно', 'Как открыть вклад в банке']
            self.assertEqual(classifier.predict(texts), ['loans', 'deposits'])
            with self.assertRaises(ValueError):
                classifier.predict([' '])
            with self.assertRaises(ValueError):
                classifier.predict(['zzzzunknownword'])
            # Словарь TF-IDF не должен выучить токены из отложенной выборки.
            from sklearn.model_selection import train_test_split
            frame, _ = load_data(ROOT / 'data/demo.csv')
            training, testing = train_test_split(frame, test_size=0.2, stratify=frame.label, random_state=42)
            vectorizer = classifier.pipeline.named_steps['tfidf']
            analyzer = vectorizer.build_analyzer()
            train_tokens = {token for text in training.text for token in analyzer(text)}
            test_tokens = {token for text in testing.text for token in analyzer(text)}
            self.assertTrue(test_tokens - train_tokens)
            self.assertTrue((test_tokens - train_tokens).isdisjoint(vectorizer.vocabulary_))

    def test_conflicting_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.csv'
            pd.DataFrame({'text': [' ТЕКСТ ', 'текст'], 'label': ['a', 'b']}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'разные метки'):
                load_data(path)


if __name__ == '__main__':
    unittest.main()
