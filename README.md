# Мои pet-проекты

Это репозиторий с моими учебными и практическими проектами по обработке
текста и машинному обучению. Каждый проект находится в отдельной папке и
может запускаться независимо.

## 1. `bank-appeals-classifier`

Классификатор обращений клиентов банка по категориям: карты (`cards`),
кредиты (`loans`), переводы (`transfers`) и вклады (`deposits`).

Что внутри:

- очистка CSV: нормализация регистра и пробелов, удаление пустых строк и
  дублей, проверка конфликтующих меток;
- `TfidfVectorizer` с `ngram_range=(1, 1)`, то есть используются униграммы
  (отдельные слова);
- `LinearSVC` с `class_weight="balanced"` для классификации обращений;
- единый `sklearn Pipeline`, который сохраняется в
  `models/pipeline.joblib`;
- оценка качества на стратифицированной выборке 80/20 с `seed=42`,
  включая accuracy, macro-F1 и матрицу ошибок;
- Python API и командная строка для предсказаний;
- HTTP API на Rust в папке `api/`, сохраняющий предсказания в SQLite.

Запуск:

```bash
cd bank-appeals-classifier
python -m pip install -r requirements.txt
python -m bank_classifier.train --data data/demo.csv
python -m bank_classifier.predict "Потерял банковскую карту"
python -m unittest discover -s tests -v
```

`data/demo.csv` — небольшой синтетический датасет для демонстрации. Он нужен
для проверки проекта и не показывает качество модели на реальных обращениях.
Подробнее о формате данных, API и ограничениях написано в
[`bank-appeals-classifier/README.md`](bank-appeals-classifier/README.md).

## 2. `feedback_classifier`

Классификатор отзывов. Он принимает CSV с текстом отзыва и меткой, очищает
данные и обучает последовательность:

```text
TF-IDF → LogisticRegression
```

Особенности проекта:

- поддерживаются настраиваемые имена колонок `text` и `label`;
- удаляются пустые записи, конфликтующие метки и дубли;
- данные делятся на train/test стратифицированно;
- модель сохраняется в `model/pipeline.joblib`;
- метрики обучения сохраняются в `reports/report.json`;
- есть отдельный скрипт предсказания и тесты.

Запуск:

```bash
cd feedback_classifier
python -m pip install -r requirements.txt
python train.py --dataset data/reviews.csv
python predict.py --model model/pipeline.joblib \
  --text "Понравилось обслуживание"
python -m unittest discover -s tests -v
```

## Общие замечания

Оба проекта используют демонстрационные данные и предназначены прежде всего
для обучения и экспериментов. Перед применением на реальных пользовательских
данных стоит отдельно проверить качество, устойчивость к новым формулировкам
и отсутствие персональной информации в отчётах.
