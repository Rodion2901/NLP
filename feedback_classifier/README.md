# Классификация отзывов

Проект обучает TF-IDF и логистическую регрессию на размеченном CSV. Предсказания используют метки из датасета.

Из папки проекта установите зависимости и запустите обучение.

```bash
python -m pip install -r requirements.txt
python train.py --dataset data/reviews.csv
```

CSV должен быть в UTF-8. По умолчанию проект читает колонки `text` и `label`. Имена можно изменить через `--text-column` и `--label-column`. Параметры `--test-size` и `--seed` по умолчанию равны `0.2` и `42`.

После обучения пайплайн сохраняется в `model/pipeline.joblib`, а отчёт с метриками — в `reports/report.json`. Относительные пути считаются от текущей папки.

```bash
python predict.py --model model/pipeline.joblib --text "Понравилось обслуживание"
python -m unittest discover -s tests -v
```
