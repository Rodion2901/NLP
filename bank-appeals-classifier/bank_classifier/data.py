"""Загрузка, проверка и удаление дублей до разделения выборки."""
from pathlib import Path
import pandas as pd

def texts_normalize(texts:str)->str:
    return " ".join(texts.lower().split())

def load_data(path: str | Path) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not {"text", "label"}.issubset(frame.columns):
        raise ValueError("CSV должен содержать столбцы text и label")
    frame = frame[["text", "label"]].copy()
    total = len(frame)
    # Такая же нормализация текста используется при инференсе.
    frame["text"] = frame["text"].map(texts_normalize)
    frame["label"] = frame["label"].str.strip()
    valid = frame["text"].str.contains(r"\w", regex=True) & frame["label"].ne("")
    invalid = int((~valid).sum())
    frame = frame.loc[valid]
    if (frame.groupby("text")["label"].nunique() > 1).any():
        raise ValueError("Одинаковые тексты имеют разные метки: исправьте разметку")
    before = len(frame)
    frame = frame.drop_duplicates("text").reset_index(drop=True)
    counts = frame["label"].value_counts()
    if len(counts) < 2:
        raise ValueError("Нужно минимум два класса")
    if counts.min() < 2:
        raise ValueError("После очистки нужно минимум два текста в каждом классе")
    return frame, {"input_rows": total, "invalid_rows": invalid,
                   "duplicates_removed": before - len(frame), "clean_rows": len(frame)}
