from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train_dates: list[str]
    test_dates: list[str]
    split_date: str


def time_train_test_split(df: pd.DataFrame, *, test_fraction: float = 0.25) -> TimeSplit:
    dates = (
        pd.to_datetime(df["as_of_date"], errors="coerce")
        .dropna()
        .drop_duplicates()
        .sort_values()
        .dt.date.astype(str)
        .tolist()
    )
    if len(dates) < 4:
        raise ValueError("Need at least 4 historical snapshot dates for time-aware validation.")
    test_count = max(1, int(round(len(dates) * float(test_fraction))))
    train_count = len(dates) - test_count
    if train_count < 2:
        train_count = len(dates) - 1
    train_dates = dates[:train_count]
    test_dates = dates[train_count:]
    return TimeSplit(train_dates=train_dates, test_dates=test_dates, split_date=test_dates[0])


def classification_summary(y_true, y_pred, y_proba) -> dict:
    y_true = pd.Series(y_true).astype(int).reset_index(drop=True)
    y_pred = pd.Series(y_pred).astype(int).reset_index(drop=True)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    out = {
        "accuracy": float((tp + tn) / len(y_true)) if len(y_true) else 0.0,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": None,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }
    if y_true.nunique() == 2:
        out["roc_auc"] = _roc_auc(y_true.to_numpy(), pd.Series(y_proba).astype(float).to_numpy())
    return out


def _roc_auc(y_true, y_score) -> float:
    pairs = pd.DataFrame({"y": y_true, "score": y_score}).dropna()
    positives = int((pairs["y"] == 1).sum())
    negatives = int((pairs["y"] == 0).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = pairs["score"].rank(method="average")
    pos_rank_sum = float(ranks[pairs["y"] == 1].sum())
    auc = (pos_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    return float(auc)
