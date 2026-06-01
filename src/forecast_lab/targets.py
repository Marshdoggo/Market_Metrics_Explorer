from __future__ import annotations

import pandas as pd


def build_top_n_target(metric_history: pd.DataFrame, *, top_n: int, horizon: int) -> pd.DataFrame:
    """Return date/ticker labels for membership in top N after horizon snapshots."""
    if metric_history is None or metric_history.empty:
        return pd.DataFrame(columns=["as_of_date", "ticker", "target", "target_as_of_date"])

    h = metric_history.copy()
    h["as_of_date"] = h["as_of_date"].astype(str).str[:10]
    h["rank"] = pd.to_numeric(h["rank"], errors="coerce")
    h = h.dropna(subset=["as_of_date", "ticker", "rank"])
    if h.empty:
        return pd.DataFrame(columns=["as_of_date", "ticker", "target", "target_as_of_date"])

    dates = (
        pd.to_datetime(h["as_of_date"], errors="coerce")
        .dropna()
        .drop_duplicates()
        .sort_values()
        .dt.date.astype(str)
        .tolist()
    )
    if len(dates) <= int(horizon):
        return pd.DataFrame(columns=["as_of_date", "ticker", "target", "target_as_of_date"])

    date_map = {dates[i]: dates[i + int(horizon)] for i in range(len(dates) - int(horizon))}
    future = h[["as_of_date", "ticker", "rank"]].copy()
    future["target"] = (future["rank"] <= int(top_n)).astype(int)
    future = future.rename(columns={"as_of_date": "target_as_of_date"})

    rows = h[["as_of_date", "ticker"]].copy()
    rows["target_as_of_date"] = rows["as_of_date"].map(date_map)
    rows = rows.dropna(subset=["target_as_of_date"])
    rows = rows.merge(future[["target_as_of_date", "ticker", "target"]], on=["target_as_of_date", "ticker"], how="left")
    rows = rows.dropna(subset=["target"])
    rows["target"] = rows["target"].astype(int)
    return rows[["as_of_date", "ticker", "target", "target_as_of_date"]]

