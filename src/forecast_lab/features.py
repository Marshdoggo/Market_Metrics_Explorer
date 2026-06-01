from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from leaderboards import HIGHER_IS_STRONGER, canonical_metric
except Exception:  # pragma: no cover - import path differs in some test runners
    HIGHER_IS_STRONGER = {}

    def canonical_metric(metric: str) -> str:
        return metric

from .targets import build_top_n_target


KEY_METRICS = [
    "Annualized Sharpe",
    "Sortino Ratio",
    "CAGR",
    "Mean Daily Return",
    "Daily Volatility (Std)",
    "Max Drawdown",
    "Downside Deviation",
    "RSI(14)",
]


def metric_history(history: pd.DataFrame, metric: str, universe: str | None = None) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame()
    metric = canonical_metric(metric)
    h = history.copy()
    if universe and "universe" in h.columns:
        h = h[h["universe"].astype(str).str.lower() == str(universe).lower()]
    if "metric" not in h.columns:
        return pd.DataFrame()
    h = h[h["metric"].astype(str) == metric].copy()
    if h.empty:
        return h
    h["as_of_date"] = h["as_of_date"].astype(str).str[:10]
    h["ticker"] = h["ticker"].astype(str).str.upper()
    h["rank"] = pd.to_numeric(h["rank"], errors="coerce")
    h["metric_value"] = pd.to_numeric(h["metric_value"], errors="coerce")
    return h.dropna(subset=["as_of_date", "ticker", "rank"])


def available_history_metrics(history: pd.DataFrame, universe: str | None = None) -> list[str]:
    if history is None or history.empty or "metric" not in history.columns:
        return []
    h = history
    if universe and "universe" in h.columns:
        h = h[h["universe"].astype(str).str.lower() == str(universe).lower()]
    metrics = h["metric"].dropna().astype(str).unique().tolist()
    preferred = [m for m in KEY_METRICS if m in metrics]
    return preferred + sorted([m for m in metrics if m not in preferred])


def snapshot_version(history: pd.DataFrame) -> str:
    if history is None or history.empty:
        return "empty"
    dates = pd.to_datetime(history.get("as_of_date"), errors="coerce").dropna()
    rows = len(history)
    latest = dates.max().date().isoformat() if not dates.empty else "unknown"
    generated = str(history.get("generated_at", pd.Series(dtype=str)).dropna().astype(str).max() or "")
    return f"{latest}:{rows}:{generated}"


def build_feature_frame(history: pd.DataFrame, *, universe: str, metric: str) -> tuple[pd.DataFrame, list[str]]:
    metric = canonical_metric(metric)
    h_metric = metric_history(history, metric, universe)
    if h_metric.empty:
        return pd.DataFrame(), []

    base_cols = ["as_of_date", "ticker", "rank", "metric_value"]
    for col in ["name", "sector", "subindustry"]:
        if col in h_metric.columns:
            base_cols.append(col)
    base = h_metric[base_cols].copy().rename(
        columns={"rank": "current_rank", "metric_value": "selected_metric_value"}
    )
    base = base.sort_values(["ticker", "as_of_date"])

    feature_cols = ["current_rank", "selected_metric_value"]
    for lag in [1, 5, 20]:
        prior = base.groupby("ticker")["current_rank"].shift(lag)
        col = f"rank_change_{lag}"
        base[col] = prior - base["current_rank"]
        feature_cols.append(col)

    for threshold in [5, 10]:
        top_col = f"is_top_{threshold}"
        days_col = f"days_in_top_{threshold}_20"
        streak_col = f"top_{threshold}_streak"
        base[top_col] = (base["current_rank"] <= threshold).astype(int)
        base[days_col] = (
            base.groupby("ticker")[top_col]
            .transform(lambda s: s.rolling(20, min_periods=1).sum())
            .astype(float)
        )
        base[streak_col] = base.groupby("ticker", group_keys=False)[top_col].apply(_current_streak)
        feature_cols.extend([days_col, streak_col])

    value_history = history.copy()
    if "universe" in value_history.columns:
        value_history = value_history[value_history["universe"].astype(str).str.lower() == str(universe).lower()]
    value_history["as_of_date"] = value_history["as_of_date"].astype(str).str[:10]
    value_history["ticker"] = value_history["ticker"].astype(str).str.upper()
    value_history["metric_value"] = pd.to_numeric(value_history["metric_value"], errors="coerce")
    present_metrics = [m for m in KEY_METRICS if m in set(value_history["metric"].dropna().astype(str))]
    if present_metrics:
        wide = (
            value_history[value_history["metric"].isin(present_metrics)]
            .pivot_table(index=["as_of_date", "ticker"], columns="metric", values="metric_value", aggfunc="last")
            .reset_index()
        )
        wide.columns.name = None
        base = base.merge(wide, on=["as_of_date", "ticker"], how="left")
        for m in present_metrics:
            feature_cols.append(m)
            median_col = f"{m} distance_from_median"
            pct_col = f"{m} percentile"
            direction = bool(HIGHER_IS_STRONGER.get(m, True))
            base[median_col] = base[m] - base.groupby("as_of_date")[m].transform("median")
            base[pct_col] = base.groupby("as_of_date")[m].rank(pct=True, ascending=not direction)
            feature_cols.extend([median_col, pct_col])

    base["universe"] = universe
    feature_cols = [c for c in feature_cols if c in base.columns]
    return base, feature_cols


def build_supervised_frame(
    history: pd.DataFrame,
    *,
    universe: str,
    metric: str,
    top_n: int,
    horizon: int,
) -> tuple[pd.DataFrame, list[str]]:
    features, feature_cols = build_feature_frame(history, universe=universe, metric=metric)
    if features.empty:
        return pd.DataFrame(), []
    labels = build_top_n_target(metric_history(history, metric, universe), top_n=top_n, horizon=horizon)
    if labels.empty:
        return pd.DataFrame(), feature_cols
    out = features.merge(labels, on=["as_of_date", "ticker"], how="inner")
    return out.sort_values(["as_of_date", "ticker"]), feature_cols


def latest_feature_rows(history: pd.DataFrame, *, universe: str, metric: str) -> tuple[pd.DataFrame, list[str], str | None]:
    features, feature_cols = build_feature_frame(history, universe=universe, metric=metric)
    if features.empty:
        return pd.DataFrame(), feature_cols, None
    dates = pd.to_datetime(features["as_of_date"], errors="coerce").dropna()
    if dates.empty:
        return pd.DataFrame(), feature_cols, None
    latest = dates.max().date().isoformat()
    return features[features["as_of_date"] == latest].copy(), feature_cols, latest


def _current_streak(values: pd.Series) -> pd.Series:
    arr = values.fillna(0).astype(int).to_numpy()
    streaks = np.zeros(len(arr), dtype=float)
    current = 0
    for i, val in enumerate(arr):
        current = current + 1 if val == 1 else 0
        streaks[i] = current
    return pd.Series(streaks, index=values.index)

