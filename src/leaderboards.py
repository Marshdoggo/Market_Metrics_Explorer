from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests


SUPPORTED_METRICS = [
    "Annualized Sharpe",
    "Sortino Ratio",
    "CAGR",
    "Mean Daily Return",
    "Daily Volatility (Std)",
    "Max Drawdown",
    "Downside Deviation",
    "RSI_14",
    "% Above 50DMA",
    "% Above 200DMA",
    "Return Skewness",
    "Return Kurtosis (Fisher)",
    "Bollinger Bandwidth (20)",
    "Dip_Buy_Score",
    "Bear_Breakdown_Score",
    "Momentum_Continuation_Score",
    "Distance_50DMA",
    "Distance_200DMA",
    "Drawdown_3M_High",
    "Return_20D",
    "Return_60D",
    "Realized_Vol_20D",
    "ATR_14_Pct",
    "Volume_Ratio_20D_60D",
]

METRIC_ALIASES = {
    "Daily Volatility": "Daily Volatility (Std)",
    "RSI": "RSI_14",
    "RSI(14)": "RSI_14",
}

HIGHER_IS_STRONGER = {
    "Annualized Sharpe": True,
    "Sortino Ratio": True,
    "CAGR": True,
    "Mean Daily Return": True,
    "Daily Volatility (Std)": False,
    "Max Drawdown": True,
    "Downside Deviation": False,
    "RSI_14": True,
    "% Above 50DMA": True,
    "% Above 200DMA": True,
    "Return Skewness": True,
    "Return Kurtosis (Fisher)": False,
    "Bollinger Bandwidth (20)": False,
    "Dip_Buy_Score": True,
    "Bear_Breakdown_Score": True,
    "Momentum_Continuation_Score": True,
    "Distance_50DMA": True,
    "Distance_200DMA": True,
    "Drawdown_3M_High": True,
    "Return_20D": True,
    "Return_60D": True,
    "Realized_Vol_20D": False,
    "ATR_14_Pct": False,
    "Volume_Ratio_20D_60D": True,
}

SNAPSHOT_COLUMNS = [
    "as_of_date",
    "universe",
    "lookback",
    "metric",
    "ticker",
    "name",
    "sector",
    "subindustry",
    "metric_value",
    "rank",
    "rank_direction",
    "generated_at",
    "source_mode",
]


def canonical_metric(metric: str) -> str:
    return METRIC_ALIASES.get(metric, metric)


def available_leaderboard_metrics(metrics_df: pd.DataFrame) -> list[str]:
    cols = set(metrics_df.columns)
    return [m for m in SUPPORTED_METRICS if m in cols]


def metric_sort_ascending(metric: str, direction: str = "strongest") -> bool:
    metric = canonical_metric(metric)
    strongest_high = HIGHER_IS_STRONGER.get(metric, True)
    if direction == "weakest":
        return strongest_high
    return not strongest_high


def leaderboard_table(
    metrics_df: pd.DataFrame,
    *,
    metric: str,
    n: int = 10,
    direction: str = "strongest",
    display_metrics: list[str] | None = None,
) -> pd.DataFrame:
    metric = canonical_metric(metric)
    if metric not in metrics_df.columns:
        return pd.DataFrame()
    df = metrics_df.copy()
    if "Ticker" not in df.columns:
        df = df.reset_index().rename(columns={"index": "Ticker"})
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=[metric])
    if df.empty:
        return df
    ascending = metric_sort_ascending(metric, direction)
    out = df.sort_values(metric, ascending=ascending).head(int(n)).copy()
    out.insert(0, "Rank", range(1, len(out) + 1))
    extra_metrics = [canonical_metric(m) for m in (display_metrics or [])]
    extra_metrics = [m for m in extra_metrics if m in out.columns and m != metric]
    cols = ["Rank", "Ticker", "Name", "Sector", "SubIndustry", metric] + extra_metrics
    return out[[c for c in cols if c in out.columns]]


def build_leaderboard_snapshots(
    metrics_df: pd.DataFrame,
    *,
    universe: str,
    as_of_date: str,
    lookback: int,
    source_mode: str,
    generated_at: str | None = None,
) -> pd.DataFrame:
    df = metrics_df.copy()
    if "Ticker" not in df.columns:
        df = df.reset_index().rename(columns={"index": "Ticker"})
    if generated_at is None:
        generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    rows: list[dict[str, Any]] = []
    for metric in available_leaderboard_metrics(df):
        x = df[["Ticker"] + [c for c in ["Name", "Sector", "SubIndustry", metric] if c in df.columns]].copy()
        x[metric] = pd.to_numeric(x[metric], errors="coerce")
        x = x.dropna(subset=[metric])
        if x.empty:
            continue
        ascending = metric_sort_ascending(metric, "strongest")
        x = x.sort_values(metric, ascending=ascending)
        rank_direction = "desc" if not ascending else "asc"
        for rank, (_, r) in enumerate(x.iterrows(), start=1):
            rows.append(
                {
                    "as_of_date": str(as_of_date)[:10],
                    "universe": universe,
                    "lookback": int(lookback),
                    "metric": metric,
                    "ticker": str(r.get("Ticker", "")).upper(),
                    "name": _clean_text(r.get("Name")),
                    "sector": _clean_text(r.get("Sector")),
                    "subindustry": _clean_text(r.get("SubIndustry")),
                    "metric_value": float(r.get(metric)),
                    "rank": int(rank),
                    "rank_direction": rank_direction,
                    "generated_at": generated_at,
                    "source_mode": source_mode,
                }
            )
    return pd.DataFrame(rows, columns=SNAPSHOT_COLUMNS)


def snapshot_path(data_repo: Path, universe: str, lookback: int) -> Path:
    return Path(data_repo) / "leaderboards" / universe / f"lookback_{int(lookback)}.parquet"


def read_snapshot_file(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    return pd.read_parquet(path)


def write_snapshot_file(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for col in SNAPSHOT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[SNAPSHOT_COLUMNS]
    out.to_parquet(path, index=False, engine="pyarrow")


def append_snapshot_rows(
    new_rows: pd.DataFrame,
    path: Path,
    *,
    force: bool = False,
) -> tuple[pd.DataFrame, int]:
    existing = read_snapshot_file(path)
    if force or existing.empty:
        combined = new_rows.copy()
        added = len(combined)
    else:
        key_cols = ["as_of_date", "universe", "lookback", "metric", "ticker"]
        existing_keys = set(map(tuple, existing[key_cols].astype(str).to_records(index=False)))
        mask = ~new_rows[key_cols].astype(str).apply(tuple, axis=1).isin(existing_keys)
        added_rows = new_rows.loc[mask].copy()
        combined = pd.concat([existing, added_rows], ignore_index=True)
        added = len(added_rows)
    if not combined.empty:
        combined = combined.drop_duplicates(
            subset=["as_of_date", "universe", "lookback", "metric", "ticker"],
            keep="last" if force else "first",
        )
        combined["as_of_date"] = combined["as_of_date"].astype(str)
        combined = combined.sort_values(["as_of_date", "metric", "rank", "ticker"])
    write_snapshot_file(combined, path)
    return combined, int(added)


def load_snapshots_from_url(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return pd.read_parquet(BytesIO(resp.content))


def latest_snapshot_date(history: pd.DataFrame, metric: str | None = None) -> str | None:
    if history is None or history.empty:
        return None
    df = history
    if metric is not None:
        df = df[df["metric"] == canonical_metric(metric)]
    if df.empty:
        return None
    dates = pd.to_datetime(df["as_of_date"], errors="coerce").dropna()
    if dates.empty:
        return None
    return dates.max().date().isoformat()


def current_leaderboard_with_change(history: pd.DataFrame, metric: str, top_n: int) -> pd.DataFrame:
    metric = canonical_metric(metric)
    h = _metric_history(history, metric)
    if h.empty:
        return pd.DataFrame()
    dates = _recent_dates(h)
    if not dates:
        return pd.DataFrame()
    current_date = dates[-1]
    prev_date = dates[-2] if len(dates) > 1 else None
    cur = h[h["as_of_date"] == current_date].sort_values("rank").head(int(top_n)).copy()
    cur = cur.rename(columns={"rank": "current_rank"})
    if prev_date:
        prev = h[h["as_of_date"] == prev_date][["ticker", "rank"]].rename(columns={"rank": "previous_rank"})
        cur = cur.merge(prev, on="ticker", how="left")
        cur["rank_change"] = cur["previous_rank"] - cur["current_rank"]
    else:
        cur["previous_rank"] = pd.NA
        cur["rank_change"] = pd.NA
    return cur


def movement_summary(history: pd.DataFrame, metric: str, top_n: int = 10, window: int = 20) -> dict[str, pd.DataFrame | bool]:
    metric = canonical_metric(metric)
    h = _metric_history(history, metric)
    if h.empty:
        return {"available": False}
    dates = _recent_dates(h)[-int(window):]
    if not dates:
        return {"available": False}
    hw = h[h["as_of_date"].isin(dates)].copy()
    current_date = dates[-1]
    previous_date = dates[-2] if len(dates) > 1 else None
    current = hw[hw["as_of_date"] == current_date].sort_values("rank")
    current_top = set(current.head(int(top_n))["ticker"])

    if previous_date:
        prev = hw[hw["as_of_date"] == previous_date].sort_values("rank")
        prev_top = set(prev.head(int(top_n))["ticker"])
        new_entrants = current[current["ticker"].isin(current_top - prev_top)]
        dropouts = prev[prev["ticker"].isin(prev_top - current_top)]
        full_delta = current[["ticker", "name", "sector", "rank", "metric_value"]].merge(
            prev[["ticker", "rank"]].rename(columns={"rank": "previous_rank"}),
            on="ticker",
            how="left",
        )
        full_delta["rank_change"] = full_delta["previous_rank"] - full_delta["rank"]
    else:
        new_entrants = current.head(int(top_n))
        dropouts = current.head(0)
        full_delta = current[["ticker", "name", "sector", "rank", "metric_value"]].copy()
        full_delta["previous_rank"] = pd.NA
        full_delta["rank_change"] = pd.NA

    persistence = (
        hw.assign(top5=hw["rank"] <= 5, top10=hw["rank"] <= 10)
        .groupby(["ticker", "name", "sector"], dropna=False)
        .agg(days_in_top_5=("top5", "sum"), days_in_top_10=("top10", "sum"), latest_rank=("rank", "last"))
        .reset_index()
        .sort_values(["days_in_top_5", "days_in_top_10", "latest_rank"], ascending=[False, False, True])
    )
    climbers = full_delta.dropna(subset=["rank_change"]).sort_values("rank_change", ascending=False).head(int(top_n))
    fallers = full_delta.dropna(subset=["rank_change"]).sort_values("rank_change", ascending=True).head(int(top_n))
    return {
        "available": True,
        "current": current_leaderboard_with_change(hw, metric, top_n),
        "new_entrants": new_entrants.head(int(top_n)),
        "dropouts": dropouts.head(int(top_n)),
        "persistence": persistence.head(int(top_n)),
        "climbers": climbers,
        "fallers": fallers,
        "has_previous": previous_date is not None,
    }


def rank_time_series(history: pd.DataFrame, metric: str, tickers: list[str], window: int = 20) -> pd.DataFrame:
    metric = canonical_metric(metric)
    h = _metric_history(history, metric)
    if h.empty or not tickers:
        return pd.DataFrame()
    dates = _recent_dates(h)[-int(window):]
    want = {str(t).upper() for t in tickers}
    out = h[h["as_of_date"].isin(dates) & h["ticker"].isin(want)].copy()
    out["as_of_date"] = pd.to_datetime(out["as_of_date"], errors="coerce")
    return out.sort_values(["as_of_date", "rank"])


def report_context(history: pd.DataFrame, metrics_df: pd.DataFrame, *, top_n: int = 5, window: int = 20) -> dict[str, Any]:
    facts: dict[str, Any] = {"available": False}
    if history is None or history.empty:
        return facts
    metrics = available_leaderboard_metrics(metrics_df)
    facts = {"available": True, "metrics": {}}
    for metric in metrics[:]:
        summary = movement_summary(history, metric, top_n=top_n, window=window)
        current = current_leaderboard_with_change(history, metric, top_n)
        facts["metrics"][metric] = {
            "current": _compact_rows(current, rank_col="current_rank"),
            "new_entrants": _compact_rows(summary.get("new_entrants", pd.DataFrame())),
            "dropouts": _compact_rows(summary.get("dropouts", pd.DataFrame())),
            "persistence": _compact_rows(summary.get("persistence", pd.DataFrame())),
            "climbers": _compact_rows(summary.get("climbers", pd.DataFrame())),
            "fallers": _compact_rows(summary.get("fallers", pd.DataFrame())),
            "has_previous": bool(summary.get("has_previous")),
        }
    return facts


def _metric_history(history: pd.DataFrame, metric: str) -> pd.DataFrame:
    if history is None or history.empty or "metric" not in history.columns:
        return pd.DataFrame()
    h = history[history["metric"] == canonical_metric(metric)].copy()
    if h.empty:
        return h
    h["as_of_date"] = h["as_of_date"].astype(str).str[:10]
    h["rank"] = pd.to_numeric(h["rank"], errors="coerce")
    h["metric_value"] = pd.to_numeric(h["metric_value"], errors="coerce")
    return h.dropna(subset=["rank"])


def _recent_dates(history: pd.DataFrame) -> list[str]:
    dates = pd.to_datetime(history["as_of_date"], errors="coerce").dropna().drop_duplicates().sort_values()
    return [d.date().isoformat() for d in dates]


def _compact_rows(df: Any, *, rank_col: str = "rank", max_rows: int = 5) -> list[dict[str, Any]]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []
    rows = []
    for _, r in df.head(max_rows).iterrows():
        row = {
            "ticker": r.get("ticker"),
            "name": r.get("name"),
            "sector": r.get("sector"),
            "rank": _safe_int(r.get(rank_col, r.get("rank"))),
            "previous_rank": _safe_int(r.get("previous_rank")),
            "rank_change": _safe_int(r.get("rank_change")),
            "metric_value": _safe_float(r.get("metric_value")),
            "days_in_top_5": _safe_int(r.get("days_in_top_5")),
            "days_in_top_10": _safe_int(r.get("days_in_top_10")),
        }
        rows.append({k: v for k, v in row.items() if v is not None and not pd.isna(v)})
    return rows


def _clean_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return round(float(value), 6)
    except Exception:
        return None
