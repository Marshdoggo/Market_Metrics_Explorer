import pandas as pd
import numpy as np
from typing import Dict, Callable
from metrics_registry import METRICS
from technical_indicators import compute_signal_scores, compute_technical_metrics


def _wide_close_to_history(prices_wide: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        str(ticker): pd.DataFrame({"Close": prices_wide[ticker]})
        for ticker in prices_wide.columns
    }

def compute_all_metrics(prices_wide: pd.DataFrame, lookback: int) -> pd.DataFrame:
    # Restrict to last lookback rows
    if lookback and lookback < len(prices_wide):
        data = prices_wide.tail(lookback)
    else:
        data = prices_wide.copy()
    out = {}
    for ticker in data.columns:
        series = data[ticker].dropna()
        if len(series) < max(60, int(lookback*0.4)):
            continue
        vals = {}
        for name, func in METRICS.items():
            try:
                vals[name] = func(series)
            except Exception:
                vals[name] = np.nan
        out[ticker] = vals
    df = pd.DataFrame.from_dict(out, orient='index')
    if df.empty:
        return df

    # Standardize RSI naming for the expanded technical layer while preserving the
    # existing computation path that produced RSI(14).
    if "RSI(14)" in df.columns and "RSI_14" not in df.columns:
        df = df.rename(columns={"RSI(14)": "RSI_14"})

    try:
        technical = compute_technical_metrics(_wide_close_to_history(prices_wide))
        if not technical.empty:
            if "RSI_14" in df.columns and "RSI_14" in technical.columns:
                df["RSI_14"] = df["RSI_14"].combine_first(technical["RSI_14"])
                technical = technical.drop(columns=["RSI_14"])
            df = df.join(technical, how="left")
    except Exception:
        pass

    df = compute_signal_scores(df)
    return df
