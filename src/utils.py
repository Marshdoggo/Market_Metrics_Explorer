import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from typing import List

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)
    return CACHE_DIR

def trading_days_ago(n: int) -> pd.Timestamp:
    # Approximate n trading days ~ n * 1.4 calendar days buffer
    return pd.Timestamp.today(tz='UTC') - pd.Timedelta(days=int(n*1.4))

def to_annualized(mean_daily_ret: float, daily_std: float, periods=252):
    sharpe = np.nan
    if daily_std and daily_std != 0:
        sharpe = (mean_daily_ret * periods) / (daily_std * np.sqrt(periods))
    return sharpe

def pct_from_ma(series: pd.Series, window: int) -> float:
    if len(series) < window:
        return np.nan
    ma = series.tail(window).mean()
    last = series.iloc[-1]
    if ma == 0:
        return np.nan
    return (last - ma) / ma * 100.0

def parse_tickers(text: str) -> List[str]:
    """Parse a comma/space-separated string of tickers (case-insensitive).
    - Trims whitespace
    - Removes leading '$'
    - Normalizes FX like 'EUR/USD' or 'EUR-USD' -> 'EURUSD'
    - Deduplicates while preserving order
    Returns a list of UPPERCASE tickers.
    """
    if not text:
        return []
    # Split on commas or whitespace
    raw = [t for t in re.split(r"[\s,]+", text) if t]
    cleaned: List[str] = []
    for t in raw:
        t = t.strip().upper()
        if not t:
            continue
        if t.startswith("$"):
            t = t[1:]
        # Normalize FX formats
        t = t.replace("/", "")  # EUR/USD -> EURUSD
        t = t.replace("-", "")  # EUR-USD -> EURUSD
        if t:
            cleaned.append(t)
    # Deduplicate preserving order
    out: List[str] = []
    seen = set()
    for t in cleaned:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out
