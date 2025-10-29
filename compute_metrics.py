import pandas as pd
import numpy as np
from typing import Dict, Callable
from metrics_registry import METRICS

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
    return df
