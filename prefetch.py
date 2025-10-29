# src/prefetch.py
from __future__ import annotations
import os
import pandas as pd
from datetime import timedelta
from universes import get_universe
from fetch_data import download_prices, download_prices_fx_window

def prewarm_equities(universes=("sp500", "nasdaq100", "dow30"), force_refresh=False):
    for u in universes:
        tickers = get_universe(u, force_refresh=False)["Ticker"].tolist()
        # This call uses your existing equity cache pathing under fetch_data.download_prices
        download_prices(tickers, force_refresh=force_refresh)

def prewarm_fx(lookback_days=252, force_refresh=False):
    pairs = get_universe("fx", force_refresh=False)["Ticker"].tolist()
    asof = pd.Timestamp.today(tz="UTC")
    download_prices_fx_window(pairs, lookback_trading_days=lookback_days, asof=asof, force_refresh=force_refresh)

def prewarm_all(force_refresh=False):
    prewarm_equities(force_refresh=force_refresh)
    prewarm_fx(force_refresh=force_refresh)

if __name__ == "__main__":
    # CLI-friendly prewarm
    prewarm_all(force_refresh=False)