# src/fetch_data.py
from __future__ import annotations
import os, io, time, json, math, textwrap
from typing import List, Iterable, Tuple, Optional, Dict

import pandas as pd
import numpy as np
import requests
import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

PRICES_PARQUET = os.path.join(DATA_DIR, "prices.parquet")
FX_YF_PARQUET  = os.path.join(DATA_DIR, "fx_yf.parquet")

# ---------------------- Utilities ----------------------

def _sanitize_symbols(symbols: Iterable[str]) -> List[str]:
    out = []
    for s in symbols or []:
        if not s: 
            continue
        s = str(s).strip().upper().replace(".", "-")
        # filter very weird things that yfinance chokes on
        if any(ch in s for ch in (" ", "/", "\\")):
            continue
        out.append(s)
    # dedupe, preserve order
    seen = set(); clean = []
    for s in out:
        if s not in seen:
            seen.add(s); clean.append(s)
    return clean

def _chunk(lst: List[str], size: int) -> List[List[str]]:
    return [lst[i:i+size] for i in range(0, len(lst), size)]

def _log(msg: str):
    print(msg, flush=True)

def _safe_concat_price_frames(frames: List[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if isinstance(f, pd.DataFrame) and not f.empty]
    if not frames:
        return pd.DataFrame()
    # frames from yfinance are wide with one column per symbol
    df = pd.concat(frames, axis=1)
    # ensure DateTimeIndex
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    # sort, drop full-null cols
    df = df.sort_index()
    df = df.dropna(axis=1, how="all")
    return df

# ---------------------- S&P500 Constituents ----------------------
# If you already have this in another file, keep using that. Minimal impl here:

_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_SP500_CACHE = os.path.join(DATA_DIR, "sp500_constituents.parquet")

def get_sp500_constituents(force_refresh: bool=False) -> pd.DataFrame:
    if os.path.exists(_SP500_CACHE) and not force_refresh:
        try:
            return pd.read_parquet(_SP500_CACHE)
        except Exception:
            pass
    _log("[fetch_data] Refreshing S&P500 list from Wikipedia…")
    tables = pd.read_html(_WIKI_URL)
    df = tables[0].rename(columns={"Symbol": "Ticker", "Security": "Name"})
    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df.to_parquet(_SP500_CACHE, index=False)
    return df[["Ticker","Name","GICS Sector","GICS Sub-Industry"]].rename(
        columns={"GICS Sector":"Sector","GICS Sub-Industry":"SubIndustry"}
    )

# ---------------------- Robust equity downloader ----------------------

def _download_chunk_multi(tickers: List[str], start=None, end=None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Try yf.download on a chunk (multi). Returns (wide_df, failed_symbols).
    Falls back to per-ticker on partial failures.
    """
    failed = []
    if not tickers:
        return pd.DataFrame(), failed

    # Multi try
    try:
        df = yf.download(
            tickers=" ".join(tickers),
            start=start, end=end,
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=False
        )
        # yfinance multi-format normalization to wide 1-col (Close) if needed
        frames = []
        for sym in tickers:
            try:
                sub = df[sym] if isinstance(df.columns, pd.MultiIndex) else df
                # prefer 'Adj Close' then 'Close'
                col = "Adj Close" if "Adj Close" in sub.columns else "Close" if "Close" in sub.columns else None
                if col is None:
                    failed.append(sym); continue
                series = sub[col].rename(sym)
                frames.append(series.to_frame())
            except Exception:
                failed.append(sym)
        return _safe_concat_price_frames(frames), failed
    except Exception as e:
        _log(f"[fetch_data] multi-chunk error ({len(tickers)}): {e!r}")
        # fall back to per-ticker below

    # Per ticker fallback for the whole chunk
    frames = []
    for sym in tickers:
        try:
            t = yf.Ticker(sym)
            hist = t.history(start=start, end=end, auto_adjust=True)
            if hist is None or hist.empty:
                failed.append(sym); continue
            col = "Close" if "Close" in hist.columns else None
            if col is None:
                failed.append(sym); continue
            frames.append(hist[col].rename(sym).to_frame())
        except Exception as e:
            failed.append(sym)
    return _safe_concat_price_frames(frames), failed


def download_prices(
    symbols: Iterable[str],
    start: Optional[pd.Timestamp]=None,
    end: Optional[pd.Timestamp]=None,
    force_refresh: bool=False,
    chunk_size: int=25,
    max_retries: int=3,
    backoff_base: float=0.8
) -> pd.DataFrame:
    """
    Wide DataFrame of prices (auto-adjusted close), columns=tickers.
    Uses on-disk cache unless `force_refresh` is True.
    """
    symbols = _sanitize_symbols(list(symbols))
    if not symbols:
        return pd.DataFrame()

    # Serve cache if available and not forcing refresh
    if os.path.exists(PRICES_PARQUET) and not force_refresh:
        try:
            cached = pd.read_parquet(PRICES_PARQUET)
            # if we requested a window, trim here
            if start is not None:
                cached = cached[cached.index >= pd.to_datetime(start)]
            if end is not None:
                cached = cached[cached.index <= pd.to_datetime(end)]
            if not cached.empty:
                _log(f"[fetch_data] Served {cached.shape[1]} tickers from cache.")
                return cached
        except Exception as e:
            _log(f"[fetch_data] cache read failed: {e!r}")

    # Fresh download (chunked + retries)
    all_frames = []
    all_failed: List[str] = []
    chunks = _chunk(symbols, chunk_size)

    for i, chunk in enumerate(chunks, 1):
        attempt = 0
        while attempt <= max_retries:
            df, failed = _download_chunk_multi(chunk, start, end)
            # If we got any data, accept (even with partial fails)
            if not df.empty or not failed:
                if not df.empty:
                    all_frames.append(df)
                all_failed.extend(failed)
                break
            # else: full failure for this chunk — backoff and retry
            attempt += 1
            sleep_s = backoff_base * (2 ** (attempt - 1))
            _log(f"[fetch_data] chunk {i}/{len(chunks)} retry {attempt} in {sleep_s:.1f}s…")
            time.sleep(sleep_s)

        # last resort: if still nothing, split the chunk into halves to salvage
        if attempt > max_retries and df.empty:
            half = max(1, len(chunk)//2)
            for sub in _chunk(chunk, half):
                sub_df, sub_failed = _download_chunk_multi(sub, start, end)
                if not sub_df.empty:
                    all_frames.append(sub_df)
                all_failed.extend(sub_failed)

    out = _safe_concat_price_frames(all_frames)

    # Merge with existing cache so we don’t lose old columns
    if os.path.exists(PRICES_PARQUET):
        try:
            base = pd.read_parquet(PRICES_PARQUET)
            out = base.combine_first(out).combine_first(out)  # keep most complete shape
        except Exception:
            pass

    if not out.empty:
        try:
            out.to_parquet(PRICES_PARQUET)
        except Exception as e:
            _log(f"[fetch_data] cache write failed: {e!r}")

    # Final trims on requested window
    if start is not None:
        out = out[out.index >= pd.to_datetime(start)]
    if end is not None:
        out = out[out.index <= pd.to_datetime(end)]

    # Log failures succinctly (Streamlit console shows them)
    failed_unique = sorted(set(all_failed))
    if failed_unique:
        _log(f"[fetch_data] Failed downloads ({len(failed_unique)}): {failed_unique[:30]}{' …' if len(failed_unique)>30 else ''}")

    return out

# ---------------------- FX windowed via yfinance (close) ----------------------

def _fx_to_yf_symbol(pair: str) -> str:
    # EURUSD -> EURUSD=X, USDJPY -> USDJPY=X
    s = str(pair).strip().upper().replace("/", "")
    return f"{s}=X"

def download_prices_fx_window(
    pairs: List[str],
    lookback_trading_days: int,
    asof: pd.Timestamp,
    force_refresh: bool=False,
    chunk_size: int=25
) -> pd.DataFrame:
    """
    Windowed FX fetch using yfinance (close). Returns wide DF columns=pairs.
    """
    yf_syms = [_fx_to_yf_symbol(p) for p in pairs]
    end = pd.Timestamp(asof).normalize()
    # Use business days as in dashboard:
    from pandas.tseries.offsets import BDay
    start = (end - BDay(int(lookback_trading_days))).normalize()

    df = download_prices(yf_syms, start=start, end=end, force_refresh=force_refresh, chunk_size=chunk_size)

    # Map back columns from 'EURUSD=X' -> 'EURUSD'
    colmap = {s: s.replace("=X","") for s in df.columns}
    df = df.rename(columns=colmap)

    # Also persist to an FX cache (optional)
    try:
        if not df.empty:
            df.to_parquet(FX_YF_PARQUET)
    except Exception:
        pass

    return df

# ---------------------- Meta (placeholder kept for API symmetry) ----------------------

def get_meta() -> Dict[str, str]:
    # Keep interface stable. Add richer metadata as needed.
    return {}