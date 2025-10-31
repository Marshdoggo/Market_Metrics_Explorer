# src/fetch_data.py
from __future__ import annotations
import os, io, time, json, math, textwrap
from typing import List, Iterable, Tuple, Optional, Dict

import pandas as pd
import numpy as np
import requests
import yfinance as yf
import plotly
try:
    # fallback price source
    import pandas_datareader.data as web  # stooq backend
except Exception:
    web = None

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


def _to_naive_ts(x):
    if x is None:
        return None
    ts = pd.to_datetime(x, errors="coerce")
    if isinstance(ts, pd.Timestamp) and ts.tz is not None:
        try:
            ts = ts.tz_convert(None)
        except Exception:
            ts = ts.tz_localize(None)
    return ts

# ---------------------- S&P500 Constituents ----------------------
# If you already have this in another file, keep using that. Minimal impl here:

_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_SP500_CACHE = os.path.join(DATA_DIR, "sp500_constituents.parquet")

# Prefer cache -> local CSV -> live Wikipedia (only when forced or nothing else works)
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
# Candidates: repo root and data/ folder
_CSV_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "sp500_tickers.csv"),
    os.path.join(DATA_DIR, "sp500_tickers.csv"),
]

def _from_local_csv() -> pd.DataFrame:
    """
    Read sp500_tickers.csv from common locations.
    Accepts either a single 'Ticker' column or 'Ticker,Name,Sector,SubIndustry'.
    Returns normalized columns: Ticker, Name, Sector, SubIndustry
    """
    df = None
    for path in _CSV_CANDIDATES:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                break
            except Exception:
                continue
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Normalize column names and values
    cols = {c.lower(): c for c in df.columns}
    if "ticker" not in cols:
        return pd.DataFrame()
    
    def _get(col, default=""):
        return df[cols[col]] if col in cols else default
    
    out = pd.DataFrame()
    out["Ticker"] = (
        df[cols["ticker"]]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(".", "-", regex=False)
    )
    out["Name"] = _get("name")
    out["Sector"] = _get("sector")
    out["SubIndustry"] = _get("subindustry")
    _log(f"[fetch_data] Loaded local S&P500 CSV: {len(out)} tickers")
    return out[["Ticker", "Name", "Sector", "SubIndustry"]]

def get_sp500_constituents(force_refresh: bool=False) -> pd.DataFrame:
    """
    Columns: Ticker, Name, Sector, SubIndustry
    Order of preference:
      1) cached parquet (fast, offline),
      2) local CSV fallback in repo,
      3) live Wikipedia (only when forced or nothing else available).
    """
    # 1) Cache first (unless force_refresh)
    if os.path.exists(_SP500_CACHE) and not force_refresh:
        try:
            df = pd.read_parquet(_SP500_CACHE)
            if not df.empty:
                return df
        except Exception:
            pass

    # 2) Local CSV fallback (unless force_refresh)
    if not force_refresh:
        local_df = _from_local_csv()
        if not local_df.empty:
            try:
                local_df.to_parquet(_SP500_CACHE, index=False)
            except Exception:
                pass
            return local_df

    # 3) Live Wikipedia (explicit refresh or last resort)
    try:
        _log("[fetch_data] Fetching S&P500 list from Wikipedia…")
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            )
        }
        html = requests.get(_WIKI_URL, headers=headers, timeout=20).text
        tables = pd.read_html(io.StringIO(html))
        df = tables[0].rename(columns={"Symbol": "Ticker", "Security": "Name"})
        df["Ticker"] = (
            df["Ticker"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
        )
        df = df[
            ["Ticker", "Name", "GICS Sector", "GICS Sub-Industry"]
        ].rename(columns={"GICS Sector": "Sector", "GICS Sub-Industry": "SubIndustry"})

        _log(f"[fetch_data] Wikipedia list fetched: {len(df)} tickers")
        try:
            df.to_parquet(_SP500_CACHE, index=False)
        except Exception:
            pass
        return df
    except Exception as e:
        _log(f"[fetch_data] Wikipedia fetch failed: {e!r}")
        # Last resort: local CSV even if force_refresh was requested
        local_df = _from_local_csv()
        if not local_df.empty:
            return local_df
        return pd.DataFrame(columns=["Ticker", "Name", "Sector", "SubIndustry"])

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
def _download_stooq(symbols: List[str], start=None, end=None) -> pd.DataFrame:
    """
    Fallback downloader using Stooq via pandas-datareader (no API key).
    Tries multiple symbol encodings (SYM, sym, sym.us).
    Returns wide Close-price DataFrame with columns=tickers.
    """
    if web is None or not symbols:
        return pd.DataFrame()

    def _try_one(sym: str) -> Optional[pd.DataFrame]:
        # Try symbol encodings that often work with stooq
        candidates = [sym, sym.lower(), f"{sym.lower()}.us"]
        for c in candidates:
            try:
                df = web.DataReader(c, "stooq", start=start, end=end)
                if df is None or df.empty:
                    continue
                df = df.sort_index()
                close_col = "Close" if "Close" in df.columns else "close" if "close" in df.columns else None
                if close_col is None:
                    continue
                return df[close_col].rename(sym).to_frame()
            except Exception:
                continue
        return None

    frames = []
    for sym in symbols:
        got = _try_one(sym.replace("=X", ""))
        if got is not None and not got.empty:
            frames.append(got)

    out = _safe_concat_price_frames(frames)
    if not out.empty:
        _log(f"[fetch_data] Stooq delivered {out.shape[1]} / {len(symbols)} symbols.")
    return out


def download_prices(
    symbols: Iterable[str],
    start: Optional[pd.Timestamp]=None,
    end: Optional[pd.Timestamp]=None,
    force_refresh: bool=False,
    chunk_size: int=10,
    max_retries: int=3,
    backoff_base: float=1.0
) -> pd.DataFrame:
    """
    Wide DataFrame of prices (auto-adjusted close), columns=tickers.
    Uses on-disk cache unless `force_refresh` is True.
    """
    symbols = _sanitize_symbols(list(symbols))
    requested_symbols = set(symbols)
    if not symbols:
        return pd.DataFrame()

    start = _to_naive_ts(start)
    end   = _to_naive_ts(end)

    # Serve cache if available and not forcing refresh
    if os.path.exists(PRICES_PARQUET) and not force_refresh:
        try:
            cached = pd.read_parquet(PRICES_PARQUET)
            if not cached.empty:
                # Coerce to a clean tz-naive DatetimeIndex before trimming
                idx = pd.to_datetime(cached.index, errors="coerce")
                mask = ~pd.isna(idx)
                if mask.any():
                    cached = cached.loc[mask]
                    cached.index = pd.DatetimeIndex(idx[mask]).tz_localize(None)
                else:
                    cached = pd.DataFrame()
            # Only keep requested symbols if present
            if not cached.empty:
                keep_cols = [c for c in cached.columns if c in requested_symbols]
                cached = cached[keep_cols] if keep_cols else pd.DataFrame()
            # if we requested a window, trim here (only when index is valid)
            if not cached.empty:
                if start is not None:
                    cached = cached[cached.index >= start]
                if end is not None:
                    cached = cached[cached.index <= end]
            if not cached.empty:
                _log(f"[fetch_data] Served {cached.shape[1]} tickers from cache.")
                return cached
        except Exception as e:
            _log(f"[fetch_data] cache read failed: {e!r}")

    # Fresh download (chunked + retries)
    all_frames = []
    all_failed: List[str] = []
    chunks = _chunk(symbols, chunk_size)
    out = pd.DataFrame()

    for i, chunk in enumerate(chunks, 1):
        attempt = 0
        yahoo_ok = False
        df = pd.DataFrame()
        while attempt <= max_retries:
            df, failed = _download_chunk_multi(chunk, start, end)
            # If Yahoo returned at least some columns, accept and move on
            if not df.empty:
                all_frames.append(df)
                all_failed.extend(failed)
                yahoo_ok = True
                break

            # If Yahoo returned nothing at all, don't keep hammering — try Stooq for the whole chunk
            _log(f"[fetch_data] chunk {i}/{len(chunks)} got 0 cols from Yahoo; trying Stooq immediately…")
            stooq_df = _download_stooq(chunk, start=start, end=end)
            if not stooq_df.empty:
                all_frames.append(stooq_df)
                # mark anything Stooq did *not* return as failed
                missing_from_stooq = sorted(set(chunk) - set(stooq_df.columns))
                all_failed.extend(missing_from_stooq)
                yahoo_ok = True
                break

            # Neither Yahoo nor Stooq gave us anything; backoff and retry Yahoo a bit
            attempt += 1
            if attempt <= max_retries:
                sleep_s = backoff_base * (2 ** (attempt - 1))
                _log(f"[fetch_data] chunk {i}/{len(chunks)} retry {attempt} in {sleep_s:.1f}s…")
                time.sleep(sleep_s)

        # If after retries we still have nothing, try splitting the chunk to salvage a few
        if not yahoo_ok and df.empty:
            half = max(1, len(chunk)//2)
            for sub in _chunk(chunk, half):
                sub_df, sub_failed = _download_chunk_multi(sub, start, end)
                if not sub_df.empty:
                    all_frames.append(sub_df)
                else:
                    # last-ditch Stooq per-sub
                    st_sub = _download_stooq(sub, start=start, end=end)
                    if not st_sub.empty:
                        all_frames.append(st_sub)
                        sub_failed = sorted(set(sub) - set(st_sub.columns))
                all_failed.extend(sub_failed)

    out = _safe_concat_price_frames(all_frames)

    # If we’re missing some symbols, try Stooq for the missing ones
    requested = set(symbols)
    have = set(out.columns) if not out.empty else set()
    missing = sorted(requested - have)
    if missing:
        _log(f"[fetch_data] Yahoo missing {len(missing)} symbols — trying Stooq fallback…")
        stooq_df = _download_stooq(missing, start=start, end=end)
        if not stooq_df.empty:
            out = _safe_concat_price_frames([out, stooq_df])

    # Merge with existing cache so we don’t lose previously-fetched columns
    if os.path.exists(PRICES_PARQUET):
        try:
            base = pd.read_parquet(PRICES_PARQUET)
            out = base.combine_first(out).combine_first(out)
        except Exception:
            pass

    if not out.empty:
        try:
            out.to_parquet(PRICES_PARQUET)
        except Exception as e:
            _log(f"[fetch_data] cache write failed: {e!r}")

    # Log succinctly what we still don’t have (after Stooq)
    have = set(out.columns) if not out.empty else set()
    still_missing = sorted(requested - have)
    if still_missing:
        _log(f"[fetch_data] Failed downloads after fallback ({len(still_missing)}): "
             f"{still_missing[:30]}{' …' if len(still_missing)>30 else ''}")

    # Final trims on requested window (only after ensuring a proper DatetimeIndex)
    if not out.empty:
        idx = pd.to_datetime(out.index, errors="coerce")
        mask = ~pd.isna(idx)
        if mask.any():
            out = out.loc[mask]
            out.index = pd.DatetimeIndex(idx[mask]).tz_localize(None)
            if start is not None:
                out = out[out.index >= start]
            if end is not None:
                out = out[out.index <= end]
        else:
            # Index was unusable; return the raw (unfiltered) result
            pass

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