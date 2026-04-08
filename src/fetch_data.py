# src/fetch_data.py
from __future__ import annotations
import os, io, time, json, math, textwrap
from collections import deque
from typing import List, Iterable, Tuple, Optional, Dict


import pandas as pd
import numpy as np
import requests
import yfinance as yf
#import plotly
try:
    # fallback price source
    import pandas_datareader.data as web  # stooq backend
except Exception:
    web = None

# Runtime/environment toggles (safer defaults on Streamlit Cloud)
RUNNING_IN_CLOUD = os.getenv("STREAMLIT_SERVER_ENABLED") == "1" or os.getenv("STREAMLIT_RUNTIME") == "1"
PREFER_STOOQ_DEFAULT = (
    os.getenv("MKTME_PREFER_STOOQ", "").lower() in ("1", "true", "yes")
    or RUNNING_IN_CLOUD
)
EQUITY_SOURCE_DEFAULT = os.getenv("MKTME_EQUITY_SOURCE", "auto").strip().lower() or "auto"
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()
ALPHAVANTAGE_DAILY_LIMIT = int(os.getenv("ALPHAVANTAGE_DAILY_LIMIT", "25") or 25)
ALPHAVANTAGE_PER_MINUTE = int(os.getenv("ALPHAVANTAGE_PER_MINUTE", "5") or 5)
ALPHAVANTAGE_PREMIUM = os.getenv("ALPHAVANTAGE_PREMIUM", "").lower() in ("1", "true", "yes")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "").strip()
TWELVEDATA_DAILY_LIMIT = int(os.getenv("TWELVEDATA_DAILY_LIMIT", "800") or 800)
TWELVEDATA_PER_MINUTE = int(os.getenv("TWELVEDATA_PER_MINUTE", "8") or 8)
TWELVEDATA_OUTPUTSIZE = int(os.getenv("TWELVEDATA_OUTPUTSIZE", "3000") or 3000)
TWELVEDATA_RATE_LIMIT_BUFFER_SECONDS = float(
    os.getenv("TWELVEDATA_RATE_LIMIT_BUFFER_SECONDS", "2") or 2
)
TWELVEDATA_MINUTE_RETRY_LIMIT = int(os.getenv("TWELVEDATA_MINUTE_RETRY_LIMIT", "2") or 2)
ALLOW_SOURCE_PROBE_FAILURE = os.getenv("MKTME_ALLOW_SOURCE_PROBE_FAILURE", "1").lower() in ("1", "true", "yes")
MAX_SYMBOLS_DEFAULT = int(os.getenv("MKTME_MAX_SYMBOLS", "0") or 0)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

PRICES_PARQUET = os.path.join(DATA_DIR, "prices.parquet")
FX_YF_PARQUET  = os.path.join(DATA_DIR, "fx_yf.parquet")
SOURCE_CANARY_SYMBOLS = ["SPY", "AAPL", "MSFT"]
_TWELVEDATA_CREDIT_HISTORY: deque[tuple[float, int]] = deque()

# NEW: allow publisher to force refresh instead of using cache
FORCE_REFRESH = os.getenv("MKTME_FORCE_REFRESH", "").lower() in ("1", "true", "yes")

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
    normalized = []
    for f in frames:
        one = f.copy()
        one.index = pd.to_datetime(one.index, errors="coerce")
        one = one[~one.index.isna()]
        if one.empty:
            continue
        if not one.index.is_unique:
            one = one.groupby(level=0).last()
        normalized.append(one.sort_index())
    if not normalized:
        return pd.DataFrame()
    # frames from yfinance are wide with one column per symbol
    df = pd.concat(normalized, axis=1)
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

    frames = []
    missing = list(tickers)

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
        for sym in tickers:
            try:
                sub = df[sym] if isinstance(df.columns, pd.MultiIndex) else df
                # prefer 'Adj Close' then 'Close'
                col = "Adj Close" if "Adj Close" in sub.columns else "Close" if "Close" in sub.columns else None
                if col is None:
                    failed.append(sym)
                    continue
                series = sub[col].rename(sym)
                one = series.to_frame().dropna(how="all")
                if one.empty:
                    failed.append(sym)
                    continue
                frames.append(one)
            except Exception:
                failed.append(sym)
    except Exception as e:
        _log(f"[fetch_data] multi-chunk error ({len(tickers)}): {e!r}")
        failed = list(tickers)

    out = _safe_concat_price_frames(frames)
    if not out.empty:
        have = set(out.columns)
        missing = [sym for sym in tickers if sym not in have]
    else:
        missing = list(dict.fromkeys(failed or tickers))

    # Per ticker fallback for anything missing from the bulk response.
    for sym in missing:
        try:
            t = yf.Ticker(sym)
            history_kwargs = {"auto_adjust": True}
            if start is None and end is None:
                # yfinance defaults Ticker.history() to a short window; ask for full history.
                history_kwargs["period"] = "max"
            else:
                history_kwargs["start"] = start
                history_kwargs["end"] = end
            hist = t.history(**history_kwargs)
            if hist is None or hist.empty:
                failed.append(sym)
                continue
            col = "Close" if "Close" in hist.columns else None
            if col is None:
                failed.append(sym)
                continue
            frames.append(hist[col].rename(sym).to_frame())
        except Exception:
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


def _alpha_vantage_series_key(payload: dict) -> str | None:
    for key in payload.keys():
        if "Time Series" in key:
            return key
    return None


def _download_alpha_vantage(symbols: List[str], start=None, end=None) -> pd.DataFrame:
    if not ALPHAVANTAGE_API_KEY:
        raise RuntimeError(
            "MKTME_EQUITY_SOURCE=alphavantage but ALPHAVANTAGE_API_KEY is not set."
        )
    if not ALPHAVANTAGE_PREMIUM:
        if len(symbols) > ALPHAVANTAGE_DAILY_LIMIT:
            raise RuntimeError(
                "Alpha Vantage free tier cannot refresh this universe: "
                f"{len(symbols)} symbols requested but the documented free limit is "
                f"{ALPHAVANTAGE_DAILY_LIMIT} requests/day."
            )
        if start is None and end is None:
            raise RuntimeError(
                "Alpha Vantage free tier only provides compact daily history "
                "(latest 100 data points) for TIME_SERIES_DAILY. "
                "This pipeline needs longer history for the 252-trading-day metrics."
            )
        if start is not None and end is not None:
            requested_span = (pd.Timestamp(end) - pd.Timestamp(start)).days
            if requested_span > 140:
                raise RuntimeError(
                    "Alpha Vantage free tier only provides compact daily history "
                    "(latest 100 data points), which is insufficient for this requested window."
                )

    session = requests.Session()
    sleep_s = 0.0 if ALPHAVANTAGE_PER_MINUTE <= 0 else max(0.0, 60.0 / ALPHAVANTAGE_PER_MINUTE)
    frames: list[pd.DataFrame] = []

    for idx, sym in enumerate(symbols):
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED" if ALPHAVANTAGE_PREMIUM else "TIME_SERIES_DAILY",
            "symbol": sym,
            "apikey": ALPHAVANTAGE_API_KEY,
            "datatype": "json",
            "outputsize": "full" if ALPHAVANTAGE_PREMIUM else "compact",
        }
        resp = session.get("https://www.alphavantage.co/query", params=params, timeout=60)
        resp.raise_for_status()
        payload = resp.json()

        if payload.get("Error Message"):
            raise RuntimeError(f"Alpha Vantage error for {sym}: {payload['Error Message']}")
        if payload.get("Note"):
            raise RuntimeError(f"Alpha Vantage rate limit response for {sym}: {payload['Note']}")
        if payload.get("Information"):
            raise RuntimeError(f"Alpha Vantage info response for {sym}: {payload['Information']}")

        series_key = _alpha_vantage_series_key(payload)
        if not series_key:
            raise RuntimeError(f"Alpha Vantage returned no time series for {sym}.")

        ts = payload.get(series_key) or {}
        if not ts:
            raise RuntimeError(f"Alpha Vantage returned an empty time series for {sym}.")

        field = "5. adjusted close" if ALPHAVANTAGE_PREMIUM else "4. close"
        s = pd.Series({k: float(v[field]) for k, v in ts.items() if field in v}, name=sym)
        one = s.to_frame()
        one.index = pd.to_datetime(one.index, errors="coerce")
        one = one[~one.index.isna()].sort_index()
        if start is not None:
            one = one[one.index >= pd.Timestamp(start)]
        if end is not None:
            one = one[one.index <= pd.Timestamp(end)]
        if one.empty:
            raise RuntimeError(f"Alpha Vantage returned no usable rows for {sym} after filtering.")
        frames.append(one)

        if idx < len(symbols) - 1 and sleep_s > 0:
            time.sleep(sleep_s)

    return _safe_concat_price_frames(frames)


def _twelvedata_symbol(symbol: str) -> str:
    s = str(symbol).strip().upper()
    if s.endswith("=X"):
        s = s[:-2]
    if len(s) == 6 and "/" not in s and s.isalpha():
        # EURUSD -> EUR/USD for FX
        return f"{s[:3]}/{s[3:]}"
    if "-" in s:
        # Twelve Data uses dot-class syntax for share classes (e.g. BRK.B, BF.B).
        return s.replace("-", ".")
    return s


def _prune_twelvedata_credit_history(now: Optional[float]=None) -> None:
    if now is None:
        now = time.monotonic()
    while _TWELVEDATA_CREDIT_HISTORY and now - _TWELVEDATA_CREDIT_HISTORY[0][0] >= 60.0:
        _TWELVEDATA_CREDIT_HISTORY.popleft()


def _seconds_until_twelvedata_capacity(required_credits: int, now: Optional[float]=None) -> float:
    if TWELVEDATA_PER_MINUTE <= 0:
        return 0.0
    if required_credits > TWELVEDATA_PER_MINUTE:
        raise RuntimeError(
            f"Twelve Data request needs {required_credits} credits but "
            f"TWELVEDATA_PER_MINUTE={TWELVEDATA_PER_MINUTE}."
        )

    if now is None:
        now = time.monotonic()
    _prune_twelvedata_credit_history(now)

    used = sum(credits for _, credits in _TWELVEDATA_CREDIT_HISTORY)
    if used + required_credits <= TWELVEDATA_PER_MINUTE:
        return 0.0

    reclaimed = 0
    for ts, credits in _TWELVEDATA_CREDIT_HISTORY:
        reclaimed += credits
        if used - reclaimed + required_credits <= TWELVEDATA_PER_MINUTE:
            return max(0.0, 60.0 - (now - ts) + TWELVEDATA_RATE_LIMIT_BUFFER_SECONDS)
    return 60.0 + TWELVEDATA_RATE_LIMIT_BUFFER_SECONDS


def _reserve_twelvedata_credits(required_credits: int) -> None:
    if TWELVEDATA_PER_MINUTE <= 0:
        return
    while True:
        now = time.monotonic()
        wait_s = _seconds_until_twelvedata_capacity(required_credits, now=now)
        if wait_s <= 0:
            _TWELVEDATA_CREDIT_HISTORY.append((now, required_credits))
            return
        _log(
            f"[fetch_data] Twelve Data waiting {wait_s:.1f}s for minute credits "
            f"before requesting {required_credits} symbols"
        )
        time.sleep(wait_s)


def _is_twelvedata_minute_limit_message(message: object) -> bool:
    text = str(message or "").lower()
    return "current minute" in text and "api credits" in text


def _download_twelvedata(symbols: List[str], start=None, end=None) -> pd.DataFrame:
    if not TWELVEDATA_API_KEY:
        raise RuntimeError(
            "MKTME_EQUITY_SOURCE=twelvedata but TWELVEDATA_API_KEY is not set."
        )
    if len(symbols) > TWELVEDATA_DAILY_LIMIT:
        raise RuntimeError(
            f"Twelve Data daily credit budget exceeded: {len(symbols)} symbols requested "
            f"but TWELVEDATA_DAILY_LIMIT={TWELVEDATA_DAILY_LIMIT}."
        )

    batch_size = max(1, TWELVEDATA_PER_MINUTE)
    frames: list[pd.DataFrame] = []
    session = requests.Session()
    chunks = _chunk(symbols, batch_size)
    failed_symbols: list[str] = []

    for i, chunk in enumerate(chunks, 1):
        params = {
            "symbol": ",".join(_twelvedata_symbol(sym) for sym in chunk),
            "interval": "1day",
            "outputsize": int(TWELVEDATA_OUTPUTSIZE),
            "order": "asc",
            "apikey": TWELVEDATA_API_KEY,
        }
        if start is not None:
            params["start_date"] = pd.Timestamp(start).strftime("%Y-%m-%d")
        if end is not None:
            params["end_date"] = pd.Timestamp(end).strftime("%Y-%m-%d")

        for attempt in range(TWELVEDATA_MINUTE_RETRY_LIMIT + 1):
            _reserve_twelvedata_credits(len(chunk))
            resp = session.get("https://api.twelvedata.com/time_series", params=params, timeout=90)
            resp.raise_for_status()
            payload = resp.json()

            if not (isinstance(payload, dict) and payload.get("status") == "error"):
                break

            message = payload.get("message") or payload
            if attempt >= TWELVEDATA_MINUTE_RETRY_LIMIT or not _is_twelvedata_minute_limit_message(message):
                raise RuntimeError(f"Twelve Data error: {message}")

            wait_s = _seconds_until_twelvedata_capacity(1) or (60.0 + TWELVEDATA_RATE_LIMIT_BUFFER_SECONDS)
            _log(
                f"[fetch_data] Twelve Data minute limit hit on batch {i}/{len(chunks)}; "
                f"retrying in {wait_s:.1f}s"
            )
            time.sleep(wait_s)

        batch_frames: list[pd.DataFrame] = []
        batch_missing: list[str] = []
        for sym in chunk:
            key = _twelvedata_symbol(sym)
            item = payload.get(key)
            if not isinstance(item, dict):
                batch_missing.append(sym)
                continue
            if item.get("status") == "error":
                batch_missing.append(sym)
                continue
            values = item.get("values") or []
            if not values:
                batch_missing.append(sym)
                continue
            one = pd.DataFrame(values)
            if "close" not in one.columns or "datetime" not in one.columns:
                batch_missing.append(sym)
                continue
            one["datetime"] = pd.to_datetime(one["datetime"], errors="coerce")
            one["close"] = pd.to_numeric(one["close"], errors="coerce")
            one = one.dropna(subset=["datetime", "close"]).set_index("datetime")[["close"]]
            if not one.index.is_unique:
                one = one.groupby(level=0).last()
            one.columns = [sym]
            if one.empty:
                batch_missing.append(sym)
                continue
            batch_frames.append(one)

        if batch_missing:
            failed_symbols.extend(batch_missing)
            _log(
                f"[fetch_data] Twelve Data missing {len(batch_missing)} symbols in batch {i}: "
                f"{batch_missing[:10]}{' …' if len(batch_missing) > 10 else ''}"
            )

        frames.extend(batch_frames)
        _log(f"[fetch_data] Twelve Data batch {i}/{len(chunks)}: {len(batch_frames)} symbols")

    out = _safe_concat_price_frames(frames)
    if failed_symbols:
        _log(
            f"[fetch_data] Twelve Data missing symbols overall ({len(set(failed_symbols))}): "
            f"{sorted(set(failed_symbols))[:30]}{' …' if len(set(failed_symbols)) > 30 else ''}"
        )
    return out


def _probe_live_equity_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    end = pd.Timestamp.utcnow().normalize().tz_localize(None)
    start = end - pd.Timedelta(days=14)
    yahoo_df, _ = _download_chunk_multi(SOURCE_CANARY_SYMBOLS, start=start, end=end)
    stooq_df = _download_stooq(SOURCE_CANARY_SYMBOLS, start=start, end=end)
    return yahoo_df, stooq_df


def _resolve_equity_source(equity_source: Optional[str]=None) -> str:
    return (equity_source or EQUITY_SOURCE_DEFAULT).strip().lower() or "auto"


def download_prices(
    symbols: Iterable[str],
    start: Optional[pd.Timestamp]=None,
    end: Optional[pd.Timestamp]=None,
    force_refresh: Optional[bool]=None,
    chunk_size: int=10,
    max_retries: int=3,
    backoff_base: float=1.0,
    prefer_stooq: Optional[bool]=None,
    equity_source: Optional[str]=None,
) -> pd.DataFrame:
    """
    Wide DataFrame of prices (auto-adjusted close), columns=tickers.
    Uses the on-disk cache unless `force_refresh` is True or the
    MKTME_FORCE_REFRESH env toggle is set.
    """
    symbols = _sanitize_symbols(list(symbols))
    requested_symbols = set(symbols)
    if not symbols:
        return pd.DataFrame()

    # Decide whether to bypass the on-disk cache.
    # Environment variable MKTME_FORCE_REFRESH acts as a global "on" switch,
    # while the function argument can also turn refresh on explicitly.
    if force_refresh is None:
        force_refresh = FORCE_REFRESH
    else:
        # Never allow the argument to disable the env flag; only to enable.
        force_refresh = bool(force_refresh) or FORCE_REFRESH

    use_disk_cache = not force_refresh

    start = _to_naive_ts(start)
    end   = _to_naive_ts(end)

    equity_source = _resolve_equity_source(equity_source)

    # Apply environment-driven defaults
    if prefer_stooq is None:
        prefer_stooq = PREFER_STOOQ_DEFAULT

    if equity_source == "twelvedata":
        return _download_twelvedata(symbols, start=start, end=end)

    if equity_source == "alphavantage":
        return _download_alpha_vantage(symbols, start=start, end=end)

    if equity_source == "stooq":
        prefer_stooq = True
    elif equity_source == "yahoo":
        prefer_stooq = False

    if force_refresh and not prefer_stooq:
        yahoo_probe, stooq_probe = _probe_live_equity_sources()
        if yahoo_probe.empty and stooq_probe.empty:
            message = (
                "Live equity source probe returned no data for canary symbols "
                f"{SOURCE_CANARY_SYMBOLS}; continuing anyway and letting per-chunk fallbacks decide."
            )
            if ALLOW_SOURCE_PROBE_FAILURE:
                _log(f"[fetch_data] {message}")
            else:
                raise RuntimeError(
                    "Live equity sources are unavailable from this environment. "
                    f"Yahoo and Stooq both returned no data for canary symbols {SOURCE_CANARY_SYMBOLS}."
                )

    # Optionally cap number of symbols to reduce rate-limit risk on Cloud
    if MAX_SYMBOLS_DEFAULT > 0 and len(symbols) > MAX_SYMBOLS_DEFAULT:
        _log(f"[fetch_data] Limiting symbols from {len(symbols)} to {MAX_SYMBOLS_DEFAULT} due to MKTME_MAX_SYMBOLS")
        symbols = symbols[:MAX_SYMBOLS_DEFAULT]

    # Serve cache if available and not forcing refresh
    if os.path.exists(PRICES_PARQUET) and use_disk_cache:
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

    # --- Early path: Stooq-only mode (default on Streamlit Cloud) ---
    if prefer_stooq:
        frames = []
        for i, chunk in enumerate(_chunk(symbols, chunk_size), 1):
            st_df = _download_stooq(chunk, start=start, end=end)
            if not st_df.empty:
                frames.append(st_df)
                # Write progressive cache so subsequent runs can reuse partial data
                try:
                    partial = _safe_concat_price_frames(frames)
                    if os.path.exists(PRICES_PARQUET) and use_disk_cache:
                        base = pd.read_parquet(PRICES_PARQUET)
                        partial = base.combine_first(partial).combine_first(partial)
                    partial.to_parquet(PRICES_PARQUET)
                except Exception:
                    pass
        out = _safe_concat_price_frames(frames)

        # Merge with existing cache so we don’t lose previously-fetched columns
        if os.path.exists(PRICES_PARQUET) and use_disk_cache:
            try:
                base = pd.read_parquet(PRICES_PARQUET)
                out = base.combine_first(out).combine_first(out)
            except Exception:
                pass

        # Final trims on requested window
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

        have = set(out.columns) if not out.empty else set()
        still_missing = sorted(set(symbols) - have)
        if still_missing:
            _log(f"[fetch_data] (stooq-only) Still missing {len(still_missing)} tickers; examples: {still_missing[:30]}{' …' if len(still_missing)>30 else ''}")
        return out

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
                # Progressive cache write for partial progress on Cloud
                try:
                    partial = _safe_concat_price_frames(all_frames)
                    if os.path.exists(PRICES_PARQUET) and use_disk_cache:
                        base = pd.read_parquet(PRICES_PARQUET)
                        partial = base.combine_first(partial).combine_first(partial)
                    partial.to_parquet(PRICES_PARQUET)
                except Exception:
                    pass
                all_failed.extend(failed)
                yahoo_ok = True
                break

            # If Yahoo returned nothing at all, don't keep hammering — try Stooq for the whole chunk
            _log(f"[fetch_data] chunk {i}/{len(chunks)} got 0 cols from Yahoo; trying Stooq immediately…")
            stooq_df = _download_stooq(chunk, start=start, end=end)
            if not stooq_df.empty:
                all_frames.append(stooq_df)
                # Progressive cache write for partial progress on Cloud
                try:
                    partial = _safe_concat_price_frames(all_frames)
                    if os.path.exists(PRICES_PARQUET) and use_disk_cache:
                        base = pd.read_parquet(PRICES_PARQUET)
                        partial = base.combine_first(partial).combine_first(partial)
                    partial.to_parquet(PRICES_PARQUET)
                except Exception:
                    pass
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
                    # Progressive cache write for partial progress on Cloud
                    try:
                        partial = _safe_concat_price_frames(all_frames)
                        if os.path.exists(PRICES_PARQUET) and use_disk_cache:
                            base = pd.read_parquet(PRICES_PARQUET)
                            partial = base.combine_first(partial).combine_first(partial)
                        partial.to_parquet(PRICES_PARQUET)
                    except Exception:
                        pass
                else:
                    # last-ditch Stooq per-sub
                    st_sub = _download_stooq(sub, start=start, end=end)
                    if not st_sub.empty:
                        all_frames.append(st_sub)
                        # Progressive cache write for partial progress on Cloud
                        try:
                            partial = _safe_concat_price_frames(all_frames)
                            if os.path.exists(PRICES_PARQUET) and use_disk_cache:
                                base = pd.read_parquet(PRICES_PARQUET)
                                partial = base.combine_first(partial).combine_first(partial)
                            partial.to_parquet(PRICES_PARQUET)
                        except Exception:
                            pass
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
    if os.path.exists(PRICES_PARQUET) and use_disk_cache:
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


def _download_yahoo_symbols(
    symbols: List[str],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    chunk_size: int=25,
    max_retries: int=3,
    backoff_base: float=1.0,
) -> pd.DataFrame:
    all_frames: list[pd.DataFrame] = []
    all_failed: list[str] = []
    chunks = _chunk(symbols, chunk_size)

    for i, chunk in enumerate(chunks, 1):
        attempt = 0
        chunk_loaded = False
        while attempt <= max_retries:
            df, failed = _download_chunk_multi(chunk, start=start, end=end)
            if not df.empty:
                all_frames.append(df)
                all_failed.extend(failed)
                chunk_loaded = True
                break
            attempt += 1
            if attempt <= max_retries:
                sleep_s = backoff_base * (2 ** (attempt - 1))
                _log(
                    f"[fetch_data] Yahoo-only chunk {i}/{len(chunks)} "
                    f"retry {attempt} in {sleep_s:.1f}s…"
                )
                time.sleep(sleep_s)
        if not chunk_loaded:
            all_failed.extend(chunk)

    out = _safe_concat_price_frames(all_frames)
    missing = sorted(set(symbols) - set(out.columns)) if not out.empty else sorted(set(symbols))
    if missing:
        _log(
            f"[fetch_data] Yahoo-only missing {len(missing)} symbols; "
            f"examples: {missing[:30]}{' …' if len(missing) > 30 else ''}"
        )
    return out

def download_prices_fx_window(
    pairs: List[str],
    lookback_trading_days: int,
    asof: pd.Timestamp,
    force_refresh: bool=False,
    chunk_size: int=25,
    equity_source: Optional[str]=None,
) -> pd.DataFrame:
    """
    Windowed FX fetch using yfinance (close). Returns wide DF columns=pairs.
    """
    resolved_source = _resolve_equity_source(equity_source)
    end = pd.Timestamp(asof).normalize()
    from pandas.tseries.offsets import BDay
    start = (end - BDay(int(lookback_trading_days))).normalize()

    if resolved_source == "twelvedata":
        return _download_twelvedata(pairs, start=start, end=end)

    yf_syms = [_fx_to_yf_symbol(p) for p in pairs]
    if resolved_source in {"auto", "yahoo"}:
        df = _download_yahoo_symbols(
            yf_syms,
            start=start,
            end=end,
            chunk_size=chunk_size,
        )
    else:
        df = download_prices(
            yf_syms,
            start=start,
            end=end,
            force_refresh=force_refresh,
            chunk_size=chunk_size,
            equity_source=resolved_source,
        )

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
