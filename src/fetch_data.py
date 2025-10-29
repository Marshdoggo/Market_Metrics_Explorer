import os, io, pandas as pd, numpy as np, yfinance as yf
import time, pathlib, json, requests
from datetime import datetime
from typing import Optional
try:
    from dotenv import load_dotenv
    load_dotenv()  # also picks up project root .env by default
except ImportError:
    pass

_HTTP = requests.Session()
_HTTP.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
})

from utils import ensure_cache_dir, trading_days_ago

CACHE_DIR = ensure_cache_dir()
TICKERS_CSV = os.path.join(CACHE_DIR, 'sp500_tickers.csv')
PRICES_PARQUET = os.path.join(CACHE_DIR, 'prices.parquet')
META_PARQUET = os.path.join(CACHE_DIR, 'meta.parquet')

# Repo root (this file lives in src/)
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
BUNDLED_TICKERS = ROOT_DIR / "sp500_tickers.csv"

def get_sp500_constituents(force_refresh: bool = False) -> pd.DataFrame:
    """
    Return a DataFrame with columns: Ticker, Name, Sector, SubIndustry.
    Prefers cached CSV; fetches Wikipedia; falls back to bundled CSV if needed.
    """
    # 1) Cached
    if os.path.exists(TICKERS_CSV) and not force_refresh:
        try:
            df = pd.read_csv(TICKERS_CSV)
            if {"Ticker","Name","Sector"}.issubset(df.columns):
                return df
        except Exception:
            pass

    # 2) Wikipedia
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        html = _HTTP.get(url, timeout=20).text
        tables = pd.read_html(html)
        df = tables[0].copy()
        # Normalize headers across wiki variants
        df.rename(columns={c: c.strip() for c in df.columns}, inplace=True)
        rename_map = {
            "Symbol": "Ticker",
            "Security": "Name",
            "GICS Sector": "Sector",
            "GICS Sub-Industry": "SubIndustry",
            "GICS sector": "Sector",
            "GICS sub-industry": "SubIndustry",
        }
        df.rename(columns=rename_map, inplace=True)
        keep = [c for c in ["Ticker","Name","Sector","SubIndustry"] if c in df.columns]
        df = df[keep]
        df["Ticker"] = df["Ticker"].astype(str).str.replace(".", "-", regex=False).str.upper().str.strip()
    except Exception:
        # 3) Fallback: bundled CSV committed in the repo
        if BUNDLED_TICKERS.exists():
            df = pd.read_csv(BUNDLED_TICKERS)
            df.rename(columns={c: c.strip() for c in df.columns}, inplace=True)
            if "Symbol" in df.columns and "Ticker" not in df.columns:
                df.rename(columns={"Symbol": "Ticker"}, inplace=True)
            if "Security" in df.columns and "Name" not in df.columns:
                df.rename(columns={"Security": "Name"}, inplace=True)
            if "GICS Sector" in df.columns and "Sector" not in df.columns:
                df.rename(columns={"GICS Sector": "Sector"}, inplace=True)
            if "GICS Sub-Industry" in df.columns and "SubIndustry" not in df.columns:
                df.rename(columns={"GICS Sub-Industry": "SubIndustry"}, inplace=True)
            keep = [c for c in ["Ticker","Name","Sector","SubIndustry"] if c in df.columns]
            df = df[keep]
            df["Ticker"] = df["Ticker"].astype(str).str.replace(".", "-", regex=False).str.upper().str.strip()
        else:
            # Last resort: empty schema
            return pd.DataFrame(columns=["Ticker","Name","Sector","SubIndustry"])

    # Persist cache/meta and return
    try:
        df.to_csv(TICKERS_CSV, index=False)
    except Exception:
        pass
    try:
        df.to_parquet(META_PARQUET, index=False)
    except Exception:
        pass
    return df

def _download_chunk(tickers: list[str], start=None, end=None) -> pd.DataFrame:
    """
    Download a chunk of tickers via yfinance and return a wide frame of Adj Close (or Close) columns.
    """
    if not tickers:
        return pd.DataFrame()

    df = yf.download(
        " ".join(tickers),
        start=start, end=end,
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    if isinstance(df.columns, pd.MultiIndex):
        closes = {}
        for t in tickers:
            try:
                sub = df[t]
            except Exception:
                continue
            s = sub.get("Adj Close", sub.get("Close"))
            if s is not None:
                closes[t] = s
        out = pd.DataFrame(closes)
    else:
        # single symbol path
        col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
        if col is None:
            return pd.DataFrame()
        out = df[[col]].copy()
        out.columns = [tickers[0]]

    return out.sort_index()

def download_prices(tickers, start=None, end=None, force_refresh: bool = False) -> pd.DataFrame:
    """
    Robust multi-ticker downloader with chunking + retries.
    Returns a wide DataFrame of prices (Adj Close if available), indexed by date.
    """
    tickers = [t.replace(".", "-").upper().strip() for t in list(tickers or [])]
    if not tickers:
        return pd.DataFrame()

    # Serve cached parquet if suitable
    if os.path.exists(PRICES_PARQUET) and not force_refresh:
        try:
            prices = pd.read_parquet(PRICES_PARQUET)
            prices.index = pd.to_datetime(prices.index).tz_localize(None)
            prices = prices.sort_index()
            if start:
                prices = prices[prices.index >= pd.to_datetime(start)]
            if end:
                prices = prices[prices.index <= pd.to_datetime(end)]
            if not prices.empty:
                return prices
        except Exception:
            pass

    if start is None:
        start = trading_days_ago(400).date()

    CHUNK = 40
    parts = []
    tickers = list(dict.fromkeys(tickers))  # de-dup preserving order
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i:i+CHUNK]
        last_exc = None
        for attempt in range(3):
            try:
                part = _download_chunk(chunk, start=start, end=end)
                if not part.empty:
                    parts.append(part)
                break
            except Exception as e:
                last_exc = e
                time.sleep(1.0 * (attempt + 1))
        # small courtesy delay for rate-limits
        time.sleep(0.25)
    if not parts:
        return pd.DataFrame()

    prices = pd.concat(parts, axis=1)
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.sort_index()
    prices.columns = [str(c).upper().strip() for c in prices.columns]
    prices = prices.ffill().bfill().dropna(how="all", axis=1)
    try:
        prices.to_parquet(PRICES_PARQUET)
    except Exception:
        pass
    return prices

def download_prices_window(
    tickers,
    lookback_trading_days: int,
    asof: Optional[datetime] = None,
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Equity helper: convert trading-day lookback to a calendar window (~1.4x buffer)
    and fetch prices via `download_prices`.
    """
    if asof is None:
        asof = pd.Timestamp.today(tz="UTC").normalize()
    cal_days = int(lookback_trading_days * 1.4)
    start = (asof - pd.Timedelta(days=cal_days)).date()
    end = asof.date()
    return download_prices(tickers, start=start, end=end, force_refresh=force_refresh)

def get_meta() -> pd.DataFrame:
    if os.path.exists(META_PARQUET):
        return pd.read_parquet(META_PARQUET)
    return get_sp500_constituents(force_refresh=False)

# ---------------- FX via Alpha Vantage (cached) ----------------
FX_CACHE = pathlib.Path(__file__).resolve().parent.parent / "data" / "fx_av"
FX_CACHE.mkdir(parents=True, exist_ok=True)

def _fx_cache_path(pair: str) -> pathlib.Path:
  return FX_CACHE / f"{pair}.parquet"

# Single canonical parquet for fast Yahoo Finance multi-symbol cache
FX_YF_PARQUET = FX_CACHE / "fx_yf_daily.parquet"

def _pairs_to_yf_symbols(pairs: list[str]) -> list[str]:
  """Map pairs like 'EURUSD' or 'EUR/USD' to yfinance symbols like 'EURUSD=X'."""
  syms: list[str] = []
  for p in pairs:
    p = p.upper().replace('/', '')
    if len(p) >= 6:
      syms.append(f"{p}=X")
  return syms

def _av_get_fx_daily(pair: str, api_key: str, outputsize: str = "full") -> pd.Series:
  """
  Returns a pandas Series of close prices indexed by datetime.
  pair format: 'EURUSD'. Uses Alpha Vantage FX_DAILY.
  """
  p = _fx_cache_path(pair)
  if p.exists():
    return pd.read_parquet(p)["close"]

  base, quote = pair[:3], pair[3:]
  url = (
    "https://www.alphavantage.co/query"
    f"?function=FX_DAILY&from_symbol={base}&to_symbol={quote}&outputsize={outputsize}&apikey={api_key}"
  )
  last_exc = None
  for i in range(3):
      try:
          r = _HTTP.get(url, timeout=30)
          r.raise_for_status()
          data = r.json()
          break
      except Exception as e:
          last_exc = e
          time.sleep(2 * (i + 1))
  else:
      raise RuntimeError(f"Alpha Vantage request failed for {pair}: {last_exc}")
  if "Time Series FX (Daily)" not in data:
    raise RuntimeError(
      f"Alpha Vantage error for {pair}: {data.get('Note') or data.get('Error Message') or 'unknown'}"
    )
  ts = data["Time Series FX (Daily)"]
  s = pd.Series({pd.to_datetime(k): float(v["4. close"]) for k, v in ts.items()}).sort_index()
  df = pd.DataFrame({"close": s})
  df.to_parquet(p, index=True)
  return s

# -------------- Fast FX downloader preferring Yahoo, fallback to Alpha Vantage --------------
def download_prices_fx_fast(
    pairs: list[str],
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Fast FX downloader:
    1) Try Yahoo Finance multi-download for all requested pairs in one call.
    2) Persist/merge into a single canonical parquet (FX_YF_PARQUET).
    3) Fall back to Alpha Vantage for any pairs Yahoo misses, and merge results.

    If `start`/`end` are not provided, defaults to ~2 years of history to
    ensure enough data for rolling indicators, Sharpe, etc.
    Returns wide DataFrame indexed by date with columns = requested pairs (close).
    """
    # Optionally drop cache if forced
    if force_refresh and FX_YF_PARQUET.exists():
        try:
            FX_YF_PARQUET.unlink()
        except Exception:
            pass

    # Default window if not specified: ~2 years
    yf_kwargs = {}
    if start is None and end is None:
        yf_kwargs["period"] = "2y"
    else:
        if start is not None:
            yf_kwargs["start"] = pd.to_datetime(start)
        if end is not None:
            yf_kwargs["end"] = pd.to_datetime(end)

    # 1) Yahoo Finance bulk download
    symbols = _pairs_to_yf_symbols(pairs)
    base = pd.DataFrame()
    if symbols:
        data = yf.download(
            symbols,
            interval="1d",
            auto_adjust=False,
            group_by="ticker",
            progress=False,
            threads=True,
            **yf_kwargs
        )
        close = pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            try:
                close = data.xs("Close", axis=1, level=1, drop_level=True)
            except Exception:
                close = data.xs("close", axis=1, level=1, drop_level=True)
            close.columns = [c.replace("=X", "") for c in close.columns]
        else:
            # Single symbol case
            if "Close" in data.columns:
                close = data[["Close"]].copy()
            elif "close" in data.columns:
                close = data[["close"]].copy()
            else:
                close = pd.DataFrame()
            if len(symbols) == 1 and not close.empty:
                close.columns = [symbols[0].replace("=X", "")]

        close = close.sort_index().dropna(how="all")

        # Merge with existing YF cache if present
        if FX_YF_PARQUET.exists():
            try:
                old = pd.read_parquet(FX_YF_PARQUET)
                close = old.combine_first(close).combine_first(old)
            except Exception:
                pass

        if not close.empty:
            try:
                close.to_parquet(FX_YF_PARQUET, index=True)
            except Exception:
                pass
            base = close

    # 2) Identify missing pairs not covered by Yahoo
    missing = [p for p in pairs if base.empty or p not in base.columns]

    if missing:
        # 3) Fall back to Alpha Vantage for the leftovers (cached per pair with rate limiting)
        av_df = download_prices_fx(missing, force_refresh=force_refresh)
        base = base.join(av_df, how="outer") if not base.empty else av_df
        # persist merged set back to canonical cache
        try:
            if FX_YF_PARQUET.exists():
                old = pd.read_parquet(FX_YF_PARQUET)
                merged = old.combine_first(base).combine_first(old)
            else:
                merged = base
            merged.to_parquet(FX_YF_PARQUET, index=True)
        except Exception:
            pass

    out = base.reindex(columns=pairs).sort_index() if not base.empty else pd.DataFrame(columns=pairs)
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out

def download_prices_fx(pairs: list[str], force_refresh: bool = False, api_key: str | None = None) -> pd.DataFrame:
  """
  Returns a DataFrame with columns = pairs (close). Uses caching and simple rate-limit spacing.
  Requires env var ALPHAVANTAGE_API_KEY.
  """
  if not api_key:
      api_key = os.getenv("ALPHAVANTAGE_API_KEY")
  if not api_key:
      raise RuntimeError("ALPHAVANTAGE_API_KEY not set")

  cols: dict[str, pd.Series] = {}
  calls = 0
  for pair in pairs:
    if force_refresh:
      cp = _fx_cache_path(pair)
      if cp.exists():
        cp.unlink()
    try:
      cols[pair] = _av_get_fx_daily(pair, api_key)
      calls += 1
      if calls % 5 == 0:
        time.sleep(60)  # ~5 req/min on AV free tier
    except Exception as e:
      print(f"[WARN] FX fetch failed for {pair}: {e}")

  if not cols:
    raise RuntimeError("No FX data retrieved")

  df = pd.DataFrame(cols).sort_index()
  df.index = pd.to_datetime(df.index).tz_localize(None)
  return df


# ----- Helper to fetch FX window using trading-day lookback and as-of -----
def download_prices_fx_window(
    pairs: list[str],
    lookback_trading_days: int,
    asof: Optional[datetime] = None,
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Convenience wrapper that converts a trading-day lookback to a calendar window (~1.4x buffer)
    and fetches FX prices using the fast Yahoo-first path with Alpha Vantage fallback.
    """
    if asof is None:
        asof = pd.Timestamp.today(tz="UTC").normalize()
    # Convert trading-day lookback to calendar days with a buffer
    cal_days = int(lookback_trading_days * 1.4)
    start = (asof - pd.Timedelta(days=cal_days)).date()
    end = asof.date()
    return download_prices_fx_fast(pairs, start=start, end=end, force_refresh=force_refresh)
