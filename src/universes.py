# src/universes.py
from __future__ import annotations

import os, time, json, pathlib
import pandas as pd
import requests

CACHE_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "cache_universes"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

WIKI_PAGES = {
    "nasdaq100": "https://en.wikipedia.org/wiki/Nasdaq-100",
    "dow30": "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
}

def _cache_path(name: str) -> pathlib.Path:
    return CACHE_DIR / f"{name}.parquet"

def _read_wiki_constituents(url: str) -> pd.DataFrame:
    """Fetch the most likely constituents table from a Wikipedia page and
    normalize odd header cases (numeric headers, first-row-as-header, etc.)."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; sp500_metrics/1.0)"}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()

    tables = pd.read_html(resp.text)
    best_df = None
    best_score = -1

    def score_table(df: pd.DataFrame) -> int:
        cols = [str(c).lower() for c in df.columns]
        score = 0
        if any(("ticker" in c) or ("symbol" in c) for c in cols):
            score += 2
        if any(("company" in c) or ("name" in c) for c in cols):
            score += 1
        return score

    for t in tables:
        df = t.copy()
        # Always coerce header types to strings first
        df.columns = [str(c) for c in df.columns]

        s = score_table(df)
        # Some pages render the first data row as headers; try promoting
        if s == 0 and len(df) > 1:
            promoted = df.iloc[1:].copy()
            promoted.columns = [str(x) for x in df.iloc[0].tolist()]
            s2 = score_table(promoted)
            if s2 > s:
                df, s = promoted, s2

        if s > best_score:
            best_df, best_score = df, s

    if best_df is None or best_score <= 0:
        raise RuntimeError(f"No suitable table found in {url}")

    return best_df

def get_universe(universe: str, force_refresh: bool = False) -> pd.DataFrame:
    """
    Returns columns: Ticker, Name, Sector, SubIndustry
    Supported: sp500 (handled elsewhere), nasdaq100, dow30, fx
    """
    u = universe.lower()
    cp = _cache_path(u)

    if u in ("nasdaq100", "dow30"):
        if cp.exists() and not force_refresh:
            return pd.read_parquet(cp)

        url = WIKI_PAGES[u]
        df = _read_wiki_constituents(url)

        # Normalize per-page quirks
        if u == "nasdaq100":
            # Most pages have columns "Ticker", "Company", "GICS Sector" …
            # Try multiple common schemas then normalize.
            cols = {str(c).lower(): c for c in df.columns}
            ticker_col = next((df[c] for k,c in cols.items() if "ticker" in k or "symbol" in k), None)
            name_col   = next((df[c] for k,c in cols.items() if "company" in k or "name" in k), None)
            sector_col = next((df[c] for k,c in cols.items() if "sector" in k), None)
            sub_col    = next((df[c] for k,c in cols.items() if "sub" in k and "industry" in k), None)

        elif u == "dow30":
            cols = {str(c).lower(): c for c in df.columns}
            ticker_col = next((df[c] for k,c in cols.items() if "symbol" in k or "ticker" in k), None)
            name_col   = next((df[c] for k,c in cols.items() if "company" in k or "name" in k), None)
            # DJIA doesn’t always list GICS—fallback to "Dow 30" tag
            sector_col = None
            sub_col    = None

        out = pd.DataFrame({
            "Ticker": ticker_col.astype(str).str.upper().str.replace(r"\.O$", "", regex=True).str.strip(),
            "Name":   name_col.astype(str).str.strip() if name_col is not None else "",
            "Sector": sector_col.astype(str).str.strip() if sector_col is not None else u.upper(),
            "SubIndustry": sub_col.astype(str).str.strip() if sub_col is not None else "",
        }).dropna(subset=["Ticker"]).drop_duplicates(subset=["Ticker"])

        out.to_parquet(cp, index=False)
        return out

    elif u == "fx":
        # Provide a curated FX universe (majors + a few crosses).
        pairs = [
            "EURUSD","GBPUSD","USDJPY","USDCHF","USDCAD","AUDUSD","NZDUSD",
            "EURJPY","EURGBP","AUDJPY","GBPJPY","CHFJPY","EURCHF","NZDJPY",
        ]
        return pd.DataFrame({
            "Ticker": pairs,
            "Name": pairs,
            "Sector": "FX",
            "SubIndustry": [p[:3] for p in pairs],   # base currency
        })

    else:
        raise ValueError(f"Unknown universe: {universe}")
