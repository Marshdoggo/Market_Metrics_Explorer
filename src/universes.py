# src/universes.py
from __future__ import annotations

import json
import pathlib
import re
from io import StringIO
from typing import Any

import pandas as pd
import requests
from fx_universe import FX_UNIVERSE_ALIASES, get_fx_universe

CACHE_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "cache_universes"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

WIKI_PAGES = {
    "nasdaq100": "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies",
    "dow30": "https://en.wikipedia.org/wiki/List_of_Dow_Jones_Industrial_Average_companies",
}

WIKI_FALLBACK_PAGES = {
    "nasdaq100": ["https://en.wikipedia.org/wiki/Nasdaq-100"],
    "dow30": [],
}

UNIVERSE_RULES = {
    "nasdaq100": {"min_rows": 90, "max_rows": 130, "default_sector": "NASDAQ-100"},
    "dow30": {"min_rows": 25, "max_rows": 35, "default_sector": "Dow 30"},
}

TICKER_ALIASES = {"ticker", "tickers", "symbol", "symbols", "ticker symbol", "stock symbol"}
NAME_ALIASES = {"company", "companies", "security", "name", "company name"}
SECTOR_ALIASES = {"sector", "gics sector"}
SUBINDUSTRY_ALIASES = {"subindustry", "sub industry", "sub-industry", "gics sub industry", "gics sub-industry"}
REQUIRED_COLUMNS = ["Ticker", "Name", "Sector", "SubIndustry"]


def _cache_path(name: str) -> pathlib.Path:
    return CACHE_DIR / f"{name}.parquet"


def _metadata_path(name: str) -> pathlib.Path:
    return CACHE_DIR / f"{name}.metadata.json"


def _flatten_column(column: Any) -> str:
    if isinstance(column, tuple):
        parts = [str(part) for part in column if str(part) and not str(part).lower().startswith("unnamed")]
        column = " ".join(parts)
    return str(column)


def _clean_label(value: Any) -> str:
    text = _flatten_column(value)
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip().lower()
    return text


def _canonical_label(value: Any) -> str:
    text = _clean_label(value)
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return text


def _clean_ticker(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"\s+", "", text.replace("\xa0", " ")).upper()
    text = text.replace(".", "-")
    text = re.sub(r"[^A-Z0-9=\-^]", "", text)
    return text


def _looks_like_ticker(value: Any) -> bool:
    ticker = _clean_ticker(value)
    return bool(re.fullmatch(r"[A-Z0-9][A-Z0-9=\-^]{0,14}", ticker)) and ticker not in {"NAN", "NONE"}


def _find_column(df: pd.DataFrame, aliases: set[str]) -> Any | None:
    exact = {_canonical_label(col): col for col in df.columns}
    for alias in aliases:
        if alias in exact:
            return exact[alias]
    for label, col in exact.items():
        if any(alias in label for alias in aliases):
            return col
    return None


def _candidate_tables_from_html(html: str) -> list[pd.DataFrame]:
    tables = pd.read_html(StringIO(html))
    candidates: list[pd.DataFrame] = []
    for table in tables:
        df = table.copy()
        df.columns = [_flatten_column(c) for c in df.columns]
        candidates.append(df)
        if len(df) > 1:
            promoted = df.iloc[1:].copy()
            promoted.columns = [_flatten_column(x) for x in df.iloc[0].tolist()]
            candidates.append(promoted)
    return candidates


def _normalize_constituents(df: pd.DataFrame, universe: str, source: str) -> pd.DataFrame:
    ticker_col = _find_column(df, TICKER_ALIASES)
    name_col = _find_column(df, NAME_ALIASES)
    sector_col = _find_column(df, SECTOR_ALIASES)
    sub_col = _find_column(df, SUBINDUSTRY_ALIASES)
    if ticker_col is None:
        raise ValueError(f"{universe}: no ticker/symbol column found in {source}")

    rules = UNIVERSE_RULES[universe]
    out = pd.DataFrame(
        {
            "Ticker": df[ticker_col].map(_clean_ticker),
            "Name": df[name_col].astype(str).str.replace(r"\[[^\]]*\]", "", regex=True).str.strip() if name_col is not None else "",
            "Sector": df[sector_col].astype(str).str.strip() if sector_col is not None else rules["default_sector"],
            "SubIndustry": df[sub_col].astype(str).str.strip() if sub_col is not None else "",
        }
    )
    out = out[out["Ticker"].map(_looks_like_ticker)].drop_duplicates(subset=["Ticker"])
    out = out[REQUIRED_COLUMNS].reset_index(drop=True)
    _validate_constituents(out, universe, source)
    return out


def _validate_constituents(df: pd.DataFrame, universe: str, source: str) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{universe}: missing required columns {missing} from {source}")
    rules = UNIVERSE_RULES[universe]
    count = len(df)
    if not rules["min_rows"] <= count <= rules["max_rows"]:
        raise ValueError(
            f"{universe}: constituent count {count} outside expected "
            f"range {rules['min_rows']}-{rules['max_rows']} from {source}"
        )
    bad = [ticker for ticker in df["Ticker"].tolist() if not _looks_like_ticker(ticker)]
    if bad:
        raise ValueError(f"{universe}: invalid ticker values from {source}: {bad[:10]}")


def _read_cached_universe(universe: str) -> pd.DataFrame | None:
    cp = _cache_path(universe)
    if not cp.exists():
        return None
    try:
        cached = pd.read_parquet(cp)
        _validate_constituents(cached, universe, str(cp))
        return cached[REQUIRED_COLUMNS].copy()
    except Exception as exc:
        print(f"[universes] Ignoring invalid cached {universe} universe at {cp}: {type(exc).__name__}: {exc}", flush=True)
        return None


def _write_cached_universe(universe: str, df: pd.DataFrame, source: str) -> None:
    cp = _cache_path(universe)
    df.to_parquet(cp, index=False)
    meta = {
        "universe": universe,
        "source": source,
        "retrieved_at_utc": pd.Timestamp.utcnow().isoformat(),
        "row_count": int(len(df)),
        "columns": REQUIRED_COLUMNS,
    }
    _metadata_path(universe).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _read_wiki_constituents(url: str, universe: str | None = None) -> pd.DataFrame:
    """Fetch and normalize the most likely constituent table from a Wikipedia page."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; MarketMetricsExplorer/1.0; +https://github.com/Marshdoggo/Market_Metrics_Explorer)"}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()

    errors: list[str] = []
    for idx, table in enumerate(_candidate_tables_from_html(resp.text)):
        if universe in UNIVERSE_RULES:
            try:
                return _normalize_constituents(table, universe, url)
            except Exception as exc:
                errors.append(f"table {idx}: {type(exc).__name__}: {exc}")
                continue

        if _find_column(table, TICKER_ALIASES) is not None:
            return table

    detail = "; ".join(errors[-5:]) if errors else "no ticker-like tables found"
    raise RuntimeError(f"No suitable table found in {url}. Last checks: {detail}")


def get_universe(universe: str, force_refresh: bool = False) -> pd.DataFrame:
    """
    Returns columns: Ticker, Name, Sector, SubIndustry.
    Supported: nasdaq100, dow30, fx and FX slices. S&P 500 is handled in fetch_data.
    """
    u = universe.lower().replace("-", "_")

    if u in ("nasdaq100", "dow30"):
        if not force_refresh:
            cached = _read_cached_universe(u)
            if cached is not None:
                return cached

        errors: list[str] = []
        for url in [WIKI_PAGES[u], *WIKI_FALLBACK_PAGES.get(u, [])]:
            try:
                out = _read_wiki_constituents(url, universe=u)
                _write_cached_universe(u, out, url)
                return out
            except Exception as exc:
                errors.append(f"{url}: {type(exc).__name__}: {exc}")

        cached = _read_cached_universe(u)
        if cached is not None:
            print(
                f"[universes] WARNING: live {u} constituent retrieval failed; "
                f"using cached known-good universe. Errors: {errors}",
                flush=True,
            )
            return cached
        raise RuntimeError(f"Unable to load {u} universe from live sources and no valid cache is available. Errors: {errors}")

    if u in FX_UNIVERSE_ALIASES:
        return get_fx_universe(u)

    raise ValueError(f"Unknown universe: {universe}")
