from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fx_universe import get_fx_universe  # noqa: E402


FetchFn = Callable[[str, pd.Timestamp | None, pd.Timestamp | None], pd.DataFrame]


def _default_fetch(symbol: str, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.DataFrame:
    import yfinance as yf

    return yf.download(
        tickers=symbol,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
        threads=False,
    )


def _close_series(frame: pd.DataFrame, symbol: str) -> pd.Series:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.Series(dtype="float64")
    if isinstance(frame.columns, pd.MultiIndex):
        if ("Close", symbol) in frame.columns:
            return frame[("Close", symbol)]
        if (symbol, "Close") in frame.columns:
            return frame[(symbol, "Close")]
    if "Close" in frame.columns:
        return frame["Close"]
    if symbol in frame.columns:
        return frame[symbol]
    return pd.Series(dtype="float64")


def check_fx_availability(
    *,
    universe: str = "fx_forex_com_core",
    output_path: str | Path = ROOT / "data" / "quality" / "fx_availability.csv",
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    fetcher: FetchFn | None = None,
) -> pd.DataFrame:
    fx = get_fx_universe(universe)
    fetch = fetcher or _default_fetch
    start_ts = pd.to_datetime(start).normalize() if start is not None else None
    end_ts = pd.to_datetime(end).normalize() if end is not None else None

    rows = []
    for item in fx.to_dict("records"):
        pair_display = str(item["pair_display"])
        symbol = str(item["yahoo_symbol"])
        row = {
            "pair_display": pair_display,
            "yahoo_symbol": symbol,
            "available": False,
            "first_date": "",
            "last_date": "",
            "rows": 0,
            "error": "",
        }
        try:
            raw = fetch(symbol, start_ts, end_ts)
            series = _close_series(raw, symbol).dropna()
            series.index = pd.to_datetime(series.index, errors="coerce")
            series = series[~series.index.isna()]
            if not series.empty:
                row["available"] = True
                row["first_date"] = series.index.min().date().isoformat()
                row["last_date"] = series.index.max().date().isoformat()
                row["rows"] = int(series.shape[0])
            else:
                row["error"] = "no usable close rows"
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    out = pd.DataFrame(rows)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Yahoo Finance availability for the FX universe.")
    parser.add_argument("--universe", default="fx_forex_com_core")
    parser.add_argument("--output", default=str(ROOT / "data" / "quality" / "fx_availability.csv"))
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()
    out = check_fx_availability(
        universe=args.universe,
        output_path=args.output,
        start=args.start,
        end=args.end,
    )
    available = int(out["available"].sum())
    print(f"Wrote {args.output}: {available}/{len(out)} available", flush=True)


if __name__ == "__main__":
    main()
