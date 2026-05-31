from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fetch_data import get_sp500_constituents  # noqa: E402
from leaderboards import append_snapshot_rows, build_leaderboard_snapshots, snapshot_path  # noqa: E402
from universes import get_universe  # noqa: E402

try:
    from compute_metrics import compute_all_metrics  # noqa: E402
except Exception:
    compute_all_metrics = None


def _ticker_frame(universe: str, price_columns: list[str] | None = None) -> pd.DataFrame:
    try:
        if universe == "sp500":
            return get_sp500_constituents(force_refresh=False)
        return get_universe(universe, force_refresh=False)
    except Exception:
        if not price_columns:
            raise
        tickers = [str(c).upper() for c in price_columns]
        return pd.DataFrame(
            {
                "Ticker": tickers,
                "Name": tickers,
                "Sector": universe.upper(),
                "SubIndustry": "",
            }
        )


def _build_metrics(prices: pd.DataFrame, tickers_df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    if compute_all_metrics is not None:
        metrics_df = compute_all_metrics(prices, lookback=lookback)
    else:
        metrics_df = _compute_metrics_fallback(prices, lookback=lookback)
    metrics_df = metrics_df.join(
        tickers_df.set_index("Ticker")[["Name", "Sector", "SubIndustry"]],
        how="left",
    )
    metrics_df = metrics_df.reset_index().rename(columns={"index": "Ticker"})
    metrics_df["Ticker"] = metrics_df["Ticker"].astype(str).str.upper()
    return metrics_df


def _compute_metrics_fallback(prices_wide: pd.DataFrame, lookback: int) -> pd.DataFrame:
    data = prices_wide.tail(lookback) if lookback and lookback < len(prices_wide) else prices_wide.copy()
    out = {}
    for ticker in data.columns:
        prices = pd.to_numeric(data[ticker], errors="coerce").dropna()
        if len(prices) < max(60, int(lookback * 0.4)):
            continue
        rets = prices.pct_change().dropna()
        if rets.empty:
            continue
        vol = float(rets.std())
        mean = float(rets.mean())
        ann_sharpe = np.nan if vol == 0 else float((mean / vol) * np.sqrt(252))
        total_return = prices.iloc[-1] / prices.iloc[0] if prices.iloc[0] else np.nan
        years = len(prices) / 252
        cagr = np.nan if years <= 0 or total_return <= 0 else float(total_return ** (1 / years) - 1)
        roll_max = prices.cummax()
        max_dd = float((prices / roll_max - 1.0).min())
        downside = np.minimum(rets, 0)
        downside_dev = float(np.sqrt(np.mean(downside**2)))
        sortino = np.nan if downside_dev == 0 else float((rets.mean() * 252) / (downside_dev * np.sqrt(252)))
        delta = prices.diff()
        up = delta.clip(lower=0).rolling(14).mean()
        down = (-delta.clip(upper=0)).rolling(14).mean()
        rs = up.iloc[-1] / down.iloc[-1] if down.iloc[-1] != 0 else np.nan
        rsi = float(100 - (100 / (1 + rs))) if rs == rs else np.nan
        ma50 = prices.rolling(50).mean().iloc[-1] if len(prices) >= 50 else np.nan
        ma200 = prices.rolling(200).mean().iloc[-1] if len(prices) >= 200 else np.nan
        ma20 = prices.rolling(20).mean()
        sd20 = prices.rolling(20).std()
        bw20 = ((ma20 + 2 * sd20) - (ma20 - 2 * sd20)) / ma20
        out[ticker] = {
            "Mean Daily Return": mean,
            "Daily Volatility (Std)": vol,
            "Annualized Sharpe": ann_sharpe,
            "Max Drawdown": max_dd,
            "CAGR": cagr,
            "Downside Deviation": downside_dev,
            "Sortino Ratio": sortino,
            "RSI(14)": rsi,
            "% Above 50DMA": np.nan if pd.isna(ma50) or ma50 == 0 else float((prices.iloc[-1] / ma50) - 1),
            "% Above 200DMA": np.nan if pd.isna(ma200) or ma200 == 0 else float((prices.iloc[-1] / ma200) - 1),
            "Return Skewness": float(rets.skew()) if len(rets) > 3 else np.nan,
            "Return Kurtosis (Fisher)": float(rets.kurt()) if len(rets) > 3 else np.nan,
            "Bollinger Bandwidth (20)": float(bw20.iloc[-1]) if len(bw20.dropna()) else np.nan,
        }
    return pd.DataFrame.from_dict(out, orient="index")


def backfill_leaderboards(
    *,
    data_repo: Path,
    universe: str,
    lookback: int,
    history_days: int,
    force: bool,
) -> dict[str, int | str]:
    universe = universe.lower()
    prices_path = data_repo / universe / "prices.parquet"
    if not prices_path.exists():
        raise FileNotFoundError(f"Price parquet not found: {prices_path}")

    prices = pd.read_parquet(prices_path)
    prices.index = pd.to_datetime(prices.index, errors="coerce")
    prices = prices.loc[~prices.index.isna()].sort_index()
    if prices.empty:
        raise RuntimeError(f"No usable prices found in {prices_path}")

    tickers_df = _ticker_frame(universe, price_columns=[str(c) for c in prices.columns])
    symbols = tickers_df["Ticker"].astype(str).tolist()
    prices = prices[[c for c in prices.columns if c in symbols]].copy()
    if prices.empty:
        raise RuntimeError(f"No price columns match the {universe} universe.")

    unique_dates = pd.DatetimeIndex(prices.index.normalize().unique()).sort_values()
    candidate_dates = list(unique_dates[-int(history_days):])
    out_frames: list[pd.DataFrame] = []
    skipped = 0

    for asof in candidate_dates:
        trailing = prices.loc[:asof]
        if len(trailing) < max(60, int(lookback * 0.4)):
            skipped += 1
            continue
        metrics_df = _build_metrics(trailing, tickers_df, lookback=lookback)
        if metrics_df.empty:
            skipped += 1
            continue
        rows = build_leaderboard_snapshots(
            metrics_df,
            universe=universe,
            as_of_date=asof.date().isoformat(),
            lookback=lookback,
            source_mode="backfill",
        )
        if rows.empty:
            skipped += 1
            continue
        out_frames.append(rows)

    if not out_frames:
        raise RuntimeError("No leaderboard snapshot rows were generated.")

    all_rows = pd.concat(out_frames, ignore_index=True)
    out_path = snapshot_path(data_repo, universe, lookback)
    _, added = append_snapshot_rows(all_rows, out_path, force=force)
    return {
        "universe": universe,
        "lookback": int(lookback),
        "history_days": int(history_days),
        "candidate_dates": len(candidate_dates),
        "skipped_dates": int(skipped),
        "generated_rows": int(len(all_rows)),
        "added_rows": int(added),
        "artifact": out_path.relative_to(data_repo).as_posix(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill leaderboard snapshot history from published price parquet.")
    parser.add_argument("--data-repo", default=os.environ.get("MKTME_DATA_REPO", str(ROOT / "mktme-data")))
    parser.add_argument("--universe", default="dow30", choices=["sp500", "nasdaq100", "dow30", "fx"])
    parser.add_argument("--lookback", type=int, default=int(os.environ.get("MKTME_LOOKBACK", "252")))
    parser.add_argument("--history-days", type=int, default=252)
    parser.add_argument("--force", action="store_true", help="Rebuild duplicate snapshot keys instead of preserving existing rows.")
    args = parser.parse_args()

    result = backfill_leaderboards(
        data_repo=Path(args.data_repo).expanduser(),
        universe=args.universe,
        lookback=int(args.lookback),
        history_days=int(args.history_days),
        force=bool(args.force),
    )
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
