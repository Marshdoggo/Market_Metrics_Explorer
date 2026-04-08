from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_report import generate_daily_report, save_report_json, save_report_markdown  # noqa: E402
from compute_metrics import compute_all_metrics  # noqa: E402
from fetch_data import download_prices, download_prices_fx_window, get_sp500_constituents  # noqa: E402
from status_store import (  # noqa: E402
    finish_pipeline_run,
    init_db,
    record_report,
    record_snapshot,
    set_status,
    start_pipeline_run,
    utc_now_iso,
    write_status_snapshot_json,
)
from universes import get_universe  # noqa: E402


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def _to_naive_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[~pd.isna(idx)]
    idx = pd.DatetimeIndex(idx[~pd.isna(idx)])
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out.index = idx
    return out.sort_index()


def _ticker_frame(universe: str) -> pd.DataFrame:
    if universe == "sp500":
        return get_sp500_constituents(force_refresh=False)
    return get_universe(universe, force_refresh=False)


def _normalize_source(raw: str | None, fallback: str) -> str:
    value = (raw or "").strip().lower()
    return value or fallback


def _source_map_from_args(args: argparse.Namespace) -> dict[str, str]:
    fallback = _normalize_source(os.environ.get("MKTME_EQUITY_SOURCE"), "auto")
    return {
        "sp500": _normalize_source(args.source_sp500, fallback),
        "nasdaq100": _normalize_source(args.source_nasdaq100, fallback),
        "dow30": _normalize_source(args.source_dow30, fallback),
        "fx": _normalize_source(args.source_fx, fallback),
    }


def _load_or_fetch_prices(
    *,
    data_repo: Path,
    universe: str,
    tickers_df: pd.DataFrame,
    lookback: int,
    force_refresh: bool,
    use_existing_parquet: bool,
    prefetched_prices: pd.DataFrame | None = None,
    equity_source: str = "auto",
) -> pd.DataFrame:
    existing_path = data_repo / universe / "prices.parquet"
    if use_existing_parquet and existing_path.exists():
        return _to_naive_dt_index(pd.read_parquet(existing_path))

    tickers = tickers_df["Ticker"].astype(str).tolist()
    if universe == "fx":
        asof = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
        fresh = _to_naive_dt_index(
            download_prices_fx_window(
                tickers,
                lookback_trading_days=lookback,
                asof=asof,
                force_refresh=force_refresh,
                equity_source=equity_source,
            )
        )
    elif prefetched_prices is not None:
        available = [ticker for ticker in tickers if ticker in prefetched_prices.columns]
        fresh = _to_naive_dt_index(prefetched_prices[available]) if available else pd.DataFrame()
    else:
        fresh = _to_naive_dt_index(
            download_prices(
                tickers,
                force_refresh=force_refresh,
                equity_source=equity_source,
            )
        )

    if existing_path.exists() and not fresh.empty:
        try:
            existing = _to_naive_dt_index(pd.read_parquet(existing_path))
            # Keep freshly fetched values where available, but backfill symbols/rows the provider missed.
            fresh = fresh.combine_first(existing)
        except Exception:
            pass
    return fresh


def _build_metrics(prices: pd.DataFrame, tickers_df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    metrics_df = compute_all_metrics(prices, lookback=lookback)
    metrics_df = metrics_df.join(
        tickers_df.set_index("Ticker")[["Name", "Sector", "SubIndustry"]],
        how="left",
    )
    metrics_df = metrics_df.reset_index().rename(columns={"index": "Ticker"})
    metrics_df["Ticker"] = metrics_df["Ticker"].astype(str).str.upper()
    return metrics_df


def _write_prices_parquet(data_repo: Path, universe: str, prices: pd.DataFrame) -> tuple[str, str]:
    out_dir = data_repo / universe
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "prices.parquet"
    prices.to_parquet(out_path, engine="pyarrow")
    rel_path = f"{universe}/prices.parquet"
    return rel_path, rel_path


def _latest_index_date(entries: list[dict[str, Any]]) -> str | None:
    dated: list[tuple[pd.Timestamp, str]] = []
    for entry in entries or []:
        raw = entry.get("date")
        ts = pd.to_datetime(raw, errors="coerce")
        if pd.isna(ts):
            continue
        dated.append((ts, ts.date().isoformat()))
    if not dated:
        return None
    dated.sort(key=lambda item: item[0], reverse=True)
    return dated[0][1]


def _sort_report_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _sort_key(entry: dict[str, Any]) -> tuple[int, pd.Timestamp]:
        ts = pd.to_datetime(entry.get("date"), errors="coerce")
        if pd.isna(ts):
            return (1, pd.Timestamp.min)
        return (0, ts)

    return sorted(entries or [], key=_sort_key, reverse=True)


def _assert_not_stale_publish(*, data_repo: Path, universe: str, prices: pd.DataFrame) -> None:
    if os.environ.get("MKTME_ALLOW_STALE_PUBLISH", "").lower() in {"1", "true", "yes"}:
        return

    existing_path = data_repo / universe / "prices.parquet"
    if not existing_path.exists():
        return

    fetched_max = pd.to_datetime(prices.index, errors="coerce").max()
    if pd.isna(fetched_max):
        return

    try:
        existing = pd.read_parquet(existing_path)
    except Exception:
        return

    existing_max = pd.to_datetime(existing.index, errors="coerce").max()
    if pd.isna(existing_max):
        return

    if fetched_max < existing_max:
        raise RuntimeError(
            f"Refusing to publish stale {universe} prices: fetched max date "
            f"{fetched_max.date().isoformat()} is older than existing published max date "
            f"{existing_max.date().isoformat()}. Set MKTME_ALLOW_STALE_PUBLISH=1 to override."
        )


def _write_report_artifacts(
    *,
    data_repo: Path,
    universe: str,
    metrics_df: pd.DataFrame,
    lookback: int,
) -> tuple[str, dict[str, str]]:
    asof_date = pd.to_datetime(metrics_df.get("AsOfDate", pd.Timestamp.utcnow())).max()
    if isinstance(asof_date, pd.Timestamp):
        asof_date = asof_date.date().isoformat()
    else:
        asof_date = pd.Timestamp.utcnow().date().isoformat()

    daily = generate_daily_report(
        metrics_df=metrics_df,
        universe=universe,
        asof_date=asof_date,
        lookback=lookback,
        primary_rank_metric="Annualized Sharpe",
        top_n=5,
    )
    daily.report["facts"] = daily.facts
    daily.report["run_utc"] = utc_now_iso()
    daily.report["markdown"] = daily.markdown

    reports_dir = data_repo / "reports" / universe
    reports_dir.mkdir(parents=True, exist_ok=True)

    json_path = reports_dir / f"{asof_date}.json"
    md_path = reports_dir / f"{asof_date}.md"
    facts_path = reports_dir / f"{asof_date}.facts.json"

    save_report_json(daily.report, json_path)
    save_report_markdown(daily.markdown, md_path)
    facts_path.write_text(json.dumps(daily.facts, indent=2), encoding="utf-8")

    index_path = data_repo / "reports" / "index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            index = {}
    else:
        index = {}

    index["generated_at"] = utc_now_iso()
    index.setdefault("base_url", "https://raw.githubusercontent.com/marshdoggo/mktme-data/main")
    index.setdefault("universes", {})
    index["universes"].setdefault(universe, [])

    entry = {
        "date": asof_date,
        "json": f"reports/{universe}/{asof_date}.json",
        "md": f"reports/{universe}/{asof_date}.md",
        "facts": f"reports/{universe}/{asof_date}.facts.json",
    }
    existing = [e for e in index["universes"][universe] if e.get("date") != asof_date]
    index["universes"][universe] = _sort_report_entries([entry] + existing)
    index["universes"][universe] = index["universes"][universe][:14]
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return asof_date, {
        "markdown_path": f"reports/{universe}/{asof_date}.md",
        "json_path": f"reports/{universe}/{asof_date}.json",
        "facts_path": f"reports/{universe}/{asof_date}.facts.json",
    }


def _write_manifest(data_repo: Path, rel_paths: dict[str, str], github_user: str) -> dict[str, Any]:
    base_url = f"https://raw.githubusercontent.com/{github_user}/mktme-data/main"
    manifest = {
        "generated_at": utc_now_iso(),
        "base_url": base_url,
        "universes": {u: {"parquet_url": f"{base_url}/{rel}"} for u, rel in rel_paths.items()},
    }
    manifest_path = data_repo / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def run_publish(
    *,
    data_repo: Path,
    universes: list[str],
    source_by_universe: dict[str, str],
    lookback: int,
    force_refresh: bool,
    trigger_type: str,
    github_user: str,
    use_existing_parquet: bool,
) -> None:
    init_db()
    run_id = start_pipeline_run(trigger_type=trigger_type, mode="legacy_publish", git_sha=_git_sha())
    set_status("pipeline.mode", "legacy_publish")
    set_status("pipeline.last_started_at", utc_now_iso())
    set_status("pipeline.last_error", None)

    rel_paths: dict[str, str] = {}
    asof_dates: dict[str, str] = {}

    try:
        data_repo.mkdir(parents=True, exist_ok=True)
        ticker_frames = {universe: _ticker_frame(universe) for universe in universes}
        prefetched_equity_prices: dict[str, pd.DataFrame] = {}
        grouped_tickers: dict[str, list[str]] = {}
        for universe in universes:
            if universe == "fx":
                continue
            source = source_by_universe.get(universe, "auto")
            grouped_tickers.setdefault(source, [])
            for ticker in ticker_frames[universe]["Ticker"].astype(str).tolist():
                if ticker not in grouped_tickers[source]:
                    grouped_tickers[source].append(ticker)
        if not use_existing_parquet:
            for source, equity_tickers in grouped_tickers.items():
                if not equity_tickers:
                    continue
                prefetched_equity_prices[source] = _to_naive_dt_index(
                    download_prices(
                        equity_tickers,
                        force_refresh=force_refresh,
                        equity_source=source,
                    )
                )

        for universe in universes:
            tickers_df = ticker_frames[universe]
            universe_source = source_by_universe.get(universe, "auto")
            prices = _load_or_fetch_prices(
                data_repo=data_repo,
                universe=universe,
                tickers_df=tickers_df,
                lookback=lookback,
                force_refresh=force_refresh,
                use_existing_parquet=use_existing_parquet,
                prefetched_prices=prefetched_equity_prices.get(universe_source),
                equity_source=universe_source,
            )
            if prices.empty:
                raise RuntimeError(f"No price data available for universe '{universe}'")
            _assert_not_stale_publish(data_repo=data_repo, universe=universe, prices=prices)

            metrics_df = _build_metrics(prices, tickers_df, lookback=lookback)
            metrics_df["AsOfDate"] = pd.to_datetime(prices.index.max()).date().isoformat()

            rel_path, snapshot_path = _write_prices_parquet(data_repo, universe, prices)
            rel_paths[universe] = rel_path
            record_snapshot(
                universe=universe,
                asof_date=pd.to_datetime(prices.index.max()).date().isoformat(),
                artifact_path=snapshot_path,
                symbol_count=int(prices.shape[1]),
                row_count=int(prices.shape[0]),
            )

            asof_dates[universe], report_paths = _write_report_artifacts(
                data_repo=data_repo,
                universe=universe,
                metrics_df=metrics_df,
                lookback=lookback,
            )
            record_report(
                universe=universe,
                asof_date=asof_dates[universe],
                markdown_path=report_paths["markdown_path"],
                json_path=report_paths["json_path"],
                facts_path=report_paths["facts_path"],
            )
            set_status(f"legacy.latest_report_date.{universe}", asof_dates[universe])

        manifest = _write_manifest(data_repo, rel_paths, github_user=github_user)
        reports_index = json.loads((data_repo / "reports" / "index.json").read_text(encoding="utf-8"))
        report_universes = reports_index.get("universes") or {}
        for universe, entries in report_universes.items():
            latest_date = _latest_index_date(entries)
            if latest_date:
                set_status(f"legacy.latest_report_date.{universe}", latest_date)
        set_status("pipeline.last_finished_at", utc_now_iso())
        set_status("pipeline.last_success_at", utc_now_iso())
        set_status("pipeline.summary", "Legacy artifact publish completed successfully from the main repo.")
        set_status("legacy.manifest_generated_at", manifest.get("generated_at"))
        set_status("legacy.report_index_generated_at", reports_index.get("generated_at"))
        set_status("legacy.manifest_universes", sorted(rel_paths.keys()))
        set_status("legacy.publisher_owner", "main_repo")
        set_status("legacy.data_repo_path", str(data_repo))

        details = {
            "published_universes": universes,
            "source_by_universe": source_by_universe,
            "latest_report_dates": asof_dates,
            "manifest_generated_at": manifest.get("generated_at"),
            "data_repo": str(data_repo),
            "used_existing_parquet": use_existing_parquet,
        }
        finish_pipeline_run(run_id, "success", details=details)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        set_status("pipeline.last_finished_at", utc_now_iso())
        set_status("pipeline.last_error", message)
        set_status("pipeline.summary", "Legacy artifact publish failed from the main repo.")
        finish_pipeline_run(run_id, "failed", error_summary=message)
        raise
    finally:
        write_status_snapshot_json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish legacy mktme-data artifacts from the main repo.")
    parser.add_argument("--data-repo", default=os.environ.get("MKTME_DATA_REPO", str(ROOT / "mktme-data")))
    parser.add_argument("--lookback", type=int, default=int(os.environ.get("MKTME_LOOKBACK", "252")))
    parser.add_argument("--github-user", default=os.environ.get("MKTME_GITHUB_USER", "marshdoggo"))
    parser.add_argument("--trigger-type", default=os.environ.get("MKTME_PIPELINE_TRIGGER", "manual"))
    parser.add_argument("--force-refresh", action="store_true", default=os.environ.get("MKTME_FORCE_REFRESH", "").lower() in {"1", "true", "yes"})
    parser.add_argument("--use-existing-parquet", action="store_true")
    parser.add_argument("--universes", nargs="+", default=["sp500"])
    parser.add_argument("--source-sp500", default=os.environ.get("MKTME_SOURCE_SP500", ""))
    parser.add_argument("--source-nasdaq100", default=os.environ.get("MKTME_SOURCE_NASDAQ100", ""))
    parser.add_argument("--source-dow30", default=os.environ.get("MKTME_SOURCE_DOW30", ""))
    parser.add_argument("--source-fx", default=os.environ.get("MKTME_SOURCE_FX", ""))
    args = parser.parse_args()

    run_publish(
        data_repo=Path(args.data_repo).expanduser(),
        universes=[u.lower() for u in args.universes],
        source_by_universe=_source_map_from_args(args),
        lookback=int(args.lookback),
        force_refresh=bool(args.force_refresh),
        trigger_type=args.trigger_type,
        github_user=args.github_user,
        use_existing_parquet=bool(args.use_existing_parquet),
    )


if __name__ == "__main__":
    main()
