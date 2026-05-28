from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from status_store import (  # noqa: E402
    finish_pipeline_run,
    init_db,
    set_status,
    start_pipeline_run,
    utc_now_iso,
    write_status_snapshot_json,
)


DEFAULT_MANIFEST_URL = "https://raw.githubusercontent.com/marshdoggo/mktme-data/main/manifest.json"
DEFAULT_REPORTS_INDEX_URL = "https://raw.githubusercontent.com/marshdoggo/mktme-data/main/reports/index.json"


def _git_sha() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True)
            .strip()
        )
    except Exception:
        return None


def _load_json(url: str) -> dict[str, Any]:
    if url.startswith("file://"):
        return json.loads(Path(url[7:]).read_text(encoding="utf-8"))
    maybe_path = Path(url).expanduser()
    if maybe_path.exists():
        return json.loads(maybe_path.read_text(encoding="utf-8"))
    import requests

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _probe_legacy_sources() -> dict[str, Any]:
    manifest_url = os.environ.get("MKTME_MANIFEST_URL", DEFAULT_MANIFEST_URL)
    reports_url = os.environ.get("MKTME_REPORTS_INDEX_URL", DEFAULT_REPORTS_INDEX_URL)

    manifest = _load_json(manifest_url)
    reports = _load_json(reports_url)

    universes = sorted((manifest.get("universes") or {}).keys())
    latest_by_universe: dict[str, str | None] = {}
    report_universes = reports.get("universes") or {}
    for universe, entries in report_universes.items():
        latest = entries[0].get("date") if isinstance(entries, list) and entries else None
        latest_by_universe[universe] = latest

    return {
        "manifest_url": manifest_url,
        "reports_index_url": reports_url,
        "manifest_generated_at": manifest.get("generated_at"),
        "reports_index_generated_at": reports.get("generated_at"),
        "manifest_universes": universes,
        "latest_report_dates": latest_by_universe,
    }


def run_probe(trigger_type: str) -> None:
    init_db()
    run_id = start_pipeline_run(trigger_type=trigger_type, mode="probe", git_sha=_git_sha())
    set_status("pipeline.mode", "probe")
    set_status("pipeline.last_started_at", utc_now_iso())
    set_status("pipeline.last_error", None)

    try:
        probe = _probe_legacy_sources()
        set_status("pipeline.last_finished_at", utc_now_iso())
        set_status("pipeline.last_success_at", utc_now_iso())
        set_status("pipeline.summary", "Legacy source probe completed successfully.")
        set_status("legacy.manifest_url", probe["manifest_url"])
        set_status("legacy.reports_index_url", probe["reports_index_url"])
        set_status("legacy.manifest_generated_at", probe["manifest_generated_at"])
        set_status("legacy.report_index_generated_at", probe["reports_index_generated_at"])
        set_status("legacy.manifest_universes", probe["manifest_universes"])
        for universe, latest_date in probe["latest_report_dates"].items():
            set_status(f"legacy.latest_report_date.{universe}", latest_date)

        details = {
            "probe": probe,
            "completed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        }
        finish_pipeline_run(run_id, "success", details=details)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        set_status("pipeline.last_finished_at", utc_now_iso())
        set_status("pipeline.last_error", message)
        set_status("pipeline.summary", "Legacy source probe failed.")
        finish_pipeline_run(run_id, "failed", error_summary=message)
        raise
    finally:
        write_status_snapshot_json()


def run_publish_from_args(args: argparse.Namespace) -> None:
    from publish_legacy_artifacts import run_publish  # noqa: WPS433

    source_by_universe = {
        "sp500": (args.source_sp500 or args.equity_source).strip().lower(),
        "nasdaq100": (args.source_nasdaq100 or args.equity_source).strip().lower(),
        "dow30": (args.source_dow30 or args.equity_source).strip().lower(),
        "fx": (args.source_fx or args.equity_source).strip().lower(),
    }
    run_publish(
        data_repo=Path(args.data_repo).expanduser(),
        universes=[u.lower() for u in args.universes],
        source_by_universe=source_by_universe,
        lookback=int(args.lookback),
        force_refresh=bool(args.force_refresh),
        trigger_type=args.trigger_type,
        github_user=args.github_user,
        use_existing_parquet=bool(args.use_existing_parquet),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the repo-local rebuild pipeline.")
    parser.add_argument(
        "--trigger-type",
        default=os.environ.get("MKTME_PIPELINE_TRIGGER", "manual"),
        help="How this run was triggered, e.g. schedule or manual.",
    )
    parser.add_argument(
        "--mode",
        default=os.environ.get("MKTME_PIPELINE_MODE", "publish"),
        choices=["probe", "publish"],
        help="Pipeline mode. Publish refreshes local data artifacts; probe only records source health.",
    )
    parser.add_argument("--data-repo", default=os.environ.get("MKTME_DATA_REPO", str(ROOT / "mktme-data")))
    parser.add_argument("--lookback", type=int, default=int(os.environ.get("MKTME_LOOKBACK", "252")))
    parser.add_argument("--github-user", default=os.environ.get("MKTME_GITHUB_USER", "marshdoggo"))
    parser.add_argument("--force-refresh", action="store_true", default=os.environ.get("MKTME_FORCE_REFRESH", "").lower() in {"1", "true", "yes"})
    parser.add_argument("--use-existing-parquet", action="store_true")
    parser.add_argument(
        "--universes",
        nargs="+",
        default=os.environ.get("MKTME_UNIVERSES", "sp500 nasdaq100 dow30 fx").split(),
        choices=["sp500", "nasdaq100", "dow30", "fx"],
    )
    parser.add_argument("--equity-source", default=os.environ.get("MKTME_EQUITY_SOURCE", "auto"))
    parser.add_argument("--source-sp500", default=os.environ.get("MKTME_SOURCE_SP500", ""))
    parser.add_argument("--source-nasdaq100", default=os.environ.get("MKTME_SOURCE_NASDAQ100", ""))
    parser.add_argument("--source-dow30", default=os.environ.get("MKTME_SOURCE_DOW30", ""))
    parser.add_argument("--source-fx", default=os.environ.get("MKTME_SOURCE_FX", ""))
    args = parser.parse_args()

    if args.mode == "probe":
        run_probe(trigger_type=args.trigger_type)
    elif args.mode == "publish":
        run_publish_from_args(args)


if __name__ == "__main__":
    main()
