from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

from status_store import (
    get_db_path,
    get_latest_pipeline_run,
    get_latest_report,
    get_latest_snapshot,
    get_recent_pipeline_runs,
    get_status_map,
)


def _parse_iso(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def _parse_date(value: Any) -> date | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value[:10])
    except Exception:
        return None


def build_dashboard_health() -> dict[str, Any]:
    db_path = get_db_path()
    latest_run = get_latest_pipeline_run()
    status_map = get_status_map()
    recent_runs = get_recent_pipeline_runs(limit=5) if latest_run else []
    latest_report = get_latest_report("sp500")
    latest_snapshot = get_latest_snapshot("sp500")

    if not db_path.exists():
        return {
            "state": "warning",
            "title": "Pipeline status is not initialized yet.",
            "summary": "This app is still relying on the legacy manifest/report flow. The rebuild status store has not been generated in this checkout yet.",
            "details": [],
            "latest_run": None,
            "recent_runs": [],
        }

    last_success_at = _parse_iso(status_map.get("pipeline.last_success_at"))
    latest_report_date = _parse_date(
        (latest_report or {}).get("asof_date") or status_map.get("legacy.latest_report_date.sp500")
    )
    now_utc = datetime.now(timezone.utc)

    details = []
    if status_map.get("pipeline.last_success_at"):
        details.append(f"Last successful pipeline run: {status_map['pipeline.last_success_at']}")
    if status_map.get("legacy.manifest_generated_at"):
        details.append(f"Legacy manifest generated_at: {status_map['legacy.manifest_generated_at']}")
    if status_map.get("legacy.report_index_generated_at"):
        details.append(f"Legacy reports index generated_at: {status_map['legacy.report_index_generated_at']}")
    if latest_report_date is not None:
        details.append(f"Legacy latest SP500 report date: {latest_report_date.isoformat()}")
    if latest_snapshot is not None:
        details.append(
            f"Latest SP500 snapshot metadata: as-of {latest_snapshot.get('asof_date')} "
            f"• rows {latest_snapshot.get('row_count')} • symbols {latest_snapshot.get('symbol_count')}"
        )

    if latest_run is None:
        return {
            "state": "warning",
            "title": "Pipeline status database exists, but no runs are recorded yet.",
            "summary": "The rebuilt status layer is wired in, but the scheduled repo-local pipeline has not written a run record yet.",
            "details": details,
            "latest_run": None,
            "recent_runs": [],
        }

    if latest_run.get("status") == "failed":
        summary = latest_run.get("error_summary") or "The most recent repo-local pipeline run failed."
        return {
            "state": "error",
            "title": "Latest repo-local pipeline run failed.",
            "summary": summary,
            "details": details,
            "latest_run": latest_run,
            "recent_runs": recent_runs,
        }

    stale = False
    if last_success_at is not None and (now_utc - last_success_at).days >= 2:
        stale = True
    if latest_report_date is not None and (date.today() - latest_report_date).days >= 3:
        stale = True

    if stale:
        title = "Pipeline health is available, but freshness looks stale."
        summary = "The repo-local pipeline last succeeded, but the observed report or run dates are lagging behind expectations."
        state = "warning"
    else:
        title = "Pipeline health looks healthy."
        summary = "The rebuild status layer is active and recent pipeline metadata is available in the app."
        state = "success"

    return {
        "state": state,
        "title": title,
        "summary": summary,
        "details": details,
        "latest_run": latest_run,
        "recent_runs": recent_runs,
    }
