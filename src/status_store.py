from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = ROOT_DIR / "data" / "runtime" / "market_metrics.db"
DEFAULT_STATUS_JSON = ROOT_DIR / "data" / "runtime" / "pipeline_status.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def get_db_path() -> Path:
    raw = os.environ.get("MKTME_STATUS_DB_PATH")
    return Path(raw).expanduser() if raw else DEFAULT_DB_PATH


def get_status_json_path() -> Path:
    raw = os.environ.get("MKTME_STATUS_JSON_PATH")
    return Path(raw).expanduser() if raw else DEFAULT_STATUS_JSON


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def db_conn():
    db_path = get_db_path()
    _ensure_parent(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> Path:
    with db_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trigger_type TEXT NOT NULL,
                mode TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                git_sha TEXT,
                error_summary TEXT,
                details_json TEXT
            );

            CREATE TABLE IF NOT EXISTS app_status (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS data_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                universe TEXT NOT NULL,
                asof_date TEXT NOT NULL,
                created_at TEXT NOT NULL,
                symbol_count INTEGER,
                row_count INTEGER,
                artifact_path TEXT NOT NULL,
                UNIQUE(universe, asof_date, artifact_path)
            );

            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                universe TEXT NOT NULL,
                asof_date TEXT NOT NULL,
                created_at TEXT NOT NULL,
                markdown_path TEXT NOT NULL,
                json_path TEXT NOT NULL,
                facts_path TEXT,
                UNIQUE(universe, asof_date, markdown_path)
            );

            CREATE INDEX IF NOT EXISTS idx_pipeline_runs_started_at
            ON pipeline_runs(started_at DESC);
            CREATE INDEX IF NOT EXISTS idx_data_snapshots_universe_asof
            ON data_snapshots(universe, asof_date DESC);
            CREATE INDEX IF NOT EXISTS idx_reports_universe_asof
            ON reports(universe, asof_date DESC);
            """
        )
    return get_db_path()


def start_pipeline_run(trigger_type: str, mode: str, git_sha: str | None = None) -> int:
    started_at = utc_now_iso()
    with db_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO pipeline_runs (trigger_type, mode, status, started_at, git_sha)
            VALUES (?, ?, 'running', ?, ?)
            """,
            (trigger_type, mode, started_at, git_sha),
        )
        return int(cur.lastrowid)


def finish_pipeline_run(
    run_id: int,
    status: str,
    *,
    error_summary: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    finished_at = utc_now_iso()
    details_json = json.dumps(details, sort_keys=True) if details is not None else None
    with db_conn() as conn:
        conn.execute(
            """
            UPDATE pipeline_runs
            SET status = ?, finished_at = ?, error_summary = ?, details_json = ?
            WHERE id = ?
            """,
            (status, finished_at, error_summary, details_json, run_id),
        )


def set_status(key: str, value: Any) -> None:
    payload = json.dumps(value, sort_keys=True)
    updated_at = utc_now_iso()
    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO app_status (key, value_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value_json = excluded.value_json,
                updated_at = excluded.updated_at
            """,
            (key, payload, updated_at),
        )


def get_status(key: str, default: Any = None) -> Any:
    init_db()
    with db_conn() as conn:
        row = conn.execute(
            "SELECT value_json FROM app_status WHERE key = ?",
            (key,),
        ).fetchone()
    if row is None:
        return default
    try:
        return json.loads(row["value_json"])
    except Exception:
        return default


def delete_status(key: str) -> None:
    init_db()
    with db_conn() as conn:
        conn.execute("DELETE FROM app_status WHERE key = ?", (key,))


def get_status_map(prefix: str | None = None) -> dict[str, Any]:
    init_db()
    sql = "SELECT key, value_json FROM app_status"
    params: tuple[Any, ...] = ()
    if prefix:
        sql += " WHERE key LIKE ?"
        params = (f"{prefix}%",)
    with db_conn() as conn:
        rows = conn.execute(sql, params).fetchall()
    out: dict[str, Any] = {}
    for row in rows:
        try:
            out[row["key"]] = json.loads(row["value_json"])
        except Exception:
            out[row["key"]] = row["value_json"]
    return out


def get_latest_pipeline_run() -> dict[str, Any] | None:
    init_db()
    with db_conn() as conn:
        row = conn.execute(
            """
            SELECT id, trigger_type, mode, status, started_at, finished_at, git_sha, error_summary, details_json
            FROM pipeline_runs
            ORDER BY started_at DESC, id DESC
            LIMIT 1
            """
        ).fetchone()
    if row is None:
        return None
    out = dict(row)
    if out.get("details_json"):
        try:
            out["details"] = json.loads(out["details_json"])
        except Exception:
            out["details"] = None
    out.pop("details_json", None)
    return out


def get_recent_pipeline_runs(limit: int = 5) -> list[dict[str, Any]]:
    init_db()
    with db_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, trigger_type, mode, status, started_at, finished_at, git_sha, error_summary
            FROM pipeline_runs
            ORDER BY started_at DESC, id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    return [dict(r) for r in rows]


def record_snapshot(
    *,
    universe: str,
    asof_date: str,
    artifact_path: str,
    symbol_count: int | None,
    row_count: int | None,
) -> None:
    init_db()
    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO data_snapshots (universe, asof_date, created_at, symbol_count, row_count, artifact_path)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(universe, asof_date, artifact_path) DO UPDATE SET
                created_at = excluded.created_at,
                symbol_count = excluded.symbol_count,
                row_count = excluded.row_count
            """,
            (universe, asof_date, utc_now_iso(), symbol_count, row_count, artifact_path),
        )


def record_report(
    *,
    universe: str,
    asof_date: str,
    markdown_path: str,
    json_path: str,
    facts_path: str | None,
) -> None:
    init_db()
    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO reports (universe, asof_date, created_at, markdown_path, json_path, facts_path)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(universe, asof_date, markdown_path) DO UPDATE SET
                created_at = excluded.created_at,
                json_path = excluded.json_path,
                facts_path = excluded.facts_path
            """,
            (universe, asof_date, utc_now_iso(), markdown_path, json_path, facts_path),
        )


def get_latest_snapshot(universe: str) -> dict[str, Any] | None:
    init_db()
    with db_conn() as conn:
        row = conn.execute(
            """
            SELECT universe, asof_date, created_at, symbol_count, row_count, artifact_path
            FROM data_snapshots
            WHERE universe = ?
            ORDER BY asof_date DESC, created_at DESC, id DESC
            LIMIT 1
            """,
            (universe,),
        ).fetchone()
    return dict(row) if row else None


def get_latest_report(universe: str) -> dict[str, Any] | None:
    init_db()
    with db_conn() as conn:
        row = conn.execute(
            """
            SELECT universe, asof_date, created_at, markdown_path, json_path, facts_path
            FROM reports
            WHERE universe = ?
            ORDER BY asof_date DESC, created_at DESC, id DESC
            LIMIT 1
            """,
            (universe,),
        ).fetchone()
    return dict(row) if row else None


def list_recent_reports(universe: str, limit: int = 5) -> list[dict[str, Any]]:
    init_db()
    with db_conn() as conn:
        rows = conn.execute(
            """
            SELECT universe, asof_date, created_at, markdown_path, json_path, facts_path
            FROM reports
            WHERE universe = ?
            ORDER BY asof_date DESC, created_at DESC, id DESC
            LIMIT ?
            """,
            (universe, int(limit)),
        ).fetchall()
    return [dict(r) for r in rows]


def write_status_snapshot_json() -> Path:
    payload = {
        "generated_at": utc_now_iso(),
        "latest_run": get_latest_pipeline_run(),
        "status": get_status_map(),
        "recent_runs": get_recent_pipeline_runs(limit=10),
    }
    out_path = get_status_json_path()
    _ensure_parent(out_path)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out_path
