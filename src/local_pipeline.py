from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
VALID_SOURCES = ["auto", "twelvedata", "alphavantage", "yahoo", "stooq"]
DEFAULT_UNIVERSES = ["sp500", "nasdaq100", "dow30", "fx"]


@dataclass(frozen=True)
class PipelineResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str


def running_in_streamlit_cloud() -> bool:
    return os.getenv("STREAMLIT_SERVER_ENABLED") == "1" or os.getenv("STREAMLIT_RUNTIME") == "1"


def local_pipeline_ui_enabled() -> bool:
    raw = os.getenv("MKTME_ENABLE_LOCAL_PIPELINE_UI", "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    return not running_in_streamlit_cloud()


def build_pipeline_command(
    *,
    universes: list[str],
    lookback: int,
    equity_source: str,
    force_refresh: bool,
    use_existing_parquet: bool,
) -> list[str]:
    selected = [u for u in universes if u in DEFAULT_UNIVERSES]
    if not selected:
        selected = ["sp500"]

    source = equity_source if equity_source in VALID_SOURCES else "auto"
    cmd = [
        sys.executable,
        str(ROOT / "pipeline" / "run_daily_pipeline.py"),
        "--mode",
        "publish",
        "--trigger-type",
        "front_end",
        "--lookback",
        str(int(lookback)),
        "--equity-source",
        source,
        "--universes",
        *selected,
    ]
    if force_refresh:
        cmd.append("--force-refresh")
    if use_existing_parquet:
        cmd.append("--use-existing-parquet")
    return cmd


def run_pipeline(
    *,
    universes: list[str],
    lookback: int,
    equity_source: str,
    force_refresh: bool,
    use_existing_parquet: bool,
    timeout_seconds: int = 60 * 60 * 3,
) -> PipelineResult:
    cmd = build_pipeline_command(
        universes=universes,
        lookback=lookback,
        equity_source=equity_source,
        force_refresh=force_refresh,
        use_existing_parquet=use_existing_parquet,
    )
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=int(timeout_seconds),
        check=False,
    )
    return PipelineResult(
        command=cmd,
        returncode=int(completed.returncode),
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
