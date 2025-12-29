"""
ai_report.py
- Generate a daily market “brief” from the computed metrics dataframe.
- Output: JSON-friendly dict + Markdown text.
- Save/load helpers so GitHub Actions can publish reports into the repo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import json
from pathlib import Path

import pandas as pd

def _fmt(v: Any, digits: int = 4) -> Optional[float]:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return round(float(v), digits)
    except Exception:
        return None


def _pick_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def _top_n(df: pd.DataFrame, by: str, n: int = 5, ascending: bool = False) -> pd.DataFrame:
    if by not in df.columns:
        return df.head(0)
    out = df.copy()
    out[by] = pd.to_numeric(out[by], errors="coerce")
    out = out.sort_values(by=by, ascending=ascending, na_position="last")
    return out.head(n)


def _sector_summary(df: pd.DataFrame, metric: str, n: int = 5) -> Dict[str, Any]:
    if "Sector" not in df.columns or metric not in df.columns:
        return {"available": False}

    x = df.copy()
    x[metric] = pd.to_numeric(x[metric], errors="coerce")
    g = x.dropna(subset=[metric]).groupby("Sector")[metric].agg(["mean", "median", "count"]).reset_index()
    if g.empty:
        return {"available": False}

    best = g.sort_values("mean", ascending=False).head(n)
    worst = g.sort_values("mean", ascending=True).head(n)

    def _rows(z: pd.DataFrame) -> List[Dict[str, Any]]:
        out = []
        for _, r in z.iterrows():
            out.append({
                "Sector": r["Sector"],
                "mean": _fmt(r["mean"], 4),
                "median": _fmt(r["median"], 4),
                "count": int(r["count"]),
            })
        return out

    return {
        "available": True,
        "metric": metric,
        "best_by_mean": _rows(best),
        "worst_by_mean": _rows(worst),
    }


import re


def _safe_num(v: Any) -> Optional[float]:
    """Coerce to float or return None."""
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return float(v)
    except Exception:
        return None


def _dist_stats(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    """Return min/median/max for a numeric column, plus count."""
    if col not in df.columns:
        return {"available": False, "column": col}
    x = pd.to_numeric(df[col], errors="coerce").dropna()
    if x.empty:
        return {"available": False, "column": col}
    return {
        "available": True,
        "column": col,
        "count": int(x.shape[0]),
        "min": _fmt(x.min(), 4),
        "median": _fmt(x.median(), 4),
        "max": _fmt(x.max(), 4),
    }


def _extreme_row(df: pd.DataFrame, col: str, highest: bool) -> Dict[str, Any]:
    """Return {Ticker, Name, value} for the extreme of `col`."""
    if col not in df.columns:
        return {"available": False, "column": col}
    x = df.copy()
    x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna(subset=[col])
    if x.empty:
        return {"available": False, "column": col}
    r = x.sort_values(col, ascending=not highest).iloc[0]
    return {
        "available": True,
        "column": col,
        "Ticker": str(r.get("Ticker")) if r.get("Ticker") is not None else None,
        "Name": None if (isinstance(r.get("Name"), float) and pd.isna(r.get("Name"))) else (str(r.get("Name")) if r.get("Name") is not None else None),
        "value": _fmt(r.get(col), 4),
    }


def build_facts_block(report: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a compact, JSON-friendly facts block from the full report."""
    facts: Dict[str, Any] = {
        "schema_version": report.get("schema_version", 1),
        "universe": report.get("universe"),
        "universe_size": report.get("universe_size"),
        "asof_date": report.get("asof_date"),
        "lookback_trading_days": report.get("lookback_trading_days"),
        "primary_rank_metric": report.get("primary_rank_metric"),
        "distribution": report.get("distribution", {}),
        "extremes": report.get("extremes", {}),
        "leaders": {},
    }

    lbs = report.get("leaderboards", {}) or {}
    # Prefer the already-sorted leaderboards from the deterministic pipeline
    facts["leaders"]["top_by_primary_metric"] = (lbs.get("top_by_primary_metric") or [])[:5]
    facts["leaders"]["top_by_sortino"] = (lbs.get("top_by_sortino") or [])[:5]
    facts["leaders"]["top_by_cagr"] = (lbs.get("top_by_cagr") or [])[:5]
    facts["leaders"]["worst_by_max_drawdown"] = (lbs.get("worst_by_max_drawdown") or [])[:5]
    facts["leaders"]["top_by_volatility"] = (lbs.get("top_by_volatility") or [])[:5]

    # Sector summary (already deterministic)
    facts["sector_summary"] = report.get("sector_summary", {})
    return facts


def facts_json_text(facts: Dict[str, Any]) -> str:
    """Compact JSON string suitable for LLM prompts."""
    return json.dumps(facts, ensure_ascii=False, separators=(",", ":"))


def _extract_numbers(text: str) -> List[str]:
    """Extract numeric literals that look like metrics (floats/percents/negatives)."""
    # capture things like -0.30, 1.9919, 17.48, 38.3%, -91.9%
    pat = r"(?<![A-Za-z0-9_])[-+]?\d+(?:\.\d+)?%?"
    return re.findall(pat, text)


def allowed_numbers_from_facts(facts: Dict[str, Any]) -> set[str]:
    """Build a whitelist of numeric strings that are allowed to appear in LLM commentary."""
    allowed: set[str] = set()

    def _walk(v: Any):
        if v is None:
            return
        if isinstance(v, (int, float)):
            # Include several common renderings to avoid false rejects
            f = float(v)
            allowed.add(str(int(f)) if f.is_integer() else str(f))
            allowed.add(f"{f:.4f}")
            allowed.add(f"{f:.3f}")
            allowed.add(f"{f:.2f}")
            allowed.add(f"{f:.1f}")
            return
        if isinstance(v, str):
            # If the string is numeric-ish, keep it
            for n in _extract_numbers(v):
                allowed.add(n)
            return
        if isinstance(v, dict):
            for vv in v.values():
                _walk(vv)
            return
        if isinstance(v, list):
            for vv in v:
                _walk(vv)
            return

    _walk(facts)

    # Allow common percent forms for drawdowns etc.
    extra = set()
    for n in list(allowed):
        if n.endswith("%"):
            continue
        try:
            f = float(n)
            extra.add(f"{f*100:.1f}%")
            extra.add(f"{f*100:.0f}%")
        except Exception:
            pass
    allowed |= extra

    return allowed


def validate_commentary_numbers(commentary: str, facts: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Return (ok, unexpected_numbers). Only checks numeric literals.

    Notes:
      - We ignore obvious date fragments (YYYY-MM-DD or YYYY/MM/DD) and small list/bullet indices.
      - Everything else numeric must be present (in some acceptable rendering) in the facts whitelist.
    """
    allowed = allowed_numbers_from_facts(facts)

    # Remove date-like substrings to prevent rejecting 2025-12-26 fragments.
    scrubbed = re.sub(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b", " ", commentary)

    nums = _extract_numbers(scrubbed)

    unexpected: List[str] = []
    for n in nums:
        # ignore bullet indices and small ordinals like 1,2,3
        if n.isdigit() and len(n) <= 2:
            continue
        # ignore 4-digit years that might appear outside dates
        if n.isdigit() and len(n) == 4:
            continue
        if n not in allowed:
            unexpected.append(n)

    return (len(unexpected) == 0), unexpected


def sanitize_llm_commentary(commentary: str) -> str:
    """Light cleanup for LLM commentary. Keeps it human-readable but avoids weird whitespace."""
    if commentary is None:
        return ""
    # Normalize newlines and strip leading/trailing whitespace
    text = str(commentary).replace("\r\n", "\n").replace("\r", "\n").strip()
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def attach_llm_commentary_to_markdown(markdown: str, commentary: str, facts: Dict[str, Any]) -> Tuple[str, bool, List[str]]:
    """Attach LLM commentary to the deterministic markdown.

    Returns:
      (new_markdown, accepted, unexpected_numbers)

    If numeric validation fails, the original markdown is returned unchanged.
    """
    cleaned = sanitize_llm_commentary(commentary)
    if not cleaned:
        return markdown, True, []

    ok, unexpected = validate_commentary_numbers(cleaned, facts)
    if not ok:
        return markdown, False, unexpected

    # Add commentary as its own section near the top (after Headlines).
    marker = "## Headlines\n"
    if marker in markdown:
        parts = markdown.split(marker, 1)
        head, rest = parts[0], parts[1]
        injected = (
            marker
            + rest.split("\n\n", 1)[0]
            + "\n\n"
            + "## Commentary (LLM)\n\n"
            + cleaned
            + "\n\n"
            + (rest.split("\n\n", 1)[1] if "\n\n" in rest else "")
        )
        return head + injected, True, []

    # Fallback: append at end
    new_md = markdown.rstrip() + "\n\n## Commentary (LLM)\n\n" + cleaned + "\n"
    return new_md, True, []


@dataclass
class DailyReport:
    report: Dict[str, Any]
    facts: Dict[str, Any]
    markdown: str


def generate_daily_report(
    metrics_df: pd.DataFrame,
    universe: str,
    asof_date: str,
    lookback: int,
    primary_rank_metric: str = "Annualized Sharpe",
    top_n: int = 5,
) -> DailyReport:
    """
    Build a TradingEconomics-style brief from the metrics table.

    - primary_rank_metric controls the “Top names” table.
    - Includes a few complementary leaderboards + sector summary when possible.
    """
    df = metrics_df.copy()

    # Ensure Ticker column exists (some flows might keep index)
    if "Ticker" not in df.columns:
        if df.index.name:
            df = df.reset_index().rename(columns={df.index.name: "Ticker"})
        else:
            df = df.reset_index().rename(columns={"index": "Ticker"})

    # Universe size (deterministic) — used to whitelist counts in LLM commentary
    try:
        universe_size = int(df["Ticker"].nunique())
    except Exception:
        universe_size = None

    base_cols = _pick_cols(df, ["Ticker", "Name", "Sector", "SubIndustry"])
    metric_cols = _pick_cols(df, [
        primary_rank_metric,
        "Sortino Ratio",
        "Mean Daily Return",
        "Daily Volatility (Std)",
        "CAGR",
        "Max Drawdown",
        "RSI(14)",
    ])

    # Distribution summaries (deterministic facts)
    dist_primary = _dist_stats(df, primary_rank_metric)
    dist_vol = _dist_stats(df, "Daily Volatility (Std)")
    dist_rsi = _dist_stats(df, "RSI(14)")

    # Extremes (deterministic facts)
    worst_dd_row = _extreme_row(df, "Max Drawdown", highest=False)
    top_rsi_row = _extreme_row(df, "RSI(14)", highest=True)
    low_rsi_row = _extreme_row(df, "RSI(14)", highest=False)

    # Leaderboards
    top_primary = _top_n(df, by=primary_rank_metric, n=top_n, ascending=False)
    top_sortino = _top_n(df, by="Sortino Ratio", n=top_n, ascending=False) if "Sortino Ratio" in df.columns else df.head(0)
    top_cagr = _top_n(df, by="CAGR", n=top_n, ascending=False) if "CAGR" in df.columns else df.head(0)

    # “Risk” boards: biggest drawdowns (most negative) and highest vol
    worst_dd = _top_n(df, by="Max Drawdown", n=top_n, ascending=True) if "Max Drawdown" in df.columns else df.head(0)
    top_vol = _top_n(df, by="Daily Volatility (Std)", n=top_n, ascending=False) if "Daily Volatility (Std)" in df.columns else df.head(0)

    def _rows(x: pd.DataFrame) -> List[Dict[str, Any]]:
        if x.empty:
            return []
        keep = base_cols + metric_cols
        keep = _pick_cols(x, keep)
        out = []
        for _, r in x[keep].iterrows():
            row: Dict[str, Any] = {}
            for c in keep:
                if c in metric_cols:
                    row[c] = _fmt(r.get(c), 6 if c in ("Mean Daily Return",) else 4)
                else:
                    val = r.get(c)
                    row[c] = None if (isinstance(val, float) and pd.isna(val)) else str(val)
            out.append(row)
        return out

    sector_block = _sector_summary(df, metric=primary_rank_metric, n=5)

    report = {
        "schema_version": 1,
        "universe": universe,
        "universe_size": universe_size,
        "asof_date": asof_date,
        "lookback_trading_days": int(lookback),
        "primary_rank_metric": primary_rank_metric,
        "leaderboards": {
            "top_by_primary_metric": _rows(top_primary),
            "top_by_sortino": _rows(top_sortino),
            "top_by_cagr": _rows(top_cagr),
            "worst_by_max_drawdown": _rows(worst_dd),
            "top_by_volatility": _rows(top_vol),
        },
        "distribution": {
            "primary_metric": dist_primary,
            "daily_volatility": dist_vol,
            "rsi14": dist_rsi,
        },
        "extremes": {
            "largest_drawdown": worst_dd_row,
            "rsi_high": top_rsi_row,
            "rsi_low": low_rsi_row,
        },
        "sector_summary": sector_block,
        "notes": [
            f"Tables are computed from the app's cached price dataset and metrics pipeline.",
            f"'top_by_primary_metric' is sorted by {primary_rank_metric} descending (higher first).",
        ],
    }

    md = _render_markdown(report)
    facts = build_facts_block(report)
    return DailyReport(report=report, facts=facts, markdown=md)


def _render_markdown(report: Dict[str, Any]) -> str:
    u = report.get("universe", "?")
    usz = report.get("universe_size")
    d = report.get("asof_date", "?")
    lb = report.get("lookback_trading_days", "?")
    prim = report.get("primary_rank_metric", "Annualized Sharpe")

    def _md_table(rows: List[Dict[str, Any]], title: str, max_rows: int = 5) -> str:
        if not rows:
            return f"### {title}\n\n_(no data)_\n"
        rows = rows[:max_rows]
        cols = list(rows[0].keys())
        lines = [f"### {title}", ""]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for r in rows:
            vals = []
            for c in cols:
                v = r.get(c)
                vals.append("" if v is None else str(v))
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")
        return "\n".join(lines)

    blocks = []
    blocks.append(f"# Market Metric Explorer — Daily Brief\n")
    uni_line = f"- Universe: **{u}**" + (f" ({usz} tickers)" if usz is not None else "")
    blocks.append(
        uni_line
        + f"\n- As-of: **{d}**"
        + f"\n- Lookback: **{lb} trading days**"
        + f"\n- Primary rank metric: **{prim}**\n"
    )

    dist = report.get("distribution", {}) or {}
    ext = report.get("extremes", {}) or {}

    blocks.append("## Headlines\n")

    prim_stats = dist.get("primary_metric", {})
    if prim_stats.get("available"):
        blocks.append(f"- {prim} distribution: min **{prim_stats.get('min')}**, median **{prim_stats.get('median')}**, max **{prim_stats.get('max')}**.")

    vol_stats = dist.get("daily_volatility", {})
    if vol_stats.get("available"):
        blocks.append(f"- Daily volatility distribution: min **{vol_stats.get('min')}**, median **{vol_stats.get('median')}**, max **{vol_stats.get('max')}**.")

    ldd = ext.get("largest_drawdown", {})
    if ldd.get("available"):
        blocks.append(f"- Largest drawdown name: **{ldd.get('Ticker')}** with max drawdown **{ldd.get('value')}**.")

    rhi = ext.get("rsi_high", {})
    rlo = ext.get("rsi_low", {})
    if rhi.get("available") and rlo.get("available"):
        blocks.append(f"- RSI extremes (14): highest **{rhi.get('Ticker')}** ({rhi.get('value')}), lowest **{rlo.get('Ticker')}** ({rlo.get('value')}).")

    blocks.append("")

    lbs = report.get("leaderboards", {})
    blocks.append(_md_table(lbs.get("top_by_primary_metric", []), f"Top by {prim}"))
    if lbs.get("top_by_sortino"):
        blocks.append(_md_table(lbs.get("top_by_sortino", []), "Top by Sortino Ratio"))
    if lbs.get("top_by_cagr"):
        blocks.append(_md_table(lbs.get("top_by_cagr", []), "Top by CAGR"))
    if lbs.get("worst_by_max_drawdown"):
        blocks.append(_md_table(lbs.get("worst_by_max_drawdown", []), "Worst by Max Drawdown"))
    if lbs.get("top_by_volatility"):
        blocks.append(_md_table(lbs.get("top_by_volatility", []), "Top by Daily Volatility (Std)"))

    sec = report.get("sector_summary", {})
    if sec.get("available"):
        blocks.append("## Sector Summary\n")
        blocks.append(f"Metric: **{sec.get('metric')}**\n")
        best = sec.get("best_by_mean", [])
        worst = sec.get("worst_by_mean", [])

        def _sec_table(rows: List[Dict[str, Any]], title: str) -> str:
            if not rows:
                return f"### {title}\n\n_(no data)_\n"
            cols = list(rows[0].keys())
            lines = [f"### {title}", ""]
            lines.append("| " + " | ".join(cols) + " |")
            lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
            for r in rows:
                lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
            lines.append("")
            return "\n".join(lines)

        blocks.append(_sec_table(best, "Best sectors by mean"))
        blocks.append(_sec_table(worst, "Worst sectors by mean"))

    notes = report.get("notes", [])
    if notes:
        blocks.append("## Notes\n")
        for n in notes:
            blocks.append(f"- {n}")
        blocks.append("")

    return "\n".join(blocks)


# ----------------------------- Save / Load -------------------------------------

def save_report_json(report: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def save_report_markdown(markdown: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")


def load_report_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))