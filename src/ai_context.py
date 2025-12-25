"""
ai_context.py
- OpenAI client creation (Streamlit secrets + env var fallback)
- Simple rate limiting helpers
- Build compact, numerically-grounded context text for the in-app chat
  including top-N table rows and a dendrogram/cluster summary for highlighted tickers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import os
from time import time

import pandas as pd

# SciPy is already in your project; used for clustering summary
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# OpenAI SDK (optional at import-time)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ----------------------------- Client / Key ------------------------------------

def get_openai_api_key(st: Any = None) -> Optional[str]:
    """
    Get API key from Streamlit secrets if available, else env var.
    Accepts `st` so this module can be used outside Streamlit too.
    """
    key = None
    if st is not None:
        try:
            key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            key = None
    if not key:
        key = os.getenv("OPENAI_API_KEY")
    return key


def get_openai_client(st: Any = None) -> Optional[Any]:
    """
    Return OpenAI() client if SDK is installed and key exists, else None.
    """
    if OpenAI is None:
        return None
    key = get_openai_api_key(st=st)
    if not key:
        return None
    return OpenAI(api_key=key)


# ----------------------------- Rate limiting -----------------------------------

def rate_limit_ok(
    session_state: Dict[str, Any],
    max_calls: int = 8,
    per_seconds: int = 60,
    key_ts: str = "_ai_call_timestamps",
) -> Tuple[bool, str]:
    now = time()
    ts = session_state.get(key_ts, [])
    ts = [t for t in ts if (now - t) < per_seconds]
    if len(ts) >= max_calls:
        retry_in = int(per_seconds - (now - min(ts)))
        session_state[key_ts] = ts
        return False, f"Rate limit hit: {max_calls}/{per_seconds}s. Try again in ~{max(1, retry_in)}s."
    ts.append(now)
    session_state[key_ts] = ts
    return True, ""


def daily_cap_ok(
    session_state: Dict[str, Any],
    max_calls: int = 80,
    key_n: str = "_ai_call_count",
) -> Tuple[bool, str]:
    n = int(session_state.get(key_n, 0))
    if n >= max_calls:
        return False, f"Daily cap reached for this session ({max_calls} calls)."
    session_state[key_n] = n + 1
    return True, ""


# ----------------------------- Context building --------------------------------

def _fmt_num(v: Any, digits: int = 4) -> str:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "nan"
        return f"{float(v):.{digits}f}"
    except Exception:
        return "nan"


def _safe_series_stats(s: pd.Series) -> Optional[Tuple[float, float, float]]:
    s2 = pd.to_numeric(s, errors="coerce").dropna()
    if s2.empty:
        return None
    return float(s2.min()), float(s2.median()), float(s2.max())


def _table_top_n(
    metrics_df: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    top_n: int = 5,
    highlight_upper: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Mimic your UI table ordering: sort by y_metric desc,
    and if highlights exist, push highlighted tickers to top.
    """
    df = metrics_df.copy()

    # Ensure helper cols exist (won't error if absent)
    if "Ticker_upper" not in df.columns and "Ticker" in df.columns:
        df["Ticker_upper"] = df["Ticker"].astype(str).str.upper()

    if highlight_upper:
        df["IsHighlighted"] = df["Ticker_upper"].isin([t.upper() for t in highlight_upper])
        df = df.sort_values(by=["IsHighlighted", y_metric], ascending=[False, False], na_position="last")
    else:
        df = df.sort_values(by=y_metric, ascending=False, na_position="last")

    cols = []
    for c in ["Ticker", "Name", "Sector", "SubIndustry", x_metric, y_metric, "Annualized Sharpe", "Sortino Ratio", "Max Drawdown", "CAGR", "Daily Volatility (Std)"]:
        if c in df.columns and c not in cols:
            cols.append(c)

    if not cols:
        return df.head(top_n)

    out = df[cols].head(top_n).copy()
    return out


def _summarize_highlight_correlations(
    prices: pd.DataFrame,
    tickers_hi: List[str],
    asof_ts: pd.Timestamp,
    lookback: int,
    max_pairs_each_side: int = 3,
) -> List[str]:
    """
    Return a short correlation summary list of strings.
    """
    lines: List[str] = []
    cols = [t for t in tickers_hi if t in prices.columns]
    if len(cols) < 2:
        return lines

    pA = prices.loc[:asof_ts, cols].copy()
    if lookback and len(pA) > lookback:
        pA = pA.tail(lookback)

    rets = pA.pct_change().dropna(how="all").dropna(axis=1, how="all")
    if rets.shape[1] < 2:
        return lines

    corr = rets.corr()
    pairs: List[Tuple[str, str, float]] = []
    ccols = list(corr.columns)
    for i in range(len(ccols)):
        for j in range(i + 1, len(ccols)):
            a, b = ccols[i], ccols[j]
            val = float(corr.loc[a, b])
            pairs.append((a, b, val))

    if not pairs:
        return lines

    pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)
    top = pairs_sorted[:max_pairs_each_side]
    bottom = pairs_sorted[-max_pairs_each_side:]

    lines.append("Pairwise correlations between highlighted tickers (daily returns):")
    lines.append("Top positively correlated pairs:")
    for a, b, v in top:
        lines.append(f"  - {a} vs {b}: corr={v:.2f}")
    lines.append("Most negatively correlated / weakest pairs:")
    for a, b, v in bottom:
        lines.append(f"  - {a} vs {b}: corr={v:.2f}")

    return lines


def _summarize_clusters(
    prices: pd.DataFrame,
    tickers_hi: List[str],
    asof_ts: pd.Timestamp,
    lookback: int,
    max_clusters: int = 3,
) -> List[str]:
    """
    Produce a compact “dendrogram summary” by clustering on distance = 1 - corr.
    Returns human-readable cluster groupings.

    NOTE: This is a summary of the same information your dendrogram visual encodes.
    """
    lines: List[str] = []
    cols = [t for t in tickers_hi if t in prices.columns]
    if len(cols) < 2:
        return lines

    pA = prices.loc[:asof_ts, cols].copy()
    if lookback and len(pA) > lookback:
        pA = pA.tail(lookback)

    rets = pA.pct_change().dropna(how="all").dropna(axis=1, how="all")
    if rets.shape[1] < 2:
        return lines

    corr = rets.corr()
    dist = 1 - corr
    dist.values[range(len(dist)), range(len(dist))] = 0.0

    # condensed distance matrix for linkage
    dist_condensed = squareform(dist.values, checks=False)
    Z = linkage(dist_condensed, method="average")

    # cluster labels: try to cap to max_clusters via "maxclust"
    try:
        labels = fcluster(Z, t=max_clusters, criterion="maxclust")
    except Exception:
        return lines

    cluster_map: Dict[int, List[str]] = {}
    for t, lab in zip(corr.columns.tolist(), labels):
        cluster_map.setdefault(int(lab), []).append(str(t))

    # Sort clusters by size desc
    clusters = sorted(cluster_map.items(), key=lambda kv: len(kv[1]), reverse=True)
    lines.append(f"Cluster summary (hierarchical, distance=1-corr), up to {max_clusters} clusters:")
    for i, (lab, members) in enumerate(clusters, start=1):
        members_sorted = sorted(members)
        lines.append(f"  - Cluster {i} ({len(members_sorted)}): {', '.join(members_sorted)}")

    return lines


def build_view_context_text(
    metrics_df: pd.DataFrame,
    prices: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    universe: str,
    asof_ts: pd.Timestamp,
    lookback: int,
    highlight_upper: Optional[List[str]] = None,
    top_table_rows: int = 5,
    include_cluster_summary: bool = True,
) -> str:
    """
    Build a compact context string for the LLM.

    Includes:
    - view metadata
    - min/median/max of x and y across the current view
    - highlighted tickers snapshot (x/y + sector/name if available)
    - top-N rows of the data table as shown (sorted by y desc + highlight pinned)
    - correlation + cluster summary for highlighted tickers (dendrogram-adjacent)
    """
    lines: List[str] = []

    lines.append(f"Universe: {universe}")
    lines.append(f"As-of date: {pd.to_datetime(asof_ts).date().isoformat()}")
    lines.append(f"Lookback window: {int(lookback)} trading days")
    lines.append(f"Scatter plot axes: X = {x_metric}, Y = {y_metric}")
    lines.append(f"Tickers in view: {len(metrics_df)}")

    # Distribution stats for x/y
    if x_metric in metrics_df.columns:
        stx = _safe_series_stats(metrics_df[x_metric])
        if stx:
            lines.append(f"{x_metric} stats: min={stx[0]:.4f}, median={stx[1]:.4f}, max={stx[2]:.4f}")
    if y_metric in metrics_df.columns:
        sty = _safe_series_stats(metrics_df[y_metric])
        if sty:
            lines.append(f"{y_metric} stats: min={sty[0]:.4f}, median={sty[1]:.4f}, max={sty[2]:.4f}")

    # Top-N table rows (as user sees it)
    try:
        tbl = _table_top_n(metrics_df, x_metric, y_metric, top_n=top_table_rows, highlight_upper=highlight_upper or [])
        if not tbl.empty:
            lines.append("")
            lines.append(f"Top {min(top_table_rows, len(tbl))} rows from the Data Table (ordered like the UI):")
            # Emit as a compact, readable "CSV-ish" block
            cols = list(tbl.columns)
            lines.append("COLUMNS: " + ", ".join(cols))
            for _, r in tbl.iterrows():
                row = []
                for c in cols:
                    v = r.get(c)
                    if c in [x_metric, y_metric, "Annualized Sharpe", "Sortino Ratio", "Max Drawdown", "CAGR", "Daily Volatility (Std)"]:
                        row.append(_fmt_num(v, digits=4))
                    else:
                        row.append(str(v) if v is not None and not (isinstance(v, float) and pd.isna(v)) else "")
                lines.append("ROW: " + " | ".join(row))
    except Exception:
        pass

    # Highlighted tickers details
    if highlight_upper:
        hu = [t.upper() for t in highlight_upper]
        df2 = metrics_df.copy()
        if "Ticker_upper" not in df2.columns and "Ticker" in df2.columns:
            df2["Ticker_upper"] = df2["Ticker"].astype(str).str.upper()

        hi = df2[df2["Ticker_upper"].isin(hu)] if "Ticker_upper" in df2.columns else pd.DataFrame()

        if not hi.empty:
            lines.append("")
            lines.append(f"Highlighted tickers present in view: {len(hi)}")
            for _, r in hi.iterrows():
                nm = r.get("Name", "")
                sec = r.get("Sector", "")
                xval = _fmt_num(r.get(x_metric), digits=4)
                yval = _fmt_num(r.get(y_metric), digits=4)
                lines.append(f"- {r.get('Ticker','?')}: Name={nm}, Sector={sec}, {x_metric}={xval}, {y_metric}={yval}")

            tickers_hi = hi["Ticker"].dropna().astype(str).unique().tolist()

            # Correlation snapshot
            corr_lines = _summarize_highlight_correlations(
                prices=prices,
                tickers_hi=tickers_hi,
                asof_ts=asof_ts,
                lookback=lookback,
            )
            if corr_lines:
                lines.append("")
                lines.extend(corr_lines)

            # Cluster summary (dendrogram-ish)
            if include_cluster_summary:
                cl_lines = _summarize_clusters(
                    prices=prices,
                    tickers_hi=tickers_hi,
                    asof_ts=asof_ts,
                    lookback=lookback,
                    max_clusters=3,
                )
                if cl_lines:
                    lines.append("")
                    lines.extend(cl_lines)

    return "\n".join(lines)


# ----------------------------- Chat call wrapper -------------------------------

@dataclass
class ChatResult:
    ok: bool
    answer: str
    error: Optional[str] = None
    context_text: Optional[str] = None


def ask_ai_about_view(
    client: Any,
    context_text: str,
    user_question: str,
    model: str = "gpt-4.1-mini",
) -> ChatResult:
    """
    Thin wrapper around Responses API call; returns a ChatResult.
    """
    system_instructions = (
        "You are a quantitative markets analyst helping a user interpret a Streamlit dashboard. "
        "Use ONLY the provided context. Be numerically concrete and concise. "
        "If the user asks for something not present in context, say so plainly and suggest what to highlight or change."
    )

    try:
        resp = client.responses.create(
            model=model,
            instructions=system_instructions,
            input=f"CONTEXT:\n{context_text}\n\nUSER_QUESTION:\n{user_question}",
        )
        answer = getattr(resp, "output_text", None) or ""
        return ChatResult(ok=True, answer=answer, context_text=context_text)
    except Exception as e:
        return ChatResult(ok=False, answer=f"Error calling OpenAI API: {e}", error=str(e), context_text=context_text)