from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import streamlit as st


UNIVERSES = ["sp500", "nasdaq100", "dow30", "fx"]
UNIVERSE_LABELS = {
    "sp500": "S&P 500",
    "nasdaq100": "Nasdaq 100",
    "dow30": "Dow 30",
    "fx": "FX",
}

PRIMARY_VOL_COL = "Realized_Vol_20D"
FALLBACK_VOL_COL = "Daily Volatility (Std)"
VOLATILITY_COLUMNS = [PRIMARY_VOL_COL, FALLBACK_VOL_COL]
REGIME_THRESHOLDS = {
    "calm": 20.0,
    "elevated": 80.0,
    "panic": 95.0,
}


def get_volatility_column(df: pd.DataFrame) -> str | None:
    for col in VOLATILITY_COLUMNS:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            if values.notna().any():
                return col
    return None


def percentile_rank(value: float, reference_series: pd.Series) -> float:
    ref = pd.to_numeric(reference_series, errors="coerce").dropna()
    if ref.empty or pd.isna(value):
        return np.nan
    return float((ref <= float(value)).mean() * 100)


def classify_vol_regime(value: float, reference_series: pd.Series) -> str:
    pct = percentile_rank(value, reference_series)
    if pd.isna(pct):
        return "Unavailable"
    if pct < REGIME_THRESHOLDS["calm"]:
        return "Calm"
    if pct < REGIME_THRESHOLDS["elevated"]:
        return "Normal"
    if pct < REGIME_THRESHOLDS["panic"]:
        return "Elevated"
    return "Panic"


def _clean_metrics_df(df: pd.DataFrame, vol_col: str | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "Ticker" not in out.columns:
        out = out.reset_index().rename(columns={"index": "Ticker"})
    if vol_col and vol_col in out.columns:
        out[vol_col] = pd.to_numeric(out[vol_col], errors="coerce")
    for col in ["Universe", "Ticker", "Name", "Sector", "SubIndustry"]:
        if col not in out.columns:
            out[col] = ""
    return out


def _summary_row(df: pd.DataFrame, label: str, vol_col: str) -> dict:
    valid = _clean_metrics_df(df, vol_col).dropna(subset=[vol_col])
    if valid.empty:
        return {
            "Universe": label,
            "Mean volatility": np.nan,
            "Median volatility": np.nan,
            "25th percentile": np.nan,
            "75th percentile": np.nan,
            "Number of assets": 0,
            "Highest-vol ticker": "",
            "Lowest-vol ticker": "",
            "Regime": "Unavailable",
        }
    highest = valid.loc[valid[vol_col].idxmax()]
    lowest = valid.loc[valid[vol_col].idxmin()]
    mean_vol = float(valid[vol_col].mean())
    return {
        "Universe": label,
        "Mean volatility": mean_vol,
        "Median volatility": float(valid[vol_col].median()),
        "25th percentile": float(valid[vol_col].quantile(0.25)),
        "75th percentile": float(valid[vol_col].quantile(0.75)),
        "Number of assets": int(valid["Ticker"].nunique()),
        "Highest-vol ticker": str(highest.get("Ticker", "")),
        "Lowest-vol ticker": str(lowest.get("Ticker", "")),
        "Regime": classify_vol_regime(mean_vol, valid[vol_col]),
    }


def compute_universe_vol_summary(metrics_by_universe: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for universe, df in metrics_by_universe.items():
        vol_col = get_volatility_column(df)
        if vol_col is None:
            continue
        rows.append(_summary_row(df, UNIVERSE_LABELS.get(universe, universe), vol_col))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Mean volatility", ascending=False, na_position="last")


def _group_vol_summary(df: pd.DataFrame, group_col: str, vol_col: str) -> pd.DataFrame:
    clean = _clean_metrics_df(df, vol_col)
    if group_col not in clean.columns or vol_col not in clean.columns:
        return pd.DataFrame()
    clean[group_col] = clean[group_col].fillna("").astype(str).str.strip()
    clean = clean[(clean[group_col] != "") & clean[vol_col].notna()].copy()
    if clean.empty:
        return pd.DataFrame()

    rows = []
    for group, group_df in clean.groupby(group_col, dropna=False):
        highest = group_df.loc[group_df[vol_col].idxmax()]
        lowest = group_df.loc[group_df[vol_col].idxmin()]
        mean_vol = float(group_df[vol_col].mean())
        rows.append(
            {
                group_col: group,
                "Mean volatility": mean_vol,
                "Median volatility": float(group_df[vol_col].median()),
                "Volatility percentile": percentile_rank(mean_vol, clean[vol_col]),
                "Number of constituents": int(group_df["Ticker"].nunique()),
                "Highest-vol ticker": str(highest.get("Ticker", "")),
                "Lowest-vol ticker": str(lowest.get("Ticker", "")),
                "Regime": classify_vol_regime(mean_vol, clean[vol_col]),
            }
        )
    return pd.DataFrame(rows).sort_values("Mean volatility", ascending=False, na_position="last")


def compute_sector_vol_summary(df: pd.DataFrame, vol_col: str) -> pd.DataFrame:
    return _group_vol_summary(df, "Sector", vol_col)


def compute_industry_vol_summary(df: pd.DataFrame, vol_col: str) -> pd.DataFrame:
    return _group_vol_summary(df, "SubIndustry", vol_col)


def compute_vol_breadth(df: pd.DataFrame, group_col: str, vol_col: str) -> pd.DataFrame:
    clean = _clean_metrics_df(df, vol_col)
    if group_col not in clean.columns or vol_col not in clean.columns:
        return pd.DataFrame()
    clean[group_col] = clean[group_col].fillna("").astype(str).str.strip()
    clean = clean[(clean[group_col] != "") & clean[vol_col].notna()].copy()
    if clean.empty:
        return pd.DataFrame()

    rows = []
    for group, group_df in clean.groupby(group_col, dropna=False):
        median = group_df[vol_col].median()
        q75 = group_df[vol_col].quantile(0.75)
        above_median = group_df[vol_col] > median
        above_q75 = group_df[vol_col] > q75
        rows.append(
            {
                group_col: group,
                "Assets": int(group_df["Ticker"].nunique()),
                "Median volatility": float(median),
                "75th percentile": float(q75),
                "% above median": float(above_median.mean() * 100),
                "% above 75th percentile": float(above_q75.mean() * 100),
                "Elevated count": int(above_q75.sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("% above 75th percentile", ascending=False)


def build_vol_leaderboard(
    df: pd.DataFrame,
    vol_col: str,
    group_col: str | None = None,
    *,
    n: int = 10,
    ascending: bool = False,
) -> pd.DataFrame:
    clean = _clean_metrics_df(df, vol_col)
    cols = ["Ticker", "Name", "Universe", "Sector", "SubIndustry", vol_col]
    if group_col and group_col in clean.columns:
        rows = []
        for group, group_df in clean.dropna(subset=[vol_col]).groupby(group_col, dropna=False):
            ranked = group_df.sort_values(vol_col, ascending=ascending).head(int(n)).copy()
            ranked[group_col] = group
            rows.append(ranked)
        if not rows:
            return pd.DataFrame()
        out = pd.concat(rows, ignore_index=True)
        cols = [group_col] + [c for c in cols if c != group_col]
    else:
        out = clean.dropna(subset=[vol_col]).sort_values(vol_col, ascending=ascending).head(int(n)).copy()
    if out.empty:
        return pd.DataFrame()
    out.insert(0, "Rank", out.groupby(group_col).cumcount() + 1 if group_col and group_col in out.columns else range(1, len(out) + 1))
    return out[[c for c in ["Rank"] + cols if c in out.columns]].rename(columns={vol_col: "Volatility"})


def compute_vol_change_from_history(history: pd.DataFrame, metric: str, window: int = 20) -> pd.DataFrame:
    if history is None or history.empty or "metric" not in history.columns:
        return pd.DataFrame()
    h = history[history["metric"].astype(str).eq(metric)].copy()
    if h.empty:
        return pd.DataFrame()
    h["as_of_date"] = pd.to_datetime(h["as_of_date"], errors="coerce")
    h["metric_value"] = pd.to_numeric(h["metric_value"], errors="coerce")
    h = h.dropna(subset=["as_of_date", "metric_value", "ticker"]).sort_values(["ticker", "as_of_date"])
    if h.empty:
        return pd.DataFrame()
    latest_date = h["as_of_date"].max()
    previous_cutoff = latest_date - pd.Timedelta(days=int(window) * 2)
    recent = h[h["as_of_date"] >= previous_cutoff].copy()
    latest = recent.sort_values("as_of_date").groupby("ticker").tail(1)
    previous = recent.sort_values("as_of_date").groupby("ticker").head(1)
    merged = latest.merge(
        previous[["ticker", "metric_value"]].rename(columns={"metric_value": "Previous volatility"}),
        on="ticker",
        how="left",
    )
    merged = merged.rename(
        columns={
            "ticker": "Ticker",
            "name": "Name",
            "sector": "Sector",
            "subindustry": "SubIndustry",
            "metric_value": "Latest volatility",
        }
    )
    merged["Volatility change"] = merged["Latest volatility"] - merged["Previous volatility"]
    cols = ["Ticker", "Name", "Sector", "SubIndustry", "Latest volatility", "Previous volatility", "Volatility change"]
    return merged[cols].dropna(subset=["Volatility change"]).sort_values("Volatility change", ascending=False)


def _normalize_vix_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["VIX"])
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]
    if "VIX" not in out.columns:
        close_col = "Close" if "Close" in out.columns else out.columns[0]
        out = out[[close_col]].rename(columns={close_col: "VIX"})
    else:
        out = out[["VIX"]]
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[~out.index.isna()].sort_index()
    out["VIX"] = pd.to_numeric(out["VIX"], errors="coerce")
    return out.dropna(subset=["VIX"])


def _vix_chart_frame(vix: pd.DataFrame) -> pd.DataFrame:
    out = _normalize_vix_frame(vix).reset_index()
    date_col = out.columns[0]
    out = out.rename(columns={date_col: "Date"})
    return out[["Date", "VIX"]].dropna(subset=["Date", "VIX"])


@st.cache_data(ttl=30 * 60)
def maybe_load_vix_data(period: str = "1y", project_root: str | None = None) -> tuple[pd.DataFrame, str | None]:
    artifact_errors: list[str] = []
    try:
        from loader import get_market_data

        stored = _normalize_vix_frame(get_market_data("vix"))
        if not stored.empty:
            return stored, None
    except Exception as exc:
        artifact_errors.append(f"manifest artifact unavailable ({type(exc).__name__}: {exc})")

    if project_root:
        try:
            local_path = Path(project_root) / "mktme-data" / "market" / "vix.parquet"
            if local_path.exists():
                stored = _normalize_vix_frame(pd.read_parquet(local_path))
                if not stored.empty:
                    return stored, None
        except Exception as exc:
            artifact_errors.append(f"local artifact unavailable ({type(exc).__name__}: {exc})")

    try:
        from fetch_data import download_vix_data

        data = download_vix_data(period=period)
        out = _normalize_vix_frame(data)
        if out.empty:
            detail = "; ".join(artifact_errors + ["No live VIX rows returned."])
            return pd.DataFrame(), detail
        return out, None
    except Exception as exc:
        detail = "; ".join(artifact_errors + [f"live fallback failed ({type(exc).__name__}: {exc})"])
        return pd.DataFrame(), detail


def _load_universe_frame(
    universe: str,
    *,
    asof_ts: pd.Timestamp,
    lookback: int,
    get_prices_for_universe: Callable[[str], pd.DataFrame],
    get_sp500_constituents: Callable[..., pd.DataFrame],
    get_universe: Callable[..., pd.DataFrame],
    compute_all_metrics: Callable[[pd.DataFrame, int], pd.DataFrame],
) -> tuple[pd.DataFrame, str | None]:
    try:
        tickers_df = get_sp500_constituents(force_refresh=False) if universe == "sp500" else get_universe(universe, force_refresh=False)
        prices_full = get_prices_for_universe(universe)
        prices_full = prices_full.copy()
        prices_full.index = pd.to_datetime(prices_full.index, errors="coerce")
        prices_full = prices_full.loc[~prices_full.index.isna()].sort_index()
        symbols = tickers_df["Ticker"].astype(str).tolist()
        keep = [c for c in prices_full.columns if c in symbols]
        prices = prices_full.loc[:asof_ts, keep]
        if prices.empty:
            return pd.DataFrame(), "No cached price rows are available for this universe."
        metrics = compute_all_metrics(prices, lookback=int(lookback))
        if metrics.empty:
            return pd.DataFrame(), "No metrics could be computed for this universe."
        metadata_cols = [c for c in ["Name", "Sector", "SubIndustry"] if c in tickers_df.columns]
        metrics = metrics.join(tickers_df.set_index("Ticker")[metadata_cols], how="left")
        metrics = metrics.reset_index().rename(columns={"index": "Ticker"})
        metrics["Universe"] = universe
        metrics["Universe Label"] = UNIVERSE_LABELS.get(universe, universe)
        return metrics, None
    except Exception as exc:
        return pd.DataFrame(), f"{type(exc).__name__}: {exc}"


def _load_history_local(project_root: str, universe: str, lookback: int) -> pd.DataFrame:
    data_repo = Path(os.environ.get("MKTME_DATA_REPO", os.path.join(project_root, "mktme-data")))
    path = data_repo / "leaderboards" / universe / f"lookback_{int(lookback)}.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _metric_card(label: str, value, help_text: str | None = None) -> None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        st.metric(label, "n/a", help=help_text)
    elif isinstance(value, (float, np.floating)):
        st.metric(label, f"{value:.2f}", help=help_text)
    else:
        st.metric(label, str(value), help=help_text)


def _round_numeric(df: pd.DataFrame, digits: int = 2) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].round(digits)
    return out


def _bar_chart(df: pd.DataFrame, x: str, y: str, color: str | None = None, title: str = "") -> None:
    if df.empty or x not in df.columns or y not in df.columns:
        return
    import plotly.express as px

    fig = px.bar(df, x=x, y=y, color=color if color in df.columns else None, title=title)
    fig.update_layout(xaxis_title="", yaxis_title=y)
    st.plotly_chart(fig, use_container_width=True)


def _scatter_chart(df: pd.DataFrame, x: str, y: str, color: str | None, title: str) -> None:
    if df.empty or x not in df.columns or y not in df.columns:
        st.info("Selected scatter metrics are not available for this view.")
        return
    import plotly.express as px

    plot_df = df.dropna(subset=[x, y]).copy()
    if plot_df.empty:
        st.info("No rows remain after filtering missing scatter values.")
        return
    hover_cols = [c for c in ["Ticker", "Name", "Universe Label", "Sector", "SubIndustry"] if c in plot_df.columns]
    fig = px.scatter(
        plot_df,
        x=x,
        y=y,
        color=color if color in plot_df.columns else None,
        hover_name="Ticker" if "Ticker" in plot_df.columns else None,
        hover_data=hover_cols,
        title=title,
    )
    fig.update_traces(marker=dict(size=8, opacity=0.82))
    st.plotly_chart(fig, use_container_width=True)


def render_volatility_dashboard(
    *,
    project_root: str,
    get_prices_for_universe: Callable[[str], pd.DataFrame],
    get_sp500_constituents: Callable[..., pd.DataFrame],
    get_universe: Callable[..., pd.DataFrame],
    compute_all_metrics: Callable[[pd.DataFrame, int], pd.DataFrame],
) -> None:
    st.header("Volatility Dashboard")
    st.caption("Realized volatility across universes, sectors, industries, and assets.")

    with st.sidebar:
        st.header("Volatility Settings")
        lookback = st.number_input("Lookback (trading days)", min_value=60, max_value=2520, value=252, step=21, key="vol_lookback")
        asof = st.date_input("As-of date", value=pd.Timestamp.utcnow().date(), key="vol_asof")
        history_window = st.selectbox("Change window", [20, 60, 120, 252], index=0, key="vol_history_window")

    asof_ts = pd.to_datetime(asof)
    metrics_by_universe: dict[str, pd.DataFrame] = {}
    load_errors: dict[str, str] = {}
    with st.spinner("Loading volatility data across universes..."):
        for universe in UNIVERSES:
            frame, error = _load_universe_frame(
                universe,
                asof_ts=asof_ts,
                lookback=int(lookback),
                get_prices_for_universe=get_prices_for_universe,
                get_sp500_constituents=get_sp500_constituents,
                get_universe=get_universe,
                compute_all_metrics=compute_all_metrics,
            )
            if error:
                load_errors[universe] = error
            metrics_by_universe[universe] = frame

    for universe, error in load_errors.items():
        st.caption(f"{UNIVERSE_LABELS.get(universe, universe)} load note: {error}")

    frames = [df for df in metrics_by_universe.values() if isinstance(df, pd.DataFrame) and not df.empty]
    if not frames:
        st.error("No volatility data is available for the selected date and lookback.")
        return
    all_metrics = pd.concat(frames, ignore_index=True)
    vol_col = get_volatility_column(all_metrics)
    if vol_col is None:
        st.error("No supported volatility column is available. Expected Realized_Vol_20D or Daily Volatility (Std).")
        return

    if vol_col != PRIMARY_VOL_COL:
        st.warning("Using Daily Volatility (Std) because Realized_Vol_20D is unavailable in the current metrics frame.")

    universe_summary = compute_universe_vol_summary(metrics_by_universe)
    sp500 = metrics_by_universe.get("sp500", pd.DataFrame())

    tabs = st.tabs(
        [
            "Market Regime",
            "Universe Vol",
            "Sector / Industry Vol",
            "Breadth",
            "Leaderboards",
            "Scatter Explorer",
            "VIX / Implied Vol",
        ]
    )

    with tabs[0]:
        st.subheader("Market Regime Overview")
        valid = all_metrics.dropna(subset=[vol_col])
        avg_by_universe = universe_summary.copy()
        highest_universe = avg_by_universe.iloc[0]["Universe"] if not avg_by_universe.empty else None
        lowest_universe = avg_by_universe.iloc[-1]["Universe"] if not avg_by_universe.empty else None
        broad_mean = float(valid[vol_col].mean()) if not valid.empty else np.nan
        broad_regime = classify_vol_regime(broad_mean, valid[vol_col])

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            _metric_card("Average realized vol", broad_mean)
        with c2:
            _metric_card("Highest-vol universe", highest_universe)
        with c3:
            _metric_card("Lowest-vol universe", lowest_universe)
        with c4:
            _metric_card("Broad regime", broad_regime)

        vix, vix_error = maybe_load_vix_data(project_root=project_root)
        vc1, vc2 = st.columns(2)
        with vc1:
            if not vix.empty:
                _metric_card("VIX", float(vix["VIX"].iloc[-1]))
            else:
                _metric_card("VIX", None, "Optional VIX load failed or returned no data.")
        with vc2:
            if not vix.empty and len(vix["VIX"].dropna()) > 20:
                _metric_card("VIX percentile", percentile_rank(float(vix["VIX"].iloc[-1]), vix["VIX"]))
            else:
                _metric_card("VIX percentile", None, "Requires historical VIX observations.")
        if vix_error:
            st.caption(f"VIX integration note: {vix_error}")

        _bar_chart(avg_by_universe, "Universe", "Mean volatility", "Regime", "Average realized volatility by universe")

    with tabs[1]:
        st.subheader("Average Volatility by Universe")
        if universe_summary.empty:
            st.info("No universe-level volatility summary is available.")
        else:
            _bar_chart(universe_summary, "Universe", "Mean volatility", "Regime", "Universe realized volatility")
            st.dataframe(_round_numeric(universe_summary), use_container_width=True, hide_index=True)

    with tabs[2]:
        st.subheader("Sector Volatility Dashboard")
        sp500_vol_col = get_volatility_column(sp500)
        if sp500.empty or sp500_vol_col is None:
            st.info("S&P 500 sector volatility is unavailable for the selected settings.")
        else:
            sector_summary = compute_sector_vol_summary(sp500, sp500_vol_col)
            if sector_summary.empty:
                st.info("Sector metadata is unavailable.")
            else:
                _bar_chart(sector_summary, "Sector", "Mean volatility", "Regime", "S&P 500 sector volatility")
                st.dataframe(_round_numeric(sector_summary), use_container_width=True, hide_index=True)

            st.subheader("Industry / Sub-Industry Volatility Dashboard")
            industry_summary = compute_industry_vol_summary(sp500, sp500_vol_col)
            if industry_summary.empty:
                st.info("Sub-industry metadata is unavailable. TODO: add richer industry/sub-industry coverage for non-S&P universes.")
            else:
                st.dataframe(_round_numeric(industry_summary), use_container_width=True, hide_index=True)

    with tabs[3]:
        st.subheader("Volatility Breadth")
        breadth = compute_vol_breadth(all_metrics, "Universe Label", vol_col)
        if breadth.empty:
            st.info("Breadth metrics are unavailable.")
        else:
            _bar_chart(breadth, "Universe Label", "% above 75th percentile", None, "Elevated volatility breadth")
            st.dataframe(_round_numeric(breadth), use_container_width=True, hide_index=True)
        st.caption("Time-series breadth against each asset's own longer-term average requires compatible historical realized-volatility snapshots.")

    with tabs[4]:
        st.subheader("Volatility Leaderboards")
        top_n = st.selectbox("Rows per leaderboard", [5, 10, 20, 50], index=1, key="vol_lb_n")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Highest volatility assets")
            st.dataframe(_round_numeric(build_vol_leaderboard(all_metrics, vol_col, n=int(top_n))), use_container_width=True, hide_index=True)
        with c2:
            st.markdown("#### Lowest volatility assets")
            st.dataframe(_round_numeric(build_vol_leaderboard(all_metrics, vol_col, n=int(top_n), ascending=True)), use_container_width=True, hide_index=True)

        st.markdown("#### Highest volatility by universe")
        st.dataframe(_round_numeric(build_vol_leaderboard(all_metrics, vol_col, "Universe Label", n=3)), use_container_width=True, hide_index=True)

        if not sp500.empty and get_volatility_column(sp500):
            st.markdown("#### Highest volatility by S&P 500 sector")
            st.dataframe(_round_numeric(build_vol_leaderboard(sp500, get_volatility_column(sp500), "Sector", n=3)), use_container_width=True, hide_index=True)

        st.markdown("#### Largest increase in volatility")
        if vol_col == PRIMARY_VOL_COL:
            st.info("Requires historical Realized_Vol_20D snapshots. Current published history contains Daily Volatility (Std), not Realized_Vol_20D.")
        else:
            history_frames = [_load_history_local(project_root, u, int(lookback)) for u in UNIVERSES]
            history_frames = [h for h in history_frames if not h.empty]
            history = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()
            change = compute_vol_change_from_history(history, vol_col, int(history_window))
            if change.empty:
                st.info("No compatible volatility history is available.")
            else:
                st.dataframe(_round_numeric(change.head(int(top_n))), use_container_width=True, hide_index=True)

    with tabs[5]:
        st.subheader("Volatility Scatter Explorer")
        presets = {
            "Volatility vs Sharpe": ("Annualized Sharpe", vol_col),
            "Volatility vs CAGR": ("CAGR", vol_col),
            "Volatility vs Max Drawdown": ("Max Drawdown", vol_col),
            "Volatility vs RSI": ("RSI_14", vol_col),
            "Volatility vs Sortino": ("Sortino Ratio", vol_col),
        }
        preset = st.selectbox("Preset", list(presets.keys()), key="vol_scatter_preset")
        x_metric, y_metric = presets[preset]
        color = st.selectbox("Color by", ["Universe Label", "Sector", "SubIndustry"], key="vol_scatter_color")
        _scatter_chart(all_metrics, x_metric, y_metric, color, preset)
        st.caption("Use this view to separate efficient risk from high-volatility, low-quality setups.")

    with tabs[6]:
        st.subheader("VIX / Implied Vol")
        vix, vix_error = maybe_load_vix_data(project_root=project_root)
        if vix.empty:
            st.info("VIX is optional and unavailable right now. The realized-volatility dashboard remains fully usable.")
            if vix_error:
                st.caption(f"Load detail: {vix_error}")
        else:
            import plotly.express as px

            latest_vix = float(vix["VIX"].iloc[-1])
            _metric_card("Latest VIX", latest_vix)
            _metric_card("VIX percentile", percentile_rank(latest_vix, vix["VIX"]))
            chart_df = _vix_chart_frame(vix)
            if chart_df.empty:
                st.info("VIX history loaded, but no plottable date/value rows are available.")
            else:
                fig = px.line(chart_df, x="Date", y="VIX", title="VIX history")
                st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Roadmap hooks: VIX futures term structure, correlation plus volatility, average pairwise correlation, "
            "volatility treemap, market-cap weighted volatility, volatility factor backtesting, and shock detection."
        )
