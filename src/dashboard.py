
import os

import sys
sys.path.append(os.path.dirname(__file__))  # ensure src/ is on sys.path

try:
    from dotenv import load_dotenv
    load_dotenv()  # load from .env at project root
except ImportError:
    pass

import streamlit as st
import pandas as pd
import json
import requests
import hashlib
import re
from datetime import date
from pathlib import Path

from fetch_data import get_sp500_constituents, get_meta
from compute_metrics import compute_all_metrics
from metrics_registry import METRICS
from leaderboards import (
    available_leaderboard_metrics,
    canonical_metric,
    current_leaderboard_with_change,
    leaderboard_table,
    load_snapshots_from_url,
    movement_summary,
    rank_time_series,
    snapshot_path as leaderboard_snapshot_path,
)
from metric_docs import METRIC_META, get_pair_guide
from universes import get_universe
from fx_universe import FX_UNIVERSE_ALIASES
from local_pipeline import VALID_SOURCES, local_pipeline_ui_enabled, run_pipeline
# --- AI helpers (chat + context) -----------------------------------------------
from ai_context import (
    get_openai_client,
    rate_limit_ok,
    daily_cap_ok,
    build_view_context_text,
    ask_ai_about_view,
)
from health_status import build_dashboard_health
from status_store import list_recent_reports
from forecast_lab.ui import render_forecast_lab
from volatility_dashboard import render_volatility_dashboard
from macro_engine.visualization.electricity_dashboard import render_electricity_dashboard
# ------------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DEFAULT_DATA_BASE_URL = "https://raw.githubusercontent.com/marshdoggo/mktme-data/main"
UNIVERSE_OPTIONS = [
    "sp500",
    "nasdaq100",
    "dow30",
    "fx_current",
    "fx",
    "fx_g10_majors",
    "fx_em_exotic",
    "fx_scandi",
    "fx_cnh",
]
UNIVERSE_LABELS = {
    "sp500": "S&P 500",
    "nasdaq100": "Nasdaq 100",
    "dow30": "Dow 30",
    "fx_current": "Current FX universe",
    "fx": "FOREX.com Core FX",
    "fx_g10_majors": "G10 majors only",
    "fx_em_exotic": "EM / exotic FX",
    "fx_scandi": "Scandi FX",
    "fx_cnh": "CNH FX",
}


def _is_fx_universe(universe: str) -> bool:
    return str(universe).strip().lower().replace("-", "_") in FX_UNIVERSE_ALIASES


# --- Manifest/Parquet loader (prefer external loader.py; fallback to inline) ----
try:
    from loader import get_prices_for_universe  # provided by src/loader.py
except Exception:
    # Inline fallback so the app still runs if loader.py is missing
    import json
    import requests
    from io import BytesIO
    from urllib.parse import urlencode

    MANIFEST_URL = os.environ.get(
        "MKTME_MANIFEST_URL",
        "https://raw.githubusercontent.com/marshdoggo/mktme-data/main/manifest.json"
    )

    @st.cache_data(ttl=60*60)
    def _load_manifest():
        r = requests.get(MANIFEST_URL, timeout=30)
        r.raise_for_status()
        return json.loads(r.text)

    @st.cache_data(ttl=60*60)
    def _load_parquet_http(url: str, cache_key: str) -> pd.DataFrame:
        if cache_key:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}{urlencode({'v': cache_key})}"
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        df = pd.read_parquet(BytesIO(r.content), engine="pyarrow")
        # Ensure UTC datetime index
        df.index = pd.to_datetime(df.index, utc=True)
        return df

    def get_prices_for_universe(universe: str) -> pd.DataFrame:
        m = _load_manifest()
        if "universes" not in m or universe not in m["universes"]:
            raise RuntimeError(f"Universe '{universe}' not present in manifest at {MANIFEST_URL}")
        url = m["universes"][universe]["parquet_url"]
        cache_key = str(m.get("generated_at") or "")
        return _load_parquet_http(url, cache_key=cache_key)
# -------------------------------------------------------------------------------


# --- Helper: safe date bounds for Streamlit date_input -------------------------
def _safe_date_bounds(prices):
    """
    Returns (min_day, max_day) as Python date objects that are safe for Streamlit date_input,
    even if the prices DataFrame is empty or has invalid datetime indices.
    """
    try:
        if hasattr(prices, "index") and len(prices.index) > 0:
            idx = pd.to_datetime(prices.index, errors="coerce").dropna()
            if len(idx) > 0:
                return idx.min().date(), idx.max().date()
    except Exception:
        pass
    today = pd.Timestamp.today(tz="UTC").normalize()
    return (today - pd.Timedelta(days=365)).date(), today.date()


def parse_tickers(text: str) -> list[str]:
    if not text:
        return []
    cleaned = []
    for raw in re.split(r"[\s,]+", text):
        ticker = raw.strip().upper().lstrip("$").replace("/", "").replace("-", "")
        if ticker:
            cleaned.append(ticker)
    return list(dict.fromkeys(cleaned))


@st.cache_data(ttl=5*60)
def _load_leaderboard_history(universe: str, lookback: int, cache_key: str) -> pd.DataFrame:
    local_data_repo = os.environ.get("MKTME_DATA_REPO", os.path.join(PROJECT_ROOT, "mktme-data"))
    local_path = leaderboard_snapshot_path(Path(local_data_repo), universe, int(lookback))
    if local_path.exists():
        return pd.read_parquet(local_path)

    base_url = os.environ.get("MKTME_LEADERBOARDS_BASE_URL", DEFAULT_DATA_BASE_URL).rstrip("/")
    url = f"{base_url}/leaderboards/{universe}/lookback_{int(lookback)}.parquet"
    return load_snapshots_from_url(url)

st.set_page_config(page_title='S&P 500 Metric Explorer', layout='wide')

st.title('📈 Market Metric Explorer')

if "highlight_tickers" not in st.session_state:
    st.session_state["highlight_tickers"] = ""
if st.session_state.get("pending_highlight_tickers"):
    st.session_state["highlight_tickers"] = st.session_state.pop("pending_highlight_tickers")

health = build_dashboard_health()
health_state = health.get("state")
health_title = health.get("title", "Pipeline health status unavailable.")
health_summary = health.get("summary", "")
health_details = health.get("details") or []

if health_state == "error":
    st.error(f"{health_title} {health_summary}".strip())
elif health_state == "warning":
    st.warning(f"{health_title} {health_summary}".strip())
else:
    st.success(f"{health_title} {health_summary}".strip())

with st.expander("Pipeline health details", expanded=False):
    latest_run = health.get("latest_run")
    if latest_run:
        st.caption(
            f"Latest repo-local run: {latest_run.get('status')}  •  "
            f"started {latest_run.get('started_at')}  •  "
            f"finished {latest_run.get('finished_at') or '(in progress)'}"
        )
        if latest_run.get("error_summary"):
            st.caption(f"Last error: {latest_run.get('error_summary')}")
    else:
        st.caption("No repo-local pipeline runs are recorded yet.")

    for detail in health_details:
        st.caption(detail)

    recent_runs = health.get("recent_runs") or []
    if recent_runs:
        recent_df = pd.DataFrame(recent_runs)
        st.dataframe(recent_df, use_container_width=True, hide_index=True)

with st.sidebar:
    app_section = st.radio(
        "App section",
        ["Metric Explorer", "Volatility Dashboard", "Macro"],
        index=0,
        key="app_section",
    )

if app_section == "Volatility Dashboard":
    render_volatility_dashboard(
        project_root=PROJECT_ROOT,
        get_prices_for_universe=get_prices_for_universe,
        get_sp500_constituents=get_sp500_constituents,
        get_universe=get_universe,
        compute_all_metrics=compute_all_metrics,
    )
    st.stop()

if app_section == "Macro":
    macro_view = st.sidebar.selectbox("Macro dataset", ["Electricity Production"], index=0)
    if macro_view == "Electricity Production":
        render_electricity_dashboard()
    st.stop()

with st.sidebar:
    st.header('Settings')
    if st.button("🔄 Refresh cached data", help="Clears Streamlit cached downloads (manifest, parquet, reports). Useful when a new publisher run landed but the app is still showing older data."):
        st.cache_data.clear()
        st.toast("Cache cleared — rerunning…")
        st.rerun()
    universe = st.selectbox(
        'Universe',
        UNIVERSE_OPTIONS,
        index=0,
        format_func=lambda value: UNIVERSE_LABELS.get(value, value),
        help='Pick the asset universe to analyze'
    )
    lookback = st.number_input('Lookback (trading days)', min_value=60, max_value=2520, value=252, step=21)
    if local_pipeline_ui_enabled():
        with st.expander("Local data pipeline", expanded=False):
            st.caption("Runs only from this checkout. It is hidden on Streamlit Cloud unless explicitly enabled.")
            pipeline_universes = st.multiselect(
                "Universes",
                UNIVERSE_OPTIONS,
                default=["sp500", "nasdaq100", "dow30", "fx"],
                format_func=lambda value: UNIVERSE_LABELS.get(value, value),
                key="local_pipeline_universes",
            )
            pipeline_source = st.selectbox(
                "Data source",
                VALID_SOURCES,
                index=VALID_SOURCES.index(os.getenv("MKTME_EQUITY_SOURCE", "auto"))
                if os.getenv("MKTME_EQUITY_SOURCE", "auto") in VALID_SOURCES
                else 0,
                key="local_pipeline_source",
            )
            pipeline_force = st.checkbox(
                "Force live refresh",
                value=True,
                help="Bypasses cached fetches and asks the provider layer for fresh data.",
                key="local_pipeline_force",
            )
            pipeline_existing = st.checkbox(
                "Reuse existing parquet only",
                value=False,
                help="Rebuilds reports and manifest from local parquet files without calling data vendors.",
                key="local_pipeline_existing",
            )
            if st.button("Run local pipeline", type="primary", key="run_local_pipeline"):
                with st.spinner("Running local data pipeline..."):
                    try:
                        result = run_pipeline(
                            universes=pipeline_universes,
                            lookback=int(lookback),
                            equity_source=pipeline_source,
                            force_refresh=bool(pipeline_force),
                            use_existing_parquet=bool(pipeline_existing),
                        )
                    except Exception as exc:
                        st.error(f"Pipeline launch failed: {type(exc).__name__}: {exc}")
                    else:
                        if result.returncode == 0:
                            st.success("Pipeline completed. Clearing cached app data...")
                            st.code(" ".join(result.command), language="bash")
                            if result.stdout.strip():
                                st.text_area("Pipeline output", result.stdout[-6000:], height=220)
                            st.cache_data.clear()
                            st.rerun()
                        else:
                            st.error(f"Pipeline failed with exit code {result.returncode}.")
                            st.code(" ".join(result.command), language="bash")
                            combined_output = "\n\n".join(
                                part for part in [result.stdout.strip(), result.stderr.strip()] if part
                            )
                            if combined_output:
                                st.text_area("Pipeline output", combined_output[-10000:], height=260)
    interactive = st.checkbox('Interactive chart (Plotly)', value=True)
    query = st.text_input(
        'Highlight tickers (comma-separated)',
        key='highlight_tickers',
        placeholder='AAPL, MSFT, NVDA  •  or  EURUSD, USDJPY',
    )
    st.caption('Data by Wikipedia (equities) and local publisher → Parquet (served via GitHub Raw).')

if universe == 'sp500':
    tickers_df = get_sp500_constituents(force_refresh=False)
else:
    tickers_df = get_universe(universe, force_refresh=False)
meta = get_meta() if not _is_fx_universe(universe) else {}

# --- Load cached prices from manifest/Parquet -------------------------------
with st.spinner('Loading cached prices…'):
    try:
        prices_full = get_prices_for_universe(universe)
    except Exception as e:
        st.error(f"Failed to load dataset for universe '{universe}'. {e}")
        st.stop()

# Subset to the universe symbols from the side bar to keep columns aligned
symbols = tickers_df['Ticker'].astype(str).tolist()
keep = [c for c in prices_full.columns if c in symbols]
prices = prices_full[keep].copy()

if prices is None or getattr(prices, 'empty', True):
    st.error("No cached price data available for this universe. "
             "Ensure the data repo published a Parquet file referenced by the manifest.")
    st.stop()
# ---------------------------------------------------------------------------

min_day, max_day = _safe_date_bounds(prices)

with st.sidebar:
    asof = st.date_input('As-of date', value=max_day, min_value=min_day, max_value=max_day)
    compare = st.checkbox('Compare to…', value=False)
    asof2 = None
    if compare:
        asof2 = st.date_input('As-of date (B)', value=max_day, min_value=min_day, max_value=max_day)

# Optional time sliders (control As-of dates)
with st.sidebar:
    use_slider = st.checkbox('Use time slider (A)', value=False)
    if use_slider:
        step = st.radio('Slider step (A)', ['Daily', 'Weekly', 'Month-end'], index=1, horizontal=True)
        # Build step options from trading days in the cache
        uA = pd.DatetimeIndex(prices.index.normalize().unique())
        if step == 'Daily':
            options_dt_A = list(uA)
        elif step == 'Weekly':
            options_dt_A = list(pd.Series(uA).groupby(uA.to_period('W-FRI')).tail(1))
        else:  # Month-end
            options_dt_A = list(pd.Series(uA).groupby(uA.to_period('M')).tail(1))

        options_A = [d.date() for d in sorted(options_dt_A)]
        # Default slider value is nearest option <= current as-of
        asof_pref_A = pd.to_datetime(asof)
        default_A = max([d for d in options_A if d <= asof_pref_A.date()], default=options_A[0])
        key_A = 'time_slider_A'
        if key_A not in st.session_state:
            st.session_state[key_A] = default_A
        # Play button auto-advances one step
        cols_play_A = st.columns([1,2])
        with cols_play_A[0]:
            if st.button('▶ Play', key='play_A'):
                idx = options_A.index(st.session_state[key_A]) if st.session_state[key_A] in options_A else 0
                if idx < len(options_A) - 1:
                    st.session_state[key_A] = options_A[idx + 1]
        with cols_play_A[1]:
            st.caption('Advance one step')
        sel_A = st.select_slider('Time slider (A)', options=options_A, value=st.session_state[key_A], key=key_A)
        asof = sel_A

    # Secondary slider for View B when comparing
    if compare:
        use_slider_B = st.checkbox('Use time slider (B)', value=False)
        if use_slider_B:
            stepB = st.radio('Slider step (B)', ['Daily', 'Weekly', 'Month-end'], index=1, horizontal=True)
            uB = pd.DatetimeIndex(prices.index.normalize().unique())
            if stepB == 'Daily':
                options_dt_B = list(uB)
            elif stepB == 'Weekly':
                options_dt_B = list(pd.Series(uB).groupby(uB.to_period('W-FRI')).tail(1))
            else:
                options_dt_B = list(pd.Series(uB).groupby(uB.to_period('M')).tail(1))

            options_B = [d.date() for d in sorted(options_dt_B)]
            asof_pref_B = pd.to_datetime(asof2 if asof2 is not None else max_day)
            default_B = max([d for d in options_B if d <= asof_pref_B.date()], default=options_B[0])
            key_B = 'time_slider_B'
            if key_B not in st.session_state:
                st.session_state[key_B] = default_B
            cols_play_B = st.columns([1,2])
            with cols_play_B[0]:
                if st.button('▶ Play', key='play_B'):
                    idx = options_B.index(st.session_state[key_B]) if st.session_state[key_B] in options_B else 0
                    if idx < len(options_B) - 1:
                        st.session_state[key_B] = options_B[idx + 1]
            with cols_play_B[1]:
                st.caption('Advance one step')
            sel_B = st.select_slider('Time slider (B)', options=options_B, value=st.session_state[key_B], key=key_B)
            asof2 = sel_B

    # Sync axes toggle for side-by-side compare
    sync_axes = st.checkbox('Sync axes ranges (A & B)', value=(True if compare else False))

asof_ts = pd.to_datetime(asof)
asof2_ts = pd.to_datetime(asof2) if asof2 is not None else None

if compare and asof2_ts is not None:
    st.write(
        f"Cache window: {min_day} → {max_day}  •  As-of A: {asof_ts.date()}  •  As-of B: {asof2_ts.date()}  •  Lookback: {lookback} trading days  •  Tickers: {prices.shape[1]}"
    )
else:
    st.write(
        f"Cache window: {min_day} → {max_day}  •  As-of: {asof_ts.date()}  •  Lookback: {lookback} trading days  •  Tickers: {prices.shape[1]}"
    )

# Helper to compute a metrics frame for a given as-of timestamp
def build_metrics_df(prices_full: pd.DataFrame, asof_ts_local: pd.Timestamp) -> pd.DataFrame:
    p = prices_full.loc[:asof_ts_local]
    df = compute_all_metrics(p, lookback=lookback)
    df = df.join(tickers_df.set_index('Ticker')[['Name','Sector','SubIndustry']], how='left')
    df = df.reset_index().rename(columns={'index': 'Ticker'})
    df['Ticker_upper'] = df['Ticker'].astype(str).str.upper()
    df['IsHighlighted'] = df['Ticker_upper'].isin(highlight) if highlight else False
    return df


highlight = parse_tickers(query)

# Build A (and B if needed) before selectors so every numeric technical metric
# is available in the manual dropdowns.
metrics_df_A = build_metrics_df(prices, asof_ts)
metrics_df = metrics_df_A  # keep old name for downstream references when not comparing
metrics_df_B = build_metrics_df(prices, asof2_ts) if compare and asof2_ts is not None else None

def _numeric_metric_options(df: pd.DataFrame) -> list[str]:
    skip = {"Ticker_upper", "IsHighlighted"}
    cols = []
    for col in df.columns:
        if col in skip:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    preferred = [
        "Annualized Sharpe",
        "Sortino Ratio",
        "CAGR",
        "Max Drawdown",
        "Daily Volatility (Std)",
        "RSI_14",
        "Dip_Buy_Score",
        "Bear_Breakdown_Score",
        "Momentum_Continuation_Score",
    ]
    return [m for m in preferred if m in cols] + [m for m in cols if m not in preferred]


def _first_available(candidates: list[str], options: list[str], fallback: str | None = None) -> str:
    for candidate in candidates:
        if candidate in options:
            return candidate
    return fallback if fallback in options else (options[0] if options else "")


numeric_metrics = _numeric_metric_options(metrics_df_A)
default_x = _first_available(["Sortino Ratio", "Annualized Sharpe"], numeric_metrics)
default_y = _first_available(["Annualized Sharpe", "Sortino Ratio"], numeric_metrics, default_x)
default_size = "(None)"
default_color = "Sector" if not _is_fx_universe(universe) else "SubIndustry"


def _option_index(options: list[str], value: str) -> int:
    return options.index(value) if value in options else 0


def _plotly_marker_sizes(df: pd.DataFrame, size_col: str | None) -> pd.Series | None:
    if not size_col or size_col not in df.columns:
        return None
    values = pd.to_numeric(df.loc[:, size_col], errors='coerce')
    if 'Drawdown' in size_col:
        values = values.abs()
    ranks = values.rank(pct=True)
    return ranks.fillna(0.0).mul(38).add(7)


def _interactive_scatter(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    color_col: str | None,
    size_col: str | None,
):
    import plotly.express as px

    plot_df = df.loc[:, ~df.columns.duplicated()].copy()
    marker_size_col = None
    if size_col and size_col in plot_df.columns:
        marker_size_col = '_MarkerSize'
        plot_df[marker_size_col] = _plotly_marker_sizes(plot_df, size_col)

    hover_data = {}
    for col in ['Ticker', 'Name', 'Sector', 'SubIndustry']:
        if col in plot_df.columns:
            hover_data[col] = True
    if size_col and size_col in plot_df.columns:
        hover_data[size_col] = True
    if color_col and color_col in plot_df.columns and color_col not in hover_data:
        hover_data[color_col] = True
    if marker_size_col:
        hover_data[marker_size_col] = False

    fig = px.scatter(
        plot_df,
        x=x,
        y=y,
        color=color_col if color_col in plot_df.columns else None,
        size=marker_size_col,
        size_max=45,
        hover_name='Ticker' if 'Ticker' in plot_df.columns else None,
        hover_data=hover_data if hover_data else None,
        title=title,
    )
    if marker_size_col:
        fig.update_traces(marker=dict(opacity=0.85))
    else:
        fig.update_traces(marker=dict(size=8, opacity=0.85))
    fig.update_layout(legend_title_text=color_col or '')
    return fig


with st.sidebar:
    st.markdown('---')
    st.header('Chart View')
    enable_color_metric = st.checkbox(
        'Enable color metric',
        value=False,
        key='main_enable_color_metric',
        help='Adds an optional color dimension to the main scatter.'
    )
    enable_size_metric = st.checkbox(
        'Enable dot size metric',
        value=False,
        key='main_enable_size_metric',
        help='Adds an optional percentile-normalized marker size dimension.'
    )
    axis_focus_pct = st.slider(
        'Axis focus (%)',
        min_value=60,
        max_value=100,
        value=100,
        step=5,
        help='100% uses the full data range. Lower values trim the most extreme outliers from the plotted axis range so clustered points look more spread out.'
    )
    axis_padding_pct = st.slider(
        'Axis padding (%)',
        min_value=0,
        max_value=20,
        value=5,
        step=1,
        help='Adds breathing room around the visible plotted range.'
    )
    log_x = st.checkbox(
        'Log scale X axis',
        value=False,
        help='Uses a logarithmic X axis when all plotted X values are positive.'
    )
    log_y = st.checkbox(
        'Log scale Y axis',
        value=False,
        help='Uses a logarithmic Y axis when all plotted Y values are positive.'
    )

# Gather user metric choices (applies to both A and B)
col1, col2 = st.columns([1,1])
with col1:
    x_metric = st.selectbox('X axis', numeric_metrics, index=_option_index(numeric_metrics, default_x))
with col2:
    y_metric = st.selectbox('Y axis', numeric_metrics, index=_option_index(numeric_metrics, default_y))

color_by = None
size_metric = None
if enable_color_metric and enable_size_metric:
    col3, col4 = st.columns([1,1])
elif enable_color_metric or enable_size_metric:
    col3 = st.container()
    col4 = None
else:
    col3 = col4 = None

if enable_color_metric:
    color_options = ['Sector','SubIndustry','(None)'] if not _is_fx_universe(universe) else ['SubIndustry','(None)']
    color_options = [c for c in color_options if c == '(None)' or c in metrics_df_A.columns] + [
        m for m in numeric_metrics if m not in {'Sector', 'SubIndustry'}
    ]
    with col3:
        color_by = st.selectbox('Color by', color_options, index=_option_index(color_options, default_color))
        if color_by == '(None)':
            color_by = None

if enable_size_metric:
    size_options = ['(None)'] + numeric_metrics
    with (col4 if col4 is not None else col3):
        size_metric = st.selectbox('Dot size', size_options, index=_option_index(size_options, default_size))
        if size_metric == '(None)':
            size_metric = None

if not enable_color_metric and not enable_size_metric:
    st.caption('Showing a two-axis scatter. Enable color or dot size in the sidebar for additional dimensions.')

# --- Metric docs & quadrant guide -------------------------------------------------
with st.expander('🧭 How to read this view (metric docs + quadrant guide)', expanded=False):
    guide = get_pair_guide(x_metric, y_metric)
    x_meta = guide["x_meta"]; y_meta = guide["y_meta"]

    cols_help = st.columns(2)
    with cols_help[0]:
        st.markdown(f"**X — {x_meta['metric']}**  ")
        st.caption(f"role: {x_meta.get('role','–')}")
        st.write(x_meta.get('desc',''))
        ins = x_meta.get('insight','')
        if ins:
            st.caption(ins)
    with cols_help[1]:
        st.markdown(f"**Y — {y_meta['metric']}**  ")
        st.caption(f"role: {y_meta.get('role','–')}")
        st.write(y_meta.get('desc',''))
        ins = y_meta.get('insight','')
        if ins:
            st.caption(ins)

    st.markdown('---')
    st.markdown('**Quadrant key (x → horizontal, y → vertical)**  \\nTR = top-right • TL = top-left • BR = bottom-right • BL = bottom-left')

    Q = guide["quadrants"]
    st.markdown(
        f"""
- **BR** (high x • low y): {Q['BR']}
- **TR** (high x • high y): {Q['TR']}
- **BL** (low x • low y): {Q['BL']}
- **TL** (low x • high y): {Q['TL']}
"""
    )

def _compute_axis_ranges(df_local: pd.DataFrame, x: str, y: str, focus_pct: int, padding_pct: int):
    sx = pd.to_numeric(df_local[x], errors='coerce').dropna()
    sy = pd.to_numeric(df_local[y], errors='coerce').dropna()
    if sx.empty or sy.empty:
        return None

    def _bounds(series: pd.Series):
        if focus_pct >= 100:
            lo = float(series.min())
            hi = float(series.max())
        else:
            trim_each_side = max(0.0, (100 - focus_pct) / 200.0)
            lo = float(series.quantile(trim_each_side))
            hi = float(series.quantile(1.0 - trim_each_side))

        if hi == lo:
            base = abs(lo) if lo != 0 else 1.0
            pad = base * max(padding_pct / 100.0, 0.05)
        else:
            pad = (hi - lo) * (padding_pct / 100.0)
        return [lo - pad, hi + pad]

    return {
        'x': _bounds(sx),
        'y': _bounds(sy),
    }

def render_view(title_label: str, metrics_df_local: pd.DataFrame, axis_ranges=None):
    # Plot
    cols_needed = [x_metric, y_metric, 'Sector', 'SubIndustry', 'Ticker', 'Name', 'Ticker_upper', 'IsHighlighted']
    for optional_col in [color_by, size_metric]:
        if optional_col:
            cols_needed.append(optional_col)
    present = list(dict.fromkeys(c for c in cols_needed if c in metrics_df_local.columns))
    missing_xy = [c for c in [x_metric, y_metric] if c not in metrics_df_local.columns]

    if missing_xy:
        st.warning(f"Selected metric(s) not available in this view: {', '.join(missing_xy)}. "
                   "Try another metric or refresh the cache.")
        return

    plot_df_local = metrics_df_local[present].dropna(subset=[x_metric, y_metric])
    st.subheader(title_label)
    can_log_x = bool((plot_df_local[x_metric] > 0).all()) if not plot_df_local.empty else False
    can_log_y = bool((plot_df_local[y_metric] > 0).all()) if not plot_df_local.empty else False
    if interactive:
        import plotly.express as px
        fig = _interactive_scatter(
            plot_df_local,
            x=x_metric,
            y=y_metric,
            title=f"{y_metric} vs {x_metric}",
            color_col=color_by,
            size_col=size_metric,
        )
        # Apply synced axis ranges if provided
        if axis_ranges is not None:
            if 'x' in axis_ranges and axis_ranges['x'] is not None:
                fig.update_xaxes(range=axis_ranges['x'])
            if 'y' in axis_ranges and axis_ranges['y'] is not None:
                fig.update_yaxes(range=axis_ranges['y'])
        if log_x and can_log_x:
            fig.update_xaxes(type='log')
        elif log_x and not can_log_x:
            st.caption(f'Log scale unavailable for X because {x_metric} includes zero or negative values in this view.')
        if log_y and can_log_y:
            fig.update_yaxes(type='log')
        elif log_y and not can_log_y:
            st.caption(f'Log scale unavailable for Y because {y_metric} includes zero or negative values in this view.')
        if highlight:
            fig.update_traces(marker=dict(opacity=0.35))
            hi = plot_df_local[plot_df_local['IsHighlighted']]
            if not hi.empty:
                fig_hi = px.scatter(
                    hi,
                    x=x_metric,
                    y=y_metric,
                    color=color_by if color_by in hi.columns else None,
                    hover_name='Ticker',
                    hover_data={c: True for c in ['Name','Sector','SubIndustry', size_metric] if c and c in hi.columns},
                )
                fig_hi.update_traces(marker=dict(size=12, symbol='diamond', opacity=0.95, line=dict(width=1)))
                for tr in fig_hi.data:
                    fig.add_trace(tr)
        # --- Quadrant crosshairs & labels (median-based) ---------------------
        try:
            x_med = float(plot_df_local[x_metric].median())
            y_med = float(plot_df_local[y_metric].median())
            fig.add_vline(x=x_med, line_dash="dot", opacity=0.35)
            fig.add_hline(y=y_med, line_dash="dot", opacity=0.35)

            # Pull quadrant texts
            guide = get_pair_guide(x_metric, y_metric)
            Q = guide["quadrants"]

            # Corner positions with small padding
            x_min, x_max = plot_df_local[x_metric].min(), plot_df_local[x_metric].max()
            y_min, y_max = plot_df_local[y_metric].min(), plot_df_local[y_metric].max()
            pad_x = (x_max - x_min) * 0.03 if x_max > x_min else 0.01
            pad_y = (y_max - y_min) * 0.03 if y_max > y_min else 0.01

            # A=TR, B=TL, C=BR, D=BL
            pos = {
                "TR": (x_max - pad_x, y_max - pad_y, "A"),
                "TL": (x_min + pad_x, y_max - pad_y, "B"),
                "BR": (x_max - pad_x, y_min + pad_y, "C"),
                "BL": (x_min + pad_x, y_min + pad_y, "D"),
            }
            for key in ["TR","TL","BR","BL"]:
                x_, y_, tag = pos[key]
                fig.add_annotation(
                    x=x_, y=y_, text=f"<b>{tag}</b>", showarrow=False, opacity=0.7,
                    xanchor="right" if key in ("TR","BR") else "left",
                    yanchor="top" if key in ("TR","TL") else "bottom",
                )
        except Exception:
            pass
        # ---------------------------------------------------------------------
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Quadrant key (A=TR, B=TL, C=BR, D=BL)"):
            guide = get_pair_guide(x_metric, y_metric)
            Q = guide["quadrants"]
            st.markdown(
                f"""
- **A (TR)** top-right — {Q['TR']}
- **B (TL)** top-left — {Q['TL']}
- **C (BR)** bottom-right — {Q['BR']}
- **D (BL)** bottom-left — {Q['BL']}
"""
            )
    else:
        try:
            from plot_metrics import scatter_xy
            fig, ax = scatter_xy(
                plot_df_local,
                x=x_metric,
                y=y_metric,
                color_by=(color_by or ''),
                title=f"{y_metric} vs {x_metric}",
                size_metric=size_metric,
            )
            if axis_ranges is not None:
                if 'x' in axis_ranges and axis_ranges['x'] is not None:
                    ax.set_xlim(axis_ranges['x'])
                if 'y' in axis_ranges and axis_ranges['y'] is not None:
                    ax.set_ylim(axis_ranges['y'])
            if log_x and can_log_x:
                ax.set_xscale('log')
            elif log_x and not can_log_x:
                st.caption(f'Log scale unavailable for X because {x_metric} includes zero or negative values in this view.')
            if log_y and can_log_y:
                ax.set_yscale('log')
            elif log_y and not can_log_y:
                st.caption(f'Log scale unavailable for Y because {y_metric} includes zero or negative values in this view.')
            st.pyplot(fig)
        except Exception as exc:
            st.caption(f"Static chart unavailable ({type(exc).__name__}); using the interactive Plotly chart instead.")
            import plotly.express as px
            fig = _interactive_scatter(
                plot_df_local,
                x=x_metric,
                y=y_metric,
                title=f"{y_metric} vs {x_metric}",
                color_col=color_by,
                size_col=size_metric,
            )
            if axis_ranges is not None:
                if 'x' in axis_ranges and axis_ranges['x'] is not None:
                    fig.update_xaxes(range=axis_ranges['x'])
                if 'y' in axis_ranges and axis_ranges['y'] is not None:
                    fig.update_yaxes(range=axis_ranges['y'])
            if log_x and can_log_x:
                fig.update_xaxes(type='log')
            elif log_x and not can_log_x:
                st.caption(f'Log scale unavailable for X because {x_metric} includes zero or negative values in this view.')
            if log_y and can_log_y:
                fig.update_yaxes(type='log')
            elif log_y and not can_log_y:
                st.caption(f'Log scale unavailable for Y because {y_metric} includes zero or negative values in this view.')
            st.plotly_chart(fig, use_container_width=True)

    # Selected vs Peers (for this view)
    st.markdown('#### Selected vs Peers')
    if highlight:
        sel = metrics_df_local[metrics_df_local['Ticker_upper'].isin(highlight)]
        if not sel.empty:
            def pct_rank(series: pd.Series, v: float) -> float:
                s = series.dropna().sort_values()
                if len(s) == 0 or pd.isna(v):
                    return float('nan')
                return float((s < v).mean())

            rows = []
            for _, r in sel.iterrows():
                sect = r['Sector'] if 'Sector' in r and pd.notna(r['Sector']) else None
                sector_df = metrics_df_local[metrics_df_local['Sector'] == sect] if sect else metrics_df_local

                rows.append({
                    'Ticker': r['Ticker'],
                    'Name': r.get('Name', ''),
                    f'{x_metric}': r[x_metric],
                    f'{y_metric}': r[y_metric],
                    f'{x_metric} pctile (Sector)': pct_rank(sector_df[x_metric], r[x_metric]),
                    f'{y_metric} pctile (Sector)': pct_rank(sector_df[y_metric], r[y_metric]),
                    f'{x_metric} pctile (Index)': pct_rank(metrics_df_local[x_metric], r[x_metric]),
                    f'{y_metric} pctile (Index)': pct_rank(metrics_df_local[y_metric], r[y_metric]),
                    f'{x_metric} sector median': sector_df[x_metric].median(),
                    f'{y_metric} sector median': sector_df[y_metric].median(),
                    f'{x_metric} index median': metrics_df_local[x_metric].median(),
                    f'{y_metric} index median': metrics_df_local[y_metric].median(),
                })

            st.dataframe(pd.DataFrame(rows))
        else:
            st.caption('Type tickers above to see a peer comparison panel.')
    else:
        st.caption('Type tickers above to see a peer comparison panel.')

    # Data table for this view
    st.markdown('#### Data Table')
    table_sort_metric = y_metric
    table_local = metrics_df_local.sort_values(by=table_sort_metric, ascending=False)
    if highlight:
        table_local = table_local.sort_values(by=['IsHighlighted', table_sort_metric], ascending=[False, False])
    st.dataframe(table_local.drop(columns=['Ticker_upper'], errors='ignore'))

# Compute shared axis ranges if requested
axis_ranges_shared = None
if compare and metrics_df_B is not None and 'sync_axes' in locals() and sync_axes:
    axisA = _compute_axis_ranges(metrics_df_A, x_metric, y_metric, axis_focus_pct, axis_padding_pct)
    axisB = _compute_axis_ranges(metrics_df_B, x_metric, y_metric, axis_focus_pct, axis_padding_pct)
    if axisA and axisB:
        axis_ranges_shared = {
            'x': [
                float(min(axisA['x'][0], axisB['x'][0])),
                float(max(axisA['x'][1], axisB['x'][1])),
            ],
            'y': [
                float(min(axisA['y'][0], axisB['y'][0])),
                float(max(axisA['y'][1], axisB['y'][1])),
            ],
        }

axis_ranges_A = axis_ranges_shared or _compute_axis_ranges(metrics_df_A, x_metric, y_metric, axis_focus_pct, axis_padding_pct)
axis_ranges_B = axis_ranges_shared or (_compute_axis_ranges(metrics_df_B, x_metric, y_metric, axis_focus_pct, axis_padding_pct) if metrics_df_B is not None else None)

st.markdown('---')

if compare and metrics_df_B is not None:
    colA, colB = st.columns(2)
    with colA:
        render_view(f"View A — {asof_ts.date()} :: {y_metric} vs {x_metric}", metrics_df_A, axis_ranges=axis_ranges_A)
    with colB:
        render_view(f"View B — {asof2_ts.date()} :: {y_metric} vs {x_metric}", metrics_df_B, axis_ranges=axis_ranges_B)
else:
    render_view(f"{y_metric} vs {x_metric}", metrics_df_A, axis_ranges=axis_ranges_A)

# --- Metric leaderboards --------------------------------------------------------
st.markdown('---')
st.subheader('Metric Leaderboards')

available_metrics = available_leaderboard_metrics(metrics_df_A)
if not available_metrics:
    st.info("No supported numeric leaderboard metrics are available for this view.")
else:
    lb_tab, move_tab, rank_tab, forecast_tab = st.tabs(["Leaderboards", "Movement", "Rank Over Time", "Forecast Lab"])

    with lb_tab:
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            lb_options = available_metrics + [
                "Worst Max Drawdown",
                "Most Defensive Max Drawdown",
            ] if "Max Drawdown" in available_metrics else available_metrics
            lb_metric_choice = st.selectbox("Metric", lb_options, key="lb_metric")
        with c2:
            lb_top_n = st.selectbox("Top N", [5, 10, 20], index=1, key="lb_top_n")
        with c3:
            lb_direction = st.selectbox("Direction", ["Highest / strongest", "Lowest / weakest"], key="lb_direction")

        if lb_metric_choice == "Worst Max Drawdown":
            lb_metric = "Max Drawdown"
            lb_dir = "weakest"
        elif lb_metric_choice == "Most Defensive Max Drawdown":
            lb_metric = "Max Drawdown"
            lb_dir = "strongest"
        else:
            lb_metric = canonical_metric(lb_metric_choice)
            lb_dir = "strongest" if lb_direction == "Highest / strongest" else "weakest"

        companion_defaults = [
            m for m in ["Annualized Sharpe", "Sortino Ratio", "CAGR", "Max Drawdown"]
            if m in available_metrics and m != lb_metric
        ][:3]
        companion_metrics = st.multiselect(
            "Show companion metrics",
            [m for m in available_metrics if m != lb_metric],
            default=companion_defaults,
            key="lb_companion_metrics",
            help="Rank by the selected metric, while displaying these additional metric columns for context.",
        )

        lb_df = leaderboard_table(
            metrics_df_A,
            metric=lb_metric,
            n=int(lb_top_n),
            direction=lb_dir,
            display_metrics=companion_metrics,
        )
        if lb_df.empty:
            st.info(f"No leaderboard rows are available for {lb_metric_choice}.")
        else:
            st.dataframe(lb_df, use_container_width=True, hide_index=True)

    history = pd.DataFrame()
    history_error = None
    try:
        history = _load_leaderboard_history(universe, int(lookback), cache_key=str(max_day))
    except Exception as exc:
        history_error = exc

    with move_tab:
        if history.empty:
            st.info(
                "No leaderboard history is published for this universe/lookback yet. "
                "Run the backfill once, then daily appends will keep this section moving."
            )
            if history_error is not None:
                st.caption(f"History load detail: {type(history_error).__name__}: {history_error}")
        else:
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                mv_metric = st.selectbox("Movement metric", available_metrics, key="movement_metric")
            with c2:
                mv_top_n = st.selectbox("Top N", [5, 10, 20], index=1, key="movement_top_n")
            with c3:
                mv_window = st.selectbox("History window", [20, 60, 120, 252], index=0, key="movement_window")
            summary = movement_summary(history, mv_metric, top_n=int(mv_top_n), window=int(mv_window))
            if not summary.get("available"):
                st.info(f"No movement history is available for {mv_metric}.")
            else:
                def _movement_display(df: pd.DataFrame, metric_name: str, rank_label: str = "Rank") -> pd.DataFrame:
                    if not isinstance(df, pd.DataFrame) or df.empty:
                        return pd.DataFrame()
                    out = df.copy()
                    rename_map = {
                        "ticker": "Ticker",
                        "name": "Name",
                        "sector": "Sector",
                        "subindustry": "SubIndustry",
                        "rank": rank_label,
                        "previous_rank": "Previous Rank",
                        "rank_change": "Rank Change",
                        "metric_value": metric_name,
                    }
                    out = out.rename(columns=rename_map)
                    cols = [
                        rank_label,
                        "Previous Rank",
                        "Rank Change",
                        "Ticker",
                        "Name",
                        "Sector",
                        "SubIndustry",
                        metric_name,
                    ]
                    keep = [c for c in cols if c in out.columns]
                    out = out[keep].copy()
                    for c in [rank_label, "Previous Rank", "Rank Change"]:
                        if c in out.columns:
                            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
                    if metric_name in out.columns:
                        out[metric_name] = pd.to_numeric(out[metric_name], errors="coerce").round(6)
                    for c in ["Ticker", "Name", "Sector", "SubIndustry"]:
                        if c in out.columns:
                            out[c] = out[c].fillna("").astype(str)
                    return out

                def _persistence_display(df: pd.DataFrame) -> pd.DataFrame:
                    if not isinstance(df, pd.DataFrame) or df.empty:
                        return pd.DataFrame()
                    out = df.rename(
                        columns={
                            "ticker": "Ticker",
                            "name": "Name",
                            "sector": "Sector",
                            "days_in_top_5": "Days Top 5",
                            "days_in_top_10": "Days Top 10",
                            "latest_rank": "Latest Rank",
                        }
                    )
                    cols = ["Ticker", "Name", "Sector", "Days Top 5", "Days Top 10", "Latest Rank"]
                    out = out[[c for c in cols if c in out.columns]].copy()
                    for c in ["Days Top 5", "Days Top 10", "Latest Rank"]:
                        if c in out.columns:
                            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
                    for c in ["Ticker", "Name", "Sector"]:
                        if c in out.columns:
                            out[c] = out[c].fillna("").astype(str)
                    return out

                current = summary.get("current", pd.DataFrame())
                if isinstance(current, pd.DataFrame) and not current.empty:
                    show = current.rename(
                        columns={
                            "ticker": "Ticker",
                            "name": "Name",
                            "sector": "Sector",
                            "metric_value": mv_metric,
                            "current_rank": "Current Rank",
                            "previous_rank": "Previous Rank",
                            "rank_change": "Rank Change",
                        }
                    )
                    st.markdown("#### Current leaderboard with rank change")
                    st.dataframe(
                        show[[c for c in ["Current Rank", "Previous Rank", "Rank Change", "Ticker", "Name", "Sector", mv_metric] if c in show.columns]],
                        use_container_width=True,
                        hide_index=True,
                    )

                c_new, c_drop = st.columns(2)
                with c_new:
                    st.markdown("#### New entrants")
                    df_new = _movement_display(summary.get("new_entrants", pd.DataFrame()), mv_metric)
                    if isinstance(df_new, pd.DataFrame) and not df_new.empty:
                        st.dataframe(df_new, use_container_width=True, hide_index=True)
                    else:
                        st.caption("None in this window.")
                with c_drop:
                    st.markdown("#### Dropouts")
                    df_drop = _movement_display(summary.get("dropouts", pd.DataFrame()), mv_metric)
                    if isinstance(df_drop, pd.DataFrame) and not df_drop.empty:
                        st.dataframe(df_drop, use_container_width=True, hide_index=True)
                    else:
                        st.caption("None in this window.")

                c_persist, c_climb, c_fall = st.columns(3)
                with c_persist:
                    st.markdown("#### Persistence")
                    df_p = _persistence_display(summary.get("persistence", pd.DataFrame()))
                    if isinstance(df_p, pd.DataFrame) and not df_p.empty:
                        st.dataframe(df_p, use_container_width=True, hide_index=True)
                    else:
                        st.caption("No persistence stats yet.")
                with c_climb:
                    st.markdown("#### Biggest climbers")
                    df_c = _movement_display(summary.get("climbers", pd.DataFrame()), mv_metric)
                    if isinstance(df_c, pd.DataFrame) and not df_c.empty:
                        st.dataframe(df_c, use_container_width=True, hide_index=True)
                    else:
                        st.caption("Needs at least two snapshots.")
                with c_fall:
                    st.markdown("#### Biggest fallers")
                    df_f = _movement_display(summary.get("fallers", pd.DataFrame()), mv_metric)
                    if isinstance(df_f, pd.DataFrame) and not df_f.empty:
                        st.dataframe(df_f, use_container_width=True, hide_index=True)
                    else:
                        st.caption("Needs at least two snapshots.")

    with rank_tab:
        if history.empty:
            st.info("No leaderboard history exists yet for rank-over-time charts.")
        else:
            c1, c2 = st.columns([2, 1])
            with c1:
                rt_metric = st.selectbox("Rank metric", available_metrics, key="rank_metric")
            with c2:
                rt_window = st.selectbox("Rank window", [20, 60, 120, 252], index=0, key="rank_window")
            default_tickers = (
                current_leaderboard_with_change(history, rt_metric, 5)
                .get("ticker", pd.Series(dtype=str))
                .dropna()
                .astype(str)
                .tolist()
            )
            options = sorted(metrics_df_A["Ticker"].dropna().astype(str).str.upper().unique().tolist())
            selected_tickers = st.multiselect(
                "Tickers",
                options,
                default=[t for t in default_tickers if t in options][:5],
                key="rank_tickers",
            )
            ts_df = rank_time_series(history, rt_metric, selected_tickers, window=int(rt_window))
            if ts_df.empty:
                st.info("Choose tickers with available history to render the rank time series.")
            else:
                import plotly.express as px

                ts_plot = ts_df.copy()
                ts_plot["as_of_date"] = pd.to_datetime(ts_plot["as_of_date"], errors="coerce")
                unique_dates = ts_plot["as_of_date"].dropna().drop_duplicates().sort_values()
                gaps = unique_dates.diff().dt.days.dropna()
                big_gaps = gaps[gaps > 7]
                if not big_gaps.empty:
                    st.warning(
                        "This leaderboard history has missing snapshot dates. "
                        "Lines are intentionally broken across gaps longer than one week."
                    )
                    gap_breaks = []
                    for ticker, group in ts_plot.dropna(subset=["as_of_date"]).sort_values("as_of_date").groupby("ticker"):
                        ticker_gaps = group["as_of_date"].diff().dt.days
                        for idx in ticker_gaps[ticker_gaps > 7].index:
                            previous_date = group.loc[group.index[group.index.get_loc(idx) - 1], "as_of_date"]
                            gap_breaks.append(
                                {
                                    "as_of_date": previous_date + pd.Timedelta(days=1),
                                    "rank": pd.NA,
                                    "ticker": ticker,
                                }
                            )
                    if gap_breaks:
                        ts_plot = pd.concat([ts_plot, pd.DataFrame(gap_breaks)], ignore_index=True)
                fig = px.line(ts_plot.sort_values(["ticker", "as_of_date"]), x="as_of_date", y="rank", color="ticker", markers=True)
                fig.update_yaxes(autorange="reversed", title="Rank")
                if len(unique_dates) == 1:
                    only_date = unique_dates.iloc[0]
                    fig.update_xaxes(range=[only_date - pd.Timedelta(days=1), only_date + pd.Timedelta(days=1)])
                fig.update_xaxes(title="As-of date")
                st.plotly_chart(fig, use_container_width=True)

    with forecast_tab:
        render_forecast_lab(
            default_universe=universe,
            lookback=int(lookback),
            max_day=max_day,
            load_history=_load_leaderboard_history,
        )

# --- Technical signals ----------------------------------------------------------
st.markdown('---')
st.subheader('Technical Signals')
st.caption(
    "Exploratory technical indicators only; scores are universe-relative and are not financial advice."
)

tech_preset = st.selectbox(
    'Technical signal preset',
    ['Manual', 'Dip Buy Candidates', 'Bear / Short Breakdown', 'Momentum Continuation'],
    index=0,
    key='technical_signal_preset',
    help='Applies defaults to the technical signal chart without changing the main explorer.'
)

tech_default_x = _first_available(["RSI_14"], numeric_metrics, default_x)
tech_default_y = _first_available(["Annualized Sharpe", "Sortino Ratio"], numeric_metrics, default_y)
tech_default_size = _first_available(["Drawdown_3M_High"], numeric_metrics, "(None)")
tech_default_color = _first_available(["Distance_200DMA", "Return_60D"], numeric_metrics, "(None)")

if tech_preset == 'Bear / Short Breakdown':
    tech_default_x = _first_available(["Return_60D"], numeric_metrics, tech_default_x)
    tech_default_y = _first_available(["Distance_200DMA"], numeric_metrics, tech_default_y)
    tech_default_size = _first_available(["ATR_14_Pct", "Realized_Vol_20D"], numeric_metrics, "(None)")
    tech_default_color = _first_available(["RSI_14"], numeric_metrics, "(None)")
elif tech_preset == 'Momentum Continuation':
    tech_default_x = _first_available(["Return_20D"], numeric_metrics, tech_default_x)
    tech_default_y = _first_available(["Return_60D"], numeric_metrics, tech_default_y)
    tech_default_size = _first_available(["Volume_Ratio_20D_60D"], numeric_metrics, "(None)")
    tech_default_color = _first_available(["RSI_14"], numeric_metrics, "(None)")

tc1, tc2 = st.columns([1, 1])
with tc1:
    tech_x_metric = st.selectbox(
        'Technical X axis',
        numeric_metrics,
        index=_option_index(numeric_metrics, tech_default_x),
        key='technical_x_metric',
    )
with tc2:
    tech_y_metric = st.selectbox(
        'Technical Y axis',
        numeric_metrics,
        index=_option_index(numeric_metrics, tech_default_y),
        key='technical_y_metric',
    )

td1, td2 = st.columns([1, 1])
with td1:
    tech_enable_color = st.checkbox('Enable technical color metric', value=False, key='technical_enable_color')
with td2:
    tech_enable_size = st.checkbox('Enable technical dot size metric', value=False, key='technical_enable_size')

tech_color_metric = None
tech_size_metric = None
if tech_enable_color and tech_enable_size:
    td3, td4 = st.columns([1, 1])
elif tech_enable_color or tech_enable_size:
    td3 = st.container()
    td4 = None
else:
    td3 = td4 = None

if tech_enable_color:
    tech_color_options = ['(None)'] + numeric_metrics
    with td3:
        tech_color_metric = st.selectbox(
            'Technical color by',
            tech_color_options,
            index=_option_index(tech_color_options, tech_default_color),
            key='technical_color_metric',
        )
        if tech_color_metric == '(None)':
            tech_color_metric = None

if tech_enable_size:
    tech_size_options = ['(None)'] + numeric_metrics
    with (td4 if td4 is not None else td3):
        tech_size_metric = st.selectbox(
            'Technical dot size',
            tech_size_options,
            index=_option_index(tech_size_options, tech_default_size),
            key='technical_size_metric',
        )
        if tech_size_metric == '(None)':
            tech_size_metric = None

if not tech_enable_color and not tech_enable_size:
    st.caption('Showing a two-axis technical scatter. Enable color or dot size for additional dimensions.')

tech_cols_needed = [tech_x_metric, tech_y_metric, 'Ticker', 'Name', 'Sector', 'SubIndustry', 'Ticker_upper', 'IsHighlighted']
for optional_col in [tech_color_metric, tech_size_metric]:
    if optional_col:
        tech_cols_needed.append(optional_col)
tech_present = [c for c in tech_cols_needed if c in metrics_df_A.columns]
tech_missing_xy = [c for c in [tech_x_metric, tech_y_metric] if c not in metrics_df_A.columns]
if tech_missing_xy:
    st.warning(f"Selected technical metric(s) not available: {', '.join(tech_missing_xy)}.")
else:
    tech_plot_df = metrics_df_A[tech_present].dropna(subset=[tech_x_metric, tech_y_metric])
    tech_fig = _interactive_scatter(
        tech_plot_df,
        x=tech_x_metric,
        y=tech_y_metric,
        title=f"{tech_y_metric} vs {tech_x_metric}",
        color_col=tech_color_metric,
        size_col=tech_size_metric,
    )
    if highlight:
        import plotly.express as px
        tech_fig.update_traces(marker=dict(opacity=0.35))
        tech_hi = tech_plot_df[tech_plot_df['IsHighlighted']]
        if not tech_hi.empty:
            tech_fig_hi = px.scatter(
                tech_hi,
                x=tech_x_metric,
                y=tech_y_metric,
                color=tech_color_metric if tech_color_metric in tech_hi.columns else None,
                hover_name='Ticker',
                hover_data={c: True for c in ['Name', 'Sector', 'SubIndustry', tech_size_metric] if c and c in tech_hi.columns},
            )
            tech_fig_hi.update_traces(marker=dict(size=12, symbol='diamond', opacity=0.95, line=dict(width=1)))
            for tr in tech_fig_hi.data:
                tech_fig.add_trace(tr)
    st.plotly_chart(tech_fig, use_container_width=True)

st.markdown('#### Technical Signal Leaderboards')


def _technical_rank_table(
    df: pd.DataFrame,
    sort_col: str,
    *,
    ascending: bool,
    support_cols: list[str],
    n: int = 10,
) -> pd.DataFrame:
    if sort_col not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out[sort_col] = pd.to_numeric(out[sort_col], errors='coerce')
    out = out.dropna(subset=[sort_col])
    if out.empty:
        return pd.DataFrame()
    out = out.sort_values(sort_col, ascending=ascending).head(int(n)).copy()
    out.insert(0, "Rank", range(1, len(out) + 1))
    cols = ["Rank", "Ticker", "Name", "Sector", sort_col] + [
        c for c in support_cols if c in out.columns and c != sort_col
    ]
    return out[[c for c in cols if c in out.columns]]


technical_leaderboards = [
    (
        "Top Dip Buy Score",
        "Dip_Buy_Score",
        False,
        ["RSI_14", "Drawdown_3M_High", "Distance_200DMA", "Annualized Sharpe", "Sortino Ratio"],
    ),
    (
        "Top Bear Breakdown Score",
        "Bear_Breakdown_Score",
        False,
        ["Return_20D", "Return_60D", "Distance_50DMA", "Distance_200DMA", "RSI_14", "ATR_14_Pct"],
    ),
    (
        "Top Momentum Continuation Score",
        "Momentum_Continuation_Score",
        False,
        ["Return_20D", "Return_60D", "Distance_50DMA", "SMA_50_200_Spread", "Volume_Ratio_20D_60D"],
    ),
    ("Most Oversold RSI", "RSI_14", True, ["Return_20D", "Drawdown_3M_High", "Distance_200DMA"]),
    ("Largest Drawdown From 3M High", "Drawdown_3M_High", True, ["RSI_14", "Return_60D", "Distance_200DMA"]),
    ("Highest Volume Surge", "Volume_Ratio_20D_60D", False, ["Return_20D", "Return_60D", "RSI_14"]),
    ("Most Extended Above 50DMA", "Distance_50DMA", False, ["RSI_14", "Return_20D", "Return_60D"]),
    ("Most Extended Below 50DMA", "Distance_50DMA", True, ["RSI_14", "Return_20D", "Return_60D"]),
]

for left, right in zip(technical_leaderboards[0::2], technical_leaderboards[1::2]):
    c_left, c_right = st.columns(2)
    for col_obj, config in [(c_left, left), (c_right, right)]:
        title, metric, ascending, support = config
        with col_obj:
            st.markdown(f"#### {title}")
            ranked = _technical_rank_table(metrics_df_A, metric, ascending=ascending, support_cols=support)
            if ranked.empty:
                st.caption(f"No rows available for {metric}.")
            else:
                board_tickers = ranked["Ticker"].dropna().astype(str).str.upper().tolist()
                if st.button("Highlight on chart", key=f"technical_highlight_{metric}_{title}"):
                    st.session_state["pending_highlight_tickers"] = ", ".join(board_tickers)
                    st.rerun()
                st.dataframe(ranked, use_container_width=True, hide_index=True)

# --- Delta panel: compare highlighted tickers A → B ---
if compare and metrics_df_B is not None and highlight:
    # Select highlighted rows from each snapshot
    selA = metrics_df_A[metrics_df_A['Ticker_upper'].isin(highlight)].copy()
    selB = metrics_df_B[metrics_df_B['Ticker_upper'].isin(highlight)].copy()

    if not selA.empty and not selB.empty:
        xA, yA = f'{x_metric} (A)', f'{y_metric} (A)'
        xB, yB = f'{x_metric} (B)', f'{y_metric} (B)'

        A_df = selA[['Ticker','Name','Sector','Ticker_upper', x_metric, y_metric]].rename(columns={x_metric: xA, y_metric: yA})
        B_df = selB[['Ticker_upper', x_metric, y_metric]].rename(columns={x_metric: xB, y_metric: yB})
        merged = pd.merge(A_df, B_df, on='Ticker_upper', how='inner')

        # Deltas and % deltas (safe divide)
        merged[f'Δ {x_metric}'] = merged[xB] - merged[xA]
        merged[f'Δ {y_metric}'] = merged[yB] - merged[yA]
        merged[f'Δ% {x_metric}'] = (merged[f'Δ {x_metric}'] / merged[xA].replace(0, pd.NA)) * 100
        merged[f'Δ% {y_metric}'] = (merged[f'Δ {y_metric}'] / merged[yA].replace(0, pd.NA)) * 100

        cols_order = [
            'Ticker','Name','Sector',
            xA, xB, f'Δ {x_metric}', f'Δ% {x_metric}',
            yA, yB, f'Δ {y_metric}', f'Δ% {y_metric}'
        ]
        out = merged[cols_order].copy()

        # Nicely rounded numbers
        num_cols = [c for c in out.columns if c not in ['Ticker','Name','Sector']]
        out[num_cols] = out[num_cols].astype(float).round(6)
        # Percent columns to two decimals
        for c in [f'Δ% {x_metric}', f'Δ% {y_metric}']:
            out[c] = out[c].map(lambda v: None if pd.isna(v) else round(v, 2))

        st.markdown('---')
        st.subheader('A → B: Δ for highlighted tickers')
        st.dataframe(out)

    else:
        st.markdown('---')
        st.caption('No overlap of highlighted tickers between A and B (or none highlighted).')

# --- Correlation dendrogram for highlighted tickers ---------------------------------
st.markdown('---')
st.subheader('Correlation dendrogram (highlighted tickers)')

# We use the A-view snapshot and lookback window to compute correlations.
if highlight:
    # Get tickers actually present in the A snapshot
    hi_rows = metrics_df_A[metrics_df_A['IsHighlighted']].copy()
    tickers_hi = hi_rows['Ticker'].dropna().astype(str).unique().tolist()

    if len(tickers_hi) < 2:
        st.caption('Highlight at least two tickers to see a dendrogram.')
    else:
        # Slice the price history up to the A-view as-of and last `lookback` days
        try:
            pA = prices.loc[:asof_ts].copy()
            # keep only highlighted columns
            cols = [t for t in tickers_hi if t in pA.columns]
            pA = pA[cols]
            # restrict to most recent `lookback` rows
            if lookback is not None and lookback > 0 and len(pA) > lookback:
                pA = pA.tail(lookback)

            # Compute daily returns and correlation matrix
            rets = pA.pct_change().dropna(how='all')
            rets = rets.dropna(axis=1, how='all')

            if rets.shape[1] < 2:
                st.caption('Not enough valid price history for a correlation view.')
            else:
                corr = rets.corr()
                try:
                    from scipy.cluster.hierarchy import dendrogram, linkage
                    from scipy.spatial.distance import squareform
                except ImportError:
                    st.caption('Install scipy to enable the highlighted-ticker correlation dendrogram.')
                else:
                    import matplotlib.pyplot as plt

                    # Convert correlation to distance (1 - corr) and then to condensed form
                    dist = 1 - corr
                    # Ensure diagonals are zero and values finite
                    dist.values[range(len(dist)), range(len(dist))] = 0.0
                    dist_condensed = squareform(dist.values, checks=False)

                    Z = linkage(dist_condensed, method='average')

                    fig, ax = plt.subplots(figsize=(8, 4))
                    dendrogram(Z, labels=corr.columns.tolist(), leaf_rotation=90, ax=ax)
                    ax.set_ylabel('Distance (1 - correlation)')
                    ax.set_xlabel('Ticker')
                    fig.tight_layout()
                    st.pyplot(fig)
        except Exception as e:
            st.caption(f'Could not compute dendrogram: {e}')
else:
    st.caption('Type tickers in the highlight box to see a correlation dendrogram.')
# -------------------------------------------------------------------------------

# --- Daily Reports (News Feed) --------------------------------------------------
st.markdown('---')
st.subheader('🗞️ Daily Reports')

# --- Digest helpers: verify publisher report facts match the UI metrics_df -------

def _stable_df_digest(df: pd.DataFrame) -> str:
    """Create a short, stable digest for a metrics dataframe.

    This lets us verify whether a publisher facts.json was derived from the *same*
    metrics used to render the UI table.
    """
    if df is None or getattr(df, 'empty', True):
        return "(empty)"

    # Prefer a small but informative subset of columns that should exist in the UI.
    preferred_cols = [
        'Ticker',
        'Annualized Sharpe',
        'Daily Volatility (Std)',
        'Max Drawdown',
        'RSI_14',
        'RSI(14)',
        'RSI(1)',
        'CAGR',
    ]
    cols = [c for c in preferred_cols if c in df.columns]
    if 'Ticker' not in cols:
        cols = ['Ticker'] + cols

    x = df[cols].copy()
    x['Ticker'] = x['Ticker'].astype(str).str.upper()

    # Sort deterministically, and round numeric values for stability
    for c in cols:
        if c == 'Ticker':
            continue
        try:
            x[c] = pd.to_numeric(x[c], errors='coerce').round(8)
        except Exception:
            pass

    x = x.sort_values('Ticker')
    payload = x.to_csv(index=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()[:12]


def _pick_first(d: dict, keys):
    for k in keys:
        if k in d:
            return d.get(k)
    return None

# ------------------------------------------------------------------------------

# Reports are published by the mktme_publisher workflow into the mktme-data repo.
DEFAULT_REPORTS_INDEX = (
    "https://raw.githubusercontent.com/marshdoggo/mktme-data/main/reports/index.json"
)
REPORTS_INDEX_URL = os.environ.get("MKTME_REPORTS_INDEX_URL", DEFAULT_REPORTS_INDEX)
DEFAULT_REPORTS_BASE_URL = "https://raw.githubusercontent.com/marshdoggo/mktme-data/main"


def _sorted_report_entries(entries):
    def _sort_key(entry):
        ts = pd.to_datetime(entry.get("date"), errors="coerce")
        if pd.isna(ts):
            return (1, pd.Timestamp.min)
        return (0, ts)

    return sorted(entries or [], key=_sort_key, reverse=True)

@st.cache_data(ttl=5*60)
def _load_reports_index(url: str, cache_key: str) -> dict:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=5*60)
def _load_report_markdown(md_url: str, cache_key: str) -> str:
    r = requests.get(md_url, timeout=30)
    r.raise_for_status()
    return r.text

@st.cache_data(ttl=5*60)
def _load_report_json(json_url: str, cache_key: str) -> dict:
    r = requests.get(json_url, timeout=30)
    r.raise_for_status()
    return r.json()

# Try to load the index. If it doesn't exist yet, show a helpful message.
idx = None
# Cache-buster: when the underlying dataset max date advances, refresh report/index fetches.
reports_cache_key = str(max_day) if 'max_day' in locals() else str(pd.Timestamp.utcnow().date())
try:
    idx = _load_reports_index(REPORTS_INDEX_URL, cache_key=reports_cache_key)
except Exception:
    idx = None

sqlite_report_entries = [
    {
        "date": rec.get("asof_date"),
        "md": rec.get("markdown_path"),
        "json": rec.get("json_path"),
        "facts": rec.get("facts_path"),
    }
    for rec in list_recent_reports(universe, limit=14)
]

if (idx is None or not isinstance(idx, dict) or "universes" not in idx) and not sqlite_report_entries:
    st.info(
        "No published reports found yet. Once your publisher workflow runs, it should create "
        "`mktme-data/reports/index.json` plus `reports/<universe>/<YYYY-MM-DD>.md`. "
        "After that, this section will populate automatically."
    )
else:
    uni_map = idx.get("universes", {}) if isinstance(idx, dict) else {}
    entries = uni_map.get(universe, []) if isinstance(uni_map, dict) else []
    entries = _sorted_report_entries(entries)

    # Controls
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        if sqlite_report_entries:
            st.caption("Source of truth: SQLite runtime metadata. Artifact files still resolve from mktme-data raw URLs.")
        else:
            st.caption(f"Source: {REPORTS_INDEX_URL}")
    with c2:
        show_n = st.selectbox("Show", [3, 5, 10, 14], index=1, help="How many recent reports to show")
    with c3:
        debug_reports = st.checkbox(
            "Debug",
            value=False,
            help="Compare publisher facts.json to the UI metrics when dates match, and show a digest check to detect mismatched metric tables."
        )

    if sqlite_report_entries:
        entries = _sorted_report_entries(sqlite_report_entries)[: int(show_n)]

    if not entries:
        st.info(
            f"No reports published yet for universe '{universe}'. "
            "Run the publisher workflow to generate the first report."
        )
    else:
        newest_report_ts = pd.to_datetime(entries[0].get("date"), errors="coerce")
        if not pd.isna(newest_report_ts) and newest_report_ts.date() > max_day:
            st.warning(
                f"Published reports are newer than the dataset loaded into the charts "
                f"({newest_report_ts.date().isoformat()} vs {max_day.isoformat()}). "
                "That usually means the parquet publish has lagged or regressed."
            )

        # base_url can be provided by the publisher; otherwise derive from index URL.
        base_url = idx.get("base_url") if isinstance(idx, dict) else None
        if not base_url:
            # crude fallback: trim to repo root
            base_url = DEFAULT_REPORTS_BASE_URL

        if not sqlite_report_entries:
            entries = entries[: int(show_n)]

        # Most recent report preview at top
        latest = entries[0]
        latest_date = latest.get("date", "(unknown)")
        latest_md_rel = latest.get("md")

        if latest_md_rel:
            latest_md_url = f"{base_url}/{latest_md_rel}"
            # Derive a facts.json URL next to the markdown (publisher writes <date>.facts.json)
            latest_facts_url = None
            if isinstance(latest_md_rel, str) and latest_md_rel.endswith(".md"):
                latest_facts_url = f"{base_url}/{latest_md_rel[:-3]}.facts.json"

            try:
                latest_md = _load_report_markdown(latest_md_url, cache_key=reports_cache_key)
                with st.expander(f"Latest report — {latest_date}", expanded=True):
                    st.markdown(latest_md)

                    # Optional debug: compare publisher facts to current UI metrics
                    if 'debug_reports' in locals() and debug_reports and latest_facts_url:
                        st.markdown('---')
                        st.caption('Debug: comparing publisher facts to current UI table (only meaningful when dates/params match).')

                        facts = None
                        try:
                            facts = _load_report_json(latest_facts_url, cache_key=reports_cache_key)
                        except Exception as e:
                            st.warning(f"Could not load report facts.json: {e}")

                        if isinstance(facts, dict):
                            # Try to read metadata from facts.json (schema-flexible)
                            facts_asof = facts.get('asof') or facts.get('as_of') or facts.get('asof_date') or facts.get('date')
                            facts_lb = facts.get('lookback') or facts.get('lookback_days') or facts.get('lookback_trading_days')
                            facts_uni = facts.get('universe')

                            meta_cols = st.columns(3)
                            with meta_cols[0]:
                                st.caption(f"facts.universe: {facts_uni}")
                            with meta_cols[1]:
                                st.caption(f"facts.asof: {facts_asof}")
                            with meta_cols[2]:
                                st.caption(f"facts.lookback: {facts_lb}")

                            # Determine whether we're comparing like-for-like
                            ui_asof = str(asof_ts.date()) if 'asof_ts' in locals() else None
                            comparable = True
                            if facts_uni and str(facts_uni) != str(universe):
                                comparable = False
                            if facts_asof and ui_asof and str(facts_asof)[:10] != ui_asof:
                                comparable = False
                            if facts_lb is not None and 'lookback' in locals():
                                try:
                                    if int(facts_lb) != int(lookback):
                                        comparable = False
                                except Exception:
                                    pass

                            if not comparable:
                                st.info(
                                    "This facts.json does not match the current UI parameters (universe/asof/lookback). "
                                    "Switch your UI As-of date to the report date (and same lookback) to compare cleanly."
                                )
                            else:
                                # Digest check: if facts.json exposes a digest, compare to UI digest
                                facts_digest = _pick_first(facts, ['data_digest', 'digest', 'df_digest'])
                                ui_digest = _stable_df_digest(metrics_df_A if 'metrics_df_A' in locals() else None)

                                dig_cols = st.columns(2)
                                with dig_cols[0]:
                                    st.caption(f"UI digest: {ui_digest}")
                                with dig_cols[1]:
                                    st.caption(f"facts digest: {facts_digest or '(missing)'}")

                                if facts_digest and str(facts_digest) != str(ui_digest):
                                    st.warning(
                                        "Digest mismatch: the publisher report appears to be derived from a different "
                                        "metrics table than what the UI is currently showing. The report numbers are not "
                                        "trustworthy until the publisher uses the same metrics dataframe as the UI."
                                    )

                                # Extract leader rows from facts in a schema-tolerant way
                                leaders = (
                                    facts.get('leaders')
                                    or facts.get('top')
                                    or facts.get('top_by_sharpe')
                                    or (facts.get('sections', {}) or {}).get('leaders')
                                )

                                leaders_list = []
                                if isinstance(leaders, list):
                                    leaders_list = leaders
                                elif isinstance(leaders, dict):
                                    for k in ['top_by_sharpe', 'sharpe', 'annualized_sharpe', 'items']:
                                        v = leaders.get(k)
                                        if isinstance(v, list):
                                            leaders_list = v
                                            break

                                # Build a comparison table using current UI metrics_df_A
                                if leaders_list and 'metrics_df_A' in locals():
                                    ui_df = metrics_df_A.copy()
                                    ui_df['Ticker_upper'] = ui_df['Ticker'].astype(str).str.upper()

                                    def _pick_num(row: dict, keys):
                                        for kk in keys:
                                            if kk in row:
                                                try:
                                                    return float(row[kk])
                                                except Exception:
                                                    return None
                                        return None

                                    cmp_rows = []
                                    for r in leaders_list[:10]:
                                        if not isinstance(r, dict):
                                            continue
                                        t = (r.get('ticker') or r.get('Ticker') or r.get('symbol') or '').strip()
                                        if not t:
                                            continue
                                        tu = t.upper()
                                        ui_match = ui_df[ui_df['Ticker_upper'] == tu]
                                        if ui_match.empty:
                                            continue
                                        ui_row = ui_match.iloc[0]

                                        fact_sharpe = _pick_num(r, ['sharpe', 'annualized_sharpe', 'Annualized Sharpe', 'ann_sharpe'])
                                        ui_sharpe = None
                                        if 'Annualized Sharpe' in ui_row.index:
                                            try:
                                                ui_sharpe = float(ui_row['Annualized Sharpe'])
                                            except Exception:
                                                ui_sharpe = None

                                        fact_vol = _pick_num(r, ['daily_vol', 'daily_volatility', 'Daily Volatility (Std)', 'vol'])
                                        ui_vol = None
                                        if 'Daily Volatility (Std)' in ui_row.index:
                                            try:
                                                ui_vol = float(ui_row['Daily Volatility (Std)'])
                                            except Exception:
                                                ui_vol = None

                                        fact_dd = _pick_num(r, ['max_drawdown', 'Max Drawdown', 'dd'])
                                        ui_dd = None
                                        if 'Max Drawdown' in ui_row.index:
                                            try:
                                                ui_dd = float(ui_row['Max Drawdown'])
                                            except Exception:
                                                ui_dd = None

                                        nm = r.get('name') or r.get('Name') or ui_row.get('Name', '')

                                        cmp_rows.append({
                                            'Ticker': tu,
                                            'Name': nm,
                                            'Sharpe (facts)': fact_sharpe,
                                            'Sharpe (UI)': ui_sharpe,
                                            'Δ Sharpe': (None if (fact_sharpe is None or ui_sharpe is None) else (fact_sharpe - ui_sharpe)),
                                            'Vol (facts)': fact_vol,
                                            'Vol (UI)': ui_vol,
                                            'Δ Vol': (None if (fact_vol is None or ui_vol is None) else (fact_vol - ui_vol)),
                                            'MaxDD (facts)': fact_dd,
                                            'MaxDD (UI)': ui_dd,
                                            'Δ MaxDD': (None if (fact_dd is None or ui_dd is None) else (fact_dd - ui_dd)),
                                        })

                                    if cmp_rows:
                                        cmp = pd.DataFrame(cmp_rows)
                                        # round numeric columns for readability
                                        for c in ['Sharpe (facts)', 'Sharpe (UI)', 'Δ Sharpe', 'Vol (facts)', 'Vol (UI)', 'Δ Vol', 'MaxDD (facts)', 'MaxDD (UI)', 'Δ MaxDD']:
                                            if c in cmp.columns:
                                                cmp[c] = pd.to_numeric(cmp[c], errors='coerce').round(6)
                                        st.dataframe(cmp)
                                    else:
                                        st.caption('No comparable leader rows found (schema mismatch) or tickers not present in the UI frame.')
                                else:
                                    st.caption('facts.json did not expose a recognizable leaders list to compare, or UI metrics were not available.')

            except Exception as e:
                st.warning(f"Could not load latest report markdown: {e}")
        else:
            st.caption("Latest report entry is missing an 'md' field in reports/index.json.")

        # Older reports
        if len(entries) > 1:
            st.markdown("#### Recent archive")
            for ent in entries[1:]:
                d = ent.get("date", "(unknown)")
                rel_md = ent.get("md")
                if not rel_md:
                    continue
                md_url = f"{base_url}/{rel_md}"
                try:
                    md = _load_report_markdown(md_url, cache_key=reports_cache_key)
                    with st.expander(f"{d}", expanded=False):
                        st.markdown(md)
                except Exception:
                    with st.expander(f"{d}", expanded=False):
                        st.caption("Could not load this report.")
# ------------------------------------------------------------------------------

# --- Chat with the view (OpenAI) ------------------------------------------------
st.markdown('---')
st.subheader('💬 Ask the AI about this view')

client = get_openai_client(st)

if client is None:
    st.info(
        "Chat is disabled until you add an OpenAI API key. "
        "For local runs, Streamlit looks in `~/.streamlit/secrets.toml` or in `src/.streamlit/secrets.toml`."
    )
else:
    if "mktme_chat" not in st.session_state:
        st.session_state["mktme_chat"] = []

    for msg in st.session_state["mktme_chat"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Ask a question about the scatter plot, dendrogram, or metrics…")

    if user_q:
        ok, msg = rate_limit_ok(st.session_state)
        if not ok:
            with st.chat_message("assistant"):
                st.warning(msg)
        else:
            ok2, msg2 = daily_cap_ok(st.session_state)
            if not ok2:
                with st.chat_message("assistant"):
                    st.warning(msg2)
            else:
                st.session_state["mktme_chat"].append({"role": "user", "content": user_q})
                with st.chat_message("user"):
                    st.markdown(user_q)

                highlight_upper = list(highlight) if highlight else []
                context_text = build_view_context_text(
                    metrics_df=metrics_df_A,
                    prices=prices,
                    x_metric=x_metric,
                    y_metric=y_metric,
                    universe=universe,
                    asof_ts=asof_ts,
                    lookback=lookback,
                    highlight_upper=highlight_upper,
                    top_table_rows=5,
                    include_cluster_summary=True,
                )

                result = ask_ai_about_view(
                    client=client,
                    context_text=context_text,
                    user_question=user_q,
                    model="gpt-4.1-mini",
                )
                answer = result.answer

                with st.chat_message("assistant"):
                    st.markdown(answer)
                st.session_state["mktme_chat"].append({"role": "assistant", "content": answer})
# -------------------------------------------------------------------------------
