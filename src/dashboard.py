import os
os.system("pip install yfinance beautifulsoup4 html5lib requests pyarrow --quiet")

import sys
sys.path.append(os.path.dirname(__file__))  # ensure src/ is on sys.path

try:
    from dotenv import load_dotenv
    load_dotenv()  # load from .env at project root
except ImportError:
    pass

import streamlit as st
import pandas as pd
from datetime import date
from fetch_data import get_sp500_constituents, download_prices, get_meta
from compute_metrics import compute_all_metrics
from metrics_registry import METRICS
from metric_docs import METRIC_META, get_pair_guide
from plot_metrics import scatter_xy
from universes import get_universe


from utils import parse_tickers

# --- Manifest/Parquet loader (prefer external loader.py; fallback to inline) ----
try:
    from loader import get_prices_for_universe  # provided by src/loader.py
except Exception:
    # Inline fallback so the app still runs if loader.py is missing
    import json
    import requests
    from io import BytesIO

    MANIFEST_URL = os.environ.get(
        "MKTME_MANIFEST_URL",
        "https://raw.githubusercontent.com/REPLACE_ME_USER/mktme-data/main/manifest.json"
    )

    @st.cache_data(ttl=60*60)
    def _load_manifest():
        r = requests.get(MANIFEST_URL, timeout=30)
        r.raise_for_status()
        return json.loads(r.text)

    @st.cache_data(ttl=60*60)
    def _load_parquet_http(url: str) -> pd.DataFrame:
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
        return _load_parquet_http(url)
# -------------------------------------------------------------------------------

# Default: prefer Stooq if toggled via Streamlit secrets / environment
PREFER_STOOQ_DEFAULT = str(os.environ.get("MKTME_PREFER_STOOQ", "0")).strip().lower() in ("1", "true", "yes")

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

st.set_page_config(page_title='S&P 500 Metric Explorer', layout='wide')

st.title('📈 Market Metric Explorer')

with st.sidebar:
    st.header('Settings')
    universe = st.selectbox(
        'Universe',
        ['sp500', 'nasdaq100', 'dow30', 'fx'],
        index=0,
        help='Pick the asset universe to analyze'
    )
    lookback = st.number_input('Lookback (trading days)', min_value=60, max_value=2520, value=252, step=21)
    refresh_tickers = st.checkbox('Refresh universe list (Wikipedia/Local)')
    refresh_prices = st.checkbox('Refresh price cache')
    prefer_stooq = st.checkbox('Prefer Stooq (faster; use if Yahoo rate-limits)', value=PREFER_STOOQ_DEFAULT)
    interactive = st.checkbox('Interactive chart (Plotly)', value=True)
    query = st.text_input('Highlight tickers (comma-separated)', value='', placeholder='AAPL, MSFT, NVDA  •  or  EURUSD, USDJPY')
    st.caption('Data by Wikipedia (equities) and local publisher → Parquet (served via GitHub Raw).')

if universe == 'sp500':
    tickers_df = get_sp500_constituents(force_refresh=refresh_tickers)
else:
    tickers_df = get_universe(universe, force_refresh=refresh_tickers)
meta = get_meta() if universe != 'fx' else {}

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


# Gather user metric choices (applies to both A and B)
col1, col2, col3 = st.columns([1,1,1])
with col1:
    x_metric = st.selectbox('X axis', list(METRICS.keys()), index=1)
with col2:
    y_metric = st.selectbox('Y axis', list(METRICS.keys()), index=0)
with col3:
    color_options = ['Sector','SubIndustry','(None)'] if universe != 'fx' else ['SubIndustry','(None)']
    color_by = st.selectbox('Color by', color_options)
    if color_by == '(None)':
        color_by = None

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
# -------------------------------------------------------------------------------

highlight = parse_tickers(query)

# Build A (and B if needed)
metrics_df_A = build_metrics_df(prices, asof_ts)
metrics_df = metrics_df_A  # keep old name for downstream references when not comparing
metrics_df_B = build_metrics_df(prices, asof2_ts) if compare and asof2_ts is not None else None

def render_view(title_label: str, metrics_df_local: pd.DataFrame, axis_ranges=None):
    # Plot
    cols_needed = [x_metric, y_metric, 'Sector', 'SubIndustry', 'Ticker', 'Name', 'Ticker_upper', 'IsHighlighted']
    present = [c for c in cols_needed if c in metrics_df_local.columns]
    missing_xy = [c for c in [x_metric, y_metric] if c not in metrics_df_local.columns]

    if missing_xy:
        st.warning(f"Selected metric(s) not available in this view: {', '.join(missing_xy)}. "
                   "Try another metric or refresh the cache.")
        return

    plot_df_local = metrics_df_local[present].dropna(subset=[x_metric, y_metric])
    st.subheader(title_label)
    if interactive:
        from plot_metrics import scatter_xy_interactive
        import plotly.express as px
        fig = scatter_xy_interactive(plot_df_local, x=x_metric, y=y_metric, title=f"{y_metric} vs {x_metric}")
        # Apply synced axis ranges if provided
        if axis_ranges is not None:
            if 'x' in axis_ranges and axis_ranges['x'] is not None:
                fig.update_xaxes(range=axis_ranges['x'])
            if 'y' in axis_ranges and axis_ranges['y'] is not None:
                fig.update_yaxes(range=axis_ranges['y'])
        if highlight:
            fig.update_traces(marker=dict(opacity=0.35))
            hi = plot_df_local[plot_df_local['IsHighlighted']]
            if not hi.empty:
                fig_hi = px.scatter(
                    hi,
                    x=x_metric,
                    y=y_metric,
                    color='Sector' if 'Sector' in hi.columns else None,
                    hover_name='Ticker',
                    hover_data={c: True for c in ['Name','Sector','SubIndustry'] if c in hi.columns},
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
        fig, ax = scatter_xy(plot_df_local, x=x_metric, y=y_metric, color_by=(color_by or ''), title=f"{y_metric} vs {x_metric}")
        st.pyplot(fig)

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
    table_local = metrics_df_local.sort_values(by=y_metric, ascending=False)
    if highlight:
        table_local = table_local.sort_values(by=['IsHighlighted', y_metric], ascending=[False, False])
    st.dataframe(table_local.drop(columns=['Ticker_upper'], errors='ignore'))

# Compute shared axis ranges if requested
axis_ranges_shared = None
if compare and metrics_df_B is not None and 'sync_axes' in locals() and sync_axes:
    def _get_xy_extents(df_local: pd.DataFrame, x: str, y: str):
        sx = df_local[x].dropna()
        sy = df_local[y].dropna()
        if sx.empty or sy.empty:
            return None
        return sx.min(), sx.max(), sy.min(), sy.max()
    extA = _get_xy_extents(metrics_df_A, x_metric, y_metric)
    extB = _get_xy_extents(metrics_df_B, x_metric, y_metric)
    if extA and extB:
        xmin = min(extA[0], extB[0]); xmax = max(extA[1], extB[1])
        ymin = min(extA[2], extB[2]); ymax = max(extA[3], extB[3])
        # Add 5% padding
        xr = xmax - xmin; yr = ymax - ymin
        pad_x = xr * 0.05 if xr > 0 else 0.01
        pad_y = yr * 0.05 if yr > 0 else 0.01
        axis_ranges_shared = {
            'x': [float(xmin - pad_x), float(xmax + pad_x)],
            'y': [float(ymin - pad_y), float(ymax + pad_y)],
        }

st.markdown('---')

if compare and metrics_df_B is not None:
    colA, colB = st.columns(2)
    with colA:
        render_view(f"View A — {asof_ts.date()} :: {y_metric} vs {x_metric}", metrics_df_A, axis_ranges=axis_ranges_shared)
    with colB:
        render_view(f"View B — {asof2_ts.date()} :: {y_metric} vs {x_metric}", metrics_df_B, axis_ranges=axis_ranges_shared)
else:
    render_view(f"{y_metric} vs {x_metric}", metrics_df_A)

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
