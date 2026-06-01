from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import streamlit as st

from .features import available_history_metrics, metric_history, snapshot_version
from .models import MODEL_OPTIONS, run_forecast, sklearn_available


def render_forecast_lab(
    *,
    default_universe: str,
    lookback: int,
    max_day,
    load_history: Callable[[str, int, str], pd.DataFrame],
) -> None:
    st.markdown("#### Forecast Lab")
    st.caption(
        "Exploratory model lab: trains on historical leaderboard states and predicts future top-N leaderboard membership. "
        "These probabilities are research signals, not investment advice."
    )

    c1, c2, c3, c4, c5 = st.columns([1.2, 1.8, 1, 1, 1.6])
    with c1:
        universe = st.selectbox("Universe", ["dow30", "sp500", "nasdaq100", "fx"], index=_universe_index(default_universe), key="fl_universe")
    ok, sklearn_detail = sklearn_available()

    history, history_error = _load_history_safe(load_history, universe, int(lookback), str(max_day))
    if history.empty:
        st.info("No leaderboard history is available for this universe/lookback yet. Run a leaderboard backfill first.")
        if history_error is not None:
            st.caption(f"History load detail: {type(history_error).__name__}: {history_error}")
        return

    metrics = available_history_metrics(history, universe)
    with c2:
        metric = st.selectbox("Rank metric", metrics, key="fl_metric") if metrics else None
    with c3:
        top_n = st.selectbox("Top N", [5, 10, 20, 50], index=2, key="fl_top_n")
    with c4:
        horizon = st.selectbox("Horizon", [5, 10, 20, 60], index=2, key="fl_horizon")
    with c5:
        model_choices = MODEL_OPTIONS if ok else ["Logistic Regression baseline"]
        model_type = st.selectbox("Model", model_choices, key="fl_model")

    if not metric:
        st.info("No rank metrics are available in the leaderboard history artifact.")
        return

    if not ok:
        st.caption(
            f"scikit-learn optional models are unavailable ({sklearn_detail}). "
            "The Logistic Regression baseline can still run."
        )

    h_metric = metric_history(history, metric, universe)
    dates = pd.to_datetime(h_metric.get("as_of_date"), errors="coerce").dropna().drop_duplicates().sort_values()
    st.caption(
        f"History artifact: {len(history):,} rows, {len(dates):,} {metric} snapshot dates, "
        f"latest {dates.max().date().isoformat() if len(dates) else 'unknown'}."
    )

    run = st.button("Train / Run Forecast", type="primary", key="fl_run")
    if not run:
        return

    with st.spinner("Training model and scoring latest snapshot..."):
        try:
            result = _cached_run_forecast(
                history,
                snapshot_version(history),
                universe,
                str(metric),
                int(top_n),
                int(horizon),
                str(model_type),
            )
        except Exception as exc:
            st.error(f"Forecast run failed: {type(exc).__name__}: {exc}")
            return

    _render_validation(result.validation)
    _render_board(result.board, result.target_description)
    _render_drilldown(result.board, result.drilldown, metric, int(top_n))


@st.cache_data(show_spinner=False, ttl=60 * 30)
def _cached_run_forecast(
    history: pd.DataFrame,
    version: str,
    universe: str,
    metric: str,
    top_n: int,
    horizon: int,
    model_type: str,
):
    _ = version
    return run_forecast(
        history,
        universe=universe,
        metric=metric,
        top_n=top_n,
        horizon=horizon,
        model_type=model_type,
    )


def _render_validation(validation: dict) -> None:
    st.markdown("#### Validation summary")
    st.caption(validation.get("split_method", "Time-aware validation"))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Samples", f"{validation.get('sample_count', 0):,}")
    m2.metric("Positive rate", _pct(validation.get("positive_class_rate")))
    m3.metric("Accuracy", _pct(validation.get("accuracy")))
    m4.metric("ROC-AUC", "n/a" if validation.get("roc_auc") is None else f"{validation['roc_auc']:.3f}")

    m5, m6, m7 = st.columns(3)
    m5.metric("Precision", _pct(validation.get("precision")))
    m6.metric("Recall", _pct(validation.get("recall")))
    m7.metric("F1", _pct(validation.get("f1")))

    st.caption(
        f"Train: {validation.get('train_date_min')} to {validation.get('train_date_max')} "
        f"({validation.get('train_rows'):,} rows). Validate: {validation.get('test_date_min')} to "
        f"{validation.get('test_date_max')} ({validation.get('test_rows'):,} rows)."
    )
    cm = validation.get("confusion_matrix") or [[0, 0], [0, 0]]
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
    st.dataframe(cm_df, use_container_width=True)
    st.caption(
        "Caveat: leaderboard states are noisy, overlapping, and autocorrelated. "
        "Validation is a time holdout, not a guarantee of future trading performance."
    )


def _render_board(board: pd.DataFrame, target_description: str) -> None:
    st.markdown("#### Probability board")
    if board.empty:
        st.info("No current tickers were scored.")
        return
    show = board.rename(
        columns={
            "ticker": "Ticker",
            "name": "Name",
            "sector": "Sector",
            "subindustry": "SubIndustry",
            "current_rank": "Current Rank",
            "selected_metric_value": "Current Metric Value",
            "forecast_probability": "Probability",
            "confidence_bucket": "Confidence",
            "latest_as_of_date": "Latest As Of",
            "target_description": "Target",
            "currently_top_n": "Currently Top N",
        }
    ).copy()
    if "Probability" in show.columns:
        show["Probability"] = pd.to_numeric(show["Probability"], errors="coerce").round(4)
    if "Current Metric Value" in show.columns:
        show["Current Metric Value"] = pd.to_numeric(show["Current Metric Value"], errors="coerce").round(6)
    st.caption(target_description)
    st.dataframe(show, use_container_width=True, hide_index=True)


def _render_drilldown(board: pd.DataFrame, drilldown: pd.DataFrame, metric: str, top_n: int) -> None:
    if board.empty:
        return
    st.markdown("#### Ticker drilldown")
    tickers = board["ticker"].dropna().astype(str).tolist()
    ticker = st.selectbox("Ticker", tickers, key="fl_drilldown_ticker")
    scored = board[board["ticker"] == ticker].head(1)
    if not scored.empty:
        r = scored.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Model probability", _pct(r.get("forecast_probability")))
        c2.metric("Current rank", int(r.get("current_rank")) if pd.notna(r.get("current_rank")) else "n/a")
        c3.metric("Currently top N", "Yes" if bool(r.get("currently_top_n")) else "No")

    path = drilldown[drilldown["ticker"].astype(str) == ticker].copy()
    if path.empty:
        st.caption("No labeled history available for this ticker.")
        return
    path["as_of_date"] = pd.to_datetime(path["as_of_date"], errors="coerce")
    path = path.sort_values("as_of_date").tail(120)
    chart_df = path[["as_of_date", "current_rank", "selected_metric_value", "target"]].rename(
        columns={
            "current_rank": "Rank",
            "selected_metric_value": metric,
            "target": f"Future top {top_n} label",
        }
    )
    st.line_chart(chart_df.set_index("as_of_date")[["Rank", metric]])
    st.dataframe(chart_df.tail(20), use_container_width=True, hide_index=True)


def _load_history_safe(load_history, universe: str, lookback: int, cache_key: str):
    try:
        return load_history(universe, lookback, cache_key), None
    except Exception as exc:
        return pd.DataFrame(), exc


def _universe_index(universe: str) -> int:
    options = ["dow30", "sp500", "nasdaq100", "fx"]
    return options.index(universe) if universe in options else 0


def _pct(value) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.1%}"
