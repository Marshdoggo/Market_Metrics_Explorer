import os
import sys

import numpy as np
import pandas as pd


sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from technical_indicators import (  # noqa: E402
    compute_atr_percent,
    compute_distance_from_ma,
    compute_drawdown_from_high,
    compute_signal_scores,
    compute_technical_metrics,
    compute_volume_ratio,
)


def test_close_only_technical_metrics_do_not_require_ohlcv():
    close = pd.Series(np.linspace(100, 140, 260), name="Close")
    df = compute_technical_metrics({"AAA": pd.DataFrame({"Close": close})})

    assert "AAA" in df.index
    assert np.isfinite(df.loc["AAA", "Distance_50DMA"])
    assert np.isfinite(df.loc["AAA", "Drawdown_3M_High"])
    assert np.isnan(df.loc["AAA", "ATR_14_Pct"])
    assert np.isnan(df.loc["AAA", "Volume_Ratio_20D_60D"])


def test_indicator_scalar_formulas():
    close = pd.Series([10, 11, 12, 13, 14, 15], dtype=float)
    assert compute_distance_from_ma(close, 3) == ((15 / 14) - 1) * 100
    assert compute_drawdown_from_high(close, 5) == 0
    assert np.isnan(compute_atr_percent(None, None, close))
    assert np.isnan(compute_volume_ratio(None))


def test_signal_scores_are_scaled_and_partial_data_tolerant():
    metrics = pd.DataFrame(
        {
            "Annualized Sharpe": [1.2, 0.3, -0.2],
            "Sortino Ratio": [1.5, 0.4, -0.1],
            "CAGR": [0.2, 0.05, -0.02],
            "RSI_14": [38, 48, 55],
            "Drawdown_3M_High": [-8, -16, -25],
            "Distance_50DMA": [2, -5, -12],
            "Distance_200DMA": [4, -8, -18],
            "SMA_50_200_Spread": [3, -2, -7],
            "Return_20D": [4, -3, -8],
            "Return_60D": [9, -5, -16],
            "Realized_Vol_20D": [18, 24, 35],
        },
        index=["AAA", "BBB", "CCC"],
    )

    scored = compute_signal_scores(metrics)
    for col in ["Dip_Buy_Score", "Bear_Breakdown_Score", "Momentum_Continuation_Score"]:
        assert col in scored.columns
        assert scored[col].dropna().between(0, 100).all()
