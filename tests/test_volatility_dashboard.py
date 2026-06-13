import os
import sys

import numpy as np
import pandas as pd


sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from volatility_dashboard import (  # noqa: E402
    _normalize_vix_frame,
    _vix_chart_frame,
    build_vol_leaderboard,
    classify_vol_regime,
    compute_industry_vol_summary,
    compute_sector_vol_summary,
    compute_universe_vol_summary,
    compute_vol_breadth,
    get_volatility_column,
)


def _sample_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB", "CCC", "DDD"],
            "Name": ["A", "B", "C", "D"],
            "Universe": ["sp500", "sp500", "sp500", "sp500"],
            "Universe Label": ["S&P 500", "S&P 500", "S&P 500", "S&P 500"],
            "Sector": ["Tech", "Tech", "Health", "Health"],
            "SubIndustry": ["Software", "Hardware", "Devices", ""],
            "Realized_Vol_20D": [10.0, 30.0, np.nan, 50.0],
            "Daily Volatility (Std)": [0.01, 0.03, 0.04, 0.05],
        }
    )


def test_get_volatility_column_prefers_realized_vol_with_fallback():
    df = _sample_metrics()
    assert get_volatility_column(df) == "Realized_Vol_20D"

    fallback = df.drop(columns=["Realized_Vol_20D"])
    assert get_volatility_column(fallback) == "Daily Volatility (Std)"

    missing = pd.DataFrame({"Ticker": ["AAA"], "Realized_Vol_20D": [np.nan]})
    assert get_volatility_column(missing) is None


def test_classify_vol_regime_uses_percentile_thresholds():
    ref = pd.Series(range(1, 101), dtype=float)
    assert classify_vol_regime(10, ref) == "Calm"
    assert classify_vol_regime(50, ref) == "Normal"
    assert classify_vol_regime(85, ref) == "Elevated"
    assert classify_vol_regime(99, ref) == "Panic"
    assert classify_vol_regime(np.nan, ref) == "Unavailable"


def test_compute_universe_summary_handles_nans_and_tiny_groups():
    df = _sample_metrics()
    summary = compute_universe_vol_summary({"sp500": df, "fx": df.head(1)})

    assert set(summary["Universe"]) == {"S&P 500", "FX"}
    sp = summary[summary["Universe"] == "S&P 500"].iloc[0]
    assert sp["Number of assets"] == 3
    assert sp["Highest-vol ticker"] == "DDD"
    assert sp["Lowest-vol ticker"] == "AAA"


def test_sector_and_industry_summaries_tolerate_missing_metadata():
    df = _sample_metrics()
    sector = compute_sector_vol_summary(df, "Realized_Vol_20D")
    industry = compute_industry_vol_summary(df, "Realized_Vol_20D")

    assert sector["Sector"].tolist()[0] == "Health"
    assert "Software" in set(industry["SubIndustry"])
    assert "" not in set(industry["SubIndustry"])

    no_sector = df.drop(columns=["Sector"])
    assert compute_sector_vol_summary(no_sector, "Realized_Vol_20D").empty


def test_breadth_metrics_work_for_small_universes():
    breadth = compute_vol_breadth(_sample_metrics(), "Universe Label", "Realized_Vol_20D")

    assert len(breadth) == 1
    row = breadth.iloc[0]
    assert row["Assets"] == 3
    assert row["Elevated count"] == 1
    assert row["% above median"] == (1 / 3) * 100


def test_build_vol_leaderboard_sorts_and_groups_missing_values():
    df = _sample_metrics()
    top = build_vol_leaderboard(df, "Realized_Vol_20D", n=2)
    low = build_vol_leaderboard(df, "Realized_Vol_20D", n=2, ascending=True)
    grouped = build_vol_leaderboard(df, "Realized_Vol_20D", "Sector", n=1)

    assert top["Ticker"].tolist() == ["DDD", "BBB"]
    assert low["Ticker"].tolist() == ["AAA", "BBB"]
    assert set(grouped["Sector"]) == {"Tech", "Health"}
    assert grouped.groupby("Sector")["Rank"].max().le(1).all()


def test_normalize_vix_frame_accepts_close_or_vix_column():
    close_df = pd.DataFrame({"Close": [12.5, None, 14.0]}, index=["2026-01-01", "2026-01-02", "2026-01-03"])
    vix_df = pd.DataFrame({"VIX": [15.0]}, index=["2026-01-04"])

    close_out = _normalize_vix_frame(close_df)
    vix_out = _normalize_vix_frame(vix_df)

    assert close_out.columns.tolist() == ["VIX"]
    assert close_out["VIX"].tolist() == [12.5, 14.0]
    assert vix_out["VIX"].tolist() == [15.0]


def test_vix_chart_frame_handles_named_date_index():
    vix = pd.DataFrame({"VIX": [17.68]}, index=pd.DatetimeIndex(["2026-06-12"], name="DATE"))

    chart_df = _vix_chart_frame(vix)

    assert chart_df.columns.tolist() == ["Date", "VIX"]
    assert chart_df["VIX"].tolist() == [17.68]
