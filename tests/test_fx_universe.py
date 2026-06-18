import os
import sys

import pandas as pd


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import fetch_data  # noqa: E402
from check_fx_availability import check_fx_availability  # noqa: E402
from fx_universe import build_fx_universe, pair_to_yahoo_symbol  # noqa: E402
from universes import get_universe  # noqa: E402


def test_pair_to_yahoo_symbol():
    assert pair_to_yahoo_symbol("EUR/USD") == "EURUSD=X"
    assert pair_to_yahoo_symbol("eurusd") == "EURUSD=X"


def test_forex_com_core_universe_count_exceeds_60():
    assert len(build_fx_universe()) > 60


def test_fx_loader_returns_expected_columns():
    df = get_universe("fx_forex_com_core")
    expected = {
        "Ticker",
        "Name",
        "Sector",
        "SubIndustry",
        "pair_display",
        "base_currency",
        "quote_currency",
        "yahoo_symbol",
        "category",
        "priority",
        "enabled",
    }
    assert expected.issubset(df.columns)
    assert "EURUSD" in set(df["Ticker"])
    assert "EURUSD=X" in set(df["yahoo_symbol"])


def test_fx_fetch_handles_bad_symbol_without_crashing(monkeypatch):
    dates = pd.date_range("2026-01-01", periods=70, freq="B")

    def fake_download_yahoo_symbols(symbols, **kwargs):
        assert "BADUSD=X" in symbols
        return pd.DataFrame({"EURUSD=X": range(len(dates))}, index=dates)

    monkeypatch.setattr(fetch_data, "_download_yahoo_symbols", fake_download_yahoo_symbols)

    out = fetch_data.download_prices_fx_window(
        ["EUR/USD", "BAD/USD"],
        lookback_trading_days=70,
        asof=pd.Timestamp("2026-04-15"),
        equity_source="yahoo",
    )

    assert out.columns.tolist() == ["EURUSD"]
    assert len(out) == 70


def test_availability_check_records_bad_symbol_without_crashing(tmp_path):
    dates = pd.date_range("2026-01-01", periods=3, freq="B")

    def fake_fetch(symbol, start, end):
        if symbol == "EURUSD=X":
            return pd.DataFrame({"Close": [1.1, 1.2, 1.3]}, index=dates)
        raise RuntimeError("missing symbol")

    out = check_fx_availability(
        universe="fx_g10_majors",
        output_path=tmp_path / "fx_availability.csv",
        fetcher=fake_fetch,
    )

    eurusd = out[out["pair_display"] == "EUR/USD"].iloc[0]
    usdjpy = out[out["pair_display"] == "USD/JPY"].iloc[0]
    assert bool(eurusd["available"]) is True
    assert int(eurusd["rows"]) == 3
    assert bool(usdjpy["available"]) is False
    assert "missing symbol" in usdjpy["error"]
