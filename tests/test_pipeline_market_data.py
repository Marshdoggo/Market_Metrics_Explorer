import json
import os
import sys

import pandas as pd


sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "pipeline"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

import publish_legacy_artifacts  # noqa: E402
from publish_legacy_artifacts import _load_or_fetch_prices, _write_manifest  # noqa: E402


def test_write_manifest_includes_market_data(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        {"sp500": "sp500/prices.parquet"},
        github_user="example",
        market_rel_paths={"vix": "market/vix.parquet"},
    )

    assert manifest["universes"]["sp500"]["parquet_url"] == "https://raw.githubusercontent.com/example/mktme-data/main/sp500/prices.parquet"
    assert manifest["market_data"]["vix"]["parquet_url"] == "https://raw.githubusercontent.com/example/mktme-data/main/market/vix.parquet"

    written = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert written["market_data"] == manifest["market_data"]


def test_fx_refresh_falls_back_to_existing_parquet_when_provider_empty(monkeypatch, tmp_path):
    fx_dir = tmp_path / "fx"
    fx_dir.mkdir()
    expected = pd.DataFrame(
        {"EURUSD": [1.08, 1.09]},
        index=pd.date_range("2026-01-02", periods=2, freq="D"),
    )
    expected.to_parquet(fx_dir / "prices.parquet")

    monkeypatch.setattr(
        publish_legacy_artifacts,
        "download_prices_fx_window",
        lambda *args, **kwargs: pd.DataFrame(),
    )

    actual = _load_or_fetch_prices(
        data_repo=tmp_path,
        universe="fx",
        tickers_df=pd.DataFrame({"Ticker": ["EURUSD"]}),
        lookback=252,
        force_refresh=True,
        use_existing_parquet=False,
        equity_source="yahoo",
    )

    pd.testing.assert_frame_equal(actual, expected, check_freq=False)
