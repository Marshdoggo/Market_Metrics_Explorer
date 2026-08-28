import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

import universes  # noqa: E402


def _rows(n=101):
    return {
        "Ticker": [f"T{i}" for i in range(n)],
        "Company": [f"Company {i}" for i in range(n)],
        "GICS Sector": ["Technology"] * n,
        "GICS Sub-Industry": ["Software"] * n,
    }


@pytest.mark.parametrize("ticker_column", ["Symbol", "Ticker", "Ticker symbol", "Ticker Symbol"])
def test_wiki_constituents_accepts_ticker_schema_variations(ticker_column):
    data = _rows()
    data[ticker_column] = data.pop("Ticker")
    out = universes._normalize_constituents(pd.DataFrame(data), "nasdaq100", "test")
    assert len(out) == 101
    assert out.columns.tolist() == ["Ticker", "Name", "Sector", "SubIndustry"]
    assert out.iloc[0]["Ticker"] == "T0"


def test_wiki_constituents_rejects_unrelated_tables():
    df = pd.DataFrame({"Year": [2024, 2025], "Note": ["foo", "bar"]})
    with pytest.raises(ValueError, match="no ticker"):
        universes._normalize_constituents(df, "nasdaq100", "test")


def test_cached_fallback_used_when_live_retrieval_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(universes, "CACHE_DIR", tmp_path)
    cached = universes._normalize_constituents(pd.DataFrame(_rows()), "nasdaq100", "seed")
    universes._write_cached_universe("nasdaq100", cached, "seed")

    def fail(*args, **kwargs):
        raise RuntimeError("upstream down")

    monkeypatch.setattr(universes, "_read_wiki_constituents", fail)
    out = universes.get_universe("nasdaq100", force_refresh=True)
    assert len(out) == len(cached)
    assert out["Ticker"].tolist() == cached["Ticker"].tolist()


def test_invalid_cache_is_not_accepted(monkeypatch, tmp_path):
    monkeypatch.setattr(universes, "CACHE_DIR", tmp_path)
    pd.DataFrame({"Ticker": []}).to_parquet(tmp_path / "nasdaq100.parquet", index=False)

    def fail(*args, **kwargs):
        raise RuntimeError("upstream down")

    monkeypatch.setattr(universes, "_read_wiki_constituents", fail)
    with pytest.raises(RuntimeError, match="upstream down"):
        universes.get_universe("nasdaq100", force_refresh=True)


def test_nasdaq_sanity_check_rejects_unrealistic_count():
    with pytest.raises(ValueError, match="outside expected range"):
        universes._normalize_constituents(pd.DataFrame(_rows(10)), "nasdaq100", "test")


def test_current_dow_fixture_returns_exactly_30_valid_constituents(monkeypatch):
    fixture_path = Path(__file__).parent / "fixtures" / "dow30_components.html"
    fixture_html = fixture_path.read_text(encoding="utf-8")

    class MockResponse:
        text = fixture_html

        @staticmethod
        def raise_for_status():
            return None

    def mock_get(url, headers, timeout):
        assert url == universes.WIKI_PAGES["dow30"]
        assert headers["User-Agent"]
        assert timeout == 20
        return MockResponse()

    monkeypatch.setattr(universes.requests, "get", mock_get)

    out = universes._read_wiki_constituents(
        universes.WIKI_PAGES["dow30"],
        universe="dow30",
    )

    assert out.columns.tolist() == universes.REQUIRED_COLUMNS
    assert len(out) == 30
    assert out["Ticker"].is_unique
    assert out["Ticker"].map(universes._looks_like_ticker).all()
    assert set(out["Ticker"]) == {
        "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
        "GOOGL", "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK",
        "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "WMT",
    }
