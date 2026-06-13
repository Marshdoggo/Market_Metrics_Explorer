import os
import sys

import pandas as pd


sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

import fetch_data  # noqa: E402


class _EmptyTicker:
    def history(self, *args, **kwargs):
        return pd.DataFrame()


class _FakeResponse:
    text = "DATE,OPEN,HIGH,LOW,CLOSE\n06/10/2026,17,18,16,17.5\n06/11/2026,18,19,17,18.5\n"

    def raise_for_status(self):
        return None


def test_download_vix_data_uses_cboe_fallback_when_yahoo_is_empty(monkeypatch):
    monkeypatch.setattr(fetch_data.yf, "download", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(fetch_data.yf, "Ticker", lambda *args, **kwargs: _EmptyTicker())
    monkeypatch.setattr(fetch_data.requests, "get", lambda *args, **kwargs: _FakeResponse())

    out = fetch_data.download_vix_data(period="1mo")

    assert out.columns.tolist() == ["VIX"]
    assert out["VIX"].tolist() == [17.5, 18.5]
