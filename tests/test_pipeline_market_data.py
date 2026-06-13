import json
import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "pipeline"))

from publish_legacy_artifacts import _write_manifest  # noqa: E402


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
