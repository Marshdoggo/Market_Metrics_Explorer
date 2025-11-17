# src/loader.py
import os, json, pandas as pd, requests
from io import BytesIO

DEFAULT_MANIFEST = "https://raw.githubusercontent.com/marshdoggo/mktme-data/main/manifest.json"
MANIFEST_URL = os.environ.get("MKTME_MANIFEST_URL", DEFAULT_MANIFEST)

def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure the index is a DatetimeIndex, and make it *tz-naive*
    so it matches the rest of the app's expectations.
    """
    idx = pd.to_datetime(df.index)

    # If there's a timezone, strip it off
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)

    df.index = idx
    return df

def _http_get(url: str, timeout=60):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r

def _load_manifest() -> dict:
    r = _http_get(MANIFEST_URL, timeout=30)
    return json.loads(r.text)

def load_parquet_http(url: str) -> pd.DataFrame:
    r = _http_get(url, timeout=60)
    df = pd.read_parquet(BytesIO(r.content), engine="pyarrow")
    return _ensure_dt_index(df)

def get_prices_for_universe(universe: str) -> pd.DataFrame:
    m = _load_manifest()
    try:
        url = m["universes"][universe]["parquet_url"]
    except Exception:
        raise RuntimeError(f"Universe '{universe}' not present in manifest {MANIFEST_URL}")
    return load_parquet_http(url)