from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd


CORE_MAJOR_PAIRS = {
    "EUR/USD",
    "GBP/USD",
    "USD/JPY",
    "USD/CHF",
    "USD/CAD",
    "AUD/USD",
    "NZD/USD",
}
G10_CURRENCIES = {"USD", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD", "NOK", "SEK", "DKK"}
SCANDI_CURRENCIES = {"NOK", "SEK", "DKK"}
EM_CURRENCIES = {"CZK", "HKD", "HUF", "ILS", "MXN", "PLN", "SGD", "TRY", "ZAR"}
ASIA_CURRENCIES = {"HKD", "JPY", "SGD"}

CURRENT_FX_PAIRS = [
    "EUR/USD",
    "GBP/USD",
    "USD/JPY",
    "USD/CHF",
    "USD/CAD",
    "AUD/USD",
    "NZD/USD",
    "EUR/JPY",
    "EUR/GBP",
    "AUD/JPY",
    "GBP/JPY",
    "CHF/JPY",
    "EUR/CHF",
    "NZD/JPY",
]

FOREX_COM_CORE_PAIRS = [
    "AUD/CAD", "AUD/CHF", "AUD/CNH", "AUD/JPY", "AUD/NOK", "AUD/NZD", "AUD/PLN", "AUD/SGD", "AUD/USD",
    "CAD/CHF", "CAD/JPY", "CAD/NOK", "CAD/PLN",
    "CHF/HUF", "CHF/JPY", "CHF/NOK", "CHF/PLN",
    "CNH/JPY",
    "EUR/AUD", "EUR/CAD", "EUR/CHF", "EUR/CNH", "EUR/CZK", "EUR/DKK", "EUR/GBP", "EUR/HKD", "EUR/HUF", "EUR/JPY", "EUR/MXN", "EUR/NOK", "EUR/NZD", "EUR/PLN", "EUR/SEK", "EUR/SGD", "EUR/TRY", "EUR/USD", "EUR/ZAR",
    "GBP/AUD", "GBP/CAD", "GBP/CHF", "GBP/DKK", "GBP/HKD", "GBP/JPY", "GBP/MXN", "GBP/NOK", "GBP/NZD", "GBP/PLN", "GBP/SEK", "GBP/SGD", "GBP/USD", "GBP/ZAR",
    "HKD/JPY",
    "NOK/JPY", "NOK/SEK",
    "NZD/CAD", "NZD/CHF", "NZD/JPY", "NZD/USD",
    "SGD/HKD", "SGD/JPY",
    "USD/CAD", "USD/CHF", "USD/CNH", "USD/CZK", "USD/DKK", "USD/HKD", "USD/HUF", "USD/ILS", "USD/JPY", "USD/MXN", "USD/NOK", "USD/PLN", "USD/SEK", "USD/SGD", "USD/TRY", "USD/ZAR",
    "ZAR/JPY",
]


@dataclass(frozen=True)
class FxPair:
    pair_display: str
    base_currency: str
    quote_currency: str

    @property
    def ticker(self) -> str:
        return f"{self.base_currency}{self.quote_currency}"

    @property
    def yahoo_symbol(self) -> str:
        return pair_to_yahoo_symbol(self.pair_display)


def normalize_fx_pair(pair: str) -> FxPair:
    raw = str(pair).strip().upper()
    parts = [part for part in re.split(r"[/\-\s]+", raw) if part]
    if len(parts) == 1 and len(parts[0]) == 6:
        base, quote = parts[0][:3], parts[0][3:]
    elif len(parts) == 2 and all(len(part) == 3 for part in parts):
        base, quote = parts
    else:
        raise ValueError(f"Unsupported FX pair format: {pair!r}")
    return FxPair(pair_display=f"{base}/{quote}", base_currency=base, quote_currency=quote)


def pair_to_yahoo_symbol(pair: str) -> str:
    fx_pair = normalize_fx_pair(pair)
    return f"{fx_pair.base_currency}{fx_pair.quote_currency}=X"


def categorize_pair(base: str, quote: str) -> str:
    pair_display = f"{base}/{quote}"
    currencies = {base, quote}
    if pair_display in CORE_MAJOR_PAIRS:
        return "major"
    if "CNH" in currencies:
        return "cnh"
    if currencies & SCANDI_CURRENCIES:
        return "scandi"
    if "USD" in currencies and currencies & EM_CURRENCIES:
        return "em_fx"
    if currencies <= ASIA_CURRENCIES:
        return "asia_cross"
    if currencies <= G10_CURRENCIES:
        return "g10_cross"
    if currencies & EM_CURRENCIES:
        return "em_fx"
    return "other_cross"


def priority_for_pair(pair_display: str, category: str) -> str:
    if pair_display in CORE_MAJOR_PAIRS:
        return "core"
    if category in {"major", "g10_cross", "scandi", "cnh"}:
        return "expanded"
    return "experimental"


def build_fx_universe(pairs: list[str] | tuple[str, ...] = FOREX_COM_CORE_PAIRS) -> pd.DataFrame:
    rows = []
    seen: set[str] = set()
    for raw_pair in pairs:
        pair = normalize_fx_pair(raw_pair)
        if pair.pair_display in seen:
            continue
        seen.add(pair.pair_display)
        category = categorize_pair(pair.base_currency, pair.quote_currency)
        rows.append(
            {
                "Ticker": pair.ticker,
                "Name": pair.pair_display,
                "Sector": "FX",
                "SubIndustry": pair.base_currency,
                "pair_display": pair.pair_display,
                "base_currency": pair.base_currency,
                "quote_currency": pair.quote_currency,
                "yahoo_symbol": pair.yahoo_symbol,
                "category": category,
                "priority": priority_for_pair(pair.pair_display, category),
                "enabled": True,
            }
        )
    return pd.DataFrame(rows)


def get_fx_universe(name: str = "fx_forex_com_core") -> pd.DataFrame:
    key = str(name).strip().lower().replace("-", "_")
    full = build_fx_universe()
    if key in {"fx", "forex_com_core", "fx_forex_com_core"}:
        return full
    if key in {"current_fx", "fx_current", "fx_legacy"}:
        return build_fx_universe(CURRENT_FX_PAIRS)
    if key in {"fx_g10_majors", "g10_majors", "majors"}:
        return full[full["category"] == "major"].reset_index(drop=True)
    if key in {"fx_em_exotic", "fx_em", "em_exotic", "em_fx"}:
        return full[full["category"] == "em_fx"].reset_index(drop=True)
    if key in {"fx_scandi", "scandi"}:
        return full[full["category"] == "scandi"].reset_index(drop=True)
    if key in {"fx_cnh", "cnh"}:
        return full[full["category"] == "cnh"].reset_index(drop=True)
    raise ValueError(f"Unknown FX universe: {name}")


FX_UNIVERSE_ALIASES = {
    "fx",
    "fx_current",
    "fx_forex_com_core",
    "fx_g10_majors",
    "fx_em_exotic",
    "fx_scandi",
    "fx_cnh",
}
