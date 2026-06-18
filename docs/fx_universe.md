# FX Universe

Market Metrics Explorer uses a FOREX.com-inspired FX universe to broaden coverage beyond the original small major/cross list.

The canonical definition lives in `src/fx_universe.py`. It includes majors, G10 crosses, Scandi pairs, CNH pairs, SGD/HKD, HKD/JPY, ZAR/JPY, and selected USD emerging-market pairs. Each row carries:

- `pair_display`, such as `EUR/USD`
- `base_currency`
- `quote_currency`
- `yahoo_symbol`, such as `EURUSD=X`
- `category`
- `priority`
- `enabled`

## Available FX Selections

- `fx_current`: the original compact FX universe
- `fx` or `fx_forex_com_core`: FOREX.com Core FX
- `fx_g10_majors`: G10 majors only
- `fx_em_exotic`: EM / exotic FX
- `fx_scandi`: Scandi FX
- `fx_cnh`: CNH FX

## Data Availability

Broker-listed tradability and market-data availability are separate concepts.

FOREX.com advertises a broad 80+ pair offering, but broker-listed spot FX pairs are not guaranteed to be available through Yahoo Finance, Twelve Data, Alpha Vantage, or any specific free market-data source. The pipeline therefore treats the universe as the desired coverage list, then verifies provider availability locally.

Run the Yahoo Finance availability check with:

```bash
python scripts/check_fx_availability.py
```

The script writes `data/quality/fx_availability.csv` with:

- `pair_display`
- `yahoo_symbol`
- `available`
- `first_date`
- `last_date`
- `rows`
- `error`

Unavailable pairs are logged and skipped during metric computation. A missing FX pair should not fail the full pipeline as long as at least one symbol has usable price history.
