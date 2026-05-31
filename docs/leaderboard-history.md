# Leaderboard History

Leaderboard snapshots are stored as parquet artifacts in the published data repo:

```text
leaderboards/<universe>/lookback_<lookback>.parquet
```

Each row represents one ticker's rank for one metric on one as-of date. Snapshot keys are:

```text
as_of_date, universe, lookback, metric, ticker
```

The daily pipeline appends only the latest as-of snapshot after a successful metrics build. Existing keys are skipped, so rerunning the pipeline for the same date does not duplicate rows.

## Backfill

Start with the Dow 30 because it is small:

```bash
python pipeline/backfill_leaderboards.py \
  --data-repo mktme-data \
  --universe dow30 \
  --lookback 252 \
  --history-days 252
```

Validate the artifact:

```bash
python - <<'PY'
import pandas as pd
p = "mktme-data/leaderboards/dow30/lookback_252.parquet"
df = pd.read_parquet(p)
print(df.shape)
print(df[["as_of_date", "metric", "ticker", "rank"]].head())
print(df.groupby("metric")["as_of_date"].nunique().sort_values())
PY
```

Then run the same command for `sp500`. A full one-year S&P 500 backfill recomputes rolling metrics for roughly 500 tickers across roughly 252 as-of dates, so treat it as a manual/local job unless CI timing has been measured in the target GitHub Actions runner.

Use `--force` only when intentionally rebuilding existing snapshot keys.
