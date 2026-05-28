#!/bin/sh
set -eu

APP_ROOT="/Users/marshallwhiteley/Desktop/Market_Metrics_Explorer"
cd "$APP_ROOT"

if [ -x "$APP_ROOT/.venv_rebuild/bin/python" ]; then
  PY="$APP_ROOT/.venv_rebuild/bin/python"
elif [ -x "$APP_ROOT/.venv311/bin/python" ]; then
  PY="$APP_ROOT/.venv311/bin/python"
elif [ -x "$APP_ROOT/.venv/bin/python" ]; then
  PY="$APP_ROOT/.venv/bin/python"
else
  PY="$(command -v python3)"
fi

export MKTME_PIPELINE_TRIGGER="${MKTME_PIPELINE_TRIGGER:-schedule}"
"$PY" "$APP_ROOT/pipeline/run_daily_pipeline.py" \
  --mode publish \
  --universes sp500 nasdaq100 dow30 fx \
  --equity-source "${MKTME_EQUITY_SOURCE:-auto}" \
  --force-refresh

DATA_REPO="${MKTME_DATA_REPO:-$APP_ROOT/mktme-data}"
if [ "${MKTME_AUTO_PUSH_DATA:-1}" = "1" ] && [ -d "$DATA_REPO/.git" ]; then
  git -C "$DATA_REPO" add -A
  if ! git -C "$DATA_REPO" diff --cached --quiet; then
    git -C "$DATA_REPO" commit -m "Update market metrics data $(date +%Y-%m-%d)"
    git -C "$DATA_REPO" push origin main
  fi
fi
