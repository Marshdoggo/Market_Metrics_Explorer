# Market Metrics Explorer Rebuild Brief

## Decision Summary

Recommendation: approve a partial rebuild, not a ground-up rewrite.

Why:

- The core product idea is working: ingest market data, compute cross-sectional metrics, publish daily summaries, and explore them in a dashboard.
- The fragile part is the delivery system around it: separate repos, GitHub raw files as runtime storage, cron hidden in a different repo, and weak health visibility.
- A partial rebuild can preserve the metric engine and UX concepts while making the system materially more reliable and easier to operate.

Decision ask:

- Approve the target architecture and phased migration below.
- Then implement in phases, starting with pipeline hardening and source-of-truth consolidation.

## Current State

Observed from repo evidence:

- The dashboard is a Streamlit app in [`src/dashboard.py`](/Users/marshallwhiteley/Desktop/Market_Metrics_Explorer/src/dashboard.py).
- The app reads price data from a manifest-backed parquet URL via [`src/loader.py`](/Users/marshallwhiteley/Desktop/Market_Metrics_Explorer/src/loader.py).
- The scheduled publisher is not in this repo. It lives in [`mktme_publisher/.github/workflows/publish.yml`](/Users/marshallwhiteley/Desktop/Market_Metrics_Explorer/mktme_publisher/.github/workflows/publish.yml).
- Published data and reports live in a separate `mktme-data` repo.
- The app treats `mktme-data/reports/index.json` as the latest-report source of truth.

Current architecture:

- Repo 1: app code
- Repo 2: publisher workflow
- Repo 3: published data artifacts
- Runtime data source: GitHub raw URLs
- Scheduler: GitHub Actions cron in the publisher repo
- Monitoring: implicit, mostly by checking whether the UI looks fresh

## Problems To Solve

### Reliability

- The critical path spans multiple repos and credentials.
- A failure in cron, checkout, secret configuration, data fetch, commit, or push can silently leave the app stale.
- GitHub raw artifacts are being used like a database and operational API.

### Operability

- There is no first-class pipeline status model.
- Freshness is inferred from files instead of stored explicitly.
- It is difficult to answer simple questions like:
  - When did the last successful fetch run?
  - Did the publish step succeed?
  - Did the app deploy with the latest data?

### Maintainability

- App, publisher, and published artifacts are split in a way that increases coordination overhead.
- The rebuild surface is bigger than necessary because compute logic and ops logic are mixed together.
- Debugging requires knowledge of multiple repos and deployment assumptions.

### Product Risk

- Stale data undermines trust more than missing features.
- Adding more universes, reports, or users will amplify the current failure modes.

## Rebuild Goals

The rebuild should:

- Make the system boringly reliable.
- Reduce moving parts and hidden dependencies.
- Preserve the useful metric and reporting logic.
- Add explicit health and freshness reporting.
- Make failures observable and recoverable.
- Keep deployment simple enough for a solo operator.

The rebuild should not:

- Rewrite metric formulas just for cleanliness.
- Replace the dashboard without a clear product gain.
- Introduce heavy infrastructure unless it directly solves a current pain.

## Recommendation

Pursue a partial rebuild with these design principles:

- One repo for app plus pipeline.
- One scheduled job system.
- One durable source of truth for metadata and status.
- One clear storage layer for snapshots and reports.
- One status surface for freshness and failures.

## Target Architecture

### Recommended Stack

Application:

- Keep Streamlit for the first rebuilt version.

Backend and jobs:

- Python application package inside the same repo.
- Scheduled pipeline via GitHub Actions initially, with cleaner job boundaries.

Storage:

- SQLite for metadata and pipeline state in the first rebuild.
- Local or object-style file storage for parquet snapshots and report artifacts.

Deployment:

- Streamlit Cloud can remain for the first rebuilt version if desired.
- If Streamlit Cloud proves limiting, move later to a single container deployment.

### Target Modules

Proposed internal layout:

- `app/`
  - Streamlit UI
- `pipeline/`
  - scheduled jobs and CLI entrypoints
- `domain/`
  - metric definitions, report generation, shared business logic
- `integrations/`
  - yfinance, Wikipedia, OpenAI, storage adapters
- `storage/`
  - snapshot writes, report writes, status writes, DB access
- `ops/`
  - health checks, validation, diagnostics

### Target Runtime Data Model

Use SQLite as the operational source of truth for:

- pipeline runs
- run steps
- run status
- fetch timestamps
- data snapshot metadata
- report metadata
- app-visible freshness fields

Keep parquet and markdown/json artifacts as files, but reference them from the DB rather than discovering them ad hoc.

Core tables:

- `pipeline_runs`
  - `id`
  - `started_at`
  - `finished_at`
  - `status`
  - `trigger_type`
  - `error_summary`
  - `git_sha`
- `data_snapshots`
  - `id`
  - `universe`
  - `asof_date`
  - `created_at`
  - `row_count`
  - `symbol_count`
  - `artifact_path`
  - `digest`
- `reports`
  - `id`
  - `universe`
  - `asof_date`
  - `created_at`
  - `markdown_path`
  - `json_path`
  - `facts_path`
  - `digest`
- `app_status`
  - `key`
  - `value`
  - `updated_at`

### Target Data Flow

1. Scheduler triggers pipeline.
2. Pipeline fetches raw data for each universe.
3. Validation runs on raw price results.
4. Snapshot parquet is written.
5. Metrics are computed from the snapshot.
6. Reports and facts artifacts are generated.
7. Metadata and freshness records are committed to SQLite.
8. App reads SQLite for latest status and latest available artifacts.
9. App renders clear freshness and health indicators in the UI.

## What To Keep

Keep with minimal changes:

- Metric definitions in [`src/metrics_registry.py`](/Users/marshallwhiteley/Desktop/Market_Metrics_Explorer/src/metrics_registry.py)
- Metric computation patterns in [`src/compute_metrics.py`](/Users/marshallwhiteley/Desktop/Market_Metrics_Explorer/src/compute_metrics.py)
- Deterministic report generation concepts in [`src/ai_report.py`](/Users/marshallwhiteley/Desktop/Market_Metrics_Explorer/src/ai_report.py)
- Streamlit dashboard interaction patterns in [`src/dashboard.py`](/Users/marshallwhiteley/Desktop/Market_Metrics_Explorer/src/dashboard.py)
- Optional grounded AI interpretation pattern in [`src/ai_context.py`](/Users/marshallwhiteley/Desktop/Market_Metrics_Explorer/src/ai_context.py)

## What To Replace

Replace or restructure:

- Multi-repo publish flow
- GitHub raw URL manifest as runtime storage contract
- `reports/index.json` as the primary freshness mechanism
- Hidden cron ownership in a separate repo
- Implicit pipeline health checks

Likely remove:

- Separate `mktme_publisher` repo as the long-term home of the scheduler
- Separate `mktme-data` repo as the primary runtime database

## Phased Migration Plan

### Phase 0: Design Lock

Goal:

- Align on architecture and delivery plan before code movement.

Deliverables:

- Approved rebuild brief
- Agreed success criteria
- Agreed hosting/storage decision for v1 rebuild

Estimated effort:

- 0.5 to 1 day

### Phase 1: Stabilize The Current System

Goal:

- Stop silent failure while the rebuild is in progress.

Scope:

- Add pipeline heartbeat output
- Add explicit `last_success` and `last_failure` records
- Add a validation step before publish
- Add clearer UI status for stale data
- Document current cron ownership and secret dependencies

Deliverables:

- Current system can self-report stale/failing state
- Failure modes become visible from the UI and logs

Estimated effort:

- 1 to 2 days

### Phase 2: Consolidate Into One Repo

Goal:

- Move scheduler and pipeline logic into the main app repo.

Scope:

- Bring publisher code into the main repo under a new package layout
- Replace cross-repo path assumptions with local package imports
- Add one pipeline entrypoint CLI
- Keep existing published artifacts working during transition

Deliverables:

- One repo contains app plus pipeline
- One workflow file owns scheduled processing

Estimated effort:

- 2 to 4 days

### Phase 3: Introduce Durable Metadata Storage

Goal:

- Replace ad hoc manifest/index discovery with explicit application state.

Scope:

- Add SQLite schema
- Write snapshot/report metadata into DB
- Add read layer for latest snapshot, latest report, freshness, and run history
- Keep artifact files on disk during this phase

Deliverables:

- App reads freshness and latest artifact metadata from SQLite
- Pipeline writes health and snapshot metadata transactionally

Estimated effort:

- 2 to 4 days

### Phase 4: Refactor App Read Path

Goal:

- Make the UI consume the new storage model.

Scope:

- Replace raw manifest/report index reads in the app
- Add a health banner and status panel
- Expose last successful pipeline run, as-of date, and report freshness
- Keep user-facing analytics behavior stable

Deliverables:

- Dashboard reflects true system state
- Stale data is explicit, not inferred

Estimated effort:

- 2 to 3 days

### Phase 5: Clean Up Legacy Publish Path

Goal:

- Remove duplicated infrastructure after the new path is proven stable.

Scope:

- Retire separate publisher repo workflow
- Retire GitHub raw manifest/report index dependency
- Archive or freeze legacy repos if desired

Deliverables:

- Single operational path
- Lower maintenance overhead

Estimated effort:

- 1 to 2 days

## Success Criteria

The rebuild is successful when:

- There is exactly one scheduled pipeline owner.
- A failed run is visible in under 1 minute via logs or status records.
- The app can display:
  - last successful fetch
  - last successful publish
  - latest as-of date
  - stale/not stale status
- The app no longer depends on `reports/index.json` for freshness truth.
- A single repo contains the code required to operate the system end-to-end.
- A new contributor can understand the pipeline in under 30 minutes.

## Risks And Tradeoffs

### If We Do Nothing

- Staleness incidents will recur.
- Confidence in the dashboard will keep dropping.
- Future features will be built on an unreliable base.

### If We Rebuild Too Much

- We spend time on architecture instead of product utility.
- We risk reintroducing bugs into working analytics logic.
- We delay shipping reliability improvements.

### Main Rebuild Risks

- Migration complexity around artifact compatibility
- Overdesigning storage or deployment too early
- Temporary duplication while old and new paths coexist

Mitigation:

- Keep the metric/report logic intact where possible.
- Migrate in phases with compatibility shims.
- Treat health visibility as the first shipping milestone.

## Recommended Implementation Order

Priority order:

1. Add current-system health visibility.
2. Consolidate publisher into this repo.
3. Add SQLite-backed status and metadata.
4. Switch app reads to the new source of truth.
5. Retire legacy publish infrastructure.

## Concrete Approval Options

### Option A: Approve Full Recommended Plan

Approve:

- Partial rebuild
- One repo
- Streamlit retained for v1 rebuilt app
- SQLite for metadata and status
- File-based artifacts retained initially

Best if:

- You want the highest reliability gain with moderate implementation risk.

### Option B: Approve Only Phase 1 Plus Phase 2

Approve:

- Immediate stabilization and repo consolidation
- Delay the SQLite migration

Best if:

- You want faster progress with lower upfront change.

Tradeoff:

- You still postpone the strongest source-of-truth improvements.

### Option C: Approve A Clean-Slate Rewrite Instead

Approve:

- New frontend
- New backend
- New storage
- New deployment model

Best if:

- You want to productize aggressively and accept a bigger investment.

Tradeoff:

- Highest cost and most delay before reliability improves.

## My Recommendation

Approve Option A.

It is the best balance of:

- reliability gain
- implementation speed
- reuse of working logic
- lower long-term ops burden

## Proposed First Execution Slice

If approved, the first build slice I would carry out is:

1. Create a new internal package layout inside this repo.
2. Move publisher logic into the repo as a pipeline module.
3. Add a SQLite status store and schema.
4. Add a health/freshness banner to the Streamlit app.
5. Add a new single-repo GitHub Actions workflow for scheduled runs.

Expected outcome of first slice:

- We can observe freshness and failures directly.
- The scheduler owner becomes unambiguous.
- The app starts reading a real source of truth for status.

## Review Notes

Questions to confirm before implementation:

- Keep Streamlit for the rebuilt v1, yes or no?
- SQLite acceptable for v1 source-of-truth metadata, yes or no?
- Keep artifact files in-repo for now, or move them to object storage in the first pass?
- Do you want the legacy multi-repo system kept alive during migration, or replaced more aggressively?
