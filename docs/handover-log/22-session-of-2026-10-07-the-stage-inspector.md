# Handover log §22. Session of 2026-10-07: the stage inspector

Part of the session log; the index is `docs/handover-log/README.md`.

A read-only Streamlit dashboard for looking at the data each stage produced.
Uncommitted.

- **Run:** `pip install -e ".[inspect]"` (or `.[dev]`, which includes it), then
  `streamlit run scripts/inspector_app.py`. The DB URL comes from
  `INSPECT_DATABASE_URL`, else `DATABASE_URL` (via `.env`); data root from
  `INSPECT_DATA_ROOT`, else `data`. Both are editable in the sidebar.
- **Layout:** queries and file readers in `pipeline/inspector/` (typed, mypy
  strict, all SQL there); UI only in `scripts/inspector_app.py` and
  `scripts/inspector_pages/`. Pages: Overview (row counts, latest runs, a
  button that runs every stage's quality gates), Crawl, Extract, Resolve,
  Reconcile, Classify, Impute, Battle drill-down (one battle across every
  table), Model (placeholder until that stage exists).
- **Read-only by construction:** every connection sets
  `default_transaction_read_only=on`; a test asserts a write raises. It is
  safe to point at the corpus database.
- **Demo data:** `python -m tests.integration.inspector_seed <data_root>`
  **drops and reseeds `DATABASE_URL`**, like the integration suite. Never run
  it against a corpus database.
- `streamlit>=1.50` (uses `width="stretch"`). pandas has no stubs here, so
  `pipeline/inspector/frames.py` ignores the import and aliases `Frame = Any`;
  installing `pandas-stubs` would turn that into an unused-ignore error.

**Verified:** ruff clean, mypy strict clean (95 files), 30 inspector tests
(9 unit, 21 integration incl. a Streamlit `AppTest` of every page, the
unreachable-DB path and the read-only guarantee), full suite 804 passed with
DB, per the worker's raw output. The lead launched the app on the seeded DB
and checked Overview, Reconcile and Impute render with charts in a browser.
**Not verified:** any real corpus (none exists yet); the other pages were
checked by AppTest only, which catches exceptions, not wrong charts.
