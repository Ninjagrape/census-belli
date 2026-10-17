# Handover log §11. Session of 2026-09-19: what changed, and what is still open

Part of the session log; the index is `docs/handover-log/README.md`.

### 11.1 Changed files (all uncommitted)

| File | Change |
|---|---|
| `pipeline/db.py` | Fixed §4.3 and §4.4. `apply_schema` now uses a raw driver cursor and commits the driver connection. |
| `config/schema.sql` | Added `battles.year_astronomical` (generated) and `idx_battles_year`. |
| `alembic/versions/2026_09_19_0002-...py` | **New.** Same column for databases built from the original baseline. |
| `tests/integration/test_db.py` | Corrected the false `-30` assertion (§4.5); added 3 tests pinning the new column. |
| `pipeline/extractors/merger.py` | Log absent model covariates (`battle_type`, `terrain`, `fortified`); see §11.3. |
| `tests/integration/test_extract.py` | Corrected a test comment whose premise was false; assert terrain logged and weather not. |
| `pipeline/quality.py` | SAVEPOINT per check so one failing gate cannot poison the rest; `begin_nested` added to the `Connection` protocol. |
| `agents/extract.yaml` | `commander_extraction_rate` rewritten against `battle_commanders`. |
| `tests/integration/test_quality_gates.py` | **New.** 5 tests pinning gate isolation and that every spec's SQL executes. |
| `tests/unit/test_quality.py` | `StubConnection` gains `begin_nested`. |
| `tests/integration/test_orchestrator_gates.py` | `ScriptedConnection` gains `begin_nested`. |
| `handover.md` | This. |

Nothing is staged and nothing is committed, per `CLAUDE.md`. Suggested message:

```
fix: make apply_schema work against a real database

Use a raw driver cursor so psycopg does not read '%' in schema comments as a
placeholder, and commit the driver connection so DDL survives the
drop_existing path. Neither had ever run against Postgres.

Add battles.year_astronomical so BC dates are readable from Python, in both
schema.sql and an Alembic migration. Correct the BC extraction test, which
asserted astronomical -30 where Postgres returns -31.

Log absent battle-level model covariates to missing_data_log, which recorded
only the required fields and the per-side ones before.

Run each quality check in its own SAVEPOINT. A failing check aborted the
transaction, so every later check in the stage reported a false failure rather
than its own verdict. Point commander_extraction_rate at battle_commanders,
which exists, instead of data.commanders_raw, which does not.
```

### 11.2 Verification status, stated honestly

Verified against live PostgreSQL 15.19: schema application from both
`apply_schema()` and Alembic, the migration's upgrade/downgrade/re-upgrade
path on real BC rows, and 284 passing tests including all 27 that had never
executed before.

`ruff` clean. `mypy --strict` clean across 37 files.

Not verified, unchanged from before: any real LLM call, any real crawl at
scale, and the entire statistical half of the project, which still does not
exist.

### 11.3 RESOLVED: what counts as a missing field?

The failing test is fixed, and the premise it was built on turned out to be
wrong. Recorded because the reasoning governs every future addition.

**The test's comment was false.** It claimed Wikidata gives no weather "and the
infobox gives no casualties for one side". Inspecting the merged fixture: both
sides carry troop reports, casualties, commanders and an outcome. Actium is
complete apart from `terrain` (empty) and `weather` (None). `missing_fields()`
already covered per-side casualties, commanders, outcome, troop totals and
scope, so it was never as narrow as it first appeared.

**The real gap** was that none of the model's *battle-level covariates* were
checked. `agents/model.yaml` builds its linear predictor from force ratio,
`battle_type`, `terrain`, defensive advantage (`fortified`) and era. Of those,
only the force-ratio and era inputs were logged when absent.

**The rule now applied:** log the absence of a field some downstream stage
actually consumes; do not log purely descriptive fields. So
`_COVARIATE_BATTLE_FIELDS = (battle_type, terrain, fortified)` are logged, and
`weather` is deliberately not -- nothing reads it, and a row per ancient battle
for a field no stage consumes would bury the missingness that matters.

Two details worth keeping:

- `fortified=False` is an answer, not an absence, and an empty `terrain` list
  is an absence. The check tests `value is None or (isinstance(value, list) and
  not value)` rather than truthiness, which would swallow the False case.
- **`battle_type` will now be logged missing on most battles**, and that is
  correct: §4.2 left nothing setting it until classify infers it. Those rows
  are the standing record of that gap.

The test now asserts both directions -- terrain logged, weather not -- so the
distinction cannot quietly erode.

### 11.4 RESOLVED: extract's quality gates were broken, in two ways

Both fixed. The second was the serious one and reached every stage.

**`commander_extraction_rate` queried a table that exists in no schema.**
It joined `data.commanders_raw`; the extract stage writes `battle_commanders`.
The gate could only ever return `relation "data.commanders_raw" does not
exist`. Rewritten against the real table. §4.1's pattern, third instance.

**One failing gate poisoned every gate after it.** Postgres aborts the whole
transaction on a failed statement, and all checks shared one connection, so
every check after a broken one returned `current transaction is aborted,
commands ignored` instead of its own verdict. `no_empty_extractions` is valid
SQL and was reported as failed purely because it ran third.

`QualityRunner._run_sql_check` now wraps each check in `begin_nested()`, a
SAVEPOINT, so a failure rolls back only that check. This was a defect in
`pipeline/quality.py`, not in any spec, so it affected **every stage's gates**.

Confirmed by removing the savepoint again: 3 of the 5 new tests in
`tests/integration/test_quality_gates.py` fail without it, with exactly the
aborted-transaction error. They are not vacuous.

**A sweep of all 9 specs now runs 22 SQL gates with no execution errors.**
`test_every_declared_sql_gate_executes` pins that, so a future spec cannot
quietly ship SQL the database rejects.

#### Changing the `Connection` protocol means changing two stubs

`pipeline/quality.py` defines a structural `Connection` Protocol, and **two**
separate test doubles implement it:

- `StubConnection` in `tests/unit/test_quality.py`
- `ScriptedConnection` in `tests/integration/test_orchestrator_gates.py`

Adding `begin_nested` to the protocol broke the second one, which is easy to
miss because its failure surfaced as unrelated gate assertions rather than an
attribute error. Both now provide a `nullcontext()` savepoint. If you add
anything else to that protocol, update both.
