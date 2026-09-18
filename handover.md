# Handover

For the next session or agent picking this project up cold.

Read `CLAUDE.md` first for what the project *is*. This file covers what state it is **in**, what is verified versus merely written, and the traps that have already cost time.

`CLAUDE.md` points every session here, and asks you to update this file before you finish. Keep it current: a stale handover is worse than none, because the next agent will trust it. If you change the state of the project, change §1, §6 and §7 to match.

Last updated: 2026-09-18. Working tree clean at `bf815ef`.

---

## 1. Current state in one table

| Stage | Spec | Implemented | Unit tested | Verified against a database |
|---|---|---|---|---|
| crawl | yes | yes | 36 tests | no |
| extract | yes | yes | 67 tests | no |
| resolve | yes | **no** | — | — |
| reconcile | yes | **no** | — | — |
| classify | yes | **no** | — | — |
| impute | yes | **no** | — | — |
| model | yes | **no** | — | — |
| evaluate | yes | **no** | — | — |
| report | yes | **no** | — | — |

Infrastructure is complete: `pipeline/db.py`, `config.py`, `quality.py`, `stages/base.py`, `llm/` (9 modules), `orchestrator.py`, Alembic with `config/schema.sql` as baseline, `docker-compose.yml`, `Makefile`, `.env.example`.

**Verify the state yourself before trusting this file:**

```bash
ruff check pipeline/ tests/ alembic/     # expect: All checks passed
python -m mypy pipeline/                 # expect: Success, 37 source files (strict)
python -m pytest tests/ -q               # expect: 255 passed, 28 skipped
```

The 28 skips are all database-dependent. That is the central fact of this handover.

---

## 2. The blocker: there is no database

The machine this was built on has **no Docker, no podman, no Postgres, no `psql`, and no `make`**. Nothing listens on 5432.

Consequence: `tests/integration/test_db.py` (19 tests), `test_crawl.py` (2) and `test_extract.py` (6) have **never executed**. Everything they cover is written but unproven:

- `apply_schema()` against real Postgres
- `get_connection()` commit/rollback semantics
- every SQL statement in `pipeline/extractors/store.py` — enum casts (`CAST(:branch AS troop_branch)`), the `extraction_method[]` array cast, the `terrain` text-array write
- the Actium fixture and the schema's CHECK/UNIQUE constraints
- `SqlCrawlLog` writing real `crawl_log` rows
- the `llm_calls` resume cache end to end

The Alembic baseline is verified **offline only** (57 DDL statements emitted). It has never been applied to a live database.

**First thing to do in the next session**, if a database is available:

```bash
make db-up            # or: docker compose up -d postgres
make db-schema        # or: python -m pipeline.db --apply-schema
make test             # or: python -m pytest tests/ -v
```

If those 28 skips turn into passes, the foundation is real. If they fail, fix that before writing any new stage. Do not build Phase 3+ on an unverified database layer.

`make` is also absent on that machine — the Makefile targets are written but unrun. Use the raw commands in parentheses above.

---

## 3. Read this before writing any code

### 3.1 `pipeline/llm` is a package, not a module

There is **no `pipeline/llm.py`** and there must not be — a module and a package of that name cannot coexist. `.claude/commands/build-all.md` Task 1.4 says "implement pipeline/llm.py"; **that instruction is stale.** The package already does everything it asks for and more.

Public API:

```python
from pipeline.llm import LLMService

service = LLMService.from_spec(spec, db_conn=conn)
result = service.complete(
    system=spec["prompt"]["system"],
    user=article_text,
    json_schema=schema,
    metadata={"battle_id": battle_id},
)
if not result.ok:
    continue          # already logged and flagged for review
upsert(result.data)
service.log_summary()
```

It handles retries, JSON-schema validation, cost accounting, and logs every call to `llm_calls`. That table doubles as a **resume cache keyed on request hash**, so a re-run must not re-pay for work already done. Read `pipeline/llm/service.py` before using it.

Providers are routed per stage in the spec, not in code:

| Stage | `params.llm_provider` |
|---|---|
| extract | gemini |
| resolve | gemini |
| classify | anthropic |

Gemini needs an **AI Studio** key (`AIza...`), not an OAuth token. See `.env.example`.

### 3.2 The quality gates are real now, and they bite

`pipeline/quality.py` executes the `quality_checks` SQL in each agent spec. Until recently `_run_single_check` was a stub returning `passed=True` unconditionally, so **every gate in every spec was inert**.

Two latent spec bugs surfaced the moment they started executing. Expect more as later stages run for the first time. When a gate fails, check whether the *spec* is wrong before assuming the *code* is.

Checks declaring a `method:` instead of SQL report as **not implemented** — deliberately failing rather than passing, so the gap stays visible. Eight are outstanding:

| Method | Used by |
|---|---|
| `diagnostics_json` | model.convergence, model.no_divergences, reconcile.model_convergence |
| `posterior_check` | model.skill_distribution_sensible |
| `per_round_check` | impute.imputation_completeness |
| `range_check` | impute.imputed_values_plausible |
| `statistical_test` | impute.distributional_check |
| `brier_score` | evaluate.calibration |
| `jaccard_overlap` | evaluate.ranking_stability |
| `held_out_accuracy` | evaluate.held_out_accuracy |

Implement each alongside the stage that produces the artefact it inspects. Register handlers in `UNIMPLEMENTED_METHODS` in `pipeline/quality.py`.

### 3.3 `retry_policy.on_failure` is honoured

`continue_with_logging` (extract, classify) does not halt the pipeline; `pause_and_alert` does. This lives on `StageResult.halts_pipeline`. It was previously read and discarded.

### 3.4 Stage contract

Every stage module exposes `run(spec: dict) -> None`, optionally `run(spec, context=None)`. The orchestrator calls `check_conforms()` immediately after import, so a mistyped or mis-signed `run` fails **before** the pipeline starts rather than hours in. A contract violation returns immediately without consuming retries, because retrying cannot fix a defect.

---

## 4. Bugs found, and the lesson

Two systematic bugs were found this session. Neither was found by writing code; both were found by verifying it.

### 4.1 The crawl gate could never pass

`agents/crawl.yaml`'s `min_battles_crawled` counted `DISTINCT battle_id` from `crawl_log`. But `battle_id` is a FK to `battles`, and `battles` has no rows until the **extract** stage. So during crawl it is necessarily NULL, the count is always 0, and as an ERROR gate with `pause_and_alert` it would have halted the pipeline after every crawl. Now counts `DISTINCT url`.

### 4.2 Every naval battle was labelled a land engagement

This is the important one.

`pipeline/extractors/infobox.py` mapped `{{Infobox military conflict}}` to a "land" variant that hardcoded `battle_type="field"`. But Wikipedia uses that generic template for **essentially every battle** — the naval-specific templates the code expected are rare. Live checks:

| Battle | Template | Extracted `battle_type` |
|---|---|---|
| Actium | Infobox military conflict | `field` |
| Trafalgar | Infobox military conflict | `field` |
| Midway | Infobox military conflict | `field` |

None carries a `ships=` or `vessels=` field to fall back on; Wikipedia puts ship counts in `strength` as free text.

`battle_type` is a covariate in `agents/model.yaml`, so the error fell precisely on the commanders whose ranking depends on it: Nelson, Yi Sun-sin, Themistocles, de Ruyter.

**This passed 73 unit tests.** The fixtures encoded the same wrong assumption as the code. It was caught only by running the parser against live Wikipedia.

> **Lesson for the next agent:** fixture-only testing validates that your code agrees with your assumptions. For a project whose entire output is a claim about the real world, periodically point the parsers at real data. It costs three HTTP requests.

Fixed: the generic template now leaves `battle_type` unset with a note, matching what the siege and rendered-HTML paths already did. **Consequence: nothing sets `battle_type` any more. `classify` must infer it** (from ship/fleet vocabulary, terrain, categories). Until then the covariate is always NULL.

---

## 5. Known constraints and footguns

### 5.1 BC dates cannot round-trip through Python

`datetime.date` has `MINYEAR == 1`. No BC date can be constructed, bound as a parameter, or **decoded from a result**. Postgres stores them fine.

```python
date(-30, 9, 2)   # ValueError: year must be in 1..9999
```

Any stage reading `battles.date_start` must filter to AD rows or project through `to_char`/`EXTRACT`. Insert BC dates as SQL literals: `DATE '0031-09-02 BC'`. Note 31 BC is astronomical year **-30**; there is no year zero.

Pinned by `tests/integration/test_db.py::test_bc_dates_cannot_round_trip_through_python`.

Given the corpus is heavily ancient, this likely needs a design decision — probably a separate integer year column. **Ask the user; do not decide unilaterally.**

### 5.2 Wikipedia rate-limits harder than the spec assumes

A smoke test of ~6 requests at 1.5s spacing drew **HTTP 429**. `agents/crawl.yaml` sets `rate_limit_wikipedia: 1.0`. Raise it, and add `Retry-After`-aware backoff — 429 is currently treated as a generic retryable failure. Fix before any real crawl.

### 5.3 Windows encoding

`alembic ... --sql` dies with `UnicodeEncodeError` because `config/schema.sql` uses box-drawing characters in its section headers and the console is cp1252. Set `PYTHONIOENCODING=utf-8`. The Makefile targets already do.

### 5.4 Shell heredocs are unreliable here

Writing long Python files via `cat <<'EOF'` failed mid-file more than once. Use the file-writing tool for substantial source files; keep Bash for inspection, patching via a short Python script, and git.

---

## 6. What is missing

Four config files are referenced by agent specs but **do not exist**. Each blocks its stage:

| File | Blocks |
|---|---|
| `config/model_default.yaml` | model |
| `config/imputation_priors.yaml` | impute |
| `config/expert_rankings.yaml` | evaluate |
| `config/sensitivity_specs.yaml` | evaluate |

Also outstanding: `README.md` (still a stub), `LICENSE`, CI config, pre-commit hooks, structlog configuration, and `tests/fixtures/gold/` (the hand-labelled set `/extraction-eval` scores against — see §8).

`pytest-asyncio` is declared in `pyproject.toml` dev extras but **not installed**, which is the `asyncio_mode` warning you will see. The crawl tests work around it with a local `asyncio.run` decorator. Installing it would let that go.

---

## 7. Next steps, in order

1. **Stand up Postgres and run the 28 skipped tests.** Nothing below is trustworthy until this passes.
2. **Phase 3 — resolve and classify.** `/build-all` Tasks 3.1 and 3.2. Give classify the `battle_type` inference job from §4.2. Its gate (`missing_data_all_logged`) is ERROR severity, so it must genuinely write `missing_data_log` rows.
3. **Phase 4 — reconcile, impute, model.** The highest-risk work in the project. **Use the `bayesian-model-reviewer` agent before committing any of it** (see §8). Write `config/model_default.yaml` and `config/imputation_priors.yaml` first.
4. **Phase 5 — evaluate and report.** `agents/report.yaml` already exists and is fairly prescriptive.
5. **Phase 6 — full-pipeline integration test, README, CI.**

Verification gates for phases 3–6 all require a database. Do not report a phase green on unit tests alone; say explicitly what was and was not verified.

---

## 8. Project-specific tooling

Four additions exist that a generic agent will not know about.

**Agents** (`.claude/agents/`):

- **`bayesian-model-reviewer`** — inference correctness for the PyMC stages: Bradley-Terry identifiability (skills are identified only up to an additive constant — needs a sum-to-zero constraint), centred vs non-centred parameterisation, ordered cutpoints, Rubin's rules pooling, imputation circularity, replacement-level uncertainty. **Use before committing Phase 4.** A mis-specified model still samples, still converges, and still emits a plausible ranking; nothing downstream will catch it.
- **`historiography-reviewer`** — domain plausibility of extracted data: ancient source inflation, troop-number scope conflation, command misattribution, anachronistic polities, BC dates, corpus selection bias. Reviews data, not code. It would have caught §4.2.

**Commands** (`.claude/commands/`):

- **`/data-audit`** — runs every spec's quality gates against the live database plus coverage-by-era/region. Distinct from `/status`, which is code health.
- **`/extraction-eval`** — prompt regression harness against hand-labelled battles. **The gold set does not exist yet**; building it is the real work, and it also unblocks `/test-stage extract`.

Both agents and both commands are unexercised. They are written, not proven.

---

## 9. Conventions worth restating

From `CLAUDE.md`, plus what was actually enforced:

- Type hints on every signature; Google-style docstrings on every public function.
- `structlog` only, never `print()` in pipeline code.
- f-strings only.
- SQLAlchemy **Core**, parameterised. Never interpolate into SQL.
- `mypy --strict` must pass. It currently does, across 37 files. Keep it that way — it caught several `Any` leaks.
- Comment on **why**, not what.
- Never commit without the user asking. Attribution lines are disabled globally.
- Check `.env` is not staged before every commit. It exists, is gitignored, and contains real keys.

---

## 10. Honest status

What is proven: the code lints, type-checks strictly, and 255 unit tests pass. The spec loader, override merging, quality-gate threshold parsing, and stage-conformance checking are exercised directly. Two real bugs were found and fixed.

What is not proven: **any interaction with a database**, any real LLM call, any real crawl at scale, and the whole statistical half of the project, which does not exist yet.

The naval-battle bug is the honest measure of where things stand. It was invisible to 73 passing tests and would have silently corrupted the published ranking for an entire class of commander. Treat "the tests pass" as a floor, not a finish line.
