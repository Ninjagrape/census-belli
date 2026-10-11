# Handover

The state of the project, the decisions that constrain future work, and the
traps that have already cost time. `CLAUDE.md` is the design; this is the state.

Keep it current and short. Update §1, §7 and §10 when the state changes. Record
a decision or footgun here only if a future session would otherwise get it
wrong; session narratives, changed-file lists and commit messages do not belong
here.

`docs/handover-log.md` is the full session-by-session history up to
2026-10-10. Code comments cite its section numbers ("handover-log §16.1").
Do not read it by default; open the cited section when a comment points there.

Last updated: 2026-10-11. Everything through the model stage is committed.
The gap review findings (`docs/todo/*.md`, tagged `(gap review)`) are unfixed.
**A real-corpus pilot (§11) found that the CLI pipeline cannot produce a
ranking today**; its findings are tagged `(pilot)` in `docs/todo/` and come
before everything else in §7.

---

## 1. Current state

| Stage | Implemented | Verified against a database | Real corpus (pilot, §11) |
|---|---|---|---|
| crawl | yes | **yes**, plus live against Wikipedia | runs; 211/211 articles, gates mis-measure |
| extract | yes | **yes** (deterministic path; LLM path fixture-only) | **broken via CLI**: writes 0 rows (P1, P2, P3) |
| resolve | yes | **yes**, incl. an end-to-end run and live Wikidata | **broken via CLI** (P1); with a context it runs, Hannibal/Caesar/Alexander unresolved |
| reconcile | yes, incl. the regime-neutral fit | on fixtures | runs, but **estimates collapse to the prior** (P4) |
| classify | first slice | on fixtures; LLM path offline/cache only | deterministic path runs; naval typing misfires |
| impute | yes | on fixtures | **fails** on the neutral scale (consequence of P4) |
| model | yes | on fixtures | never reached; 13 of 154 battles would enter |
| evaluate | **no** | — | — |
| report | **no** | — | — |

The baseline below was re-run on 2026-10-11 and matched exactly (959 passed,
0 skipped; model suite 30 passed, 1 xfailed).

**Verify before trusting this file:**

```bash
export DATABASE_URL="postgresql+psycopg://general_war:general_war@127.0.0.1:5432/general_war"
ruff check pipeline/ tests/ scripts/ alembic/   # expect: All checks passed
python -m mypy pipeline/                         # expect: Success, 112 source files (strict)
python -m pytest tests/ -q -m "not live and not model"   # expect: 959 passed, 0 skipped
python -m pytest tests/model -m model            # recovery suites; slow (~15 min)
```

Without `DATABASE_URL` the integration tests skip rather than fail, and a green
run with skips looks almost identical to one without. **Any skip under that
marker means you are not testing the half of the project that matters.** The
live tests (`test_crawl_live.py`, `test_llm_live.py`, `test_resolve_live.py`)
are gated on `GENERAL_WAR_LIVE_CRAWL=1` / `GENERAL_WAR_LIVE_LLM=1`.

The `asyncio_mode` warning is expected: `pytest-asyncio` is declared but not
installed, and the crawl tests drive their own event loop. Installing it would
change how those tests are collected.

---

## 2. Environment

- **Postgres 15.19, native** (winget). No Docker, podman or `make`.
  `host 127.0.0.1 port 5432 db/user/password general_war`. `psql` is under
  `C:\Program Files\PostgreSQL\15\bin\`, not on PATH. pip-installable Postgres
  (`pgserver`, `postgresql-wheel`) has no Windows wheel; do not retry it.
- **`.env`** is read by `pipeline.config.load_env()` (BOM sniffed: the file here
  is UTF-16). Exported variables win. The file holds only an unused Gemini
  key, so export `DATABASE_URL` yourself.
- **Python 3.14 locally, 3.11 in CI.** `pymc>=5.28,<6` is pinned because it
  installs on both; pymc 6 needs 3.12. Local anthropic is 0.111 although the
  project declares `>=1.0`; mypy is clean against both, runtime under 1.x is
  untested.
- **No C compiler**, so PyTensor's Python fallback is unusably slow. `nutpie`
  (numba) is the default `nuts_sampler`.
- CI installs only what is declared. Stub packages (`types-*`) are invisible at
  runtime and easy to forget. After adding a top-level directory, run
  `pip install -e . --dry-run` (package discovery is pinned to `pipeline*`).

---

## 3. Read before writing code

### 3.1 `pipeline/llm` is a package

There is no `pipeline/llm.py` and must not be.

```python
from pipeline.llm import LLMService
service = LLMService.from_spec(spec, db_conn=conn)
result = service.complete(system=..., user=..., json_schema=schema, metadata={"battle_id": bid})
if not result.ok:
    continue          # already logged and flagged for review
```

Retries, schema validation, cost accounting, and a row in `llm_calls` per call.
`llm_calls` is also the **resume cache keyed on request hash**, and the offline
path correlates on the same hash. Providers are routed per stage in the spec
(`params.llm_model`, `params.llm_mode`); see §9.4. In offline mode
`complete()` returns `CallStatus.DEFERRED`, which is never written to
`llm_calls` and is not a failure. `params.candidate_source: local` resolves
with no LLM and leaves ambiguous groups `unresolved`.

### 3.2 Quality gates

`pipeline/quality.py` executes each spec's `quality_checks`. Each check runs in
its own SAVEPOINT. When a gate fails, **check whether the spec is wrong before
assuming the code is**: several gates have measured tables their stage never
writes (see §4). `method:` checks dispatch through `METHOD_HANDLERS`
(`diagnostics_json`, `imputation_gate`, `model_gate`); an unregistered method
reports "not implemented" rather than passing. Outstanding, all evaluate's:
`brier_score`, `jaccard_overlap`, `held_out_accuracy`.

`pipeline/quality.py`'s `Connection` protocol has **two** test doubles:
`StubConnection` (tests/unit/test_quality.py) and `ScriptedConnection`
(tests/integration/test_orchestrator_gates.py). Change both.

### 3.3 Stage contract and retry policy

Every stage exposes `run(spec)` or `run(spec, context=None)`; the orchestrator
calls `check_conforms()` on import so a bad signature fails before the pipeline
starts. `retry_policy.on_failure: continue_with_logging` does not halt the
pipeline; `pause_and_alert` and `complete_with_warnings` do
(`StageResult.halts_pipeline`).

### 3.4 Schema changes go in two places

Every schema change needs an Alembic migration **and** the same change in
`config/schema.sql`: integration fixtures build the database with
`apply_schema()`, which never runs Alembic. Guard the migration with
`IF NOT EXISTS` so both paths converge. `apply_schema()` and the baseline
migration use a raw driver cursor because psycopg reads the `%` in schema
comments as a placeholder, even through `exec_driver_sql`.

---

## 4. The lesson

Almost every serious defect in this project was invisible to a green suite and
found by pointing code at something real: live Wikipedia, live Wikidata, a real
Postgres, or a number a reader thought looked wrong. Treat "the tests pass" as a
floor. Recurring classes, with the log section that records each:

- **Fixtures encoding the code's own assumption.** Every naval battle labelled
  `field` passed 73 unit tests (log §4.2); a BC-year test asserted the wrong
  convention (§4.5); the circularity comparator passed its integration test with
  a fake force ratio in it (§23.2).
- **Gates whose SQL measures something the stage does not write**, or divides a
  number by itself: log §4.1, §4.6, §11.4, gap review A2.
- **robots.txt applied to an API.** `query.wikidata.org` disallows `/sparql`,
  and routing SPARQL through `Fetcher.fetch` silently disabled every query in
  the project (log §4.7). `www.wikidata.org/w/api.php` is also disallowed. API
  calls go through `fetch_raw`; ask before wrapping any new one in `fetch`.
- **Identifiers recalled from memory are wrong.** Q-ids (Nelson, Agrippa, Red
  Cliffs, Hannibal) and article titles have each been wrong when recalled.
  Look them up live.
- **Exists is not correct.** Disambiguation pages and redirects to campaign
  articles return 200 and pass title checks (log §18.3, §18.9).
- **Year-shaped non-years.** Arsht's `-5000` sentinel, Wikidata dates at
  precision below 9 (decade, century), and Wikidata's signed years (`+1850-...`;
  a fixture writing `1850-01-01` is rejected).
- **Clean diagnostics, prior-only posterior.** A parameter the data cannot
  identify samples at rhat 1.00 and returns its prior (log §19.8-19.9). Check
  prior-to-posterior contraction, not just rhat. The pilot's reconcile did this
  for every side at once (§11, P4).
- **Tests that hand the stage what the CLI does not.** Integration tests pass a
  `StageContext` with a connection; the orchestrator passes none, so extract and
  resolve wrote nothing from the CLI while every test was green (§11, P1).
  Verify a stage through `python -m pipeline.orchestrator`, not only pytest.

---

## 5. Footguns

- **BC dates cannot round-trip through Python** (`datetime.date` MINYEAR is 1).
  Read `battles.year_astronomical`, a generated column: 31 BC is -30 there and
  -31 from a bare `EXTRACT(YEAR ...)`. Only astronomical numbering subtracts
  correctly across 1 BC/AD. Wikidata and Arsht both write historical years, so
  convert on the way in and join on `year_astronomical`. A battle known only by
  year is written `YYYY-01-01` with `date_precision='year'`.
- **One integration suite at a time against `DATABASE_URL`.** Every module
  fixture drops and rebuilds `public`. Overlapping runs hang silently or throw
  `UndefinedTable`; check `pg_stat_activity` / `pg_blocking_pids()` before
  suspecting the code. Load any real corpus into a separate database
  (e.g. `general_war_corpus`). `python -m tests.integration.inspector_seed`
  also drops `DATABASE_URL`.
- **Module-scoped fixtures that run a stage need `try/finally` rollback**, or an
  exception leaks a connection holding row locks and later tests wait forever.
- **PyMC:** never name a variable like a dimension (ArviZ returns the
  dimension's labels); pass every per-observation array as `pm.Data`, because
  numpy constants are compiled into nutpie's code and compile time explodes
  with corpus size.
- `yaml.safe_load` turns `2026-09-24` into a `date`; convert before writing
  JSON.
- `logging_config.configure()` is process-global; a test that calls it needs
  the restore fixture in tests/unit/test_infrastructure.py.
- `alembic ... --sql` needs `PYTHONIOENCODING=utf-8` on Windows (box-drawing
  characters in schema.sql).
- Shell heredocs mangle backslashes and have failed mid-file. Use the
  file-writing tool for source files and Windows paths.
- Workers: tell them to format only files they own (one ran `ruff format`
  across the tree). A worker cut off by a usage limit may have finished more or
  less than its last line says; check `git status` and the brief's items.
- `claude-haiku-5-5` thinks unless told not to (handled by the
  `thinking_on_by_default` trait) and cannot take temperature 0.

---

## 6. What is missing

- `config/expert_rankings.yaml` and `config/sensitivity_specs.yaml`, referenced
  by `agents/evaluate.yaml`.
- The extraction gold set (`tests/fixtures/gold/` for `/extraction-eval`).
- Everything in `docs/todo/`.

---

## 7. Next steps, in order

0. **Fix the pilot findings P1-P4 (§11)**, in that order, each with a test that
   fails first. P1 to P3 are mechanical; P4 is a statistical diagnosis (Sonnet,
   then `bayesian-model-reviewer`). Then the gap review items (A1, A2, A4, A5,
   A8, A10, A13) and the rest of §11.
1. **Re-run the curated corpus** (`config/sources_seed_arsht.yaml`) into
   `general_war_corpus` through the CLI, not a driver. `data/raw/` from the
   pilot can be reused (crawl takes ~2h); drop the extract batch files first.
2. **Classify, second slice.** Run classify (offline: it writes the batch file
   and halts), `/process-llm-batch <file>`, re-run classify; populate `is_defender`; LLM missingness for
   `unclassified` rows. Until this runs, every multi-commander side is
   `default_split`, credits nobody, and `exclusion_concentration` will fire.
3. **Reconcile and impute open items** (`docs/todo/04-reconcile.md`,
   `06-impute.md`), above all propagating the shared reconcile error into impute.
4. **Evaluate and report.** Fix the two BLOCKING report items first
   (`docs/todo/09-report.md`).
5. Full-pipeline integration test.

**LLM throughput**: the pilot measured 22,591 unique extract requests for 211
battles (~107 per battle), so the earlier full-corpus estimate of 6,000-12,000
is far too low. Before answering any, check why passages are so many (1,652
were flagged for review) and whether passage size or filtering should change.
The offline path has never run on a real request file; pilot it on ~10
battles' requests and measure requests per session.

Say explicitly what was and was not verified against a database.

---

## 8. Project tooling

- **`bayesian-model-reviewer`** agent: gate after any change to reconcile,
  impute, model or WAR pooling. A mis-specified model still samples and still
  ranks; nothing downstream catches it.
- **`historiography-reviewer`** agent: domain plausibility of extracted data.
  Reviews data, not code.
- **`/data-audit`**: every spec's gates against the live DB plus coverage by
  era/region. **`/extraction-eval`**: prompt regression harness, waiting on the
  gold set. **`/resolve-eval`**: threshold sweep, zero LLM quota.
  **`/process-llm-batch`**: answers a deferred-request file with
  `llm-batch-worker` subagents (`.claude/agents/`, Read and Write only).
- **Stage inspector**: `streamlit run scripts/inspector_app.py`, read-only by
  construction (every connection sets `default_transaction_read_only`), safe to
  point at a corpus database. DB from `INSPECT_DATABASE_URL`, else
  `DATABASE_URL`. The Model page is still a placeholder (gap review A16).

---

## 9. Standing design decisions

What a future session needs before changing a stage. The reasoning behind each
is in the log section cited.

### 9.1 Crawl

- `Fetcher` honours `Retry-After` (both RFC 9110 forms); a server delay raises
  the wait, never lowers it, capped at 120s. `rate_limit_wikipedia` is 2.0
  (1.5 drew 429s). The live crawl test fails if any fetch needs a retry.
- Named corpora come from `wikipedia_battle_articles:` in the seed file. An
  evaluation corpus names no list pages or SPARQL queries, so its membership
  cannot drift with Wikipedia edits.

### 9.2 Extract

- **Log a missing field only if a downstream stage consumes it** (log §11.3).
  `battle_type`, `terrain`, `fortified` are logged; `weather` is not.
  `fortified=False` is an answer, an empty list is an absence.
- The generic `{{Infobox military conflict}}` template asserts no `battle_type`;
  classify infers it.
- DBpedia contributes no sides: it flattens combatants and commanders into
  unordered lists, and a commander on the wrong side is an error the model
  cannot see. They go to notes, "unassigned to sides". Infobox wikitext is the
  only source of side assignment.

### 9.3 Resolve

- **Prefer a split to a merge.** A false merge pools two careers invisibly; a
  false split is visible. A link must clear the fuzzy threshold, be
  date-compatible, and beat the runner-up by a margin; `UNCERTAIN` from the LLM
  splits.
- **The lifespan date gate does the real work.** It abstains when either side
  is undated. Among exact matches, date-confirmed candidates beat dateless
  namesakes. A half-dated candidate is bounded by `MAX_PLAUSIBLE_AGE_YEARS`
  (100), or an ancient with no death date is eligible for every later battle.
  Polity is a tiebreak only.
- "Unknown" never becomes a general; a failed LLM call leaves the group
  unresolved, not new; grouping is exact-key only, and spellings reunite later
  through their shared Q-id.
- Resolve upserts `battle_commanders` and sets only the columns it owns, so a
  re-run never undoes classify.
- **Duplicate battles are reported, never merged**
  (`data/processed/duplicate_battles.jsonl`). Merging needs its own evidence and
  review step.
- Lookups ask in `en` and `mul` (Wikidata person labels are migrating to `mul`;
  Nelson has no `en` label). The label service returns a bare Q-id when it finds
  no label; `_preferred_name` and `_publishable_name` stop that being published
  as a name.
- The SPARQL query aggregates with `GROUP_CONCAT`; the multi-valued OPTIONAL
  cross-product made it 15x slower.
- **Label search (`wbsearchentities`) is deferred on purpose**: its hit is
  usually the sole candidate, which skips the ambiguity margin and auto-links at
  ~0.62 confidence where evidence is weakest, and live probes are noisy. Measure
  the residual class with a gold set first.

### 9.4 LLM routing (offline-first)

- **Gemini is removed** (2026-10-11). Every LLM stage names an Anthropic model:
  `claude-haiku-5-5` for extract and resolve, `claude-sonnet-5-5` for classify.
  Haiku 5.5 and Sonnet 5.x reject temperature, so no spec sets one.
- `params.llm_mode: offline` (default) builds `OfflineProvider`: cache hits are
  served from `llm_calls`, misses return `DEFERRED` and are flushed to
  `data/llm_requests/<stage>_<ts>.jsonl` at the end of the stage. The stage
  returns a `StageOutcome`; the orchestrator skips that stage's quality gates
  (they measure completeness of data still waiting) and **halts the pipeline**,
  CLI exit code 3. Deterministic outputs are still written.
- Extract with deferred passages: battle `needs_review` with a "pending" note,
  no `missing_data_log` rows, nothing in `llm_calls`. A re-run after import
  equals a run that had the answers first time (integration test
  `test_offline_extract.py`). Resolve leaves deferred groups `unresolved`
  (method `llm_deferred`).
- `/process-llm-batch`: `scripts/llm_offline.py split` (compact chunks, system
  prompt and schema once per chunk, `--skip-answered` for resume after a usage
  limit), one `llm-batch-worker` per chunk on the chunk's `agent_model`,
  `validate`, `import --dir`. Import validates every answer against its
  request's schema and records provider `claude_code`.
- **The hash names the spec's provider and model**, not who answered, so an
  offline answer and an `llm_mode: api` call share the cache. Changing
  `llm_model` invalidates every cached answer for that stage (classify's move
  from sonnet-4-6 to sonnet-5-5 did this; nothing real was cached).
- Tests: `tests/conftest.py` redirects `REQUEST_DIR` to a temp dir, or offline
  runs write batch files into the real `data/`.
- Never cram many battles into one prompt: it breaks the per-request cache and
  the per-passage schema. Chunking groups independent requests, it does not
  merge them.

### 9.5 The Arsht reference set

`tests/fixtures/gold/arsht/` and `config/sources_seed_arsht.yaml`, rebuilt by
`python -m scripts.arsht_reference --verify-titles --with-infobox --write-seed`.
22 generals, 211 battles, all dated.

- **A reference set, not a gold set.** His numbers are one model's output on one
  corpus. Use them for roster coverage, a correlation target, and hygiene
  findings, never as facts. Napoleon's total matches the article (16.678 vs
  16.679); other generals differ by up to 0.4, so the article and the published
  pages are two runs.
- `arsht_infobox.jsonl` (raw infobox strength text beside his parsed numbers)
  is the one thing extract can be diffed against without hand-labelling.
- Disambiguation titles are repointed by a hand-made table
  (`_DISAMBIGUATION_TARGETS`), deliberately not a nearest-year heuristic, and
  the seed writer refuses any still-flagged title.

### 9.6 Reconcile

- **Bias belongs to the claim, not the document.** One infobox field often
  carries both modern and ancient figures, so the inflation term is keyed on a
  claim regime parsed from the report's context. Only ~2.2% of real strength
  strings carry a readable marker, so the era fallback does ~98% of the work.
  A multi-regime field is `unlabelled` unless the report's own value picks the
  nearest marker; `modern_scholarly` is the zero-bias anchor.
- The inflation prior is fitted to a 45-case gold set
  (`tests/fixtures/gold/reconcile/`, agent-built, unreviewed) by
  `scripts/fit_inflation_prior.py`: pooled ~3.2x, against the spec's original
  1.35x. The era contrast is fitted too (it is negative).
- **In an all-unlabelled corpus `m0` is not identified**; the inflation
  adjustment is entirely prior. Assert on `m0 + offset_bar`, never `m0`.
  `_UNLABELLED_WEIGHT = 0.5` (~2.5x deflation of every unmarked figure) is a
  judgement call and the most consequential constant in the stage.
- **Copies add no information**: `merge_cross_regime_duplicates` then
  `collapse_lineages` keep one row per lineage before the fit. Zeroing singleton
  lineage effects instead was tried and made singletons overconfident.
- **Regime-neutral troops fit** (for force ratios): a second exact fit where the
  unlabelled offset takes the faithful-component prior and `a_era` takes
  `N(0, 0.25)`; labelled regimes keep their priors. It exists because
  asymmetric labelling (Gaugamela) tilts published ratios. Columns
  `est_troops_neutral*`, draws in
  `data/processed/reconcile/neutral_draws_run{id}.npz`; no fallback to published
  ratios. Published counts stay adjusted.
- One `model_runs` row per run, covering all quantities. Estimates are written
  only when the fit meets the write thresholds; an unconverged run is recorded
  but leaves the previous estimates.
- `est_troops_total = exp(median(mu))`, not `mean(exp(mu))`. A one-report side's
  95% interval spans roughly 0.24x-4.3x and must be allowed to. The width gates
  currently measure the shared prior offset (threshold 4.5); do not narrow
  intervals to pass a gate.
- The absolute level is set by `b_type`'s zero-sum, so adding a source type
  shifts every absolute figure. It cancels in ratios.
- One-sided-bound sides are labelled `bound_only`. Pure censoring is too weak
  for them (the recovery ablation is an xfail); `pm.ExGaussian` is the
  principled replacement.

### 9.7 Classify

- One commander: `field_commander`, weight 1.0. Infobox commanders all arrive as
  role `unknown`, so the fallback is the main path: multi-commander sides get
  `default_split` at 1/n and wait for the LLM. The model credits nobody on a
  `default_split` side.
- With roles known: 2+ field commanders are co-commanders; non-field roles get
  fixed weights (sovereign/nominal 0.05, supreme/theatre 0.10, subordinate 0.08,
  capped at 0.30 in total). An LLM answer that does not match every commander on
  the side is rejected whole. A side whose rows are all `llm` is never
  recomputed by a re-run.
- `battle_type`: naval from categories, naval troop branches or ship vocabulary
  (bare "squadron" is not enough); siege from name or category; `field` only
  when an article was read with no other signal, otherwise `unknown`. Sieges
  are written `siege_offensive` + `fortified = TRUE` until a per-side siege flag
  exists.
- Missing covariates are classified **MAR**, not MCAR: missingness plausibly
  tracks documentation level.

### 9.8 Impute

`agents/impute.yaml` and `config/imputation_priors.yaml` carry the rationale.

- One joint PyMC model replaces MICE/PMM (one substantively missing variable).
  Per-battle random effect and noise are marginalised analytically.
- Observed sides enter as reconcile's posterior with reconcile's side prior
  divided out, so nothing is shrunk twice; a side too weak after division is
  treated as missing.
- **Overimputation**: every side, observed included, is drawn per round, or
  errors-in-variables attenuates `beta_force` downstream. Outcome is
  conditioned on; commander identity enters only as a size effect.
- `scale: neutral` is primary and the only scale that writes
  `missing_data_log`; `published` is a sensitivity run. Primary is MAR;
  `mnar_delta` serves evaluate's sensitivity grid. Stale draws files are refused.
- Imputing troops as latents inside the Bradley-Terry model is the fully
  congenial alternative; it departs from the spec and is the user's call.

### 9.9 Model

`agents/model.yaml` and `config/model_default.yaml` carry the rationale.

- Two-sided battles only (3+ excluded and counted). Side A is the lower
  `side_id`, so **orientation must carry no information**: ordinal logit with
  symmetric cutpoints, and when the two sides' grades disagree the outcome is
  interval-censored. Battle-level terms (`beta_type`, era, terrain) are dropped
  because under arbitrary orientation they can only learn "side A wins"; no
  `mu_era`, because eras never meet.
- Sides with no creditable commander are latents (`mu_unknown`,
  `mu_unattributed`), not dropped battles; dropping them removed most
  Napoleonic and modern battles.
- **WAR is expected WAR**: expected score at fitted skill minus at replacement
  (40th percentile per draw), split by attribution weight. Realised WAR is kept
  in diagnostics only. `win_prob_*` columns hold expected score, not P(win).
- One fit per imputed round; mixture pooling; stored sd is Rubin's `sqrt(T)`.
  Every round must converge (one retry) or nothing is written. The pooled
  `bradley_terry_hierarchical` row is written last.
- The circularity comparator is a joint latent fit on per-battle `x_diff`, so
  absolute era levels cannot leak in as fake force ratios.
- **Disclosures the ranking must carry**: cross-era comparison rests on the
  shared skill prior (the comparison graph is disconnected); technology is not
  modelled; naval and aerial share one population and slope; `total_war` is
  volume-weighted, so a coverage gap by era is a rank gap; attribution weights
  are fixed data.

---

## 10. Honest status

Built and verified against a real Postgres: crawl, extract, resolve, and
reconcile, classify, impute and model on fixtures. The recovery suites pass on
synthetic data. The real-corpus pilot (§11) showed that "verified against a
database" meant "verified when a test passes the connection in": through the
CLI, extract and resolve write nothing, and reconcile's real-data estimates
ignore the reports.

Not proven: any real corpus producing a ranking, the offline batch path on a real
request file (`/process-llm-batch` has never run), any billed API call
(refusal and truncation branches untested), CI (no git remote), and every gate threshold,
which has only run against empty or fixture databases, apart from the pilot's
(§11). The gold sets are agent-built and unreviewed by a human. Evaluate and
report do not exist, so there is no ranking.

---

## 11. Real-corpus pilot, 2026-10-11

The 211-battle Arsht seed run stage by stage via the CLI into database
`general_war_corpus` (left in place, with `data/raw`, `data/processed`,
`data/imputed` and three 204 MB extract batch files in `data/llm_requests`).
No LLM request was answered. Where the CLI wrote nothing, a scratch driver
passing `StageContext(db_conn=conn)` and wrapping each extract write in a
savepoint was used to reach later stages; nothing from it is in the repo.
P1-P4 were re-checked against the code and the database by the lead; the rest
are the worker's observations, with evidence, not yet re-checked.

**Blocking**

- **P1. The orchestrator never passes a `StageContext`.** `run_stage` calls
  `runner_module.run(spec)` (`orchestrator.py:169`). Extract logs
  `extract_database_write_skipped` and writes 0 rows; resolve spends ~6 min on
  SPARQL and writes 0 generals. Crawl, reconcile, classify, impute and model
  open their own connection and are unaffected. Extract's LLM service also has
  no connection, so no `llm_calls` audit trail.
- **P2. One bad row aborts all of extract's writes.** `write_battle` shares one
  transaction with no savepoint (`extract.py:686-693`): the first
  `InvalidDatetimeFormat` turns the remaining 207 battles into
  `InFailedSqlTransaction` and the commit rolls back. The comment there claims
  the opposite. Check resolve and classify write loops for the same pattern.
- **P3. Unparsed date text reaches `date_start`.** 56 of 211 battles (27%) fail
  to insert, mostly ancient or multi-day: `"212 BC"`, `"Spring 208 BC"`,
  `"21–22 May 1809"`, `"5–6 June 1967 ( 1967-06-05 – 1967-06-06 )"`,
  `"15 November [ O.S. 3 November ] 1812 – ..."`, `"January 635 AD (13 AH)"`.
  58 records have `date_precision` None. Losing them biases the corpus further
  toward the modern era (already 121 of 155 written battles are 17th-20th c.).
- **P4. Reconcile's real-data estimates ignore the reports.** Converged fit
  (5000 draws; the default 1000 missed ESS 400 at 359.6), yet all 275 sides lie
  in 33,798-37,171 (log-sd 0.0117): Kursk Soviet max report 2,500,000 gives
  36,947; Actium Octavian (report 3,000) 35,069; Gaugamela Achaemenid
  (1,000,000) 35,440. Corr(ln est, ln max report) 0.31. Every source is
  `wikipedia_infobox` and 1728 of 1755 lineages are `regime_unlabelled`. The
  per-side posterior sd (~0.26) is wide while the medians are near-identical,
  which suggests side effects are shrunk to the corpus level or the likelihood
  is mis-indexed; undiagnosed. Impute then finds 275 weak sides and 0 battles
  to condition on (`impute_no_observed_sides`), so the model never runs.

**Wrong output**

- Extract stores sub-components as `branch='total'`: Cannae Rome has `Total:
  86,400` plus "25,000 Romans", "30,000 allies" as separate totals; Carthage
  likewise. Ranges become a lower and an upper bound row.
- Resolve: Hannibal (25 mentions), Julius Caesar (9) and Alexander (4) are
  `llm_deferred`, so Cannae and Gaugamela credit Hasdrubal or Parmenion. 13
  well-known people got "no candidate returned" (Schwarzenberg, MacDonald,
  Grouchy, Platov, Amr ibn al-As); Blücher failed the date gate. Luftwaffe,
  Stavka and Army Group B became generals. Austerlitz's "Russian Empire" side
  holds the Austrian commanders. Side polity strings like `"Octavian 's
  forces"` (stray space).
- Classify typed Marathon, Malvern Hill, Yom Kippur War and Operation Torch as
  naval; `field` is never produced (133 of 155 `unknown`).
- The seed includes wars, not battles ("1948 Arab–Israeli War", "Yom Kippur
  War"). Trafalgar is not in it.
- Model design on this data: 154 battles in scope, 108 with no outcome, 33
  with no creditable side, **13 included, 13 generals**. 216 of 308 side
  outcomes are blank, so outcome extraction is the next coverage limit after
  attribution.

**Gates and orchestration**

- Crawl `min_battles_crawled` counts every 200 URL including citations (1986
  here), so it measures citations, not battles. `wikidata_coverage` 0: a named
  corpus fetches no Wikidata entities, `data/raw/wikidata` is empty.
  Citation error rate 0.21 (doi.org 403 x114, 328 read timeouts).
- The orchestrator's stage retry reruns a fixed-seed sampler, so a
  non-converged reconcile fails identically twice and doubles the runtime.
- Reconcile `unfillable_sides_are_logged` 16 while the stage logged 3
  unfillable (naval-excluded reports likely); `ci_width_reasonable` returns
  NULL. Classify `missing_data_all_logged` 30 (error). Impute's gates read
  `data/imputed/neutral/gate_report.json` even under `scale=published`; model
  rejects `scale=published`.
- Extract: 11,337 `citation_line_malformed` warnings (cause not found); every
  re-run writes a new full batch file with identical hashes; 35,277 deferred
  calls dedupe to 22,591 requests, which is the real extract LLM bill for 211
  battles (~107 per battle, far above §7's estimate).

**Runtimes**: crawl 7,034 s; extract 128 s; resolve 353 s cold, 140 s with the
SPARQL cache; reconcile 206 s per attempt at defaults, ~310 s at 5000 draws;
impute (published) 82 s per attempt.
