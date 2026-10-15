# Handover

The state of the project, the decisions that constrain future work, and the
traps that have already cost time. `CLAUDE.md` is the design; this is the state.

Keep it current and short. Update §1, §7 and §10 when the state changes. Record
a decision or footgun here only if a future session would otherwise get it
wrong; session narratives, changed-file lists and commit messages do not belong
here.

`docs/handover-log.md` is the full session-by-session history up to
2026-10-12 (§25 is the real-corpus pilot, §26 the sweep that fixed it). Code comments cite its section numbers ("handover-log §16.1").
Do not read it by default; open the cited section when a comment points there.

Last updated: 2026-10-15 (documentation audit). `main` is committed through
6c91f1a (report stage). The real corpus
passes crawl through evaluate on the CLI (model on 59 of 210 battles until
classify's LLM slice runs).

---

## 1. Current state

| Stage | Implemented | Verified against a database | Real corpus via CLI (2026-10-12, `general_war_corpus2`) |
|---|---|---|---|
| crawl | yes | **yes**, plus live against Wikipedia | **passes all gates**; 211/211 articles, Wikidata coverage 1.0 |
| extract | yes | **yes** (deterministic path; LLM path fixture-only) | **passes**; 236 battles, 272/420 side outcomes; 14,944 LLM requests deferred |
| resolve | yes | **yes**, incl. live Wikidata | **passes**; 678 linked, 189 unresolved, 146 LLM requests deferred |
| reconcile | yes, incl. the regime-neutral fit | **yes** (integration suite) | **passes all error gates** (run 7): infobox sigma 0.52, spread_ratio 0.60, singleton_slope 0.54, 6 low-precision sides widened; only `calibration_set_is_not_empty` warns |
| classify | first slice | on fixtures; LLM path offline/cache only | **passes**, 254 LLM requests deferred (exit 3) |
| impute | yes | **yes** (integration suite) | **passes** at 2000 draws / 1500 tune (max rhat 1.0092 vs 1.01: thin margin) |
| model | yes | on fixtures | **passes** at 3000 draws (run 54: rhat 1.004, ESS 1445); 59 of 210 battles included; 4 warning gates (§7 step 1) |
| evaluate | yes | **yes** (integration 12, model CV 4) | **runs, 4 of 5 gates pass** (2026-10-14, model run 54): held-out accuracy 0.70, Brier 0.222; named_checks not_evaluable |
| report | yes (6c91f1a) | **yes** (integration 10, seeded run, no fit) | **runs, all gates pass except `exclusion_concentration`** (by design: output marked provisional, exit 1); 3 headline generals, 58 below `min_battles` |

Counts on `main` at 6c91f1a (2026-10-14): non-model suite 1645 passed, 0
skipped, ~17.5 min. Model suite per file (2026-10-13): impute_recovery 5, model_recovery
6, reconcile_contamination 3, reconcile_neutral 5, reconcile_recovery 17 + 1
xfailed (~11 min alone). Run the model files one per command; a single
`tests/model` run exceeds a 10-minute tool timeout.

**Verify before trusting this file:**

```bash
export DATABASE_URL="postgresql+psycopg://general_war:general_war@127.0.0.1:5432/general_war"
ruff check pipeline/ tests/ scripts/ alembic/   # expect: All checks passed
python -m mypy pipeline/                         # expect: Success, 147 source files (strict)
python -m pytest tests/ -q -m "not live and not model"   # expect: 0 failed, 0 skipped (~1645 passed)
PYTHONIOENCODING=utf-8 alembic heads             # expect: one head, a7b8c9d0e1f2
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
(`diagnostics_json`, `imputation_gate`, `model_gate`, `evaluation_gate`); an
unregistered method reports "not implemented" rather than passing.
`evaluation_gate` reads `evaluation_report.json` and fails as stale when
the report's `model_run_id` is not the newest converged pooled run.

`pipeline/quality.py`'s `Connection` protocol has **two** test doubles:
`StubConnection` (tests/unit/test_quality.py) and `ScriptedConnection`
(tests/integration/test_orchestrator_gates.py). Change both.

### 3.3 Stage contract and retry policy

Every stage exposes `run(spec)` or `run(spec, context=None)`; the orchestrator
calls `check_conforms()` on import so a bad signature fails before the pipeline
starts. `retry_policy.on_failure: continue_with_logging` does not halt the
pipeline; `pause_and_alert` and `complete_with_warnings` do
(`StageResult.halts_pipeline`).

`--continue-past-deferred` runs later stages on deterministic data after an
offline LLM deferral (exit code 3 at the end; the result is provisional). A
gate failure retries only with `retry_policy.retry_on_gate_failure: true`
(crawl only); reconcile, impute and model have `max_stage_retries: 0` (fixed
seed; model retries non-convergence itself, reconcile and impute do not).
`--stages all` skips unbuilt stages. Gate thresholds may say
`{params.<name>}`; a missing param fails the check. `imputation_gate` reads
the run's `params.scale`.

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
  for every side at once (log §25, P4).
- **Tests that hand the stage what the CLI does not.** Integration tests pass a
  `StageContext` with a connection; the orchestrator passes none, so extract and
  resolve wrote nothing from the CLI while every test was green (log §25, P1).
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
- **CI installs from `requirements/ci-lock.txt`, not `pyproject.toml`.** A
  fixed seed is bit-reproducible on one dependency set (verified 2026-10-13,
  in and across processes), so a locked CI result depends on the code alone.
  A new dependency goes in `pyproject.toml` and the lock is regenerated (command
  in its header) in its own commit, or CI will not have it.
- **Model tests retry convergence, never claims.** `fit_until_converged` in
  `tests/model/conftest.py` refits on `seed + k * RETRY_SEED_OFFSET` (3
  attempts, as `pipeline/modelers/fitting.py` does) only when a convergence
  gate is missed; a coverage or recovery assertion is never retried. A new
  model test whose fit must converge uses it instead of asserting a gate on
  one fit.
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
- **Run model fits in the foreground.** A backgrounded PyMC run stalled on
  Windows; a foreground command over 10 minutes is moved to the background by
  the tool, so run `tests/model` one file per command.
- **Workers editing `agents/*.yaml` during a test run break that run**: stages
  read their spec at runtime. Never overlap a code-editing worker with a
  verification run on the same tree.
- **Parallel workers** each get their own database (`general_war_wt_*`).
  Three workers once picked the same Alembic revision id, so check
  `alembic heads` after any merge.
- Auto-mode's permission classifier may block a subagent from connecting to
  `general_war_corpus2`; ask the user rather than routing the work through
  another agent.
- `claude-haiku-5-5` thinks unless told not to (handled by the
  `thinking_on_by_default` trait) and cannot take temperature 0.

---

## 6. What is missing

- The extraction gold set (`tests/fixtures/gold/` for `/extraction-eval`).
- Everything in `docs/todo/`.

---

## 7. Next steps, in order

**Do not lower `min_spread_ratio`** (§9.6).

1. **Model stage on the corpus passes its error gates** (2026-10-13, 3000
   draws in `config/model_default.yaml`, was 1500: ESS 660 < 800). Warnings to
   work through: `circularity_check` (joint latent beta_force gap 0.276 vs
   limit 0.224; skill Spearman 1.000, top-50 overlap 46), `top_generals_cross_component`
   (46 of the top 50 outside the largest component, so their level rests on
   the shared prior), `exclusion_concentration`, `exclusions_reported`.
   The corpus model includes only 59 of 210 battles: 77 have no
   creditable side (every multi-commander side is `default_split` until
   classify's LLM slice runs) and 74 no outcome. Treat any corpus ranking as a
   mechanics check, not a result, until step 2 is done.
2. **Answer the deferred LLM batches** (`data/llm_requests/extract_20261012T095006Z.jsonl`,
   14,944 requests; `resolve_20261012T095357Z.jsonl`, 146) with
   `/process-llm-batch`, starting with resolve (small, and it unblocks Hannibal,
   Caesar, Alexander). Pilot extract on ~10 battles first and measure requests
   per session. 19,790 of the remaining extract passages are `web_secondary`
   citations (~94 per battle); whether to keep them is an open cost decision.
   Then classify's LLM slice (roles; until it runs every multi-commander side is
   `default_split` and credits nobody).
3. **Reconcile and impute open items** (`docs/todo/04-reconcile.md`,
   `06-impute.md`), above all propagating the shared reconcile error into impute.
   From 2026-10-13: the 6 low-precision troop sides (59, 176, 287, 371, 373,
   411) are the implausible figures ("2.3M", "3.8M", "754", "300", campaign
   totals), so the MCSE flag is acting as a scope-error detector; audit them
   with side 419 (battle 236, four ranges and three points of different scopes)
   and sides 217, 274, 420 (campaign totals above 1M) at extract. Six-Day War,
   Yom Kippur War and the 1982 Lebanon War are wars in the battle corpus.
   Casualties: re-measure pi_contam under `contaminated` now that ranges merge
   (0.41 before) before reading it as killed/wounded/captured scope variety;
   a one-sided scope component and a casualty-scope design column are the
   candidates. Singletons now follow their own report (slope 0.54), so scope
   errors on the 87% singleton sides are not detectable by the model. Impute
   passes rhat 1.0092 against 1.01 on hierarchical scales `tau_p`, `tau_c`;
   another corpus may need more draws.
4. **Report**: committed (6c91f1a). The model-side WAR interval and
   P(adjacent order) (§9.11) are built on `wip/outstanding`, not yet on
   `main`. A real ranking needs step 2 first. Evaluate open items: the calibration
   gate sits on the imputed CV arm (§9.10); the +/-1 sd `g_unlabelled`
   sensitivity is on `wip/outstanding` (d91dcbc).
5. Full-pipeline integration test on fixtures (crawl through model).

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
  `DATABASE_URL`.

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
- Gates read a marker row
  `crawl-plan://battle-articles?targets=..&fetched=..&entities=..&floor=..`
  (http_status 0) written to crawl_log at the end of each run; error-rate SQL
  excludes it. Named corpora fetch each article's Wikidata entity via
  `Special:EntityData` (robots allows it), Q-id read from the article HTML.

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
- Arm-of-service figures get their branch, contingents/nationalities `other`;
  a side with only components gets one derived total marked `is_lower_bound`,
  `page_or_section='infobox:derived_total'`. Victor matching uses demonym
  stems plus a small table, still exactly-one-side.
- Battles are keyed by `battles.source_slug` (the crawl filename stem), so
  same-name battles stay apart. Offline batch files with identical hash sets
  are reused.

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
- Non-person filter by name pattern and P31; title/particle stripping and a
  search fallback for zero-candidate names (fixture-tested only, not live).
  Generals get born/died and `*_year_astronomical` columns.

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
- **A stated range is one point, not two bounds** (2026-10-13). Loading
  "A–B" as a lower and an upper censored row let sigma grow for free (each
  censored probability tends to 0.5 as sigma grows, with no -log sigma term)
  and collapsed every singleton on the corpus. `merge_range_pairs` (claims.py)
  runs **after** lineage collapse (before it, a Wikidata copy of B survived as
  an independent point) and merges bounds sharing side, source key, quantity,
  branch, casualty type, scope and context into one row at sqrt(A·B) with
  `range_var = (log(B/A)/4)^2` added to its variance (the range read as a
  central 95% interval; a uniform reading would be /sqrt(12)). A == B becomes
  a plain point. Lone bounds stay censored.
- **Default likelihood is `student_t`.** The `contaminated` family (two-Normal
  mixture, pi ~ Beta(1, 9), sd 2.0) is kept as an ablation only: on the corpus
  pi_contam was 0.02 for troops and student_t gave the better singleton slope.
- **Write gate**: global variables need rhat <= 1.05, bulk ESS >= 400, 0
  divergences, and side variables rhat <= 1.05; failing any of those withholds
  the whole fit. Side quantiles are also checked for quantile MCSE (<= 5% of
  CI width and <= 0.05 log at the median); a side failing only that is
  **written with each bound widened by 2 x its MCSE**, listed in diagnostics
  `mcse_failed_side_ids`, and capped by the error gate `sides_low_precision`
  (5%). Never withhold such a side: impute would fill it from the era prior
  (MNAR on exactly the extreme singletons), discarding the report.
- **Neutral draws carry a database fingerprint** (`current_database()` and the
  run's `started_at`); impute loads `neutral_draws_run{latest written run}.npz`
  and refuses a file from another database. Run ids are per database and the
  test and corpus databases share `data/processed/reconcile/`.
- `data_era` is centred over the unlabelled rows it applies to; a non-finite
  rhat/ESS fails the write gate; withheld sides are logged;
  `ci_width_reasonable` and the singleton gate measure multi-claim and
  single-claim sides by `est_troops_method`.

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
- Articles are found by `source_slug`, then title (`battles.wikipedia_url`
  never matched crawl's hashed filenames). An existing non-`unknown`
  battle_type is never rewritten, so reset it before re-running classify on an
  old database. `is_defender` is set only for "Siege of X" where one side
  names X.

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
- `scale=published` runs write model types with a `_published` suffix, so the
  neutral ranking stays `MAX(run_id)` of the plain type.
- The circularity comparator is a joint latent fit on per-battle `x_diff`, so
  absolute era levels cannot leak in as fake force ratios.
- **Disclosures the ranking must carry**: cross-era comparison rests on the
  shared skill prior (the comparison graph is disconnected); technology is not
  modelled; naval and aerial share one population and slope; `total_war` is
  volume-weighted, so a coverage gap by era is a rank gap; attribution weights
  are fixed data.

### 9.10 Evaluate

`agents/evaluate.yaml` carries the rationale.

- **Writes files only** (`data/processed/evaluation/run{N}/`, plus
  `data/processed/evaluation_report.json`), never `model_runs` or estimate
  rows, so `MAX(run_id)` still names the published run. Targets the newest
  converged pooled run with estimates and refuses a design or imputed rounds
  that no longer match it.
- **CV masks, it does not rebuild**: `loglik_weight` 0 on held-out battles
  (`build_model` and `build_joint_latent_model`; None leaves the graph
  identical). Held-out x_diff comes from the measurement-only prior, never the
  imputed rounds, which encode the outcome. Two arms: imputed (gated, what is
  published) and joint latent. Synthetic null with leaky imputation: both arms
  at chance, imputed arm Brier 0.34 (confidently wrong when imputation is
  circular), leaky predictor 0.76; that is the test guarding leakage.
- **Listing order predicts the winner**: side A won 79% of decided corpus
  battles. The model ignores orientation, so judge accuracy against 0.5 and
  calibration in the favourite's frame (`calibration_favourite`), never the
  A-frame bins.
- Evaluation fits use `eval_draws` 1500 and an ESS floor of
  `min(model, eval_min_ess_bulk=400)`; rhat and divergences stay the model's.
  At the model's 800 the corpus folds failed on ESS 715-760.
- Variants run on the same rounds and seeds as a baseline refit; a second
  baseline on disjoint rounds is the noise floor (corpus: top-20 overlap 17).
  `realised_war` ranks on `realised_mean`; `exclude_extreme_ratio` filters on
  measured troops only; `observed_only` is also compared with the same battle
  subset on imputed rounds; `weak_hyperpriors` keeps skills shrunk.
- Named checks by Q-id; rank ranges are reported, not gated; not_evaluable
  fails. The Arsht list is a recomputed 22-general subset.
- Corpus run, defaults, ~24 min (cv 556 s, sensitivity 859 s); run it in the
  background (it finished fine there). Imputed arm accuracy 0.70 (0.56-0.80,
  n=53), Brier 0.222; joint 0.60, 0.223; larger army wins 0.45 (n=33), model
  0.70 on that mask. equal_attribution and observed_only overlap exactly 12 of
  20. Napoleon 2nd, Alexander 24th: mechanics check, 59 battles. Outcome
  grades: no decisive level is ever observed in the corpus (extract writes
  victory/defeat only).

### 9.11 Report

`agents/report.yaml` carries the params; `pipeline/reporters/` the code.

- **Presentational only**: reads the run named by `LATEST_EVALUABLE_RUN`,
  that run's `model_diagnostics/run{N}/` files and evaluate's
  `evaluation/run{N}/`; writes files, never the database. Nothing is re-ranked:
  generals with `n_battles >= min_battles` keep their stored rank, the rest are
  listed unranked.
- **Provisional, not refused.** A failed blocking model gate
  (`exclusion_concentration` always counts, although the model records it as a
  warning), or a missing or stale evaluation report, still writes every output
  with `provisional: true` and a banner naming each reason; the spec's error
  gates then fail the stage. No evaluable run writes nothing and deletes the
  old manifest. A failed evaluate gate counts only at error severity, read from
  `agents/evaluate.yaml` (the evaluation report stores none); unknown gates block.
- `model_gate` checks bind `root` to the spec's `diagnostics_root`
  (`_SPEC_PARAM_BINDINGS`); before that, the report's `exclusion_concentration`
  read a fixed path whatever the override.
- `manifest.json` carries `model_run_id` and stage-side gates
  (`export_matches_db`, `evaluation_current`) in evaluate's gate format, read by
  the existing `evaluation_gate` handler through spec param `report_path`, so a
  manifest for an older run fails as stale.
- **Rank overlap is approximate**: `total_war +/- z*war_sd`, because the model
  stores no WAR interval or WAR draws. Exported as `war_normal_approx_lo/hi`,
  disclosed in every output. `bayesian-model-reviewer` (2026-10-14) judged it a
  poor proxy for ordering: generals' WAR is correlated (shared replacement
  level, shared battles) and skewed. **Follow-up, built on `wip/outstanding`
  (998f8e6), not yet on `main`**:
  in `pool_reductions` (`summarise.py`, `war_by_round` is already there) store
  `war_ci_lo/hi` via `widen_interval` like skill, and `p_above_next_rank` =
  mean over pooled draws of WAR_i > WAR_next; persist WAR draws in
  `pooled_draws.npz`; same in `summarise_single`. Migration + schema.sql,
  model re-review, corpus refit; then report drops the approximation.
- No Jinja (not a declared dependency): `string.Template`. Names from Wikipedia
  are inlined as JSON with `<`, `>`, `&` escaped and rendered via `textContent`.
- `explorer.html` is ~4.9 MB, almost all inlined plotly.js, so it works offline.

---

## 10. Honest status

Built and verified against a real Postgres: crawl, extract, resolve, and
reconcile, classify, impute, model and evaluate on fixtures. The recovery
suites pass on synthetic data. The real-corpus pilot (log §25) showed that
"verified against a database" meant "verified when a test passes the
connection in". On the real corpus (`general_war_corpus2`) crawl through model
pass their error gates through the CLI (§1), and evaluate runs with 4 of 5
gates passing, but every LLM request is still deferred, so the model sees 59 of
210 battles.

Not proven: any real corpus producing a ranking, the offline batch path on a real
request file (`/process-llm-batch` has never run), any billed API call
(refusal and truncation branches untested), CI on the current `main`
(`origin` exists; no session has checked a run), and every gate threshold,
which has only run against fixture databases and the real corpus (§1). The gold sets are agent-built and unreviewed by a human. Evaluate runs
on the corpus (§9.10) but the model covers 59 of 210 battles, so its numbers
are a mechanics check. Report runs on the corpus and correctly refuses to call
its output publishable (`exclusion_concentration`), so there is no published
ranking.
