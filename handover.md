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

Last updated: 2026-10-10. Everything through the model stage is committed. The
gap review findings (`docs/todo/*.md`, tagged `(gap review)`) are unfixed.

---

## 1. Current state

| Stage | Implemented | Verified against a database |
|---|---|---|
| crawl | yes | **yes**, plus live against Wikipedia |
| extract | yes | **yes** (deterministic path; LLM path fixture-only) |
| resolve | yes | **yes**, incl. an end-to-end run and live Wikidata |
| reconcile | yes, incl. the regime-neutral fit | on fixtures; never on a real corpus |
| classify | first slice | on fixtures; LLM path offline/cache only, no billed call |
| impute | yes | on fixtures; never on a real corpus |
| model | yes | on fixtures; never on a real corpus |
| evaluate | **no** | — |
| report | **no** | — |

**Verify before trusting this file:**

```bash
export DATABASE_URL="postgresql+psycopg://general_war:general_war@127.0.0.1:5432/general_war"
ruff check pipeline/ tests/ scripts/ alembic/   # expect: All checks passed
python -m mypy pipeline/                         # expect: Success, 113 source files (strict)
python -m pytest tests/ -q -m "not live and not model"   # expect: 925 passed, 0 skipped
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
  is UTF-16). Exported variables win. The file holds only `GEMIINI_API_KEY`
  (misspelt, user to fix), so export `DATABASE_URL` yourself.
- **Gemini key prefix `AQ.A...` is valid.** Do not judge a key by its prefix;
  make one call and read the error.
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
(`params.llm_provider`): extract and resolve use gemini, classify anthropic.

Resolve builds its client lazily, so a run where everything resolves
deterministically needs no key; the first genuinely ambiguous commander raises
`LLMConfigError` and the stage stops (`pause_and_alert`). A real resolve run
needs `google-genai` and `GEMINI_API_KEY`. `params.candidate_source: local`
resolves with no LLM and leaves ambiguous groups `unresolved`.

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
  prior-to-posterior contraction, not just rhat.

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

0. **Fix the pre-pilot gap review items** (A1, A2, A4, A5, A8, A10, A13). Without
   A1 the pilot crawl halts on its own gate; without A4 same-name battles merge.
1. **Run the curated corpus** (`config/sources_seed_arsht.yaml`, 211
   title-verified battles) through crawl, extract and resolve, into a separate
   database (§5). Extract's deterministic path needs no LLM quota. This turns
   every gate threshold from unmeasured into measured.
2. **Classify, second slice.** Run the pending multi-commander sides through the
   offline path (`python -m scripts.llm_offline export --stage classify`,
   `/process-llm-batch`, import); populate `is_defender`; LLM missingness for
   `unclassified` rows. Until this runs, every multi-commander side is
   `default_split`, credits nobody, and `exclusion_concentration` will fire.
3. **Reconcile and impute open items** (`docs/todo/04-reconcile.md`,
   `06-impute.md`), above all propagating the shared reconcile error into impute.
4. **Evaluate and report.** Fix the two BLOCKING report items first
   (`docs/todo/09-report.md`).
5. Full-pipeline integration test.

**The blocker for a full corpus is not code**: the Gemini key is free tier, 20
requests/day, and the 3,000-battle corpus needs 6,000-12,000 extract calls
(§9.4). Settle it before planning a real run.

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
  **`/process-llm-batch`**: the offline LLM path.
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

### 9.4 LLM routing and quota

- Gemini 3.8 Flash for extract and resolve, Anthropic for classify; fallback
  `claude-haiku-5-5` (named in the specs, read by no code).
- **Free tier, 20 requests/day**: 3,000 battles x 2-4 passages is 300-600 days.
  The cap is the problem, not cost (one call $0.0014, corpus order $50-200).
- The workaround is the offline path: `pipeline/llm/offline.py` exports
  uncached requests to JSONL, `/process-llm-batch` answers them in a Claude Code
  session, `scripts/llm_offline.py import` writes them to `llm_calls` on the
  same `request_hash`.
- If batching is ever revisited: provider batch APIs (independent requests, one
  per passage, `request_hash` as `custom_id`) are fine. Cramming many battles
  into one prompt is rejected: it breaks the per-request cache and the
  per-passage schema.

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
synthetic data.

Not proven: any real corpus past resolve, any billed LLM call (refusal and
truncation branches untested), CI (no git remote), and every gate threshold,
which has only run against empty or fixture databases. The gold sets are
agent-built and unreviewed by a human. Evaluate and report do not exist, so
there is no ranking.
