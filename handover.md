# Handover

The state of the project, the decisions that constrain future work, and the
traps that have already cost time. `CLAUDE.md` is the design; this is the state.

Keep it current and short. Update §1, §7 and §10 when the state changes. Record
a decision or footgun here only if a future session would otherwise get it
wrong; session narratives, changed-file lists and commit messages do not belong
here.

`docs/handover-log.md` is the full session-by-session history (§25 the
real-corpus pilot, §26 the sweep that fixed it, §27 the 2026-10-14..16
"fix all outstanding items" sessions with every worker's verified numbers,
§28 the 2026-10-16 corpus repair and classify batch).
Code comments cite its section numbers ("handover-log §16.1"). Do not read it
by default; open the cited section when a comment points there.

Last updated: 2026-10-16 (evening). `main` holds the 2026-10-16 corpus-repair
commit (fixes A-E in §7). On the corpus, extract through impute have run on
it and the classify LLM batch is answered; a `historiography-reviewer` pass
then found the corpus **not fit for a refit** (§7). The next session builds
the Phase 1 root-cause fixes and the Phase 2 overrides file.

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

The table's corpus column predates 2026-10-16. Corpus state now
(`general_war_corpus2`, Alembic `c9d0e1f2a3b4`): 236 battles, 210 with exactly
2 sides (26 are 0-side shells, see §7); resolve 784 linked, 43 unresolved,
0 deferred; reconcile passes every error gate (spread 0.65; warnings:
prior_predictive 0.062 vs 0.05, calibration set empty); classify 0 deferred,
attribution llm 267 sides / rule_single 133 / default_split 17; impute run 57
(rhat 1.004, ESS 1375, 156 of 420 sides imputed). Diagnostic model fit, run
79 (`data/logs/corpus2_model_20261016.log`; **its ranking is not valid**, §7):
converged (max rhat 1.003, no divergences); 133 of 210 battles included, 70
dropped for no outcome (was 59 included on 2026-10-14); circularity check
passes (beta_force gap 0.167 vs 0.191, skill Spearman 0.98). Failing gates:
`exclusion_concentration` (7 top generals under 50% credited),
`implausible_career_span` (1 general over 60 years, a merged identity: R2),
`top_generals_cross_component` (33 of the top 50 outside the largest
component; the missing outcomes cut cross-era links: R1). Use these as the
baseline Phase 1 should move.

Counts last measured by the lead (2026-10-16, `main` with fixes A-E):
non-model suite **2071 passed, 0 skipped**, ~12-18 min
(`data/logs/main_nonmodel_suite_20261016e.log`); model files (measured before
A-E, which touch none of them) evaluate_cv 4, impute_recovery 5,
model_recovery 7, reconcile_contamination 3, reconcile_neutral 5,
reconcile_recovery 19 + 1 strict xfail. Run the model files one per command;
a single `tests/model` run exceeds a 10-minute tool timeout.

**Verify before trusting this file:**

```bash
export DATABASE_URL="postgresql+psycopg://general_war:general_war@127.0.0.1:5432/general_war"
ruff check pipeline/ tests/ scripts/ alembic/   # expect: All checks passed
python -m mypy pipeline/                         # expect: Success, 151 source files (strict)
python -m pytest tests/ -q -m "not live and not model"   # expect: 0 failed, 0 skipped (~2050 passed)
PYTHONIOENCODING=utf-8 alembic heads             # expect: one head, b8c9d0e1f2a3
python -m pytest tests/model/<file>.py -m model  # one file per command; ~30 min in total
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
- **Agent `isolation: worktree` branches from `origin/main`**, which can lag
  local `main`. For parallel workers, create worktrees yourself from the right
  base (`git worktree add -b wip/<name> .claude/worktrees/w-<name> <base>`)
  and point workers at them. Auto mode refuses copying uncommitted diffs
  between worktrees; commit to a local throwaway branch instead (the user
  approved this for 2026-10-14..16).
- A worktree has no `data/`. A directory junction to the main `data/` works,
  but **remove it (`cmd /c rmdir`) before `git worktree remove`**.
- A database built by `apply_schema()` has no `alembic_version`, so
  `alembic upgrade head` replays every migration from the baseline. They are
  guarded and converge, but check the data after (`general_war_corpus2` was
  fine and is now stamped).
- **Re-running a corpus stage means re-running everything after it, in
  order: extract, resolve, reconcile, classify, impute.** Extract rewrites
  sides and missing_data_log; classify without a fresh reconcile logs false
  troop gaps (2026-10-16, fixes B and D).
- Never poll another task's output with a shell `until ...; sleep` loop: if
  that task dies the loop runs forever. Rely on completion notifications.
- An `llm-batch-worker` can stall on a slow API ("no progress for 600s");
  relaunch that chunk, `split --skip-answered` makes it safe.
- `claude-haiku-5-5` thinks unless told not to (handled by the
  `thinking_on_by_default` trait) and cannot take temperature 0.

---

## 6. What is missing

- The extraction gold set (`tests/fixtures/gold/` for `/extraction-eval`).
- Everything in `docs/todo/`.

---

## 7. Next steps, in order

**2026-10-16 fixes, committed** (each reviewed, tested, suite green):
- **A. battle_type provenance**: `battles.battle_type_source`
  ('extract'|'classify', migration `c9d0e1f2a3b4`). Classify's "inferred"
  note used to live in `missing_data_log`, which extract rewrites per battle,
  so rule types froze. No backfill: a pre-existing rule type with NULL source
  is never revisited (reset it to 'unknown'). Populated covariate rows that
  extract re-logs are marked observed and kept out of missingness requests
  (stage and offline export share one helper).
- **B. stale `battle_sides`**: extract keyed sides by label and never deleted
  old ones (69 of 210 corpus battles had 3-6 sides; **the 2026-10-14 refit ran
  on this**). Extract now deletes a battle's sides it did not write this run,
  with dependents, only when it wrote >= 1 side; keeps a stale side holding
  `llm_extraction` rows while an LLM pass is deferred; never adopts a QID
  owned by another slug; skips cleanup for a battle written twice in one run.
  A relabel deletes resolve's links on that side: **re-run resolve onward
  after extract.**
- **C. orphan generals**: resolve prunes generals no row references, as its
  last write before the gates.
- **D. stage order**: **extract, resolve, reconcile, classify.** Classify run
  without a fresh reconcile logs `troop_total` "no reconciled troop estimate"
  on every side whose estimate went missing (impute selects targets from this
  log). Reconcile now deletes those classify rows for sides it estimates.
- **E. DBpedia dates**: `dbp:date` can be a citation's archive date
  (Herdonia 210 BC became 2015-12-19) or lose the era (Dyrrhachium 48 BC
  became 1948). The merger rejects a DBpedia date more than a year from the
  infobox's, and replaces any chosen date that contradicts an explicit year in
  the title.

Corpus snapshot from before the side repair:
`data/backups/general_war_corpus2_20261016_pre_side_repair.dump`. Classify
batch `classify_20261016T062311Z` (399/400) and the follow-ups are imported;
every stage answers from the cache.

**`historiography-reviewer` verdict on the classified corpus: not fit for a
refit.** Attribution logic is mostly sound; the data feeding it is not. The
user chose: root-cause fixes first (Phase 1), then a reviewed overrides file
for what remains (Phase 2), then refit.

1. **Phase 1, root causes** (independent; parallel workers in their own
   worktrees from `main`, each with its own database; §5):
   - **R1 outcomes (largest)**: 70 of 204 decided non-campaign battles have no
     outcome on either side and are dropped (`battles_no_outcome`): Waterloo
     168, Leipzig 104, Borodino 38, Jena 94, Acre 216, Gettysburg 78,
     Chancellorsville 56, Fredericksburg 71, Shiloh 145, Issus 93, Granicus
     183, Chaeronea 50, Marathon 113, Artemisium 22, Second El Alamein 211.
     Napoleon loses 15 battles (5 defeats), Lee about 15. Cause: infobox
     result text ("Coalition victory", "Union victory") is not mapped onto
     side labels ("United States"). Decided: deterministic mapping (polity
     aliases, coalition/allied/union terms, named belligerents) in extract,
     then a targeted offline Sonnet request for battles still without an
     outcome (new request kind, a few dozen requests via
     `/process-llm-batch`). The user ruled out answering extract's 14,705
     deferred requests for this.
   - **R2 resolve lifespan guard**: reject a mention whose battle year is
     after the person's death or before birth+15, per mention, and split a
     name group whose years exceed one lifetime. Would catch 13 of 17 date
     conflicts: Munda 45 BC linked to Pompey the Great (d. 48 BC); Ticinus
     157 side 282 Scipio Nasica Q453860 (b. 228 BC) at 1.0 for P. Cornelius
     Scipio cos. 218; Trebia 195 side 351 Sempronius Tuditanus for Ti.
     Sempronius Longus; "Charles of Lorraine" Q3659993 (1684-1751) at
     Chotusitz 58, Soor 149, Prague 177 for Charles Alexander (1712-80);
     Bellegarde Q110721608 (b. 1798) at Aspern and Wagram; Prince Henry
     Q457386 (b. 1747) at Prague; Yi Okki at Noryang. Also the group-years
     issue in step 4 (Caesar's group includes 1948, Hannibal's 2015).
   - **R3 campaign typing**: Operation Overlord 207, Barbarossa 206,
     Bagration 205, Battle of Moscow 122 (precision 'decade'), Siege of
     Leningrad 223 (Zhukov 0.35 "victory"), Operation Torch 208, Suez Crisis
     234 are typed as battles; possibly Siege of Tobruk 228, Siege of Corinth
     219. Rules to add, each tested against this list: "Operation X" titles,
     a siege spanning over a year, decade-or-coarser date precision.
2. **Phase 2, `config/overrides.yaml`** (design: versioned, reviewed, applied
   by the owning stage with `extraction_method`/`attribution_method`
   'manual', survives re-runs, every entry cites a source), for what Phase 1
   leaves: inverted outcomes Thebes 156 (Alexander charged a loss) and
   Toulon 229 (Royalist victory); New Carthage 128 Mago Barca Q335604 (wrong
   Mago); merges Scipio Africanus (Q2253 holds only Baecula, no-QID
   "Publius Cornelius Scipio" holds New Carthage, Ilipa, Great Plains, Zama),
   Daun (3 rows), Charles Alexander; wrong side Al-Qa'qa at Zumail 175,
   Ariobarzanes at Issus, "Jena" as a general at Jena-Auerstedt; Artaphernes
   the elder at Marathon (0.45); attribution Actium side 11 (Agrippa 0.60 vs
   the 0.7 expert check, Octavian 0.2) and Bulge 180 (Eisenhower top at 0.28;
   should be supreme_commander at low weight). Re-check after Phase 1 which
   remain.
3. **Refit** in stage order (extract, resolve, reconcile, classify with any
   new batch, impute, model, evaluate, report); `historiography-reviewer`
   again before reading the ranking; `bayesian-model-reviewer` on reconcile's
   prior_predictive warning and on whether shrinkage over-pools (step 4).
4. **Open, not decided** (each needs data or a user call):
   - From the 2026-10-16 historiography review, to document or decide:
     troop scope is effectively constant (1303 of 1305 reports 'engaged'; the
     context strings are infobox fragments); reconcile compresses force
     ratios (small sides x1.5-4.5, e.g. Uhud Muslims 754 to 2679; sides over
     150k halved, Berlin 2.3M to 454k; Marathon Persians 68k against a modern
     20-25k), which undercredits underdog wins; 405 of 553 credited generals
     have one decided battle and 12 have six or more; `polity` equals
     `side_label` on all 420 sides (107 unstandardised strings); parse errors
     ("2.6-2.9 million" read as 3, "40-45,000" as 40); 21 duplicate battle
     pairs are each a 0-side shell holding the QID plus the real row without
     one (one article crawled under two slug hashes): not double counted, but
     the real rows lost their QIDs (Arsht join, named checks); 17 sides stay
     default_split because answers broke the rank-0 rule (14) or dropped a
     duplicated infobox name (3).
   - Bayesian review leftovers: L8 (`z_skill` carries the common shift that
     raw skill was dropped from the convergence gate for); L-h (count sides
     whose weak classification differs between raw and residual covariance
     on the corpus).
   - Resolve: stray modern years in groups (Caesar's includes 1948,
     Hannibal's 2015, likely access dates) keep modern namesakes through the
     date gate; Lefebvre, Moreau, Bernadotte, Bülow and Wŏn Kyun are not
     returned by exact label lookup (label search, deferred by §9.3 until a
     gold set exists); `resolvers/mentions.py` uses `part_of[0]`, which can
     be a Q-id, as war context.
   - Model: the recovery Spearman margin is thin (worst 0.825 vs 0.8 over
     4 seeds); the joint-latent comparator's beta_force sits ~0.15 below truth
     at 400 battles per era.
   - The user's calls, unchanged: extract's 14,705 deferred requests and the
     web_secondary keep/cut decision, the calibration gate's arm, the gold
     sets, CI on the `origin` remote, the billed API smoke test.

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
- **Candidates are date-gated and ranked before the prompt cap of 8**
  (`rank_for_prompt`: exact name, date-confirmed, score, sitelinks, Q-id).
  Capping in the endpoint's Q-id order dropped Julius Caesar and Hannibal
  Barca from their own lists (log §27).
- **Prompts must be byte-stable**: the request hash is the cache key, and
  SPARQL `GROUP_CONCAT` order is arbitrary, so multi-valued fields are sorted
  before rendering. Unsorted, resolve never converged across re-runs. Resolve
  renders with `render_template` (single braces); classify uses `str.format`.
- An LLM `UNCERTAIN` (or a Q-id not offered) is method `llm_uncertain` and the
  corpus-local general carries `UNCERTAIN_SPLIT_NOTE`; `no_duplicate_generals`
  excludes those deliberate splits (they count in the warning instead).

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
- One-sided-bound sides are labelled `bound_only`. **Lone bounds use
  `bound_likelihood: exgaussian` by default** (2026-10-16): "up to X" is log y
  = eta + noise + Exp, "at least X" mirrored; one shared lambda per direction,
  Gaussian tails (less robust to a mis-extracted bound than the Student-t
  points). Recovery RMSE 0.30 vs censored 2.14 (lower bounds), 0.33 vs 1.92
  (upper). `censored` stays an option; the strict xfail ablation and
  `test_unmerged_range_pairs_inflate_sigma` pin it.
- Noise prior `sigma_sq_type ~ InverseGamma(3, 4/3)`: finite variance, mode
  of the old IG(2, 1). IG(3, 2) (mean-preserving) over-shrank two-report
  sides.
- A source key has its own effect only if it **co-reports** with another key
  in >= `min_battles_per_source_effect` battles (S3); tau_u/tau_sigma are
  hierarchical only with >= 5 own-effect keys (N1); in a type mixing own and
  pooled keys the deviations are plain Normal and b_type is settled by the
  deviation prior, so the absolute level depends softly on tau_u's prior
  (S4; no own-effect keys on the current corpus). A period mean and `a_era`
  are not separately identified on unlabelled rows (N3).
- Casualty component ratios vary by side: a non-centred effect per
  (side, non-reference level).
- Diagnostics: identified-scale widths use each side's own precision-weighted
  offset; `n_identifying_sides` counts sides linking a regime to the anchor's
  connected component; prior predictive checks on the published and neutral
  fits (a sampler failure fails the warning gate). The neutral npz carries
  `g_unlabelled`, `a_era`, `offset_bar`, `m0` draws and per-side era and
  unlabelled share for impute.
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
  are written `siege_offensive` + `fortified = TRUE`; `siege_defensive` is never
  produced, the per-side `is_defender` carries who was besieged. `campaign`
  (title "War" without Battle/Siege, "Invasion of", "Conquest of", or a span
  over 180 days unless the title is "Battle of"/"Siege of") wins over every
  type and is excluded by the model. `amphibious` and `guerrilla` need a lead
  mention plus density. Terrain is rule-based with false-positive guards.
- The role prompt carries infobox order, unresolved-commander counts and seven
  few-shots reviewed by `historiography-reviewer` (alternating consuls are
  equal co-field commanders). The LLM also answers per-side `is_defender`,
  written only where the siege-title rule left NULL.
- Missingness: casualties missing while troops are present is its own rule
  (MNAR for the loser); a polity-anchored list of 20th-century authoritarian
  wartime states makes their missing figures MNAR; rows the rules leave
  unclassified go to the LLM (offline).
- Missing covariates are classified **MAR**, not MCAR: missingness plausibly
  tracks documentation level.
- Articles are found by `source_slug`, then title (`battles.wikipedia_url`
  never matched crawl's hashed filenames). An existing non-`unknown`
  battle_type is never rewritten, so reset it before re-running classify on an
  old database (this may be why the 2026-10-16 corpus run typed 0 battles).

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
- **Reconcile's shared error is propagated** (2026-10-15): side draws are
  regressed on theta = (g_unlabelled, a_era); battles are built from the
  residuals, and each round shifts every likelihood mean by its loading times
  one stratified (Latin hypercube) theta draw shared by all battles. The
  shift is carried back through the prior division **with m0's co-movement**
  (m0 = corpus_level - offset_bar moves with g): B + P^-1 (B - 1 c')/s0^2.
  Imputed and weak sides co-move through a per-era loading table. Weakness is
  judged on the raw covariance, so the switch never changes the missing set.
- Imputing troops as latents inside the Bradley-Terry model is the fully
  congenial alternative; it departs from the spec and is the user's call.

### 9.9 Model

`agents/model.yaml` and `config/model_default.yaml` carry the rationale.

- Two-sided battles, plus 3+ side battles whose outcomes split cleanly into
  two camps (merged; an unnamed member's troop share is a latent component,
  not credit for named allies); others are excluded and counted, as are
  `campaign` battles. Side A is the lower
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
- Defender term beta_def * (d_A - d_B), with complement-fill when one side is
  flagged, built only with >= 10 battles known. `beta_force` is the **land**
  slope; naval/aerial get a fixed offset `delta_water_air` (a two-group
  hierarchy left beta_force prior-dominated and loosened the circularity
  gate). WAR keeps the side's posture and slope. Convergence is gated on
  `z_skill` and centred skill, not raw skill.
- **Disclosures the ranking must carry**: cross-era comparison rests on the
  shared skill prior (the comparison graph is disconnected); technology is not
  modelled; naval and aerial share one skill population; `total_war` is
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
- **Rank overlap uses the model's WAR draws**: `war_ci_lo/hi` (pooled, widened
  by Rubin's T like skill) and P(above the next **headline** general),
  recomputed by report from `pooled_draws.npz`'s `war_draws` over the
  eligible ordering (unwidened mixture draws, disclosed). The old
  `total_war +/- z*war_sd` approximation is a fallback for runs without them,
  disclosed only then.
- No Jinja (not a declared dependency): `string.Template`. Names from Wikipedia
  are inlined as JSON with `<`, `>`, `&` escaped and rendered via `textContent`.
- `explorer.html` is ~4.9 MB, almost all inlined plotly.js, so it works offline.

---

## 10. Honest status

Built and verified against a real Postgres: every stage, through the CLI on
the real corpus (`general_war_corpus2`) from crawl to impute on 2026-10-16,
with the classify and resolve LLM batches answered offline and served from the
cache. The recovery suites pass on synthetic data.

The 2026-10-16 work showed the corpus data, not the model, is the binding
constraint: re-runs had silently duplicated sides and frozen battle types
(fixed), and a domain review found a third of decided battles without an
outcome, namesake misattributions at full weight, and wars typed as battles
(§7, open). Until Phase 1 and 2 land there is no valid ranking; the
2026-10-14 numbers ran on duplicated sides and should not be cited.

Not proven: any real corpus producing a publishable ranking, any billed API
call (refusal and truncation branches untested), CI on the current `main`
(`origin` exists; no session has checked a run), and every gate threshold
beyond fixtures and this corpus. The gold sets are agent-built and unreviewed
by a human.
