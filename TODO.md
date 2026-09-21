# General WAR — TODO

Last verified against the code on 2026-09-20: 410 passed, 10 skipped, ruff clean,
mypy --strict clean across 49 files. Run with DATABASE_URL exported or the
integration tests skip silently. The 10 skips are the opt-in live tests
(GENERAL_WAR_LIVE_CRAWL, GENERAL_WAR_LIVE_LLM) and are correct; more than 10
means DATABASE_URL is unset.

`handover.md` is the state of the project and the traps; this is the task list.
Stages 1-3 are built and verified against a real database. Stages 4-9 are
specified but unimplemented, which is the bulk of what is left.

## Project Setup
- [x] CLAUDE.md project spec
- [x] Database schema (config/schema.sql)
- [x] Agent specs for all 9 pipeline stages
- [x] Pipeline orchestrator skeleton
- [x] pyproject.toml with dependencies
- [x] Seed sources config
- [x] .gitignore
- [x] README.md (public-facing, not the CLAUDE.md)
      - states plainly that the statistical half does not exist and that there
        is no published ranking; lists the limitations bearing on believing one
- [x] LICENSE (GPL-3.0)
- [x] Alembic migration init from config/schema.sql
      - applied to a live database 2026-09-19; upgrade/downgrade/re-upgrade verified
- [x] Alembic migration 0002: battles.year_astronomical
      - also added to schema.sql, because the integration fixtures call
        apply_schema() and never run Alembic. Both paths need every future change.
- [x] .env.example with required env vars (DATABASE_URL, ANTHROPIC_API_KEY, GEMINI_API_KEY)
      - [x] pipeline.config.load_env() now reads it, and the orchestrator CLI
        calls it. An already-exported variable wins, so CI secrets and a
        deliberate export are never overridden.
      - [x] BOM sniffing: the .env on this machine is UTF-16-LE, so python-dotenv
        failed on its first byte. See handover.md 15.5.
      - [ ] **the key in .env is spelled GEMIINI_API_KEY (two I's), and the file
        holds nothing else — no DATABASE_URL, no ANTHROPIC_API_KEY.** Left alone
        deliberately: it is a credentials file. User to fix.
- [x] Postgres for local development
      - PostgreSQL 15.19 installed natively via winget; no container runtime on
        this machine. docker-compose.yml is written but has never been run.
- [x] CI config (GitHub Actions: lint, type check, test)
      - runs the suite against a real postgres:15 service and **fails the build
        if any test skips**, because a green run full of skips is the blind spot
        that hid 4.3 and 4.4. Excludes the `live` marker, so CI never touches
        Wikipedia. Also round-trips the Alembic migration.
      - [ ] never actually run: there is no git remote
- [x] Pre-commit hooks (ruff, mypy, and a hook refusing to commit a .env)
      - [ ] `pre-commit install` has not been run on this machine
      - ruff-format is registered but staged `manual`: applying it reformats the
        whole tree and belongs in its own commit

## Infrastructure
- [x] Database connection module (pipeline/db.py) using SQLAlchemy Core
- [x] Config loader that merges agent YAML specs with CLI overrides
- [x] Structured logging setup (pipeline/logging_config.py)
      - console renderer on a terminal, JSON otherwise; LOG_LEVEL honoured;
        httpx and sqlalchemy.engine held at WARNING
      - NOTE: configure() is process-global. A test that calls it needs the
        restore fixture in tests/unit/test_infrastructure.py, or it silences
        unrelated tests elsewhere in the suite. See handover.md 15.6.
- [x] LLM client wrapper (pipeline/llm/) that handles retries, token logging, cost tracking
      - provider-agnostic: anthropic + gemini, routed per stage via agents/<stage>.yaml
      - llm_calls table doubles as the resume cache, keyed on request hash
      - [ ] smoke-test the Gemini path against a live key (refusal + truncation
        branches). **Harness written: tests/integration/test_llm_live.py, 5 tests,
        gated on GENERAL_WAR_LIVE_LLM=1. Never run** — blocked on the free-tier
        quota below. Refusal and truncation are the branches worth paying for:
        either one mistaken for success writes wrong data with an ok status.
- [x] Quality check runner (actually execute the SQL checks in orchestrator.py)
      - each check runs in its own SAVEPOINT; without it one failing check
        aborted the transaction and every later check reported a false failure
      - all 22 SQL gates across all 9 specs execute without error
      - [ ] gates have only ever run against an empty database, so their SQL is
        proven valid, their thresholds are not
- [x] Stage runner base class / protocol that each pipeline/stages/*.py implements

## Stage 1: Crawl (pipeline/stages/crawl.py)
Implemented. 76 unit + 2 integration tests. The integration tests now run
against a real database.
- [x] Wikipedia battle list page parser (pipeline/crawlers/wikipedia.py)
- [x] Individual battle article fetcher with rate limiting and robots.txt (fetcher.py)
- [x] Citation URL extractor from Wikipedia article references (citations.py)
- [x] Citation fetcher (allow-listed domains only) (citations.py)
- [x] Wikidata SPARQL client using queries from sources_seed.yaml (wikidata.py, seeds.py)
- [x] DBpedia RDF fetcher (dbpedia.py)
- [x] Crawl state persistence (resume from interruption) (state.py)
- [x] Crawl log DB writer (log.py) — verified against Postgres
- [x] Deduplication of battle URLs across different list pages
- [x] Integration test with 5-10 known battle URLs
      - tests/integration/test_crawl_live.py, 10 real articles, opt-in via
        GENERAL_WAR_LIVE_CRAWL=1, never run in CI. **Run 2026-09-19: all 5 pass.**
        Builds its fetcher from the shipped agents/crawl.yaml, so a rate limit
        too fast for Wikipedia fails the test rather than being dodged by it.
      - it also pins the 4.2 naval regression against live HTML, and that
        robots.txt still permits /wiki/
- [x] 429-aware backoff before any real crawl
      - Fetcher honours Retry-After (both RFC 9110 forms), raising the wait but
        never lowering it, capped at 120s. rate_limit_wikipedia 1.0 -> 2.0.
      - measured live: 10 consecutive fetches, zero retries, zero 429s

## Stage 2: Extract (pipeline/stages/extract.py)
Implemented. 74 unit + 6 integration tests. Every SQL write is now verified
against Postgres, including the enum casts and the text-array column.
- [x] MediaWiki infobox parser using mwparserfromhell (infobox.py, infobox_html.py)
  - [x] Handle military conflict infobox template
  - [x] Handle campaignbox templates
  - [x] Handle variant infobox formats (naval, aerial, siege)
  - note: the generic template no longer asserts battle_type. It is used for
    essentially every battle including naval ones, so asserting "field" from it
    mislabelled every naval engagement. classify must infer it now.
- [x] Article body text cleaner (textnorm.py, article.py)
- [x] LLM extraction batch runner (article.py + pipeline/llm/)
  - [x] Chunk long articles into passages under context limit
  - [x] Send each passage with the extract prompt from agents/extract.yaml
  - [x] Parse and validate JSON responses against the output_schema
  - [x] Log every LLM call (input hash, output, tokens, cost)
  - [ ] never run against a live LLM key; all of this is fixture-tested only.
        Harness ready (tests/integration/test_llm_live.py); blocked on the
        free-tier quota. See handover.md 15.7.
- [x] Wikidata JSON mapper (wikidata_mapper.py)
- [x] DBpedia RDF mapper (pipeline/extractors/dbpedia_mapper.py)
      - contributes typed dates, coordinates, part_of, place and result. BC dates
        become Postgres literals: DBpedia writes 31 BC as -031-09-02, three year
        digits not four.
      - **returns sides=[] deliberately.** DBpedia flattens combatant1/combatant2
        and commander1/commander2 into unordered lists — Trafalgar reports one
        combatant ("Spain") and four commanders with no marker of which fleet
        anyone was in. Splitting that into sides would be invention. They go to
        notes marked "unassigned to sides". See handover.md 15.3.
- [x] Merge extractions from infobox + body + Wikidata + DBpedia (merger.py)
      - citations are merged via the LLM pass; DBpedia now enters at priority 35
- [x] Write to data/processed/*.jsonl and DB tables (store.py)
- [x] Unit tests for infobox parsing with fixture HTML
- [x] Integration test: extract known battle (Actium) and verify commander/troop fields
- [x] Log absent model covariates (battle_type, terrain, fortified) to missing_data_log

## Stage 3: Resolve (pipeline/stages/resolve.py)
- [x] Wikidata entity lookup by name (SPARQL: find humans with matching label + military occupation)
      - verified against the live endpoint 2026-09-19. Doing so found that
        robots.txt had been silently disabling every SPARQL query in the
        project, in crawl as well as resolve. See handover.md 4.7.
- [x] Query coverage: entities matched by alias were being discarded after
      the query found them, which is why "Duke of Wellington" returned
      nothing. Fixed by selecting ?name. See handover.md 14.1.
- [x] Exact-literal lookup could not reach Horatio Nelson (Q83235). **The
      diagnosis in 14.7 was wrong** — he has no `en` label at all; the name
      lives on his **`mul`** label, Wikidata's multilingual code, onto which
      person labels have been migrating since 2024. The lookup asked only for
      `"..."@en`. Fixed by asking in both tags and setting the label service to
      `"en,mul"`; verified live 2026-09-20, and Nelson now links
      deterministically with no LLM call. See handover.md 16.1.
- [ ] Label search (wbsearchentities / SPARQL CONTAINS): **deferred, not
      dropped.** Its premise is refuted by the above, and it would fire only on
      zero-candidate names — where its hit is usually the *sole* candidate and
      `matcher.py` skips the ambiguity-margin test, auto-linking at ~0.62
      confidence exactly where evidence is weakest. Probed live it is noisy
      (paintings, pubs, a racehorse). The residual class is unmeasured; the
      gold set below should size it before anything is built. See handover.md 16.6.
- [ ] **Blocker for any real run: the Gemini key is free tier, 20 requests per
      day.** Stated properly: 3,000 battles x ~2-4 passages each is 6,000-12,000
      extract calls, so **300-600 days**. The project cannot run. The blocker is
      the *request cap*, not cost — the one measured call was $0.0014, putting
      the corpus at order $50-200 — so any paid tier removes it. See handover.md 16.8.
      - [x] **Offline processing path** replaces the batch API design:
            `pipeline/llm/offline.py` exports uncached requests to JSONL,
            the user processes them via Claude.ai upload or `/process-llm-batch`
            in a Claude Code session (Pro subscription, not billed API), and
            `scripts/llm_offline.py import` writes responses to `llm_calls`.
            Correlates on `request_hash`. See `.claude/commands/process-llm-batch.md`.
      - [ ] ~50-battle **cost pilot** on both providers to replace the estimate
            with a measurement, reusing the extrapolation in
            `.claude/commands/extraction-eval.md` section 5.
      - [x] Resolve the spec conflict: `CLAUDE.md` updated to reflect that
            extract/resolve use Gemini, classify uses Anthropic. Haiku 4.5
            documented as fallback in both agent specs.
- [x] Exact match linker (name -> Wikidata ID)
- [x] Fuzzy match with context (rapidfuzz + lifespan gate + polity tiebreak)
      - the date gate does the real work: a candidate who was not alive cannot
        have commanded. Era is the hard gate; polity is a tiebreak only,
        because polity labels are anachronistic more often than they are wrong.
- [x] LLM disambiguation for ambiguous cases (prompt from agents/resolve.yaml)
      - verified live 2026-09-19: google-genai 2.24.0 installed, one real
        gemini-3.8-flash call resolved the genuine Yi Sun-sin ambiguity
        correctly, wrote its llm_calls row, and the repeat call was served
        from the cache at zero cost. See handover.md 13.3.
- [x] General deduplication (merge records sharing a Wikidata ID)
- [x] Alias table builder
- [x] Battle deduplication (same event under different names)
      - detects and reports to data/processed/duplicate_battles.jsonl;
        deliberately does not merge. See handover.md 12.4.
- [x] Resolution audit log writer (data/processed/resolution_log.jsonl)
- [x] Unit tests for fuzzy matching edge cases (59 tests)
- [x] Integration test: two spellings of one commander resolve to one entity
      - 18 integration tests plus an end-to-end run of resolve.run() against
        live Postgres. Agrippa / Marcus Vipsanius Agrippa across two battles.
- [ ] Tune fuzzy_threshold, ambiguity_margin and battle_duplicate_threshold
      against hand-labelled data. Re-measured live 2026-09-20 after the mul fix
      on an 8-commander spot check: Nelson, Wellington, Napoleon, Agrippa and
      Scipio all link deterministically, Yi Sun-sin correctly defers to the LLM,
      and both Hannibal cases defer rather than link. **Zero wrong links**,
      which is the number that matters. Still a spot check, not a gold set.
      - [x] `tests/fixtures/gold/resolve/` — 15 mentions, 12 battles, synthetic
            cached candidates. Agent-labelled, unreviewed. See handover.md §17.4.
      - [x] `scripts/resolve_sweep.py` + `.claude/commands/resolve-eval.md`.
            Zero LLM quota. At defaults (85.0/6.0): 0 wrong links, 11 correct,
            1 ambiguous. Gold set too clear-cut to discriminate thresholds.
- [x] `tests/integration/test_resolve_live.py` — 5 live regression tests,
      gated on GENERAL_WAR_LIVE_CRAWL=1. Covers mul label, Q-id label shape,
      date gate separation, deterministic resolution. See handover.md §17.3.

## Stage 4: Reconcile (pipeline/stages/reconcile.py)
- [ ] Source-bias model specification in PyMC
  - [ ] Hierarchical model: per-source bias and precision
  - [ ] Ancient source prior (positive bias for inflation)
  - [ ] Likelihood: observed reports ~ Normal(true_value * source_bias, source_precision)
- [ ] Fit the model on battles with 3+ source reports (calibration set)
- [ ] Apply calibrated source parameters to all battles
- [ ] Compute best estimates + 95% CI for each side's troop totals and casualties
- [ ] Write estimates to battle_sides, updated biases to sources
- [ ] Convergence diagnostics check
- [ ] Unit test with synthetic source disagreement data

## Stage 5: Classify (pipeline/stages/classify.py)
- [ ] Command role classifier
  - [ ] Deterministic rules for obvious cases (single commander = field_commander)
  - [ ] LLM classification for multi-commander sides (prompt from agents/classify.yaml)
  - [ ] Hierarchy builder (set reports_to_bc_id, hierarchy_rank)
  - [ ] Attribution weight assigner (role-based heuristic, then model-refinable)
  - [ ] Validation: weights per side sum to ~1.0
- [ ] `battle_type` inference — extract no longer sets it (see Stage 2), so the
      covariate is NULL on essentially every battle. Infer from ship/fleet
      vocabulary, terrain and categories. extract already logs it as missing,
      so `missing_data_log` tells you which battles need it.
- [ ] Missingness classifier
  - [ ] Heuristic rules by era and data pattern
  - [ ] LLM classification for ambiguous cases
  - [ ] Write to missing_data_log
        - extract writes rows for the required fields, the per-side fields and
          the three battle-level model covariates. The rule is: log the absence
          of a field some stage consumes, not every field that happens to be
          NULL. classify assigns the missingness class to those rows.
- [ ] Its `missing_data_all_logged` gate is ERROR severity, so it halts the
      pipeline if it does not genuinely write those rows
- [ ] Integration test: classify Actium (Agrippa = field_commander, Octavian = sovereign)

## Stage 6: Impute (pipeline/stages/impute.py)
- [ ] Era/polity prior loader from config/imputation_priors.yaml
- [ ] MICE implementation in PyMC
  - [ ] Conditional model: troop_total ~ era + polity + war + battle_type + outcome
  - [ ] Predictive mean matching to preserve distribution
  - [ ] Generate N imputed datasets
- [ ] Write imputed datasets to data/imputed/round_{n}/
- [ ] Update missing_data_log with imputation details
- [ ] Distributional checks (imputed values plausible within era)
- [ ] config/imputation_priors.yaml with documented era/polity priors

## Stage 7: Model (pipeline/stages/model.py)
- [ ] Bradley-Terry hierarchical model in PyMC
  - [ ] Latent skill parameters per general
  - [ ] Skill ~ Normal(mu_era, sigma_era) hierarchical prior
  - [ ] Covariates: log force ratio, defensive flag, battle type, terrain
  - [ ] Era random effects
  - [ ] Ordered outcome (5-level) via cumulative link
  - [ ] Attribution-weighted skill aggregation per side
- [ ] Non-centered parameterisation option for difficult posteriors
- [ ] CmdStanPy fallback if PyMC fails to converge
- [ ] WAR calculation from posterior
  - [ ] Define replacement level (40th percentile skill)
  - [ ] Per-battle WAR = outcome - P(win | replacement, covariates, opponent)
  - [ ] Aggregate WAR with uncertainty
- [ ] Fit on each imputed dataset
- [ ] Combine posteriors via Rubin's rules
- [ ] Also fit on observed-only data for comparison
- [ ] Write to model_runs, general_skill_estimates, battle_war_details
- [ ] Convergence diagnostics (rhat, ESS, divergences)
- [ ] Unit test with small synthetic tournament data

## Stage 8: Evaluate (pipeline/stages/evaluate.py)
- [ ] Held-out prediction (5-fold CV on battles)
- [ ] Brier score / calibration plot
- [ ] Posterior predictive checks (simulate outcomes, compare to observed)
- [ ] Sensitivity analysis runner
  - [ ] Refit with each variant from agents/evaluate.yaml
  - [ ] Compute rank correlation and top-20 overlap across variants
- [ ] Expert ranking comparison (load config/expert_rankings.yaml, compute overlap)
- [ ] Named sanity checks (Agrippa > Octavian, Belisarius > Justinian)
- [ ] Evaluation report generator (JSON + summary text)
- [ ] Diagnostic plots (rank stability, CI widths, skill vs n_battles)
- [ ] config/expert_rankings.yaml (curated from published lists)

## Stage 9: Report (pipeline/stages/report.py)
- [x] agents/report.yaml spec
- [ ] Rankings table generator (top N with CIs)
- [ ] Interactive explorer (Plotly or HTML artifact)
  - [ ] Searchable general lookup
  - [ ] Per-general battle breakdown with WAR contributions
  - [ ] Uncertainty visualisation (CI overlap between adjacent ranks)
  - [ ] Sensitivity comparison view
- [ ] Static report output (markdown + figures for the README/blog)
- [ ] CSV/JSON export of final rankings

## Known constraints
- RESOLVED: Wikipedia rate limiting. `rate_limit_wikipedia` is now 2.0 and the
  fetcher honours `Retry-After` on every retryable status, preferring it over the
  computed backoff and capping it at 120s. Measured live on 2026-09-19: ten
  consecutive article fetches, zero retries, zero 429s.
  `tests/integration/test_crawl_live.py` fails if any fetch needs a second
  attempt, so a spec that is too fast again is a test failure rather than a
  corrupted crawl.
- `battle_type` must be inferred in classify (tracked as a task under Stage 5).
  The generic
  {{Infobox military conflict}} template carries no domain evidence, so extract
  now leaves battle_type unset for it, which is almost every battle. Nothing
  sets it yet, so the covariate is currently always NULL. `missing_data_log`
  now records this per battle rather than leaving it silent.
- BC dates cannot round-trip through Python. `datetime.date` has MINYEAR == 1,
  so no pre-1 AD date can be constructed, bound as a parameter, or decoded from
  a result. Postgres stores them fine.
  **Read `battles.year_astronomical` instead** — a generated integer column
  added for exactly this. Astronomical numbering, so 31 BC is -30, where a bare
  `EXTRACT(YEAR ...)` returns -31. Only the astronomical form subtracts
  correctly across the BC/AD boundary, and getting it wrong would put a silent
  off-by-one into every date-derived covariate.
  See tests/integration/test_db.py, which pins both forms.
- `alembic ... --sql` needs `PYTHONIOENCODING=utf-8` on Windows: schema.sql uses
  box-drawing characters that cp1252 cannot encode. The Makefile targets set it,
  but `make` is not installed on this machine, so run the raw commands.
- **`export DATABASE_URL` before running the test suite.** Without it the 28
  integration tests skip rather than fail, so a green run with 28 skips looks
  almost identical to a green run with none.
- Shell heredocs mangle backslashes in this environment. Writing a Windows path
  through one silently produced `C:\Program Files\PostgreSQL` + a literal
  backspace. Use the file-writing tool for anything containing backslashes.

## Tooling (.claude/)
- [x] bayesian-model-reviewer agent — inference correctness for the PyMC stages
- [x] historiography-reviewer agent — domain plausibility of extracted data
- [x] /data-audit command — data-level health check, distinct from /status
- [x] /extraction-eval command — prompt regression harness
- [ ] tests/fixtures/gold/ — 20 hand-labelled battles the eval harness scores against
      (the real work behind /extraction-eval; also unblocks /test-stage extract and classify)
- [ ] Implement the non-SQL quality check methods registered in pipeline/quality.py
      (diagnostics_json, posterior_check, per_round_check, range_check,
      statistical_test, brier_score, jaccard_overlap, held_out_accuracy).
      These currently report as unimplemented rather than passing.
- [x] **Wire a real DB connection into the orchestrator CLI.** Done. main() now
      loads .env, configures logging, opens a connection and passes it down.
      `--no-db` keeps the old behaviour explicitly; a missing DATABASE_URL exits 2
      rather than running nine stages and reporting nine identical gate failures.
      Verified: the CLI now returns real gate verdicts.

## Stretch Goals
- [ ] Web UI for browsing results and exploring individual generals
- [ ] Battle map visualisation (geocoded battles on a world map)
- [ ] Temporal skill curves (how a general's skill estimate evolves over their career)
- [ ] Strategic vs tactical skill decomposition
- [ ] Campaign-level analysis (did a general perform better in certain campaigns)
- [ ] "What if" simulator (swap a general into a historical battle, predict outcome)
- [ ] Non-English Wikipedia sources (French, German, Chinese, Arabic battle articles)
