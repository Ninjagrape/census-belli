Build the entire General WAR project as an agentic pipeline. You are the orchestrating agent. Your job is to delegate each piece of work to a sub-agent, verify the output, and proceed.

## Delegation Pattern

For each task, use the Task tool to spawn a sub-agent with a focused instruction. The sub-agent handles the implementation; you handle sequencing, verification, and integration.

## Build Sequence

Work through this sequence. After each step, verify the output before moving to the next. If a step fails, diagnose, fix or re-delegate, and only proceed once it passes.

### Phase 1: Infrastructure
Delegate these as individual tasks:

**Task 1.1** — "Create docker-compose.yml with a Postgres 15 service (port 5432, db general_war). Create .env.example with DATABASE_URL and ANTHROPIC_API_KEY. Create a Makefile with targets: db-up, db-down, db-reset, db-schema, install, lint, test."

**Task 1.2** — "Implement pipeline/db.py: SQLAlchemy Core async engine factory reading DATABASE_URL from env. Include apply_schema(engine) that reads config/schema.sql and executes it. Include get_connection() context manager. All queries must use parameterised statements."

**Task 1.3** — "Implement pipeline/config.py: load_agent_spec(stage_name) reads agents/{stage}.yaml, returns typed dict. merge_overrides(spec, overrides) deep-merges CLI overrides into the spec's params section."

**Task 1.4** — "Implement pipeline/llm.py: AnthropicClient class wrapping the anthropic SDK. Constructor takes model, temperature, max_tokens from agent spec params. Methods: extract(prompt, text) -> dict for structured extraction, classify(prompt, text) -> dict for classification, disambiguate(prompt, candidates) -> dict for entity resolution. Every call must: retry up to 3 times with exponential backoff on rate limits or server errors, log via structlog (model, input_hash, output, token_count, estimated_cost), validate JSON output and return raw on parse failure with a logged warning."

**Task 1.5** — "Implement pipeline/quality.py: QualityRunner class. Takes a DB connection and a list of quality check dicts from an agent spec. For each check: execute the SQL query, parse the threshold string (supports '>=', '<=', '>', '<', '=' followed by a number), compare, return a QualityCheckResult. Wire this into pipeline/orchestrator.py replacing the placeholder _run_single_check."

**Task 1.6** — "Implement pipeline/stages/base.py: define a StageRunner Protocol with a single run(spec: dict) -> None method. Update pipeline/orchestrator.py to validate that imported stage modules conform to this protocol."

After all 1.x tasks complete:
- Verify: run `ruff check pipeline/` and `mypy pipeline/` — fix any issues
- Verify: write and run tests/integration/test_db.py (stand up DB, apply schema, insert fixture data, query v_battle_overview)
- Commit: "feat: project infrastructure"

### Phase 2: Data Collection (Crawl + Extract)
These are the longest stages. Delegate each sub-module:

**Task 2.1** — "Implement pipeline/stages/crawl.py following the spec in agents/crawl.yaml. Sub-modules needed: pipeline/crawlers/wikipedia.py (battle list parser, article fetcher), pipeline/crawlers/wikidata.py (SPARQL client using queries from config/sources_seed.yaml), pipeline/crawlers/citations.py (citation URL extractor and fetcher for allow-listed domains). Use httpx async client with rate limiting. Persist crawl state to data/raw/crawl_state.json for resume. Write all fetch attempts to crawl_log table."

**Task 2.2** — "Implement pipeline/stages/extract.py following the spec in agents/extract.yaml. Sub-modules needed: pipeline/extractors/infobox.py (mwparserfromhell parser for military conflict infoboxes), pipeline/extractors/article.py (body text cleaner and LLM extraction batcher using pipeline/llm.py with the prompt from agents/extract.yaml), pipeline/extractors/wikidata_mapper.py (Wikidata JSON to schema), pipeline/extractors/merger.py (merge extractions from all sources per battle, prefer structured sources over LLM extraction, flag disagreements). Output to data/processed/*.jsonl and the DB."

After 2.x tasks:
- Verify: integration test with 5 known battles (Actium, Austerlitz, Cannae, Gettysburg, Stalingrad)
- Verify: quality checks from agents/crawl.yaml and agents/extract.yaml pass on the test data
- Commit: "feat: crawl and extract stages"

### Phase 3: Entity Resolution + Classification
**Task 3.1** — "Implement pipeline/stages/resolve.py following agents/resolve.yaml. Three-pass approach: exact Wikidata match, fuzzy contextual match (rapidfuzz + era/polity overlap), LLM disambiguation for remainder. Write to generals, general_aliases, update battle_commanders.general_id. Log every resolution decision to data/processed/resolution_log.jsonl."

**Task 3.2** — "Implement pipeline/stages/classify.py following agents/classify.yaml. Two sub-tasks: (A) command role classification using LLM for multi-commander sides, with deterministic rules for single-commander sides, writing command_role, hierarchy_rank, reports_to_bc_id, attribution_weight to battle_commanders. (B) missingness classification for all NULL fields in battle_sides, writing to missing_data_log. Use the prompts from agents/classify.yaml."

After 3.x tasks:
- Verify: Actium test case — Agrippa is field_commander, Octavian is sovereign, Agrippa's attribution_weight > 0.7
- Verify: quality checks pass
- Commit: "feat: resolve and classify stages"

### Phase 4: Statistical Pipeline (Reconcile + Impute + Model)
**Task 4.1** — "Implement pipeline/stages/reconcile.py following agents/reconcile.yaml. Hierarchical source-bias model in PyMC: per-source log-bias and precision, ancient source prior with positive bias. Fit on battles with 3+ reports, apply to all. Write best estimates + CIs to battle_sides, calibrated biases to sources."

**Task 4.2** — "Implement pipeline/stages/impute.py following agents/impute.yaml. MICE via PyMC with predictive mean matching. Condition on era, polity, war, battle_type, outcome. Generate N=10 complete datasets in data/imputed/round_{n}/. Update missing_data_log."

**Task 4.3** — "Implement pipeline/stages/model.py following agents/model.yaml. Bradley-Terry hierarchical model in PyMC: latent skill per general with era-level hierarchical prior, covariates (log force ratio, defensive flag, battle type), ordered outcome (5-level cumulative link), attribution-weighted skill aggregation. WAR calculation with 40th-percentile replacement level. Fit on each imputation round, combine via Rubin's rules. Write to model_runs, general_skill_estimates, battle_war_details. Include convergence diagnostics."

After 4.x tasks:
- Verify: model converges on a small synthetic dataset (tests/fixtures/synthetic_tournament.json)
- Verify: convergence diagnostics pass (rhat < 1.02, ESS > 800, divergences < 10)
- Commit: "feat: reconcile, impute, and model stages"

### Phase 5: Evaluation + Reporting
**Task 5.1** — "Implement pipeline/stages/evaluate.py following agents/evaluate.yaml. Held-out 5-fold CV, Brier score, posterior predictive checks, sensitivity analysis (refit with each variant), expert ranking comparison, named sanity checks (Agrippa > Octavian, Belisarius > Justinian). Write evaluation_report.json and diagnostic plots."

**Task 5.2** — "Implement pipeline/stages/report.py and create agents/report.yaml. Generate: rankings table (top 100 with CIs), per-general battle breakdown, sensitivity comparison, and a static markdown report with embedded figures for the README. Also generate CSV and JSON exports of the final rankings."

After 5.x tasks:
- Verify: evaluation report is generated and all sanity checks pass
- Commit: "feat: evaluate and report stages"

### Phase 6: Integration
**Task 6.1** — "Write a comprehensive integration test in tests/integration/test_full_pipeline.py that runs the entire pipeline on a small fixture dataset (20 battles, 30 generals) and verifies: all tables are populated, quality checks pass for all stages, the model converges, rankings are produced with credible intervals, and the Agrippa > Octavian check passes."

**Task 6.2** — "Write a README.md covering: project overview, how to set up and run, architecture diagram (mermaid), methodology summary, how to interpret results, and how to contribute."

**Task 6.3** — "Set up GitHub Actions CI: lint (ruff), type check (mypy), unit tests, integration tests (with Postgres service container). Add badges to the README."

Final commit: "feat: full pipeline integration, README, CI"

## Orchestration Rules

- Never proceed to the next phase until the current phase's verification steps pass.
- If a sub-agent's output has lint or type errors, fix them before committing.
- If a test fails, diagnose whether it's a test issue or an implementation issue. Fix the root cause.
- After each phase, run the full test suite to catch regressions: `pytest tests/ -v`
- Keep TODO.md updated throughout. Check off items as they're completed.
- Each commit should be atomic and pass all tests.
