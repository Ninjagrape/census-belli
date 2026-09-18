# General WAR — TODO

## Project Setup
- [x] CLAUDE.md project spec
- [x] Database schema (config/schema.sql)
- [x] Agent specs for all 9 pipeline stages
- [x] Pipeline orchestrator skeleton
- [x] pyproject.toml with dependencies
- [x] Seed sources config
- [x] .gitignore
- [ ] README.md (public-facing, not the CLAUDE.md)
- [ ] LICENSE (MIT or similar)
- [ ] Alembic migration init from config/schema.sql
- [x] .env.example with required env vars (DATABASE_URL, ANTHROPIC_API_KEY, GEMINI_API_KEY)
- [ ] Docker compose for local Postgres
- [ ] CI config (GitHub Actions: lint, type check, test)
- [ ] Pre-commit hooks (ruff, mypy)

## Infrastructure
- [ ] Database connection module (pipeline/db.py) using SQLAlchemy Core
- [ ] Config loader that merges agent YAML specs with CLI overrides
- [ ] Structured logging setup (structlog config)
- [x] LLM client wrapper (pipeline/llm/) that handles retries, token logging, cost tracking
      - provider-agnostic: anthropic + gemini, routed per stage via agents/<stage>.yaml
      - llm_calls table doubles as the resume cache, keyed on request hash
      - [ ] smoke-test the Gemini path against a live key (refusal + truncation branches)
- [x] Quality check runner (actually execute the SQL checks in orchestrator.py)
- [ ] Stage runner base class / protocol that each pipeline/stages/*.py implements

## Stage 1: Crawl (pipeline/stages/crawl.py)
- [ ] Wikipedia battle list page parser (extract links to individual battle articles)
- [ ] Individual battle article fetcher with rate limiting and robots.txt
- [ ] Citation URL extractor from Wikipedia article references
- [ ] Citation fetcher (allow-listed domains only)
- [ ] Wikidata SPARQL client using queries from sources_seed.yaml
- [ ] DBpedia RDF fetcher
- [ ] Crawl state persistence (resume from interruption)
- [ ] Crawl log DB writer
- [ ] Deduplication of battle URLs across different list pages
- [ ] Integration test with 5-10 known battle URLs

## Stage 2: Extract (pipeline/stages/extract.py)
- [ ] MediaWiki infobox parser using mwparserfromhell
  - [ ] Handle military conflict infobox template
  - [ ] Handle campaignbox templates
  - [ ] Handle variant infobox formats (naval, aerial, siege)
- [ ] Article body text cleaner (strip markup, keep structure)
- [ ] LLM extraction batch runner
  - [ ] Chunk long articles into passages under context limit
  - [ ] Send each passage with the extract prompt from agents/extract.yaml
  - [ ] Parse and validate JSON responses against the output_schema
  - [ ] Log every LLM call (input hash, output, tokens, cost)
- [ ] Wikidata JSON mapper (Wikidata fields to schema columns)
- [ ] DBpedia RDF mapper
- [ ] Merge extractions from infobox + body + Wikidata + DBpedia + citations
- [ ] Write to data/processed/*.jsonl and DB tables
- [ ] Unit tests for infobox parsing with fixture HTML
- [ ] Integration test: extract known battle (Actium) and verify commander/troop fields

## Stage 3: Resolve (pipeline/stages/resolve.py)
- [ ] Wikidata entity lookup by name (SPARQL: find humans with matching label + military occupation)
- [ ] Exact match linker (name -> Wikidata ID)
- [ ] Fuzzy match with context (rapidfuzz + era/polity/war overlap check)
- [ ] LLM disambiguation for ambiguous cases (prompt from agents/resolve.yaml)
- [ ] General deduplication (merge records sharing a Wikidata ID)
- [ ] Alias table builder
- [ ] Battle deduplication (same event under different names)
- [ ] Resolution audit log writer
- [ ] Unit tests for fuzzy matching edge cases
- [ ] Integration test: resolve "Napoleon" / "Napoleon Bonaparte" / "Emperor Napoleon" to one entity

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
- [ ] Missingness classifier
  - [ ] Heuristic rules by era and data pattern
  - [ ] LLM classification for ambiguous cases
  - [ ] Write to missing_data_log
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
- [ ] Wire a real DB connection into the orchestrator CLI once pipeline/db.py exists
      (run_stage and run_pipeline already accept db_conn; main() still passes None)

## Stretch Goals
- [ ] Web UI for browsing results and exploring individual generals
- [ ] Battle map visualisation (geocoded battles on a world map)
- [ ] Temporal skill curves (how a general's skill estimate evolves over their career)
- [ ] Strategic vs tactical skill decomposition
- [ ] Campaign-level analysis (did a general perform better in certain campaigns)
- [ ] "What if" simulator (swap a general into a historical battle, predict outcome)
- [ ] Non-English Wikipedia sources (French, German, Chinese, Arabic battle articles)
