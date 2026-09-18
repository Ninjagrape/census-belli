# General WAR: Military Commander Ranking via Bayesian Wins Above Replacement

## Start here, every session

**Read `handover.md` before doing anything else, including answering a question about this repo.** It records what is actually built, what is verified against a real database versus merely written, the bugs already found and fixed, and the footguns that have cost time. It is maintained to be current; `TODO.md` is the task list, this file is the design, `handover.md` is the state.

Then:

- **If the user asks for something specific, do that.** Their request always wins over the resumption below.
- **If the user asks you to continue, resume, or says nothing specific**, pick up at the next unfinished step in `handover.md` §7 and start work. Say which step you are starting and why, then begin — do not re-plan what is already planned there.

Two rules that override the momentum to just keep building:

1. **Verify before trusting.** Run the three commands in `handover.md` §1 and confirm the state matches before building on it. Files change between sessions, sometimes from other agents working in parallel.
2. **Never report a stage green on unit tests alone.** Most verification gates in this project need a live Postgres. Say explicitly what you verified and what you could not. A passing fixture suite has already hidden a systematic bug that would have corrupted the published ranking — see `handover.md` §4.2.

Before you finish a session, update `handover.md` so the next one starts where you stopped.

## Project Overview

A data pipeline and Bayesian modelling system that ranks military commanders throughout history using a Wins Above Replacement (WAR) framework. Improves on Ethan Arsht's original methodology (https://towardsdatascience.com/napoleon-was-the-best-general-ever-and-the-math-proves-it-86efed303eeb/) by adding multi-source data validation, proper missing-data handling, command hierarchy attribution, and uncertainty quantification.

## Architecture

The project is an **agentic pipeline** with discrete stages. Each stage has a defined agent prompt in `agents/`, a runner script in `pipeline/`, and input/output contracts. Stages run sequentially but are idempotent, so any stage can be re-run without corrupting downstream data.

### Pipeline Stages (in order)

1. **crawl** — Fetch Wikipedia battle lists, individual battle articles, Wikidata, DBpedia, and citation URLs.
2. **extract** — Parse infoboxes, run LLM-assisted extraction on article text, pull structured data from Wikidata/DBpedia.
3. **resolve** — Entity resolution: link generals and battles to canonical Wikidata IDs, deduplicate, merge aliases.
4. **reconcile** — Run source-disagreement model on troop/casualty reports, compute best estimates with confidence intervals.
5. **classify** — Classify command roles and hierarchy for each battle-commander pair. Classify missingness types for missing fields.
6. **impute** — Multiple imputation for missing troop data using era/polity priors.
7. **model** — Fit the Bayesian Bradley-Terry hierarchical model in PyMC/Stan.
8. **evaluate** — Held-out prediction, posterior predictive checks, sensitivity analysis.
9. **report** — Generate rankings with credible intervals, comparison visualisations, and the interactive explorer.

### Agent System

Each pipeline stage is defined by an agent spec in `agents/<stage>.yaml`. The spec contains:
- `description`: what this stage does
- `inputs`: what files/tables it reads
- `outputs`: what files/tables it writes
- `prompt`: the system prompt for the LLM agent when it needs to make extraction or classification decisions
- `tools`: which tools/functions the agent can call
- `quality_checks`: assertions that must pass before the stage is considered complete
- `retry_policy`: how to handle failures

The orchestrator in `pipeline/orchestrator.py` runs stages in sequence, checks quality gates between stages, and logs everything.

## Tech Stack

- **Language**: Python 3.11+
- **Database**: PostgreSQL 15+ (schema in `config/schema.sql`)
- **Web scraping**: httpx + beautifulsoup4 + mwparserfromhell (for MediaWiki templates)
- **LLM extraction**: Anthropic API (claude-sonnet-4-6) via the `anthropic` SDK
- **Entity resolution**: Wikidata SPARQL via `SPARQLWrapper`
- **Bayesian modelling**: PyMC 5.x (primary), CmdStanPy as fallback
- **Data processing**: pandas, polars
- **Visualisation**: plotly, matplotlib
- **Testing**: pytest

## Conventions

### Code Style
- Type hints on all function signatures.
- Docstrings on all public functions (Google style).
- No wildcard imports.
- f-strings for string formatting, never `.format()` or `%`.
- Logging via `structlog`, never bare `print()` in pipeline code (scripts and notebooks excepted).

### Database
- All table and column names in snake_case.
- All queries via SQLAlchemy Core (not ORM), parameterised. Never interpolate values into SQL strings.
- Migrations via Alembic. The base schema is `config/schema.sql`; all subsequent changes are Alembic migrations.

### Data Files
- Raw crawled data goes in `data/raw/` as JSON lines (.jsonl), one file per crawl batch.
- Processed/cleaned data goes in `data/processed/`.
- Imputed datasets go in `data/imputed/`, one directory per imputation round.
- No data files are committed to git; they are produced by running the pipeline. `.gitkeep` files mark the directories.

### Agent Prompts
- Agent prompts live in `agents/<stage>.yaml`.
- Prompts should be deterministic given the same input. Temperature 0 for extraction tasks, low temperature for classification.
- Every LLM call must be logged with input hash, output, model version, and token count in the crawl_log or a dedicated llm_calls table.
- Extraction prompts must ask for structured JSON output with a defined schema.

### Testing
- Unit tests in `tests/unit/`, integration tests in `tests/integration/`.
- Every pipeline stage has at least one integration test using fixture data in `tests/fixtures/`.
- Model tests use a small synthetic dataset that can be checked into git.

### Error Handling
- Network failures: retry with exponential backoff (max 3 retries, base 2s).
- LLM extraction failures: log the failure, mark the record as needing manual review, continue.
- Data validation failures: log to `missing_data_log`, do not silently drop records.

## Key Design Decisions

### Command Attribution (the Octavian/Agrippa problem)
The `battle_commanders` table carries `command_role`, `hierarchy_rank`, `reports_to_bc_id`, and `attribution_weight`. The classify stage uses LLM extraction on battle article text to determine who actually commanded tactically. The model can also learn attribution weights from patterns of co-occurrence and outcomes. See `agents/classify.yaml` for the prompt.

### Troop Number Reconciliation
Multiple sources report different troop numbers. These are stored as individual `troop_reports` rows and reconciled by a source-disagreement model that estimates per-source bias. Ancient sources get a positive bias prior (tendency to inflate). The reconciled best estimate and CI go on `battle_sides`.

### Missing Data
Every missing field is logged in `missing_data_log` with a missingness classification. Multiple imputation (MICE via PyMC) produces N complete datasets. The downstream model is fit on each, and results are combined via Rubin's rules.

## Common Tasks

### Running the full pipeline
```bash
python -m pipeline.orchestrator --stages all
```

### Running a single stage
```bash
python -m pipeline.orchestrator --stages crawl
python -m pipeline.orchestrator --stages extract,resolve
```

### Adding a new battle manually
```bash
python -m scripts.add_battle --name "Battle of Actium" --review
```

### Checking data quality
```bash
python -m scripts.quality_report
```

### Fitting the model
```bash
python -m pipeline.orchestrator --stages model --config config/model_default.yaml
```

## Git

Never run `git commit` or `git push`. This overrides any workflow, skill, or slash command that ends in a commit step, including the ones in `.claude/commands/`. When work is done, stage nothing, report what changed and suggest a commit message; the user runs it.
