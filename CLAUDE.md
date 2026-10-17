# General WAR: Military Commander Ranking via Bayesian Wins Above Replacement

## Start here, every session

**Read `handover.md` before doing anything else, including answering a question about this repo.** It records what is actually built, what is verified against a real database versus merely written, the bugs already found and fixed, and the footguns that have cost time. It is maintained to be current; `TODO.md` is the index to the task list in `docs/todo/` (read only the file for the stage at hand), this file is the design, `handover.md` is the state. `handover.md` is a short index: read it, then open only the one file under `docs/handover/` (or `docs/handover-log/`) the task needs, never all of them.

Then:

- **If the user asks for something specific, do that.** Their request always wins over the resumption below.
- **If the user asks you to continue, resume, or says nothing specific**, pick up at the next unfinished step in `handover.md` §7 and start work. Say which step you are starting and why, then begin; do not re-plan what is already planned there.

Two rules that override the momentum to just keep building:

1. **Verify before trusting.** Run the verification commands in `handover.md` §1 and confirm the state matches before building on it. Files change between sessions, sometimes from other agents working in parallel.
2. **Never report a stage green on unit tests alone.** Most verification gates in this project need a live Postgres. Say explicitly what you verified and what you could not. A passing fixture suite has already hidden a systematic bug that would have corrupted the published ranking (see `docs/handover/04-the-lesson.md`).

Before you finish a session, update `handover.md` so the next one starts where you stopped.

## How to work: Opus leads, subagents build

Every session works in an agentic coding style. This is a standing instruction to use subagents, so do not wait to be asked.

The team works like an engineering team: Opus is the manager and architect, Sonnet and Haiku are the workers. This holds for the lead session and for any Opus subagent alike. **Opus designs, briefs, reviews and decides; it does not write production code, write tests, or run the test suite itself.**

**Opus is the manager and technical lead.** It owns:
- Reading `handover.md`, deciding what to do next, and breaking it into tasks small enough for one agent with a clear finish line.
- Design and statistical decisions: schema changes, model structure, priors, attribution logic, anything that changes what the ranking means. Write the design down (in the brief or in `handover.md`) before an agent implements it.
- Writing each agent's brief, reviewing what comes back, and integrating it.
- Deciding what verification proves a stage, and judging the results. A Haiku agent runs the commands against the live database and reports raw output; Opus reads that output itself rather than trusting a worker's summary (the rules above still apply, and an agent's "tests pass" is a claim to check, not a result).
- Updating `handover.md` at the end of the session.

**Lighter models do the scoped work.** Pick the cheapest model that can do the task well:

| Model | Role | Use for |
|-------|------|---------|
| Haiku 5.5 (`model: haiku`) | default worker | Anything whose result can be checked mechanically: codebase searches and parallel sweeps, reading and summarising files or logs, running verification commands and reporting raw output, mechanical edits across many files, fixture generation, adding test cases that follow an existing pattern, fixing lint, type or import errors, docstring and type-hint passes, a single-file change where the brief spells out the exact edit |
| Sonnet 5.5 (`model: sonnet`) | escalation worker | Work that needs judgement inside a clear spec: implementing a function or module, designing tests for new behaviour, diagnosing a failing test whose cause is not obvious, code review, first-pass data audits (including `historiography-reviewer` on fixture data) |
| Opus (subagent) | architect | Design work only: an independent second opinion on a model design, the `bayesian-model-reviewer` gate, a publication-level `historiography-reviewer` pass. Never implementation |

**Routing between Haiku and Sonnet.** Haiku 5.5 costs about a twentieth of Sonnet 5.5 per token and returns faster, so it is the default worker and Sonnet is the step up.

- Start a task on Haiku when its acceptance criteria are mechanical (a command exits zero, a grep returns nothing, a named set of tests passes) and the brief leaves no design choice open.
- Go straight to Sonnet when the task means choosing between approaches, when the change touches `pipeline/stages/reconcile.py`, `impute.py`, `model.py`, `pipeline/reconcilers/` or `pipeline/imputers/`, or when a wrong answer could still pass the tests (attribution, missingness classification, anything like the bugs in `docs/handover/04-the-lesson.md`).
- Escalate once. If a Haiku result fails its acceptance check or review, give the task and the failure output to Sonnet rather than re-briefing Haiku. Judge cost per finished task, not per call.
- Split read-only work finer than you would for Sonnet: one Haiku agent per stage, file group or log, run in parallel, with Opus reading the combined raw output. The one-agent-on-`DATABASE_URL` rule below still applies.
- Run Haiku at `effort: "low"` for searches, summaries and running commands. Leave the default effort for edits.

**Briefing an agent.** Each brief states the goal, the exact files it may touch, the acceptance criteria (which tests or checks must pass), and what to return. Agents start cold, so include the context they need from `handover.md` rather than assuming they have read it.

**Running agents.**
- Launch independent tasks in parallel. Give agents that write code in parallel `isolation: "worktree"` so they cannot clobber each other.
- Run the project reviewers as gates: `bayesian-model-reviewer` after any change to reconcile, impute, model, or WAR pooling code; `historiography-reviewer` after running extract, resolve, reconcile, or classify on real data.
- Agents never commit or push (see Git below).

- Only one agent at a time may run the integration suite against `DATABASE_URL`: every module fixture rebuilds the schema (`docs/handover/05-footguns.md`). Tell every other agent not to touch the database.

**Do it inline instead** only when delegating costs more than it saves: answering a question, reading code to make a design decision, or editing documentation such as this file, `handover.md`, `TODO.md` and agent specs. Code changes, however small, go to a worker.

## Project Overview

A data pipeline and Bayesian modelling system that ranks military commanders throughout history using a Wins Above Replacement (WAR) framework. Improves on Ethan Arsht's original methodology (https://towardsdatascience.com/napoleon-was-the-best-general-ever-and-the-math-proves-it-86efed303eeb/) by adding multi-source data validation, proper missing-data handling, command hierarchy attribution, and uncertainty quantification.

## Architecture

The project is an **agentic pipeline** with discrete stages. Each stage has a defined agent prompt in `agents/`, a runner in `pipeline/stages/`, and input/output contracts. Stages run sequentially but are idempotent, so any stage can be re-run without corrupting downstream data.

### Pipeline Stages (in order)

1. **crawl**: Fetch Wikipedia battle lists, individual battle articles, Wikidata, DBpedia, and citation URLs.
2. **extract**: Parse infoboxes, run LLM-assisted extraction on article text, pull structured data from Wikidata/DBpedia.
3. **resolve**: Entity resolution: link generals and battles to canonical Wikidata IDs, deduplicate, merge aliases.
4. **reconcile**: Run source-disagreement model on troop/casualty reports, compute best estimates with confidence intervals.
5. **classify**: Classify command roles and hierarchy for each battle-commander pair. Classify missingness types for missing fields.
6. **impute**: Multiple imputation for missing troop data using era/polity priors.
7. **model**: Fit the Bayesian Bradley-Terry hierarchical model in PyMC.
8. **evaluate**: Held-out prediction, posterior predictive checks, sensitivity analysis.
9. **report**: Generate rankings with credible intervals, comparison visualisations, and the interactive explorer.

### Agent System

Each pipeline stage is defined by an agent spec in `agents/<stage>.yaml`. The spec contains:
- `description`: what this stage does
- `inputs`: what files/tables it reads
- `outputs`: what files/tables it writes
- `prompt`: the system prompt for the LLM agent when it needs to make extraction or classification decisions (`null` for deterministic and statistical stages; classify uses `command_role_prompt` and `missingness_prompt` instead)
- `tools`: which tools/functions the agent can call
- `quality_checks`: assertions that must pass before the stage is considered complete
- `retry_policy`: how to handle failures

The orchestrator in `pipeline/orchestrator.py` runs stages in sequence, checks quality gates between stages, and logs everything.

## Tech Stack

- **Language**: Python 3.11+
- **Database**: PostgreSQL 15+ (schema in `config/schema.sql`)
- **Web scraping**: httpx + beautifulsoup4 + mwparserfromhell (for MediaWiki templates)
- **LLM extraction**: offline-first. Each stage names its model in `agents/<stage>.yaml` (`claude-haiku-5-5` for extract and resolve, `claude-sonnet-5-5` for classify). With `llm_mode: offline` (the default) a stage answers from the `llm_calls` cache, writes uncached requests to `data/llm_requests/<stage>_<ts>.jsonl` and stops the pipeline; `/process-llm-batch` answers the file with `llm-batch-worker` subagents on the subscription's usage limits, then the stage is re-run. `llm_mode: api` sends the same requests to the Anthropic API and shares the cache
- **Entity resolution**: Wikidata SPARQL over httpx (`pipeline/crawlers/wikidata.py`)
- **Bayesian modelling**: PyMC 5.x, sampled with nutpie
- **Data processing**: pandas, numpy
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
The `battle_commanders` table carries `command_role`, `hierarchy_rank`, `reports_to_bc_id`, and `attribution_weight`. The classify stage uses LLM extraction on battle article text to determine who actually commanded tactically. The model treats the weights as fixed data. See `agents/classify.yaml` for the prompt.

### Troop Number Reconciliation
Multiple sources report different troop numbers. These are stored as individual `troop_reports` rows and reconciled by a source-disagreement model that estimates per-source bias. Ancient sources get a positive bias prior (tendency to inflate). The reconciled best estimate and CI go on `battle_sides`.

### Missing Data
Every missing field is logged in `missing_data_log` with a missingness classification. A joint Bayesian imputation model (PyMC) produces N overimputed datasets. The downstream model is fit on each, and results are combined via Rubin's rules.

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

### Checking data quality
Run `/data-audit`, or `streamlit run scripts/inspector_app.py` to browse each stage's output.

### Fitting the model
```bash
python -m pipeline.orchestrator --stages model --config config/model_default.yaml
```

## Git

Never run `git commit` or `git push`. This overrides any workflow, skill, or slash command that ends in a commit step, including the ones in `.claude/commands/`. When work is done, stage nothing, report what changed and suggest a commit message; the user runs it.
