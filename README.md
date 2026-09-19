# General WAR

Ranking military commanders throughout history with a Bayesian Wins Above
Replacement model, and being honest about how much we do not know.

The idea comes from [Ethan Arsht's "Napoleon was the best general ever, and the
math proves it"](https://towardsdatascience.com/napoleon-was-the-best-general-ever-and-the-math-proves-it-86efed303eeb/),
which borrowed baseball's WAR framework for generals. This project keeps the
framework and rebuilds the data and the statistics underneath it.

> **Status: partially built.** Three of nine pipeline stages are implemented
> and verified against a real database. The statistical half of the project —
> the part that actually produces a ranking — does not exist yet. See
> [Project status](#project-status) before assuming any number here is real.
> There is no published ranking, and there should not be one until the
> [known limitations](#known-limitations) below are closed.

## What problem this is trying to solve

"Who was the greatest general?" is normally answered with a list and an
argument. Turning it into a measurement means answering some harder questions
first, and most of the work here is in those rather than in the model:

**A win is not a win.** Winning while outnumbered three to one is not the same
achievement as winning with three to one in your favour. WAR asks what a
*replacement-level* commander would have been expected to do with the same
army, on the same ground, against the same opponent, and credits the
difference.

**The troop numbers are mostly wrong.** Ancient sources inflate, chroniclers
copy each other, and modern estimates disagree by factors of three. Rather than
picking one figure, every reported number is stored as its own row with its
source, and a source-disagreement model estimates each source's bias — ancient
authors get a prior that expects inflation — to produce a best estimate with a
credible interval.

**Missing data is not missing at random.** A battle with no recorded troop
numbers is usually a small or poorly documented one, and dropping those rows
would quietly bias the corpus towards famous battles. Every missing field is
logged and classified, then multiply imputed, and the model is fitted across
the imputed datasets and pooled.

**Who actually commanded?** Octavian was present at Actium; Agrippa ran the
battle. Crediting the nominal commander is the single largest source of error
in a naive version of this, so command role and hierarchy are classified per
battle, and each commander carries an attribution weight.

**The uncertainty is the result.** Every general gets a posterior, not a point
score. Where two commanders' intervals overlap, the honest answer is that the
data cannot separate them, and the output says so rather than ranking them
anyway.

## How it works

A nine-stage pipeline. Each stage has a spec in `agents/<stage>.yaml` declaring
its inputs, outputs, prompt and quality gates, and a runner in
`pipeline/stages/`. Stages are idempotent, so any one can be re-run without
corrupting what came after it.

| # | Stage | What it does | Status |
|---|-------|--------------|--------|
| 1 | `crawl` | Fetch Wikipedia battle lists and articles, Wikidata, DBpedia, citations | Built, verified |
| 2 | `extract` | Parse infoboxes, map Wikidata and DBpedia, LLM-extract from article text | Built, verified |
| 3 | `resolve` | Link commanders and battles to canonical Wikidata entities | Built, verified |
| 4 | `reconcile` | Source-disagreement model over troop and casualty reports | Not built |
| 5 | `classify` | Command roles, hierarchy, attribution weights, missingness types | Not built |
| 6 | `impute` | Multiple imputation for missing troop data | Not built |
| 7 | `model` | Hierarchical Bayesian Bradley-Terry fit, WAR calculation | Not built |
| 8 | `evaluate` | Held-out prediction, posterior predictive checks, sensitivity analysis | Not built |
| 9 | `report` | Rankings with credible intervals, interactive explorer | Not built |

Between stages the orchestrator runs that stage's quality gates — real SQL
against the database — and halts if an ERROR-severity gate fails.

## Getting started

Requires Python 3.11+ and PostgreSQL 15+.

```bash
git clone https://github.com/<owner>/census-belli.git
cd census-belli
pip install -e ".[dev]"
```

Configure the environment. Copy the example and fill it in:

```bash
cp .env.example .env
```

| Variable | Needed for | Notes |
|---|---|---|
| `DATABASE_URL` | everything | `postgresql+psycopg://user:password@localhost:5432/general_war` |
| `GEMINI_API_KEY` | extract, resolve | Provider is routed per stage in the agent specs |
| `ANTHROPIC_API_KEY` | classify | |
| `LOG_LEVEL` | optional | `DEBUG`, `INFO` (default), `WARNING`, `ERROR` |

`.env` is read on startup and never overrides a variable you have already
exported. Save it as **UTF-8** — a PowerShell redirect writes UTF-16, which
most tooling cannot read.

Create the database and apply the schema:

```bash
createdb general_war
python -m pipeline.db --apply-schema
alembic upgrade head
```

Then run the pipeline, or a range of it:

```bash
python -m pipeline.orchestrator --stages all
python -m pipeline.orchestrator --stages crawl,extract
python -m pipeline.orchestrator --stages model --config config/model_default.yaml
```

A Docker Compose file is included for the database if you would rather not
install PostgreSQL locally.

## Development

```bash
export DATABASE_URL="postgresql+psycopg://general_war:general_war@127.0.0.1:5432/general_war"

ruff check pipeline/ tests/ alembic/
python -m mypy pipeline/          # strict
python -m pytest tests/ -q
pre-commit install
```

**Export `DATABASE_URL` before running the tests.** Without it the integration
tests skip rather than fail, and a green run with skips looks almost identical
to a green run without — which has twice hidden a real defect here. CI
provisions a PostgreSQL service and fails the build if anything skips.

Some tests make real requests to Wikipedia and DBpedia. They are skipped unless
you ask for them, and they never run in CI:

```bash
GENERAL_WAR_LIVE_CRAWL=1 python -m pytest tests/integration/test_crawl_live.py -v
```

They are worth running. Every systematic bug found in this project so far was
found by pointing the code at something real, not by reading it and not by
adding fixtures — including one that had silently disabled an entire data
source across two stages, and one that labelled every naval battle a land
engagement while 73 unit tests passed.

## Project status

Verified against a live PostgreSQL 15: the database layer, the schema and its
Alembic migration, the crawl log, the extract writer, and the resolve stage end
to end. Verified against live services: Wikipedia and DBpedia fetching and
parsing, Wikidata SPARQL lookup, and one real LLM disambiguation call.

Not yet built: everything statistical — stages 4 and 6 through 8 — and the
classify stage they depend on.

Not yet proven: any crawl at corpus scale, and the quality gates' thresholds,
which execute correctly but have mostly run against an empty database.

## Known limitations

Recorded here because they bear on whether any eventual ranking should be
believed:

- **Corpus selection bias.** The battle list comes from English Wikipedia,
  which covers European and American military history far better than anywhere
  else. A commander whose battles are not written up cannot rank, and this is
  not a gap the model can correct for.
- **`battle_type` is currently never set.** Wikipedia's generic infobox
  template carries no domain evidence, so extract deliberately leaves the field
  unset rather than guessing; the classify stage is meant to infer it and does
  not exist yet. Until then the covariate is null everywhere.
- **Commanders known by a title may not resolve.** The Wikidata lookup matches
  labels and aliases literally, so someone whose common name is neither —
  Horatio Nelson is the worked example — becomes a corpus-local entity instead
  of linking to their real record.
- **Duplicate battles are detected, not merged.** Merging is irreversible and
  wrong more often than it looks, so suspected pairs are reported for review.
- **Attribution is a modelling choice, not a fact.** Who "really" commanded is
  contested for many battles, and the attribution weights encode a position on
  that. The sensitivity analysis in stage 8 exists to show how much the ranking
  depends on it.

## Repository layout

```
agents/          One YAML spec per pipeline stage: prompts, params, quality gates
pipeline/        Stage runners, crawlers, extractors, resolvers, LLM client, DB layer
config/          schema.sql and the model/imputation configuration
alembic/         Migrations; schema.sql is the baseline
tests/           unit/ needs nothing; integration/ needs PostgreSQL
data/            Crawl and pipeline output. Not committed.
docs/            Design notes
```

`CLAUDE.md` is the design document and the conventions. `handover.md` is the
detailed state of the work, including every bug found so far and what it cost
to find — the most useful thing to read before changing anything.

## Licence

GPL-3.0. See [LICENSE](LICENSE).

Battle and commander data is drawn from Wikipedia, Wikidata and DBpedia, which
carry their own licences (CC BY-SA and CC0 respectively).
