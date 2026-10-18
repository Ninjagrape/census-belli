# General WAR

Ranking military commanders throughout history with a Bayesian Wins Above
Replacement model, and being honest about how much we do not know.

The idea comes from [Ethan Arsht's "Napoleon was the best general ever, and the
math proves it"](https://towardsdatascience.com/napoleon-was-the-best-general-ever-and-the-math-proves-it-86efed303eeb/),
which borrowed baseball's WAR framework for generals. This project keeps the
framework and rebuilds the data and the statistics underneath it.

> **Status: a first reviewed ranking exists, and it is not publishable.** All
> nine pipeline stages run end to end on a real 236-battle corpus, with 54
> cited manual corrections applied. The latest fit (model run 108, October
> 2026) converges, predicts held-out battles at 75% accuracy, and was checked
> by a domain review and a statistical review. Both found no fatal defect, and
> both found problems serious enough that the numbers below must not be
> cited: troop figures, the depth of the ranking below the top 15, and
> corpus coverage. See [Findings so far](#findings-so-far) and
> [Known limitations](#known-limitations).

## What this adds to the original

Arsht's version scraped one Wikipedia infobox per battle, fitted an expected
outcome from the troop numbers in it, and credited each general with the
difference between what happened and what was expected, as a single number.
This project keeps that idea and changes almost everything underneath it:

| | Original approach | This project |
|---|---|---|
| **Troop numbers** | One infobox figure, taken as given | Every report from every source kept, then reconciled by a model that estimates each source's bias. Ancient inflation is a prior fitted to 45 cited cases, keyed on whether a figure is an ancient claim or a modern estimate rather than on the source document |
| **Force ratios** | Ratio of the two infobox figures | Computed on a regime-neutral scale, so a battle where one side's figure is a modern estimate and the other's an ancient claim is not tilted |
| **Missing data** | Rows dropped or filled with a sentinel | Every missing field logged and classified (MCAR/MAR/MNAR), then multiply imputed by a joint Bayesian model, with results pooled by Rubin's rules |
| **Who commanded** | Infobox commanders credited as listed | Command roles and hierarchy classified per battle; credit split by attribution weight, so Agrippa, not Octavian, gets Actium |
| **Who is who** | Names as written | Commanders linked to Wikidata identities, with a matcher built to prefer splitting two people over wrongly merging them, and a lifespan guard against namesakes |
| **Skill** | Sum of outcome surpluses | Hierarchical Bayesian Bradley-Terry model on graded outcomes, with skills shrunk towards a population and uncertainty carried through |
| **WAR** | Realised: outcome minus expectation | Expected WAR against a 40th-percentile replacement commander, so a lucky 3-0 record does not read as certain greatness |
| **Output** | A point score per general | A posterior per general with credible intervals; overlapping generals are reported as inseparable |
| **Corrections** | None | A cited, reviewed overrides file for the errors no rule can catch, re-applied on every run and guarded so a correction cannot silently stop applying |
| **Checking it** | Nothing beyond the fit | Quality gates between every stage, parameter-recovery tests on synthetic data, a check that imputation is not feeding the model its own outcomes, held-out prediction, sensitivity analysis, and independent domain and statistical reviews |

## What problem this is trying to solve

"Who was the greatest general?" is normally answered with a list and an
argument. Turning it into a measurement means answering some harder questions
first, and most of the work here is in those rather than in the model:

**A win is not a win.** Winning while outnumbered three to one is not the same
achievement as winning with three to one in your favour. WAR asks what a
*replacement-level* commander would have been expected to do with the same
army, against the same opponent, and credits the difference.

**The troop numbers are mostly wrong.** Ancient sources inflate, chroniclers
copy each other, and modern estimates disagree by factors of three. Rather than
picking one figure, every reported number is stored as its own row with its
source, and a source-disagreement model estimates each source's bias to produce
a best estimate with a credible interval.

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
`pipeline/stages/`. Stages are idempotent, so any one can be re-run, and
re-running a stage means re-running every stage after it.

| # | Stage | What it does | Status (2026-10-18) |
|---|-------|--------------|--------|
| 1 | `crawl` | Fetch Wikipedia battle lists and articles, Wikidata, DBpedia, citations | Built, run on the corpus |
| 2 | `extract` | Parse infoboxes, map Wikidata and DBpedia, decide outcomes, apply cited corrections | Built, run; the LLM passage pass is deliberately left unanswered |
| 3 | `resolve` | Link commanders to Wikidata identities, guard against namesakes | Built, run, all gates pass |
| 4 | `reconcile` | Source-disagreement model over troop and casualty reports | Built, run, converged |
| 5 | `classify` | Battle types, command roles, attribution weights, missingness types | Built, run, all gates pass |
| 6 | `impute` | Multiple imputation of troop strengths (20 rounds) | Built, run, converged |
| 7 | `model` | Hierarchical Bayesian Bradley-Terry fit per round, pooled; WAR | Built, run, converged (195 of 210 battles) |
| 8 | `evaluate` | Held-out prediction, predictive checks, sensitivity analysis | Built, run, 5 of 5 gates pass |
| 9 | `report` | Rankings with credible intervals, interactive explorer | Built, run; output for internal reading only |

Between stages the orchestrator runs that stage's quality gates, real SQL
against the database or checks on the stage's own artefacts, and halts if an
error-severity gate fails.

## Methodology

This section describes what the pipeline does on the current corpus. The
standing design notes, with the reasoning behind each choice, are under
`docs/handover/9.*.md`.

### Corpus

The battle list is Ethan Arsht's reference set, crawled from English Wikipedia
with each battle's Wikidata and DBpedia records and its citation pages: 236
battles and 420 belligerent sides, ancient to modern. 26 battle rows are
duplicate shells with no sides and never reach the model; campaigns
(operations, wars, sieges over two years, and battles dated only to a decade
or century) are typed as such and excluded from the fit.

### Extraction

Infoboxes are parsed from wikitext (with an HTML fallback) into dates,
belligerents, commanders, strengths and casualties. Every strength and
casualty figure becomes its own report row with its source, its claim regime
(an ancient claim, a modern scholarly estimate, or unlabelled) and its scope.
BC dates are stored with astronomical year numbering so that intervals
subtract correctly across 1 BC / AD 1, and each date carries its precision.

Outcomes come from the infobox result line ("Union victory", "Roman pyrrhic
victory"), mapped onto the sides deterministically. Battles still without an
outcome get one targeted LLM request (Claude Sonnet) reading the article's
lead; the grade (decisive, plain, pyrrhic) never comes from the LLM.

All LLM work runs offline by default: a stage writes its uncached requests to
a batch file and stops, the batch is answered by Claude Code subagents
(Haiku for extraction and resolution, Sonnet for classification), every
answer is cached against a hash of its request, and the stage is re-run.
Extract's article-passage pass, about 14,700 requests, has deliberately not
been answered, so troop figures and their scope come from infoboxes only.

### Resolving who is who

Commander mentions are grouped by normalised name and matched against
Wikidata person candidates fetched over SPARQL: exact label or alias matches
first, then fuzzy matches above a threshold that must beat the runner-up by a
margin, then an LLM disambiguation for the groups still ambiguous. A lifespan
guard rejects any link where the battle falls outside the candidate's command
window (from age 15 to death, widened when Wikidata gives only a decade or
century), detaches mentions that fall outside it, and splits a name group
whose battles span more than one career. When the LLM is uncertain, the
mention becomes a separate corpus-local general rather than a guess. Links the
stage no longer makes are pruned, and the affected side's attribution is reset
so it is recomputed.

### Reconciling troop numbers

Reports are de-duplicated into lineages (copies add no information), then fit
by a source-disagreement model in PyMC: each side has a latent true strength,
each source type and claim regime a bias, with a Student-t likelihood for
point figures and dedicated likelihoods for "up to" and "at least" bounds.
Ancient inflation is a prior fitted to 45 cited cases (pooled about 3.2x).
Because asymmetric labelling would tilt force ratios, the force ratio the
model uses comes from a second, regime-neutral fit. Each side gets a best
estimate with a 95% interval; a side known from one report keeps an honestly
wide interval (roughly 0.24x to 4.3x).

### Classifying command and missingness

A side with one named commander gives that commander full credit. A side with
several is classified by an LLM reading the article (role, hierarchy rank,
who reports to whom, attribution weight), checked against rules: weights sum
to 1, one top rank or co-commanders, no reporting cycles. An answer that fails
the rules is rejected and the side falls back to an equal split. Battle types
(field, siege, naval, campaign) are set by rules. Every missing field is
classified as missing completely at random, at random, or not at random.

### Imputation

Troop strengths are imputed by one joint Bayesian model: log strength by era,
battle type, outcome, a polity random effect and a commander size effect, with
a per-battle effect. Observed sides enter as reconcile's posterior with
reconcile's prior divided out, so nothing is shrunk twice, and every side is
re-drawn in each round (overimputation) to avoid attenuating the force effect.
Reconcile's shared measurement error is carried into every round. Twenty
imputed datasets are written; missing-at-random is the primary assumption,
with a not-at-random shift in the sensitivity grid.

### The model and WAR

For each imputed dataset, a hierarchical ordinal Bradley-Terry model is fitted
in PyMC (sampled with nutpie): the graded outcome of a battle depends on the
difference in the two sides' attribution-weighted commander skill, the
log force ratio, and a defender term, through symmetric ordinal cutpoints so
that which side is listed first carries no information. Skills are drawn from
one shared population, sides with no creditable commander are latent, and
naval and aerial battles get a fixed offset. The twenty fits are pooled and
their uncertainty combined by Rubin's rules; every round must converge or
nothing is written.

WAR is expected WAR: in each battle, the side's expected score with its
commanders' fitted skill minus its expected score with a replacement-level
commander (the 40th percentile of the skill distribution, per posterior draw),
split among the side's commanders by attribution weight and summed over their
battles. Generals with at least three credited battles form the headline
ranking; the rest are listed without a rank.

### Manual corrections

Some errors are row-level and no rule can catch them: a namesake linked to the
wrong person, a coalition labelled by its first member, a battle whose result
the infobox leaves to the prose. These go in `config/overrides.yaml`, one
entry per correction, each with a citation and a reviewer. The stage that
owns each field re-applies its entries on every run, after its own logic, and
a gate fails the run if any entry stops matching its target. The current file
holds 54 entries, drafted by a domain-review agent and read and approved by
the project owner.

### Evaluation and review

Evaluate refits the model in five cross-validation folds and scores held-out
battles (accuracy against 0.5, Brier score), runs prior and posterior
predictive checks, compares the force effect against a joint latent-troops
model to measure circularity (imputation conditioned on outcomes), and refits
eleven sensitivity variants (equal attribution, observed-only, weak
hyperpriors, field commanders only, three-level outcomes and others) to see
whether the top of the ranking survives them. Before any ranking is read, two
independent reviews run: a historiography review of the data and a Bayesian
review of the inference.

## Findings so far

**Provisional, internal. Do not cite.** Model run 108, October 2026, on the
236-battle corpus with the 54 corrections applied. 732 generals were credited;
89 have three or more credited battles. Every row's interval overlaps the next
row's, so neighbouring positions are not firmly ordered.

| # | General | Battles | WAR (95% CI) |
|---|---|---|---|
| 1 | Napoleon | 43 | 8.5 (4.6 to 12.4) |
| 2 | Julius Caesar | 15 | 5.6 (2.7 to 8.5) |
| 3 | Hannibal | 17 | 4.6 (1.1 to 8.1) |
| 4 | Khalid ibn al-Walid * | 13 | 4.2 (2.1 to 6.2) |
| 5 | Frederick II of Prussia | 14 | 3.5 (0.5 to 6.6) |
| 6 | Robert E. Lee | 27 | 3.0 (-2.0 to 7.6) |
| 7 | Alexander the Great | 9 | 2.0 (0.7 to 3.5) |
| 8 | Scipio Africanus | 5 | 2.0 (0.6 to 3.4) |
| 9 | Ulysses S. Grant | 15 | 1.5 (-0.6 to 3.7) |
| 10 | George Meade | 11 | 1.2 (-1.1 to 3.3) |
| 11 | Yi Sun-sin | 5 | 0.9 (0.0 to 2.0) |
| 12 | Belisarius | 5 | 0.9 (-0.2 to 2.1) |
| 13 | George B. McClellan | 7 | 0.9 (-0.8 to 2.5) |
| 14 | Pyrrhus * | 3 | 0.7 (-0.4 to 1.9) |
| 15 | Bernard Montgomery | 3 | 0.6 (-0.1 to 1.3) |

\* Rests on inflated ancient troop figures (see below).

**What holds up.**
- The fit is sound: all twenty rounds converged with no divergences, and the
  statistical review found the sampler, pooling, ordinal link and WAR
  arithmetic correct.
- Held-out accuracy is 75% on 181 battles (against 50% for a coin flip on
  orientation-free prediction), with a Brier score of 0.18.
- The top of the ranking is stable: every sensitivity variant keeps at least
  15 of the top 20, most keep 19 or 20.
- The domain review judged the top eight a defensible set, and the six
  generals whose skill interval excludes zero the six a specialist would
  expect.

**What does not, yet.**
- **Troop numbers.** Where an infobox gives both an ancient figure and a
  labelled modern estimate, extraction keeps only the ancient one (Hazir: about
  57,900 against a modern estimate of about 7,000). Khalid's and Pyrrhus's
  placements rest on these. Reconciliation also compresses force ratios toward
  even in every era, and most Eastern Front strengths are imputed rather than
  observed. Fixing this is the next piece of work.
- **The ranking below about 15.** WAR divides a side's result among its
  commanders by weight, so subordinates inherit their leader's success: from
  rank 16 down, many headline generals (Ney, Lannes, Massena, Davout) have a
  skill estimate no tighter than the prior but a positive WAR from Napoleon's
  battles. Read ranks 16 to 89 as "fought on successful sides", not as
  individual merit.
- **Cross-era placement is assumed, not measured.** Commanders from different
  eras never fight each other, so 41 of the top 50 sit outside the largest
  connected group of battles and are placed relative to it by the shared skill
  prior alone.
- **Coverage.** The corpus has no battles from 700 to 1199, 1300 to 1499, or
  the 1600s; 42 of the 89 headline generals are Revolutionary or Napoleonic.
  Commanders with five or more battles win 74% of their decided battles,
  those with one or two win 31%: the corpus is built from famous generals'
  careers, and their opponents mostly appear once.

The full reviews are in `docs/handover-log/29-*` (historiography) and
`docs/handover-log/30-*` (Bayesian); the plan that follows from them is in
`docs/handover/07-next-sessions-plan.md`.

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
| `ANTHROPIC_API_KEY` | optional | Only for a stage set to `llm_mode: api`. The default, `offline`, needs no key: LLM requests go to a batch file answered by `/process-llm-batch` in Claude Code |
| `LOG_LEVEL` | optional | `DEBUG`, `INFO` (default), `WARNING`, `ERROR` |

`.env` is read on startup and never overrides a variable you have already
exported. Save it as **UTF-8**: a PowerShell redirect writes UTF-16, which
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

A stage that defers LLM requests stops the pipeline and names its batch file;
answer it with `/process-llm-batch <file>` in Claude Code and re-run the stage.
`--continue-past-deferred` runs later stages on deterministic data meanwhile.

A Docker Compose file is included for the database if you would rather not
install PostgreSQL locally. It has never been run, so treat it as a starting point.

## Development

```bash
export DATABASE_URL="postgresql+psycopg://general_war:general_war@127.0.0.1:5432/general_war"

ruff check pipeline/ tests/ scripts/ alembic/
python -m mypy pipeline/          # strict
python -m pytest tests/ -q -m "not live and not model"
pre-commit install
```

**Export `DATABASE_URL` before running the tests.** Without it the integration
tests skip rather than fail, and a green run with skips looks almost identical
to a green run without, which has twice hidden a real defect here. CI
provisions a PostgreSQL service and fails the build if anything skips. The
statistical recovery tests are marked `model` and run one file at a time
(about 30 minutes in total).

Some tests make real requests to Wikipedia and DBpedia. They are skipped unless
you ask for them, and they never run in CI:

```bash
GENERAL_WAR_LIVE_CRAWL=1 python -m pytest tests/integration/test_crawl_live.py -v
```

They are worth running. Every systematic bug found in this project so far was
found by pointing the code at something real, not by reading it and not by
adding fixtures, including one that had silently disabled an entire data
source across two stages, and one that labelled every naval battle a land
engagement while 73 unit tests passed.

## Inspecting pipeline data

A read-only Streamlit dashboard shows the data at each stage: table sizes, crawl and extract completeness, resolve outcomes, reconcile reports against estimates, classify attribution, imputation rounds, quality gates and a per-battle drill-down.

```bash
pip install -e ".[inspect]"
streamlit run scripts/inspector_app.py
```

The database URL defaults to `INSPECT_DATABASE_URL`, else `DATABASE_URL`, and can be changed in the sidebar. Every connection is opened with `default_transaction_read_only=on`, so the dashboard cannot write.

The report stage also writes `data/processed/report/explorer.html`, a
self-contained interactive explorer of the ranking, and `report.md` with
figures.

## Project status

Verified against a live PostgreSQL 15 and the real corpus (October 2026):
every stage from crawl to report, through the command line, with every
error-severity gate passing; offline LLM batches answered and served from the
cache; the 54 manual corrections applied on every run. The non-model test
suite (about 2,400 tests) was last run in full against a live database with
the corrections in place, and passed apart from two tests that assumed an
empty corrections file; those two have since been fixed and pass on their
own. The statistical recovery tests pass on synthetic data.

Not yet proven: a ranking fit to publish (see below); any billed API call;
quality-gate thresholds beyond fixtures and this one corpus; CI on the current
code. The gold sets used to check extraction, resolution and reconciliation
were built by agents and are not yet reviewed by a person.

## Known limitations

Recorded here because they bear on whether any eventual ranking should be
believed:

- **Corpus selection bias.** The battle list comes from English Wikipedia and
  one curated list, which cover European and American military history far
  better than anywhere else, and whole centuries are absent. A commander whose
  battles are not written up cannot rank, and the model cannot correct for it.
- **Fame bias.** The corpus is largely the career lists of famous generals,
  against opponents who mostly appear once, which flatters the famous.
- **Cross-era comparison rests on a shared prior.** Commanders from different
  eras never meet, so ranking Caesar against Napoleon depends on the assumption
  that skill is drawn from one population. Technology is not modelled.
- **Troop figures.** Labelled modern estimates are currently lost when an
  ancient figure shares the side; scope ("engaged" versus "total") is a
  default rather than read from the text; reconciliation compresses force
  ratios toward even.
- **WAR is shared side performance.** A subordinate on a successful side earns
  positive WAR without evidence of their own skill, so the headline below the
  top 15 measures who fought on winning sides as much as individual merit.
- **Attribution is a modelling choice, not a fact.** Who "really" commanded is
  contested for many battles. A side's lone infobox-named commander gets full
  credit even where the article shows they were absent (McClellan at Glendale).
  The sensitivity analysis shows how much the ranking depends on it.
- **The prose was not read.** The LLM pass over article text is unanswered,
  so everything comes from infoboxes, structured data and targeted requests.
- **Some commanders do not resolve, and some resolve to the wrong person.**
  The matcher works on labels and aliases with a lifespan guard; namesakes it
  misses are corrected one by one in the overrides file, and about twenty more
  are drafted.
- **Duplicate battles are detected, not merged.** Merging is irreversible and
  wrong more often than it looks, so suspected pairs are reported for review.

## Repository layout

```
agents/          One YAML spec per pipeline stage: prompts, params, quality gates
pipeline/        Stage runners, crawlers, extractors, resolvers, LLM client, DB layer
config/          schema.sql, model and imputation configuration, overrides.yaml
alembic/         Migrations; schema.sql is the baseline
tests/           unit/ needs nothing; integration/ needs PostgreSQL; model/ is slow
scripts/         Offline LLM batch tooling, the inspector, one-off analyses
data/            Crawl and pipeline output. Not committed.
docs/            Design notes, handover, task lists, session log
```

`CLAUDE.md` is the design document and the conventions. `handover.md` is a
short index to the current state of the work; its standing decisions and known
traps are one file per topic under `docs/handover/`, and it is the most useful
thing to read before changing anything. `TODO.md` indexes the task lists in
`docs/todo/`. `docs/handover-log/` is the full history, one file per session,
including every bug found so far and what it cost to find.

## Licence

GPL-3.0. See [LICENSE](LICENSE).

Battle and commander data is drawn from Wikipedia, Wikidata and DBpedia, which
carry their own licences (CC BY-SA and CC0 respectively).
