Audit whether the data is fit to model. Read-only.

This is not `/status`. `/status` checks code health — what is implemented, whether it lints, whether tests pass. This checks whether the *data in the database* can support the claims the project intends to make. A pipeline that runs cleanly on a biased, sparse, or mis-attributed corpus still produces a ranking, and that ranking is still wrong.

Scope: `$ARGUMENTS` may name a single stage to audit (`crawl`, `extract`, `resolve`, `reconcile`, `classify`, `impute`, `model`, `evaluate`, `report`). With no argument, audit everything.

## 1. Connect and establish scale

Read `DATABASE_URL` from the environment. If the database is unreachable, say so plainly and stop — do not report zeroes as if they were findings.

Report row counts for: `battles`, `battle_sides`, `battle_commanders`, `generals`, `general_aliases`, `troop_reports`, `casualty_reports`, `sources`, `missing_data_log`, `crawl_log`, `model_runs`, `general_skill_estimates`, `battle_war_details`.

An empty database is a valid state early in the project. Report it as "pipeline not yet run" rather than as a list of failures.

## 2. Quality gate matrix

For every stage in scope, load `agents/<stage>.yaml` and run its `quality_checks` through the real runner:

```python
from pipeline.orchestrator import load_agent_spec
from pipeline.quality import QualityRunner

runner = QualityRunner(conn)
results = runner.run_all(load_agent_spec(stage).get("quality_checks", []))
```

Render a matrix: stage, check name, actual value, threshold, severity, pass/fail.

Checks declaring a `method:` rather than SQL report as not implemented — `diagnostics_json`, `posterior_check`, `per_round_check`, `range_check`, `statistical_test`, `brier_score`, `jaccard_overlap`, `held_out_accuracy`. Count these separately and state the number plainly. They are unverified, not passing, and the distinction matters when deciding whether a stage is really green.

## 3. Coverage and corpus skew

This section is the reason the command exists. Report, as counts and percentages:

- **By century.** `date_start` bucketed. Expect a heavy post-1500 concentration.
- **By region.** Join through `wars.region` where present, else `battles.location_name`.
- **By battle_type.** The `battle_type` enum. Watch for everything defaulting to `unknown` or `field`.
- **By era.** `generals.era`, which drives the hierarchical prior in the model stage and the era priors in `agents/impute.yaml`.

Name the skew explicitly. "62% of battles fall after 1700; pre-500 battles are 4% of the corpus" is the finding — not a caveat to bury. This is the corpus bias that most threatens the published ranking, and `config/sources_seed.yaml` makes it structural: all 14 seed lists are English Wikipedia.

## 4. Completeness

- `battle_sides` with no `troop_reports` row, and the share of the total.
- `battle_sides` where `est_troops_total IS NULL` after the reconcile stage has run.
- `battle_commanders` still at `command_role = 'unknown'` — compare against classify's `all_commanders_classified` threshold of 200.
- `missing_data_log` rows still at `missingness_class = 'unclassified'`.
- `battles` with `needs_review = true`, and the most common `review_notes` values.
- `battles` where `date_start IS NULL`, and where `date_precision` is still the `'day'` default for pre-500 battles, which is almost certainly a default that was never set.

## 5. Source disagreement

Query `v_troop_source_agreement` (already defined in `config/schema.sql`). Report the 20 sides with the highest `cv`, with battle name, `n_reports`, `min_reported`, `max_reported`.

These are where reconciliation does the most work and where extraction errors hide. A `cv` above 1.0 usually means either a genuine ancient-source dispute or a scope conflation — one source reporting theatre strength against another reporting engaged troops. Flag which you think it is.

Also report the distribution of `troop_reports.scope`. If it is overwhelmingly `engaged`, the extraction prompt is defaulting rather than discriminating.

## 6. Attribution review queue

Query `v_attribution_review_queue` (confidence < 0.5 or unknown role). Report the count and the 20 lowest-confidence rows.

Then check the structural invariants nothing else enforces:

- Every `side_id` should have exactly one commander at `hierarchy_rank = 0`. Report sides with zero or more than one.
- `reports_to_bc_id` should form an acyclic chain within a side. Report any cycle.
- `attribution_weight` per side should sum near 1.0. Report the worst deviations, and note that classify's own check tolerates 0.15 across up to 50 sides.

## 7. Model readiness

- Distribution of battles per general. Report the count with 1, 2, 3-5, 6-10, 11-25, and 25+ battles.
- How many generals clear `min_battles: 3` from `agents/report.yaml`.
- The long tail with 1-2 battles will be shrunk almost entirely to the era mean by the hierarchical prior. State how many that is, since it determines how much of the corpus is genuinely being ranked rather than imputed from its era.
- Win rate by side across the corpus. It should sit near 50% by construction. A meaningful skew means defeats are missing asymmetrically, which is the survivorship problem `agents/classify.yaml` anticipates in its MNAR heuristics.

## 8. Report

Produce:

- A one-line verdict: is the data fit to model, fit with caveats, or not yet.
- The gate matrix.
- The three findings that most threaten the validity of a published ranking, each with the query result that supports it.
- The single most impactful next action.

Make no changes. This command writes nothing to the database, the filesystem, or git. If you find something that needs fixing, say what and where, and stop.
