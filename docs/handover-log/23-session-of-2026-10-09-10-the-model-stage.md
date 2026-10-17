# Handover log §23. Session of 2026-10-09/10: the model stage

Part of the session log; the index is `docs/handover-log/README.md`.

Opus designed and judged; two Sonnet workers built (`pipeline/modelers/` core,
then the stage around it); Haiku workers ran every verification; the
`bayesian-model-reviewer` gated the design and then the code. Opus wrote no
production code. Uncommitted.

### 23.1 What the model is, and where it departs from the spec

The full rationale is in `agents/model.yaml` and the comments of
`config/model_default.yaml`. In short:

- **Unit: two-sided battles.** 3+ side battles are excluded and counted (camp
  merging is a TODO). Side A is the lower `side_id`, which follows infobox
  order, so orientation must carry no information.
- **Ordinal logit, five levels, symmetric cutpoints** `(-c2, -c1, c1, c2)`.
  Symmetry makes swapping sides exactly `eta -> -eta, k -> 4-k`, so the
  likelihood cannot learn listing order (the §21.2 bug class). Pyrrhic is a
  win. If the two sides' grades disagree within a class, the outcome is
  interval-censored over both, so the grade used does not depend on which
  side is A; contradictory classes exclude the battle.
- **`eta = S_A - S_B + beta_force (x_A - x_B) + beta_def (d_A - d_B)`.** The
  spec's additive `beta_type`, `alpha_era` and terrain terms are **dropped**:
  they are battle-level, so under an arbitrary orientation they can only learn
  "side A wins". No `mu_era` on skill: eras never meet, so it would be
  identified by the prior alone. The defender term drops out while
  `is_defender` is all NULL (it is, today).
- **Skills** non-centred, one population, `sigma_skill ~ HalfNormal(1)`.
  Reported skills are centred per draw.
- **Sides with no creditable commander are latents, not dropped battles**
  (reviewer CRITICAL). An unresolved side is `mu_unknown + sigma_skill z`; a
  side with 2+ commanders all `default_split` is `mu_unattributed + sigma_skill
  z` and credits nobody, but still informs its opponent. Dropping those battles
  had removed most Napoleonic and modern battles and would have mechanically
  demoted their generals under a `total_war` ranking.
- **WAR is expected WAR (xWAR)**: per draw, the side's expected score at its
  fitted skill minus at replacement (40th percentile of skills in that draw),
  split by attribution weight. Arsht's realised WAR (`y - E_rep`) was
  rejected as primary: it ignores the general's own skill uncertainty and
  bypasses shrinkage, so a 3-0 general gets a big WAR with a tight sd. It is
  kept in the diagnostics npz. `win_prob_model`/`win_prob_baseline` hold
  **expected score** (`P(win) + 0.5 P(indecisive)`), not P(win).
- **MI**: one fit per imputed neutral round; mixture of thinned draws for
  point summaries and ranks; stored sd is Rubin's `sqrt(T)`; intervals are
  mixture quantiles widened by `sqrt(T / Var_mix)`. FMI per general is a lower
  bound while the §21.7 shared-error limitation stands.
- **Write rule**: every round must pass rhat <= 1.02, ess_bulk >= 800 and 0
  divergences, after one retry at a new seed and target_accept 0.95. Rounds are
  never dropped. Otherwise no estimates are written and the pooled row says
  `converged: "false"`. Stale or mixed impute manifests are refused.
- **Runs**: `bradley_terry_round` per round, `bradley_terry_joint_latent`, and
  the pooled `bradley_terry_hierarchical` row **last** (report reads
  `MAX(run_id)`; see 23.6).

### 23.2 The circularity comparator, and the defect the code review found

Impute conditions on outcome, so the main fit's force ratios are partly built
from the outcomes it explains. The comparator is one **joint latent-x fit**
with no outcome conditioning, on the same battles. The first version gave each
missing side its absolute era prior (e.g. ancient N(9.2, 2.0)) independently
of its measured opponent, so the gap between the era mean and the corpus level
became a fake force ratio. On the fixture this gave `beta_force` -0.017 (sd
0.027, implausibly tight) against 0.111 pooled; on realistic scales it still
shifted by about a posterior sd. **Fix:** the latent is now each battle's
`x_diff`, with prior `N(m_A - m_B, sqrt(sd_A^2 + sd_B^2))` when both are
measured and `N(0, sqrt(sd_obs^2 + 2 sigma_w^2))` otherwise (`sigma_w` 1.0 in
config). Absolute era levels drop out. After the fix: fixture gap 0.161 against
a limit of 0.159, and a recovery test with a third of sides unmeasured covers
the true slope (0.317, 95% [0.016, 0.609], sd 1.54x the fixed-x fit; the bound
is 1.6x, justified in the test).

The lesson repeats §4.2: the integration test passed with the defect in. Only
a number that looked wrong to a reader (an sd of 0.027 on 90 battles) caught
it.

### 23.3 Gates (all warnings except the existing errors)

`convergence` (now also fails on `diagnostics.converged != 'true'`),
`no_divergences`, `top_generals_have_intervals` (filters on the latest
hierarchical run; returns -1 when it wrote nothing), the posterior predictive
category check (level 0.99: at 0.95 a correct model fails ~20% of the time
over five categories), `prior_predictive_sensible` (on expected score, which
is exactly 0.5 at a zero gap), `fmi_top_generals`, `circularity_check`,
`exclusion_concentration`, `extreme_ratio_dependence` (WAR from battles won
outnumbered by more than e^2: the Plassey/Cajamarca class, where an unmodelled
technology gap reads as skill), `implausible_career_span` (> 60 years, a
false-merge detector), `top_generals_cross_component`, `exclusions_reported`.
Dataset gates are read from `run{id}/gate_report.json` by a new `model_gate`
method in `pipeline/quality.py`, which shares a reader with impute's.

On the integration fixture `circularity_check` and
`top_generals_cross_component` fire, as they should: the fixture has two eras
that never meet, and only 90 battles.

### 23.4 Disclosures the ranking carries

Cross-era comparison rests on the shared skill prior (the comparison graph is
disconnected; `mu_unknown`/`mu_unattributed` are a soft link). Technology is
not modelled; one log-ratio slope for all eras. Naval and aerial battles share
the skill population and slope (counts in the pooled diagnostics).
`total_war` is volume-weighted, so a coverage gap by era is a rank gap.
Attribution weights are fixed data, renormalised per side, not learned.

### 23.5 Verification, stated honestly

Run by a Haiku worker against live PostgreSQL 15.19 on 2026-10-10, raw output
read by Opus:

- `ruff check pipeline/ tests/ scripts/ alembic/`: clean.
- `mypy --strict pipeline/`: clean, **113 source files**.
- `pytest tests/ -m "not live and not model"`: **925 passed, 0 skipped, 0
  failed**, 8 minutes.
- `pytest tests/model/test_model_recovery.py -m model`: 6 passed, 46s. Recovery:
  max rhat 1.009, min ess 883, 0 divergences; within-era Spearman 0.909 /
  0.852; `beta_force` 0.440, 95% [0.247, 0.636] (truth 0.5); 95% skill
  coverage 0.967.
- `tests/integration/test_model.py`: 12 passed (rows, pooled row last, WAR
  decomposition reconciles in SQL, write rule on a failing round, stale
  manifest refused, dry run, every gate executes).
- Two `bayesian-model-reviewer` passes (design: build after changes; code:
  commit after fixes). Every finding was applied except LOW 8 (rhat on raw
  `skill`), which is in TODO.md.

**Not verified:** no real corpus has been through the model; every number is
synthetic or from a fixture. The rest of `tests/model` (reconcile and impute
recovery) was not rerun this session; nothing upstream changed. The recovery
test's Spearman > 0.8 criterion ranged 0.797-0.922 across the seeds the worker
tried, so it sits near the information limit; the committed seed passes.

### 23.6 Still open

- **BLOCKING for report:** its gates use `MAX(run_id) FROM model_runs`;
  evaluate's refits will take that over. Filter on `model_type =
  'bradley_terry_hierarchical' AND diagnostics->>'converged' = 'true'`.
  `exclusion_concentration` must block publication.
- New evaluate variants from the review: `exclude_extreme_ratio`,
  `three_level_outcome` (Wikipedia's "decisive" is editor-dependent and tracks
  fame), `realised_war`, `equal_attribution`.
- TODO.md Stage 7: 3+ side camp merging, force slope by battle type, CmdStanPy
  fallback, rhat on centred skills.
- Workers hit the API session and weekly limits twice this session, mid-task.
  Both were resumed or picked up from disk without damage. Check `git status`
  and grep for the brief's items rather than trusting a cut-off worker's last
  line.
