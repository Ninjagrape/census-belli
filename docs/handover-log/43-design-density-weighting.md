# §43 design: density-weighted WAR aggregation and within-war ranks

Decided by the user 2026-11-10 (handover-log §42 section 6, items 1 and 3).
This file is the implementation brief. It changes what `total_war` means, so
bayesian-model-reviewer reviews it before code is written.

## 1. Goal

`total_war` scales with credited volume, and corpus density turns volume into
rank (BMR M3): the ACW has 101-115 credited battles a year, Napoleonic and WW2
about 21-28. Down-weight each battle's contribution to a general's total by
how crowded its period is, without touching the likelihood. Add a within-war
rank so the headline does not rest only on the cross-era comparison, which is
not identified (BMR M2).

## 2. Definitions

- **Included battles:** the battles in the model design (`ModelDesign`), the
  same set for every imputation round.
- **Density** of included battle `b` with year `y_b`
  (`ModelDesign.year`, from `battles.year_astronomical`):
  `d_b = #{c included, year known : |y_c - y_b| <= w} / (2w + 1)`, with
  `w = window_years = 1`. The count includes `b`. The density is global: it
  counts every war in the window, not only `b`'s war (section 6, Q1).
- **Aggregation weight:** `a_b = min(1, (d_ref / d_b) ** power)`, with
  `d_ref = density_ref = 29` and `power = 0.5`. An undated battle
  (`year` NaN) gets `a_b = 1` and does not count towards anyone's density;
  log how many.
- `d_ref = 29` was measured in §42 as the 75th percentile of `d` outside the
  ACW in run 52. A run-52 measurement under this exact definition is recorded
  in §43 before the refit; if it differs, the user picks the value. The config
  holds a number, not a percentile rule.
- Every commander entry `e` on battle `b` carries `a_e = a_b`.

## 3. What changes

| Quantity | Before | After |
|---|---|---|
| Likelihood | unweighted | unchanged |
| `xwar_entry`, `realised_entry` (per entry) | as now | unchanged |
| `total_xwar[g]` per draw | `sum_e xwar_entry[e]` | `sum_e a_e * xwar_entry[e]` |
| `total_realised[g]` | `sum_e realised_entry[e]` | `sum_e a_e * realised_entry[e]` |
| `war_per_battle` | `total_war / n_included_battles` | unchanged: unweighted `sum_e entry_xwar / n_included_battles` (section 8, Q2); so `war_per_battle * n_battles != total_war`, documented |
| `rank` (`rank_by: total_war`) | unweighted total | weighted total |
| `battle_war_details.war_contribution` | entry xWAR | unchanged (entry xWAR) |
| `battle_war_details.aggregation_weight` | n/a | new column, `a_e` |
| `war_decomposition_reconciles` | `total_war = sum(war_contribution)` | `total_war = sum(war_contribution * COALESCE(aggregation_weight, 1))` |

The weighting is linear and applied per draw before Rubin pooling, so pooling,
intervals and `p_above_next_rank` need no change.

`war_aggregation: {method: none}` must reproduce the current totals exactly
on the same draws. That is the unweighted variant and the regression test.

## 4. Code (Sonnet, in a worktree)

1. `config/model_default.yaml`: a new block, with a comment saying why:
   ```yaml
   war_aggregation:
     method: density        # density | none
     density_ref: 29.0
     power: 0.5
     window_years: 1
   ```
   `pipeline/modelers/config.py` parses and validates it (`density_ref > 0`,
   `power >= 0`, `window_years >= 0`, `method` in the enum). It is recorded in
   `model_runs.config` with the rest of the config.
2. A new module `pipeline/modelers/aggregation.py` (design.py is already 672
   lines): pure functions `battle_density(years, window_years) -> np.ndarray`
   and `aggregation_weights(design, cfg) -> np.ndarray` (per entry, aligned
   with `design.cmd_side`). Numpy only, sort-based, no DB.
3. `pipeline/modelers/war.py`: `compute_war` takes `entry_weight: np.ndarray |
   None = None` (None means all ones) and applies it in the two total loops
   only. The module docstring gains a step saying what the weight is and that
   entries are stored unweighted.
4. `pipeline/modelers/summarise.py`: `reduce_draws` passes the weights;
   `assert_decomposition` checks the weighted sum.
5. `pipeline/stages/model.py`: `_estimate_rows` writes the weighted
   `war_per_battle`; `_detail_rows` writes `aggregation_weight`; the weight
   distribution (min, median, share below 1, n undated, d percentiles) is
   logged and added to `model_runs.diagnostics` under `war_aggregation`.
6. `pipeline/modelers/store.py`: `WarDetailRow.aggregation_weight` and the
   insert SQL.
7. An Alembic migration after `0010`: add
   `battle_war_details.aggregation_weight DOUBLE PRECISION` (nullable; NULL on
   older runs means 1). `config/schema.sql` matches, with a comment.
8. `agents/report.yaml`: `war_decomposition_reconciles` uses the weighted sum.
   `agents/model.yaml`: the WAR definition text.
9. Evaluate post-hoc variants (`kind: post_hoc`: re-reduce the baseline
   draws, no refit) in `pipeline/evaluators/variants.py` and
   `agents/evaluate.yaml`:
   - `war_unweighted` (`method: none`);
   - `war_dref_low` / `war_dref_high` (`density_ref` 20 / 40);
   - `war_power_low` / `war_power_high` (`power` 0.25 / 1.0);
   - `rule_single_half`: entries whose `battle_commanders.attribution_method
     = 'rule_single'` get their aggregation weight multiplied by 0.5 (the
     interim stand-in for §42 decision 2 until the classify fix lands).
   Each reports top-N overlap with the baseline, as the other variants do.
10. Report (`pipeline/reporters/`): section 5.

## 5. Report: global plus within-war

- The global table stays, ranked on the weighted total. Its caption and the
  disclosures say "density-weighted total WAR" and why, in one sentence.
- **Within-war WAR** of general `g` in war `k`: the sum over `g`'s entries on
  battles with `battles.war_id = k` of `war_contribution * aggregation_weight`
  (from `battle_war_details`; no new model quantity). A general who fought in
  several wars has one row per war.
- **Within-war rank:** the rank of that sum among the generals credited in
  war `k` (1 is best), on posterior means. The global table gets a column with
  the general's rank in their primary war (the war with the largest
  within-war WAR), shown as "3rd of 212, American Civil War". Battles with
  NULL `war_id` go to a "no war recorded" bucket that is not ranked.
- **Within-component rank:** the rank of the weighted total among generals in
  the same comparison component (`component_id` is already in the explorer
  payload). A second column.
- Explorer: a war selector showing the within-war table.
- Disclosures: within-war ranks are on posterior means, without intervals;
  the cross-war comparison rests on prior-centred components (M2).

## 6. Questions for the reviewer

1. **Global or per-war density.** As decided, `d` counts every war in the
   window, so a small war fought alongside the ACW, or alongside WW2's many
   fronts, is also down-weighted. Per-war density avoids that but depends on
   how finely `war_id` splits wars. Is global acceptable, or should density be
   computed within `war_id` (global for NULL)?
2. **`war_per_battle`.** Should it be the weighted mean `total / sum a`, or
   stay the unweighted mean so it reads as per-battle quality independent of
   the aggregation?
3. **Replacement level.** `s_rep` stays the 40th percentile over all credited
   generals (one global level, M3). Weighting shrinks ACW volume but does not
   change that most entries carry positive xWAR. Is that acceptable for this
   change, with the within-war rank carrying the honest comparison?
4. Anything that would stop `method: none` reproducing the current totals.

## 7. Tests and acceptance

- Unit: `battle_density` (inclusive window edges, a single battle, all
  undated, BC years); `aggregation_weights` (cap at 1, `power = 0` gives all
  ones, undated gives 1); `compute_war` with `entry_weight=None` equals the
  current output exactly; weighted totals equal `sum(a * entry)` per draw;
  `assert_decomposition` passes on weighted sums and fails on a broken one;
  config validation errors; the variants parse and `war_unweighted` equals
  the old totals; the within-war rank on a fixture with a general in two wars
  and a NULL-war battle.
- Integration (scratch DB, one suite at a time): migration up and down; the
  model stage on fixtures writes `aggregation_weight` and the report gate
  passes.
- ruff, mypy and the full unit suite.

## 8. Review: bayesian-model-reviewer, GO with changes (2026-11-10)

The weight is a fixed per-entry constant shared by every draw and round, so
the weighted total is linear per draw and Rubin, `mixture`, `widen_interval`
and `p_above_next_rank` apply unchanged. Merged camps are fine (`a_e`
multiplies the `w_i / W` share); latent sides count towards density. The
amendments below are **binding** and override sections 2-7 where they differ.

1. **Post-hoc routing.** `_post_hoc` (`pipeline/evaluators/sensitivity.py`)
   returns None unless the percentile or `rank_by` changes, and `repool`
   (`pipeline/evaluators/fit.py`) takes no weights. Both take a
   `war_aggregation` override; the no-change check compares the full
   aggregation config; `variants.py` validates the new option keys.
2. **Every `reduce_draws` caller.** `reduce_draws` takes the aggregation
   config and computes weights from the design it is given. Callers: the model
   stage, the joint comparator (`pipeline/stages/model.py`), the refit variants
   (`pipeline/evaluators/fit.py`). Refit variants (`battle_filter`,
   `design_transform`) use the **baseline** design's weights keyed by
   `battle_id` (1 for a battle not in the baseline), so a variant isolates its
   own effect and does not re-measure density.
3. **`cmd_method`.** Add `cmd_method: np.ndarray` (aligned with `cmd_side`)
   to `ModelDesign`, from `CommanderInput.attribution_method`, so
   `rule_single_half` can be built from the design.
4. **Report readers.** `pipeline/reporters/load.py` reads `war_contribution`
   as the per-battle `expected_war`: load `aggregation_weight`, show the entry
   xWAR and the weighted contribution, so the explorer's battle rows sum to
   the total. Update `pipeline/inspector/queries/model.py` too.
5. **Primary war** = the war with the largest `sum a_e` (most weighted
   battles), ties to the larger within-war WAR. Not the best war, which would
   flatter.
6. **Within-war uncertainty.** Carry `war_id` (per battle) into `ModelDesign`;
   `reduce_draws` computes per-(general, war) weighted totals per draw; pool
   with `mixture`; report the within-war rank on the pooled mean with a 90%
   rank interval (rank per draw within the war, 5th-95th percentile). The table
   caption says what the interval is.
7. **Density scope.** Add `scope: global | component` to `war_aggregation`.
   Default `global` (the user's decision); `component` (density counted within
   the general-general comparison component, the M2 identification unit) is an
   evaluate variant `war_density_component`. Reason: global density also
   down-weights small wars fought beside a dense one (Paraguay, Schleswig 1864,
   Taiping get about 0.5 next to the ACW). The run log lists the 20 generals
   outside the densest component with the lowest mean weight. The user may
   switch the default after seeing the variant.
8. **MINORs.** Realised totals are weighted (as section 3) and labelled
   density-weighted in the explorer payload. `method: none` writes
   `aggregation_weight = 1.0` (NULL only on runs before the migration).
   Density uses integer years (`np.rint`, assert integral) so window edges are
   exact; at the ends of the corpus's year range the window is half empty (note
   it, no fix). Within-war denominators apply `min_battles` as the global table
   does. The run-52 re-measurement under the `/(2w+1)` definition comes before
   `density_ref` is committed.
9. **`method: none` exactness.** Keep the existing per-entry loop for the
   None path (no `np.add.at` or matmul, which change summation order). Test
   with `np.array_equal`.
10. **Disclosure.** Within one war `a_e` is nearly constant, so the within-war
    rank is still close to an unweighted volume rank: it is honest about
    cross-era identification, not about volume.

Added tests: `_post_hoc`/`repool` with only the aggregation changed gives a
different total; refit-variant weights equal the baseline's by `battle_id`;
`p_above_next_rank` and the Rubin `war_sd` on weighted totals equal those from
`a * entry` draws directly; a merged camp plus a latent opponent with `a < 1`
still decomposes; `rule_single_half` halves only `rule_single` entries; the
primary-war tie-break; the within-war rank interval on a fixture; a
report-gate integration test with a deliberately broken weight fails.

## 9. Measurement and implementation

**Measured 2026-11-10 on run 52** (1535 included battles, 0 undated; ACW =
war_id 3, 436 battles counting theatre wars 36/66, which have none included):

| d under /(2w+1), w=1 | p5 | p25 | p50 | p75 | p90 | p95 | max |
|---|---|---|---|---|---|---|---|
| all | 0.9 | 9.8 | 27.7 | 62.7 | 101.3 | 127.3 | 127.3 |
| ACW | 52.3 | 91.0 | 101.3 | 127.3 | 127.3 | 127.3 | 127.3 |
| non-ACW | 0.67 | 5.7 | 19.0 | **30.0** | 36.0 | 38.7 | 127.3 |

Non-ACW p75 is 30.0 (design: 29). With d_ref 29: ACW median a 0.535 (all
below 1), non-ACW median 1.0 (25% below 1). With 30: ACW median 0.544.
`war_id` coverage 82.5% (269 included battles NULL, the "no war recorded"
bucket). Wars 27 and 28 are both "Arab-Byzantine wars" (capitalisation
duplicate in `wars`): merge before within-war ranks are published.
Run 52 gates false: fmi_top_generals, top_generals_cross_component,
posterior_predictive_category_check.

## 10. Session 2026-11-10 (second half): data-fix drafts and rule_single

**DB access.** No `DATABASE_URL` was set anywhere; earlier sessions used the
allow-listed `export DATABASE_URL=...` prefix or a run script holding it
(refit7.sh, gone with w-c3seed). The classifier blocks agents reading the
password from `.claude/settings.local.json`. The user set `DATABASE_URL` in the
Windows User environment, but Postgres rejected that password (auth failed for
`general_war`): re-set it with the working password, then agents read it via
`os.environ` falling back to `winreg` HKCU\Environment and swap the DB name.

**Drafted, awaiting sign-off** (`43-drafts/misc_draft_ov0700-0704.yaml`; loads
with config, 486 entries; slugs and side labels match one row each):
- ov-0700 Actium side 125 is **Octavian's** (Agrippa's own side): infobox
  "16,000 infantry on ships / 3,000 archers"; the parser put 3,000 in `total`.
  Point 19,000, engaged.
- ov-0701 Chester Station, ov-0702 Swift Creek: **will not apply as drafted.**
  `match_attribution` (pipeline/classifiers/overrides.py) requires the listed
  generals to equal the side's linked commanders, and the side links only
  Beauregard (Robert Ransom Jr. is not in `generals`; whether Bushrod Johnson,
  Q434337, is on the Swift Creek side is unchecked). Options for the user:
  (a) `commander_link: none` on the Beauregard mention, so the side becomes
  unattributed (nobody credited); (b) a new override path that adds a
  commander to a side (extract/resolve change). (a) is available now.
- ov-0703 "Capt. Cleaver" (Second Mesilla CSA): a real junior officer, not a
  parse error; unlinking him gives Hunter the side, mirroring ov-0608.
- ov-0704 Brice's Cross Roads CSA 3,500 engaged (all cavalry).

**Lone-branch sides** (`43-drafts/lone_branch_sides_refit7.txt`): 72 sides were
promoted in refit 7 (87 events; the "62" earlier was the prior-dominated
count): cavalry 30, infantry 30, irregular 12. Of the cavalry ones, about 17 are
whole cavalry forces (Brice's, Massard Prairie, Pine Bluff, Collierville,
Cabin Creek II, Columbus 1865, Legnica, Ticinus, ...), so the upward offset is
wrong for most of them; 4 are mis-scoped (Yellow Bayou, Utica, Vosges, Ruspina)
and 2 are junk ("2 Cavalry troops", "1 cavalry company"). Infantry sides
include unit-designation junk (1, 2, 20, 32).
**Design note (for the user and BMR):** cavalry-only promotion should carry no
offset: promote a lone cavalry figure as an ordinary total (`lone_branch`
False), keep the TN(log 1.5, 0.6) offset for infantry-only. Mis-scoped and
junk rows get overrides or fall to the 100 floor. Irregular: not reviewed.
This is a reconcile prior change, so it goes through bayesian-model-reviewer
and lands before refit 8.

**Nested pairs and campaign rows:** not drafted (DB access). The brief is
ready to re-run as is: credited = rows in `battle_war_details` for the latest
run; signals P361 (`data/corpus3/raw/wikidata/Q*.json`) and infobox `partof`
(`data/corpus3/raw/battles_html/*.wikitext`); ids from ov-0620; flag parents
whose credited children are fragments (Bulge?); flag Berlin and Kursk.

**Decisions 2026-11-11 (user).**
- Retypes ov-0620..0632 approved as drafted (`43-drafts/retypes_draft_ov0620.yaml`):
  nested parents Siege of Petersburg, Siege of Knoxville, Wilmington; operation
  rows Manassas Station Ops, Ambiorix, Barvenkovo-Lozovaya, Tikhvin, Melitopol,
  Polesskoe, Moonsund, St. Lawrence, Battle of Britain, Van Buren raid. Kept as
  battles (children are fragments, or adjacent not nested): Berlin, Gazala,
  Bulge, Stalingrad/Kalach, Warsaw 1939, Kursk, the pockets. The P361/partof
  signals find only 6 credited pairs; the review's "55" was not reproduced
  (campaignbox children are unexpanded templates), so recall is low. Dummy
  1 January dates (Kursk, Atlanta, Nashville, ...) distort date-based nesting.
- ov-0700, 0703, 0704 approved. Chester Station / Swift Creek: option (a),
  `commander_link: none` on Beauregard (both battles Butler v Beauregard,
  indecisive; Ransom not in `generals`, Bushrod Johnson linked to neither).
- Cavalry-only lone-branch figures: no offset (ordinary total); infantry-only
  keeps TN(log 1.5, 0.6). Reconcile change, BMR gate, before refit 8.
- rule_single: write the W < 1 design brief for BMR review; prompt change and
  LLM batch only after. `rule_single_half` variant ships meanwhile.
- `density_ref` = 30.0 (the measured value).

**rule_single finding (blocks the classify-only fix).** A sole commander's
weight cannot be cut below 1 today: `validate_side` requires weights to sum to
1, and `_classify_side` (pipeline/modelers/design.py) renormalises a named
side to 1. Only merged camps carry a named share W < 1 with the rest as a
latent. So "cut weight where there is no evidence of tactical command" needs a
model change: let an ordinary side carry W < 1, the remainder a
`mu_unattributed` share, which the WAR code already supports for camps. Design
choice for the user and BMR before any prompt change or LLM batch. The
classify side of it: route single-commander sides to the LLM with a narrow
question (directed the fighting, or department/theatre commander above an
unnamed field commander), answered only for the about 1,000+ rule_single sides
(ask usage % first).
