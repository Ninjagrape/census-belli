# Handover log §20. Sessions of 2026-09-29 to 2026-10-02: reconcile closed out, classify begun

Part of the session log; the index is `docs/handover-log/README.md`.

Run as Opus-led with Sonnet subagents (`CLAUDE.md`). Plan:
`~/.claude/plans/wrap-up-reconcile-as-sleepy-sphinx.md`, which follows the
reconcile design in `start-work-on-the-enchanted-lemon.md` rather than
re-planning it. Two scope decisions were the user's: reconcile got the **core
close-out only** (the four model refinements in §19.9's last paragraph stay
open), and classify is **deterministic first**, with its LLM step wired
through the offline cache path and no billed call.

### 20.1 What was built

| File | What it is |
|---|---|
| `pipeline/reconcilers/summarise.py` | InferenceData to `SideEstimate` / `SourceBias` + diagnostics: `worst`, `identified_level` (`m0 + offset_bar`), `n_identifying_sides` per regime, prior-to-posterior `contraction` |
| `pipeline/reconcilers/store.py`, `report.py` | Every reconcile write; `data/processed/reconciliation_report.json` |
| `pipeline/stages/reconcile.py` | The stage runner. **One `model_runs` row per invocation, covering both quantities** (§20.2 item 4) |
| `tests/model/` | The parameter-recovery suite, 14 tests, `-m model` |
| `pipeline/classifiers/` | `records`, `load`, `roles`, `requests`, `battle_type`, `missingness`, `store` |
| `pipeline/stages/classify.py` | The stage runner; `params.llm_mode: offline` (default, cache only) or `api` |
| `scripts/llm_offline.py` | `_classify_requests` now calls the stage's own builder, so request hashes match (pinned by a test) |

### 20.2 Seven defects, all found by running something, none by reading it

1. **The model's `level` variable shared its name with the `level`
   dimension.** ArviZ then returns the dimension's labels (`engaged`,
   `available`, ...) for `idata.posterior["level"]`, so `summarise_fit` crashed
   on every real fit: no estimate could ever have been written. Found by the
   recovery test; invisible to the summarise unit tests, which built their
   InferenceData by hand. Renamed `corpus_level`.
   `tests/unit/test_reconcile_model.py` builds the model and fails if any
   variable is named like a dimension; confirmed non-vacuous by reverting.
2. **The inflation prior's `fitted_on: 2026-09-24` is a `date` after
   `yaml.safe_load`**, and summarise copied it into a payload that promises
   plain JSON. The store hid it with `default=str`; the report writer then
   crashed. Fixed at the source. The round-trip test now uses a real date;
   with an empty provenance it could never have caught this.
3. **That crash hung the integration suite for 40 minutes, twice.** The
   module-scoped `reconciled` fixture had no `try/finally`, so the stage's
   exception leaked a connection holding row locks on `battle_sides`, and
   every later test that ran the stage waited on them forever. Diagnosed from
   `pg_stat_activity` and `pg_locks`, not from test output, which a hang never
   prints. The fixture now rolls back on failure.
4. **Two `model_runs` rows per stage run would have blinded two gates.**
   `model_convergence` reads only the newest run, so the troops fit's
   convergence would never have been gated; `estimates_are_not_stale`
   compares troop estimates to `MAX(run_id)`, which would have been the
   casualties run, so every troop estimate would have read stale. Found in
   review before the code ran. One run now covers both quantities, with
   diagnostics `{worst, troops, casualties}`; this also closed a
   `sources.bias_run_id` provenance gap.
5. **A classify re-run overwrote the LLM's judgement with a constant.** A side
   the LLM classified looks internally consistent on the next run, so the
   deterministic rule recomputed it from the fixed weight table (Agrippa 0.85
   became 0.867) and relabelled it `rule_roles`. The subagent recorded this as
   intended and loosened its idempotency test to match. It was not intended:
   every re-run would have replaced the Octavian/Agrippa reading of an article
   with a table. A side whose rows are all `attribution_method = 'llm'` is now
   left untouched; the test asserts run 1 equals run 2 and fails without the rule.
6. **`ci_width_reasonable` could never pass.** §19.9 predicted it; the first
   write measured 4.13 against `< 2.2`. In a near-all-unlabelled corpus the
   unlabelled offset is prior-driven and shared by every side, and its prior
   sd (about 0.94 from `moments_for`) alone floors `LN(hi/lo)` at about 3.7
   (3.3 to 3.7 across regimes). Threshold raised to `< 4.5` with the
   arithmetic in `agents/reconcile.yaml`. Intervals were not narrowed.

7. **The model could not compile a real corpus.** Every per-observation
   array (indices, `log_y`, roundness, the censoring row sets) entered the
   graph as a numpy constant, and PyTensor's numba backend compiles constants
   into the generated code as literals. nutpie's compile then grew faster
   than linearly in the corpus. Measured on synthetic corpora, compile only,
   no sampling:

   | observations | before | after |
   |---|---|---|
   | 160 | 55s | 34s |
   | 320 | 307s | 29s |
   | 670 | not finished after 5 min, 11.5 GB, machine at 0.2 GB free | 31s |

   A real corpus of ~3,000 battles could never have been fitted. It also
   explains the recovery run that went 70 minutes without reporting. The
   arrays now enter as `pm.Data` (`_data` in `model.py`, names prefixed
   `data_` so they cannot collide with a dimension). The model is unchanged;
   all 43 reconcile unit and integration tests pass on it, including the
   same-seed re-run test. **Censoring was not the cause**: 320 observations
   compiled in 334s without it. The 11 GB figure came from the 302-side
   recovery corpus on the same code.

Plus one judgement reversed: the classify brief's missingness heuristics
marked missing `battle_type`, `terrain` and `fortified` as **MCAR**. They are
now **MAR**. A missing covariate plausibly tracks documentation level, and
after classify a still-`unknown` battle_type means no article was read, an
observed fact. MCAR is the special case of MAR, so the weaker assumption is
the safe one for impute.

### 20.3 Classify, first slice: what it does

- **Roles.** One commander: `field_commander`, weight 1.0. Several with
  consistent existing roles: kept, weights from a role table normalised per
  side (`_ROLE_BASE_SCORE` in `pipeline/classifiers/roles.py`). Otherwise the
  spec's fallback, 0.6/0.4 by listing order or equal for 3+, role `unknown`,
  flagged for the LLM. **Listing order is `bc_id` order**, a proxy: neither
  extract's infobox order nor `apparent_role` is persisted anywhere classify
  can read.
- **LLM.** `offline` mode looks each pending side's request up in
  `llm_calls` by hash (`pipeline.llm.call_log.find_completed_call`) and
  applies a cached answer; a miss stays `default_split`. No billed call has
  been made; the Actium test seeds a cached answer.
- **battle_type.** Naval from categories, naval-branch troop reports or ship
  vocabulary; siege from the name or a category; `field` **only when an
  article was actually read** and carried no other signal, because defaulting
  everything else to field would be §4.2 in reverse. No article: stays
  `unknown`. Inferring a type marks the battle's `missing_data_log` row
  `observed` with a `classify:` note rather than deleting it.
- **Missingness.** Heuristics only, on `unclassified` rows of fields impute
  consumes; never touches resolve's `mnar` on `commander_general_id`.
  Ambiguous rows stay `unclassified`.

### 20.4 Verification, stated honestly

**Against live PostgreSQL 15.19:** `ruff` clean across `pipeline/ tests/
scripts/`; `mypy --strict` clean on 69 files; **648 passed, 0 skipped**
(`-m "not live and not model"`; 29 deselected, being the 15 live and 14 model
tests). The 12 reconcile stage tests write real estimates and run all 8
gates; the 11 classify integration tests run the stage, the cache path and
all 3 gates.

Gate values on the 5-battle seeded reconcile corpus, which is plumbing and not
evidence about thresholds: `model_convergence` fails at ess 68 (the tests
sample 200 draws on 2 chains, deliberately); `calibration_set_is_not_empty`
reads 2 against 20, correct for 5 battles; all error-severity gates pass.

**The recovery suite, first full run** (`pytest tests/model -m model`,
2026-10-02, after the §20.2 item 7 fix): runs in 5m19s, **9 passed, 5
failed**. §20.9 fixed two of the model causes; the table is kept because the
diagnosis is the reusable part. Before the fix it
never finished. The failures, read rather than loosened:

| test | measured | required | diagnosis |
|---|---|---|---|
| convergence of the main fit | rhat 1.033, ess 172 | rhat < 1.01 | **model**: `tau_lineage` ess 172, posterior 0.075 +/- 0.055, collapsed toward zero, and `tau_sigma` rhat 1.03. Almost every lineage has one report, so the lineage scale is unidentified and its 668 non-centred offsets form a funnel. **Fails the production gate too** (ess >= 400). These are §19.9's open `tau_lineage` and variance-prior items, now measured. |
| bound-as-point vs censored | naive RMSE 0.358, censored 1.958 | naive >= 1.4x censored | **model**: the likelihood direction is right, but a side whose reports are all upper bounds has no information on how far below the figure the truth lies, so at the ablation's 80% bounds the corpus level slides. Treating "up to X" as pure censoring is too weak; a soft bound (point near X with a downward offset and extra variance) is the likely fix. A design decision, not taken. |
| 5 shared-lineage vs 5 independent | width ratio 1.007 | > 1.3 | **test**: `add_lineage_test_sides` draws independent noise for the "shared" side and only labels it one lineage, so the truth never repeats a claim. Must draw one shared error per lineage. |
| RMSE of log estimates | 0.367 | < 0.35 | marginal; re-judge after the convergence fix |
| singleton vs 4-report width | 2.0x | >= 2.5x | marginal; re-judge after the convergence fix |

Passed: source-type biases and rank order, inflation learned from data,
scope offsets, two-sided coverage, singleton coverage, `ess(m0)` with the
anchor present, all-unlabelled identified level, all-ancient era centring,
split-source bias.

**Not verified:** no real corpus has been through reconcile or classify; the
database holds test fixtures only (§7 step 0 is still the gate). No billed LLM
call has been made for classify. The gold sets are agent-built and unreviewed
(§19.6).

### 20.5 Decisions recorded for later stages

- **Siege convention.** The `battle_type` enum has no neutral siege value, so
  classify writes `siege_offensive`, meaning "a siege, besieger's viewpoint",
  plus `fortified = TRUE`. **Flag for the model stage:** `agents/model.yaml`'s
  linear predictor uses `beta_type[battle_type]` and `I(A is defender)`, both
  side-relative, and nothing in the schema yet records which side defended.
  That needs a per-side flag before the model stage can use either term.
- **`ci_width_reasonable` measures the prior, not the sources.** The better
  gate is interval width net of the shared regime factor; not built.

### 20.6 Still open

- Of the four §19.9 refinements, the lineage one is done (§20.10); the
  prior predictive check, the per-side casualty ratio and `InverseGamma(2, 1)`
  remain open, by decision.
- **The reviewer's remaining findings (§20.10)**, none applied, all for the
  user to prioritise. HIGH 2 must be settled before the model stage.
- Classify: LLM missingness classification; the offline batch for pending
  sides; the defender flag (§20.5).

### 20.7 Review gates

The `bayesian-model-reviewer` pass **ran on 2026-10-02** (§20.10). The
`historiography-reviewer` pass on classify hit the account's usage limit and
**has not happened**; run it before relying on classify's role weights.

### 20.8 A gotcha this session paid for

**Never run two integration suites against `DATABASE_URL` at once.** Every
module fixture drops and rebuilds the `public` schema, so parallel agents
block each other on locks, and a hang prints nothing. CI runs files serially,
so only parallel local agents hit this. If a run goes quiet, read
`pg_stat_activity` with `pg_blocking_pids()` before assuming the sampler is
slow. Wrap long local pytest runs in `timeout` so a hang fails fast.

### 20.9 The recovery suite forced two more model fixes (2026-10-02)

After the user chose to fix the lineage term rather than defer it:

1. **Lineage effects only for repeated claims.** *Superseded by §20.10: the
   reviewer showed this made singletons overconfident.* `_lineage_effect` in
   `model.py` gives a random effect only to lineages with two or more
   reports, in a new `lineage_shared` dimension; singletons contribute zero.
   A one-report lineage's offset cannot be told apart from that report's
   noise. Before: `tau_lineage` ess 172, posterior 0.075. After: 0.23, and
   no longer among the weakest variables.
2. **`z_sigma` zero-sum within source type.** The per-source noise
   deviations were a plain Normal, so their mean within a type was the same
   parameter as that type's `sigma_sq_type`: the §19.9 `u_source` ridge, on
   the noise scale instead of the bias. Before: `tau_sigma` ess 201, rhat
   1.02, `sigma_sq_type` sd 0.16. After: sd 0.03, `tau_sigma` gone from the
   weakest list.

Two test corrections, each argued rather than loosened:

- The shared-lineage test side drew independent noise per report and only
  *labelled* them one lineage, so the truth never repeated a claim. It now
  copies one figure five times. The test passes.
- The singleton/four-report width ratio required >= 2.5. Four independent
  reports narrow an interval by sqrt(4) = 2 at most, and shared terms pull it
  lower; measured 2.0. Now >= 1.8.

At this point the suite stood at 10 passed, 1 xfail, 2 failing; §20.10 is
where it ended.

Verified after both fixes: ruff clean, `mypy --strict` clean on 69 files,
**648 passed, 0 skipped** against live Postgres.

### 20.10 The bayesian-model-reviewer pass, and the lineage collapse

The review confirmed the `z_sigma` fix, measured convergence as acceptable at
the spec's 2000 draws (max rhat 1.006, ess 584), and found the RMSE threshold
wrong rather than the model (RMSE 0.366 against a stated posterior sd of
0.377; coverage 0.56 / 0.84 / 0.98 at 50 / 80 / 95%, slightly conservative).

**It also showed §20.9's lineage change was unsound (HIGH 1).** Zeroing
singleton lineage effects dropped their variance from sigma^2 + tau^2 to
sigma^2 while `sigma_source` stays shared. In production a lineage is a
Wikidata or DBpedia copy within 1% of a Wikipedia figure, so near-zero scatter
drags `sigma_source` down and non-copied reports inherit it: singleton 95%
coverage fell to 0.65-0.72 on a stress corpus, against 0.90-0.92 before. The
`tau_lineage` collapse that motivated the change was an artefact of a recovery
corpus with one shared lineage.

**Fix applied, the reviewer's preferred one:** `collapse_lineages` in
`claims.py`, called at the top of `build_design`, keeps one row per lineage
(the originating, non-derived report). Copies add no information, so the model
no longer has a lineage term, `tau_lineage` or `lineage_prior_sd` at all.
Copies still count in `n_reports` and `n_sources`. A source whose only
reports were copies gets no fitted bias and keeps its prior, which is correct.
The design and summarise test helpers defaulted every report to lineage 0,
which collapsed whole test corpora to one row; they now default each report to
its own lineage, as `assign_lineages` does in production.

Also applied: the RMSE test now asserts calibration (RMSE within 0.85-1.15 of
the stated posterior sd, four-report sides below 0.25) instead of a fixed
0.35, and the main fit samples the spec's 2000 draws.

**Result: recovery suite 13 passed, 1 xfail (censoring), 0 failed, 5m10s.**
Main fit max rhat 1.007, ess 466, 0 divergences. Full suite 648 passed,
0 skipped; ruff and `mypy --strict` clean.

**Findings not applied, for the user to prioritise:**

- **HIGH 2: asymmetric labelling biases force ratios.** The unlabelled
  offset cancels in a force ratio only when both sides share a regime mix. At
  Gaugamela the Persian field carries modern and ancient labels and lands near
  the modern figure, while the Macedonian 47,000 is unlabelled and is deflated
  about 2.5x, tilting the ratio toward "Alexander outnumbered" and inflating
  his WAR. Labelled fields cluster on famous ancient battles, so this lands on
  the top of the ranking. **§19.9 point 3's "the ranking is largely
  insulated" is wrong in this case.** Settle before the model stage: count
  battles whose sides differ in dominant regime; apply the unlabelled
  adjustment only where both sides are unlabelled, or compute force ratios on
  a regime-neutral scale; at minimum run a `_UNLABELLED_WEIGHT = 0`
  sensitivity variant.
- **MEDIUM 1: both width gates now measure the prior.** The shared unlabelled
  offset floors `LN(hi/lo)` near 3.7, so the error-severity
  `singleton_estimates_are_honestly_uncertain` (below 0.8) can never fire on a
  mostly-unlabelled corpus. Fix: compute widths of `mu_side + offset_bar` (the
  identified scale) in summarise and gate those at 0.8 and 2.2; keep 4.5 on
  the published intervals. Also, the YAML comment's "measured 4.13 on the
  first real write" was the 5-battle seeded fixture, not real data.
- **MEDIUM 2: `n_identifying_sides` counts the wrong thing.** Two reports in
  one regime identify sigma, not `g_regime`. Count co-occurrence with the
  anchor's connected component instead. `contraction` still catches the
  failure meanwhile.
- **MEDIUM 4: censoring is defensible on the real mix, with guards.** In
  `arsht_infobox.jsonl` 18% of quantities carry bounds but only 0.6% of sides
  are one-sided-only; ranges become interval censoring, which is right. Label
  one-sided-only sides `bound_only` in `_side_estimates` (a single range is
  currently labelled `source_disagreement`, because its two endpoints are two
  lineages) and rebuild the xfail ablation on the production mix. Principled
  replacement later: `pm.ExGaussian` on bound rows.
- **MEDIUM 5: the absolute level is set by `b_type`'s zero-sum.** `mu_side` is
  what the average source *type* would report for an anchor claim, not "what a
  modern scholar would say" as `model.py`'s device 4 claims, so adding a
  source type shifts every absolute figure. It cancels in ratios. Document it,
  or pin an observed reference type.
- **LOW:** `z_side` as `ZeroSumNormal` removes the `corpus_level` ridge
  (corr -0.948 to -0.007); `model_convergence` should use `max_rhat: 1.01`;
  centre `data_era` within the unlabelled rows; write estimates only after the
  convergence gate passes, so an unconverged run cannot overwrite good ones.

### 20.11 Wrap-up decisions, 2026-10-02

Taken by the user at the end of the session:

- **HIGH 2 (asymmetric labelling): force ratios move to a regime-neutral
  scale.** Not built; it needs schema work, so it is the first task next
  session. Spec: reconcile writes, beside each published `est_troops_total`,
  a second per-side estimate that carries no claim-regime adjustment, so the
  two sides of a battle are on a common footing whatever their labels. New
  `battle_sides` columns in **both** `config/schema.sql` and an Alembic
  migration (§5.1). The model stage builds `log(force_ratio)` from those
  columns; published troop counts stay adjusted. Before building, decide
  exactly what "no claim-regime adjustment" means for a side whose reports
  are themselves ancient claims (Persian 1,000,000): the neutral scale must
  not reintroduce the inflation the regime term exists to remove. Run the
  `bayesian-model-reviewer` on that definition before coding it.
- **Done: `bound_only` method label.** A side whose every report bounds it
  from one direction gets `est_*_method = 'bound_only'`; a range (an upper
  and a lower bound) does not. Pinned in `tests/unit/test_reconcile_summarise.py`.
- **Done: write only after convergence.** `_fit_quantity` writes estimates and
  biases only when the fit's worst diagnostics meet `write_max_rhat` (1.05),
  `write_min_ess_bulk` (400) and `write_max_divergences` (0), matching the
  `model_convergence` gate. An unconverged run is still recorded with its
  diagnostics, so the gate fails, but the previous estimates stand. Pinned by
  `test_an_unconverged_fit_writes_no_estimates`. The integration tests relax
  the thresholds in `_load_spec` because they sample 200 draws on purpose.
- **Deferred to next session:** the `historiography-reviewer` pass on
  classify, and the remaining §20.10 items (identified-scale width gates,
  `n_identifying_sides`, the `z_side` ridge, `max_rhat: 1.01`, era centring
  within unlabelled rows, the `b_type` reference-type documentation).

Final state: ruff clean, `mypy --strict` clean on 69 files, **650 passed,
0 skipped** against live Postgres; recovery suite 13 passed, 1 xfail.
Nothing committed.

---
