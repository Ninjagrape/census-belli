# Handover log §21. Session of 2026-10-06/07: outstanding items closed, impute built

Part of the session log; the index is `docs/handover-log/README.md`.

Run under the team structure now written into `CLAUDE.md`: Opus (lead and
Opus reviewers) designed, briefed and judged; Sonnet workers implemented;
Haiku ran every verification command against the database. Opus wrote no
production code.

### 21.1 The regime-neutral troops fit (§20.10 HIGH 2, §20.11)

**Definition, decided after a `bayesian-model-reviewer` design pass:** the
neutral estimate is `mu_side` from a **second, exact troops fit** on the same
design in which the unlabelled regime's offset takes the faithful-component
prior `N(faithful_mu, faithful_sd)` (that is, `_UNLABELLED_WEIGHT = 0`) and
the unlabelled era term `a_era` takes `N(0, 0.25)`. Labelled regimes keep
their fitted priors, so an attributed ancient 1,000,000 is still deflated.
`build_model(..., neutral=True)`; troops only.

Rejected along the way, and why:
- A per-draw linear-Gaussian shift of the published fit. The reviewer showed
  it leaves out the shift in the shrinkage target (`m0 = corpus_level -
  offset_bar` moves when `g_unl` does), which under-shifts singletons by
  0.09-0.13 log, a bias that tracks report count and therefore fame. It also
  could not represent Student-t downweighting, censored-row information, or
  the re-identification of the labelled regimes and source biases.
- Pinning `g_unl` at exactly 0. Throws away data that identify an
  unlabelled-vs-anchor gap.
- A common constant on every side. Cancels in the ratio and changes nothing.

Also built:
- `merge_cross_regime_duplicates` (`claims.py`), run before
  `collapse_lineages` in both fits: an unlabelled row agreeing within 1% with
  a labelled row on the same side, quantity and branch/casualty type is
  dropped, so an unattributed infobox copy of Herodotus's figure is not read
  as faithful.
- `battle_sides`: `est_troops_neutral`, `_lo`, `_hi`, `_log_mean`, `_log_sd`,
  `est_troops_neutral_run_id` (own provenance, independent of
  `est_troops_run_id`, since one fit can pass the write thresholds while the
  other does not) and `is_defender BOOLEAN` (all NULL until classify fills
  it). Migration `d4e5f6a7b8c9`; Alembic round trip verified.
- `data/processed/reconcile/neutral_draws_run{run_id}.npz` (`run_id`,
  `side_id`, `battle_id`, `draws` [<=1000, n_sides] log scale, `m0_mean`,
  `s0_mean`), the interface impute reads. Written only when the neutral fit
  passes the write thresholds. **No fallback to published ratios.**
- Gates `neutral_fit_converged` (error) and `neutral_estimates_are_not_stale`
  (warning). The `troops_neutral` diagnostics block is kept out of the
  combined `worst`, so `model_convergence` still means the published fits.

**Measured, on a fixture where every source type appears on both sides of 8
battles** (the first fixture had each type on one side only, so `b_type` and
`s0` absorbed the between-side difference and shrank a raw ratio of 0.80 to
0.31; the reviewer caught it):

| case | published log ratio | neutral |
|---|---|---|
| Gaugamela, Persian labelled, Macedonian unlabelled | 1.115 | 0.843 |
| same, Macedonian relabelled `modern_scholarly` | 0.737 (+ faithful_mu 0.128 = 0.865) | — |
| both sides unlabelled | 0.725 | 0.714 |

Neutral matches "relabel as modern scholarly" within 0.022, and the published
tilt of 0.27 in the mixed case is what HIGH 2 predicted. On the 302-side
recovery corpus the neutral fit takes ~73s, rhat 1.009, ess 462: only 15%
headroom over the 400 write threshold.

### 21.2 Classify: the historiography review, and what it fixed

The `historiography-reviewer` pass that §20.7 deferred ran on the rules and
`arsht_infobox.jsonl`. Its central finding: infobox commanders all arrive as
role `unknown` (`infobox.py` sets `apparent_role="unclear"`), so the
deterministic **fallback is the main path**, not the edge case, and
0.6/0.4-by-listing-order favoured heads of state and nominal commanders, the
first-listed in infoboxes. Estimated (not measured) 60-70% of multi-commander
sides got materially wrong weights. Listing order cannot even be measured:
the reference file is alphabetical and `bc_id` is resolve's insert order.

Fixed: an LLM answer that fails to match every commander on the side
(full name, then unique surname) is rejected whole, where before the
unmatched commander took weight 1.0 (about half the side); the default split
is 1/n; two or more field commanders are co-commanders (Cannae); non-field
roles get fixed weights (sovereign/nominal 0.05, supreme/theatre 0.10,
subordinate 0.08, capped at 0.30 in total) so Agrippa keeps 0.79 at Actium
with subordinates listed; `validate_side` checks weights, rank 0 and
`reports_to` cycles after every path; bare "squadron" no longer makes a
battle naval, and a lone naval troop branch does not either; the prompt
matches the code and has a `not_a_person` escape.

Deferred, in TODO.md: infobox order and unresolved-commander counts,
non-person commanders in resolve, few-shot cases (Munda, Mutina, Chaeronea,
Salamis, Mongols, Moltke), per-side siege, casualty-specific missingness.
**Decision for the model stage: it must not consume `default_split` sides
without an LLM or manual role.**

### 21.3 Team structure

`CLAUDE.md` "How to work" now states it plainly: Opus is manager and
architect, Sonnet and Haiku are workers; Opus does not write production code,
tests, or run the suite; a Haiku agent runs verification and Opus reads the
raw output; one agent at a time on `DATABASE_URL`.

2026-10-09: routing updated for Haiku 5.5 ($0.10/$0.50 per MTok against
Sonnet 5.5's $2/$10). Haiku is now the default worker for any task with
mechanical acceptance criteria; Sonnet is the single escalation step and owns
anything touching the statistical stages. Rules are in `CLAUDE.md` "How to
work".

Same day, the pipeline's LLM fallback moved from `claude-haiku-4-5-20251001`
to `claude-haiku-5-5` in `agents/extract.yaml` and `agents/resolve.yaml`.
Nothing reads `llm_fallback_*` in code yet; it tells a human what to put in
`llm_model` when Gemini runs out. Two things had to change for that swap to
work, both done:

- `pipeline/llm/capabilities.py` has a `claude-haiku-5-5` entry ($0.10/$0.50,
  no sampling params). The old dated ID never matched the `claude-haiku-4-5`
  key, so cost would have been recorded as NULL.
- Haiku 5.5 thinks unless told not to. New trait `thinking_on_by_default`;
  `AnthropicClient._build_params` sends `thinking: disabled` for such models
  when thinking was not requested and effort is unset/low/medium/high (at
  xhigh/max that is a 400, so it sends nothing). Tests in
  `tests/unit/test_anthropic_params.py`.

Verified: unit suite 681 passed, ruff clean (Haiku worker, raw output read).
Not verified: no live call to `claude-haiku-5-5` has been made. Temperature 0
is unavailable on it, so run `/extraction-eval` on it before relying on it
(TODO.md).

### 21.4 The impute stage

Design reviewed before coding and the code reviewed after (verdict: ready to
commit, no CRITICAL/HIGH). `agents/impute.yaml` and `config/imputation_priors.yaml`
carry the full rationale. In short:

- **One joint PyMC model, not MICE/PMM.** With one substantively missing
  variable, chained equations reduce to one conditional model. `y_s = a_era +
  beta_out + beta_type + beta_def + polity RE + commander RE + u_b + eps_s`,
  categoricals as `ZeroSumNormal` so none is collinear with `a_era`. `u_b` and
  `eps_s` are marginalised analytically: per battle an MvNormal with
  covariance `tau_b^2 J + sigma^2 I + P_lik^-1`, batched by observed-side
  count, every array as `pm.Data`.
- **Observed sides enter as a likelihood**: reconcile's per-battle posterior
  (from the npz) with reconcile's own side prior `N(m0, s0^2)` divided out, so
  nothing is shrunk twice. A side too weak after division (precision <= 0.5)
  is treated as missing.
- **Overimputation.** Each of 20 rounds takes one posterior draw and draws
  *every* side's latent from the conditional Gaussian, observed ones included.
  Drawing observed sides independently of the outcome would attenuate
  `beta_force` downstream (errors-in-variables survives Rubin's rules). An
  oracle test with true parameters gives a slope ratio of 1.00 +/- 0.05 over
  10 seeds; the fitted recovery run gives 0.844 against a naive 0.605, and
  the reviewer traced the gap to the weakly identified sigma/tau_b split, not
  to a defect.
- **Outcome is conditioned on** (the compatibility condition for proper MI);
  commander identity enters only as a size effect, never skill.
- **Scales.** `params.scale: neutral` is primary and the only scale that
  writes `missing_data_log`; `published` is a sensitivity run that writes
  files only. Outputs `data/imputed/{scale}/round_{n}/sides.csv` + manifest,
  and a per-scale `gate_report.json`.
- **MNAR**: primary run is MAR; `mnar_delta` (default 0) shifts imputed MNAR
  sides, for evaluate's grid {-0.5, -0.25, +0.25, +0.5}.
- **Stale input is refused**: the stage raises if the newest draws file's
  run is not the latest reconcile run whose neutral fit was written.
- Gates: completeness, `imputed_ratio_plausible` (only battles with an
  imputed side, median across rounds; Rorke's Drift is a real 3.4, and a
  per-round gate on observed sides fails real battles), the absolute range on
  the published scale only, KS per era as a warning, convergence.

### 21.5 Things a future session would otherwise rediscover

- **Concurrent suites against one database produce `UndefinedTable` errors,
  not hangs.** One verification run here reported 14 reconcile ERRORs and a
  crawl failure, which vanished on a clean exclusive rerun. A Haiku agent had
  been cut off by a rate limit mid-suite and restarted; the most likely cause
  is two runs (or a run plus schema edits) overlapping. Check `tasklist` and
  `pg_stat_activity` before believing an isolation bug.
- **Integration tests cannot exercise a converged `neutral_fit_converged`.**
  The gate SQL hard-codes ess 400 and the integration tests sample 200 draws
  on purpose, so the passing branch is unreachable there; the integration
  test now asserts only that it executes. The failing branch is covered.
  Parameterising the gate thresholds from the spec would close this.
- A worker ran `ruff format pipeline/` across the tree and reverted the
  untouched files by hand. Tell workers to format only files they own.

### 21.6 Verification, stated honestly

All run by worker agents against live PostgreSQL 15.19, raw output read by
Opus, on 2026-10-07:

- `ruff check pipeline/ tests/ scripts/ alembic/`: clean.
- `mypy --strict pipeline/`: clean, **83 source files**.
- Alembic `upgrade head` / `downgrade -1` / `upgrade head`: clean through
  `d4e5f6a7b8c9`.
- `pytest tests/ -m "not live and not model"`: **774 passed, 0 skipped,
  0 failed**, exit 0. (Wall time 9h33m because the laptop slept overnight;
  a clean run takes about 7 minutes.)
- `pytest tests/model -m model`: **24 passed, 1 xfail** (the censoring
  ablation, as before), exit 0, 15m46s.
- Two `bayesian-model-reviewer` passes (design, then code) and one
  `historiography-reviewer` pass, all findings triaged above.

**Not verified:** no real corpus has been through reconcile, classify or
impute; every number above is from fixtures or synthetic corpora. No billed
LLM call. The `historiography-reviewer` error rates for classify are
judgement estimates, not measurements.

### 21.7 Still open

- **Shared reconcile error is not propagated into impute (reviewer M1).**
  Only within-battle covariance is kept and battles are drawn independently,
  so the uncertainty in `g_unlabelled` and `a_era` that moves every
  mixed-regime battle together averages out instead of appearing as
  between-round variance. `beta_force` uncertainty is understated for exactly
  the Gaugamela class. Proper fix: export regime draws in the npz and apply
  one joint draw per round. Meanwhile evaluate must run a +/-1 sd
  `g_unlabelled` sensitivity.
- `est_troops_method` still counts lineages over all reports, so an
  unlabelled copy that the cross-regime merge drops still counts as a second
  lineage.
- The neutral fit's ESS headroom is thin (462 vs 400 on the recovery corpus).
- `missing_data_log.imputed_lo/hi` are the 2.5/97.5 percentiles of 20 rounds,
  close to the min and max; the note on each row says so.
- Imputing troops as latents inside the Bradley-Terry model is the fully
  congenial alternative to MI. It departs from the spec; **the user's call.**
