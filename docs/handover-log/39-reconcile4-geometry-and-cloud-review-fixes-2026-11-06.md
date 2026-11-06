# §39. Corpus3 reconcile run 4, the ESS geometry diagnosis, and fixes from the cloud-commit review (2026-11-06)

The §38 plan ran as far as (c). The Kalka bug is fixed. More draws and a low-rank mass matrix did not clear the troop ESS gate (130 -> 160), so the cause is the model's geometry, and a reparameterisation is being diagnosed on a scratch DB. In parallel, a code review of the cloud commits found one HIGH and four MEDIUM bugs in main's extract and resolve. All five are fixed in main's working copy, uncommitted.

## 1. Commits and uncommitted state

| Where | What | State |
|---|---|---|
| main 519e60d | docs for §38 | committed this session |
| w-c3seed (wip/corpus3-seed, 2808819) | `drop_superseded_by_override` in `pipeline/reconcilers/load.py` (+ counter in records.py, 5 tests); opt-in `mcmc_low_rank_mass_matrix` (`SamplerSettings`, `build_sample_kwargs`, 16 tests); `agents/reconcile.yaml` tune 4000 / draws 8000 / low-rank false | uncommitted |
| main working copy | extract head-noun fix (textnorm.py, infobox.py, tests); wikilink identity fixes (links.py, wikilinks.py, 2 new test files); overrides ov-0035/0105/0108 retired, ov-0126 retargeted; footgun in 05-footguns.md; this file | uncommitted |

Patches of both main fixes are also in the session scratchpad (`extract_fix.patch`, `wikilink_fix.patch`).

Later the same night:
- **Tilde ranges.** `_DASH_RANGE_RE` in textnorm.py did not accept `~`, `～` or `〜` between numbers, so Yonan's "500~1,400" became two points. Fixed, with 9 tests in `tests/unit/test_extract_tilde_range.py`. Corpus impact: bound flags only, on Yonan (both sides) and Yongin (Toyotomi Japan). Passage hashes unchanged.
- **Overrides ov-0600..0606** (user-approved, from the noise-gate audit in section 4) were added to both overrides files: main 471 entries, w-c3seed 474. Each target matches exactly one corpus3 side.

Suggested commits on main (the user runs them):
1. `fix(extract,resolve): head noun decides equipment counts; tilde ranges; wikilink identity must name the mention`, covering:
   - textnorm.py, infobox.py, links.py, wikilinks.py;
   - tests: test_infobox_derived_personnel, test_infobox_wikilinks, test_wikilink_extract_fixes, test_wikilink_resolve_fixes, test_extract_tilde_range.
2. `chore(overrides): retire ov-0035/0105/0108, retarget ov-0126, add ov-0600..0606` (config/overrides.yaml).
3. `docs: handover-log §39, editable-install footgun` (handover.md, 05-footguns.md, handover-log README and §39).

w-c3seed commits wait for A+B to be reviewed.

## 2. (a) Kalka: an override did not replace a modern infobox row

- Root cause: `_apply_troop_strength` (`pipeline/extractors/override_apply.py:222-262`) stores the override row labelled "(modern estimates)". The only supersede rule, `drop_superseded_by_modern`, keeps every modern-regime row, so Kalka's infobox "~60,000-100,000 (modern estimates)" pair survived next to ov-0103.
- Fix: `drop_superseded_by_override` runs after it. On a side with an override troop row it drops every other troop row and counts it in `excluded_superseded_by_override`. `drop_superseded_by_modern` now never drops an override row.
- Corpus3 (read-only check): 30 overridden sides; only Kalka changes (2 rows). The 6 unkept override rows are on campaign battles, which the loader excludes by design.
- Verified: unit 2429, ruff, mypy; integration `-k "reconcile or prior_check"` on scratch `general_war_erarot_it` 82 passed 0 skipped (`w-c3seed/data/logs/c3_it_reconcile_20261106.log`).
- bayesian-model-reviewer (general-purpose Opus with the spec): no BLOCKER. Exclusion is right. MINORs for later:
  1. After the fix, `manual_entry` never co-reports, so its `sigma_type_modern` just returns its prior. Consider pinning it and leaving it out of `noise_not_leaking`.
  2. Guard: log any side where a `manual_entry` key shares a side with another key.
  3. Ancient override rows get ancient noise on top of their range variance (conservative, undocumented).
  4. Split sampler params per quantity: casualties do not need the troop draws.

## 3. (b)+(c) Reconcile run 4: still fails ESS

- Run 3's `diagnostics->'troops'->'worst_variable'`: ESS `corpus_level`, rhat `sigma_type_modern`. Its noise gate 1.514 was `wikipedia_infobox` (0.334 / 0.22), not `manual_entry` (0.249).
- User chose the low-rank mass matrix (nutpie 0.16 `adaptation="low_rank"`), tune 4000, draws 8000 (memory: about 3 GB of trace per fit).
- Backup `data/backups/general_war_corpus3_20261106_pre_reconcile4.dump`. Log `w-c3seed/data/logs/corpus3_reconcile4_20261106.log`. Load: `excluded_superseded_by_override` 2, troop reports 2489.
- **Troop fit after 6 h 17 min: rhat 1.035, min ESS 160, 0 divergences. Not written.** Low-rank barely helped (run 3: 130 from 4 x 5000).
- The user stopped the run during the neutral fit. Run 4's `model_runs` row on corpus3 is unfinished; no estimates were written.
- Reading: the data identify only `corpus_level + s_b*mean(zb)` (about 1650 battles pin their levels), so the sampler walks a long ridge. A sampler setting cannot fix that. A Sonnet diagnosis (section 4) tests sum-to-zero `zb`/`zd`, which changes the parameterisation: bayesian-model-reviewer and user sign-off are needed before it lands.

## 4a. Noise-gate audit (multi-row sides, baseline trace)

- 34 sides have 2+ rows, all infobox; 33 have 2+ point rows. Their pooled within-side sd is 0.369, and the top 5 sides carry 47% of the within-side sum of squares (about 0.29 without them).
- Top sides:
  - Yonan, both sides: a tilde range split into two points (a bug, now fixed in main's extract).
  - Tarutino, Jena, Polotsk, Mariazell, Rovereto: available vs engaged pairs, which the scope offset absorbs.
  - Odesa 1941 (both sides), First Badajoz, Dürenstein (both sides), Ratisbon: initial vs total, or beginning vs end.
  - Ajnadayn: two modern estimates, 10k vs 20k.
- Table: scratchpad `essdiag/audit/multirow_sides.md`.
- historiography-reviewer drafted ov-0600..0607. The user approved 0600..0606, held 0607 (Ratisbon: the French 72,000 may be a total), and Ajnadayn gets no override (genuine scholarly disagreement).

## 4. ESS diagnosis (scratch DB `general_war_c3_diag`, restored from the pre_reconcile4 dump)

Sonnet diagnosis, 4 x 1000 after 1000 tune. Scripts and traces are in session scratchpad `essdiag/`. The design is `essdiag/DESIGN_A_B.md`, and it should be copied into the repo if adopted.

- **My first hypothesis was wrong.** Global sum-to-zero `zb`/`zd` did not help: `corpus_level` ESS went 16 -> 8. The ridge between `corpus_level` and `mean(zb)` is mild (corr -0.65).
- **Reviewer (Opus, from the traces):** the correlations among m0, offset_bar, g_unlabelled, a_era and e are algebraic identities, and those free coordinates have ESS 75-114. The real blockers:
  1. The per-era-bin mean of the non-centred Student-t `zb` duplicates the era level `u_k`, a prior-only ridge (corr -0.70 to -0.94; `u_k` ESS 9-40). This is the N3 lesson again, through `b`.
  2. `log(s_d/sigma_ibx)` has ESS 11 and rhat 1.31. Single-row sides identify only `s_d^2 + sigma^2` (0.480 +- 0.022), so moving along the ridge rescales all 2088 `zd` at once. No mass matrix can do that.
  3. MAJOR: `s_b` vs rms(`zb`), corr -0.87.
- **Design, user-approved 2026-11-06:**
  - (A) centre `zb` within era bin. This is a stated prior change, with an expected |dlog mu_side| < 0.01.
  - (B) marginalise `d` exactly on the ~1970 single-row point sides (troops, Normal family only). The posterior is unchanged. It includes the `prior_check` mu_side fix.
  - Optional (C) a centred `b` as a test-only arm.
- **Noise gate:** it really sits at about 1.49-1.51, so it stays on a knife edge after any ESS fix. The data alone put infobox noise near 0.37, coming from the ~34 multi-row sides. Hypothesis: merges remove agreeing copies, so multi-row sides are selected for disagreement. **User chose:** audit those sides first (ranking script, then historiography-reviewer), and decide on re-stating the gate only after that. Raising `noise_modern_median` to fit the data was rejected.

RESULTS of the A+B build and the 3-seed scratch check: see section 8.

## 5. Cloud-commit review (Sonnet, 2808819..40172bf) and fixes on main

The cloud session expired; this replaces the review it was meant to return.

- **HIGH, fixed:** `is_equipment_count` scanned the whole phrase, so "7,000 cavalry mounted on horses" and "5,000 infantry supported by artillery batteries" were equipment. Real headcounts dropped from totals, and `_complete_total` promoted stray ship counts (1-6) or a small component (Actium Antony 2,000) to the side total.
  - Fix: the first personnel or arm noun decides (`_ARM_NOUN_RE`), more vessel words, and a non-personnel count under 100 is never promoted to total.
  - On the 3,304 corpus3 articles the change alters branch labels only, never values: 38 sides `total -> other` (ship and unit counts) and Fort Pulaski Union 10,000 `other -> total`.
  - Left alone: Giarabub (3, from "2 squadrons" + "1 battalion") and Arlington Mills ("squad of 9", now no total). Both are below the plausibility floor.
- **MEDIUM, fixed (wikilink identity):**
  - A link only becomes the mention's identity when it names the mention (not "son of [[X]]"). A new `ignored_name_mismatch` counter covers the rejects.
  - Links attach by whole-word match with name coverage, so a substring can no longer attach them.
  - Targets are decoded with `unquote` and `html.unescape`.
  - Unanswered titles are counted (`unanswered_title`) and warned about (`wikilink_fetch_failures`).
  - **No quality gate yet:** a gate needs a `pipeline/quality.py` handler. Fetcher retries were already 3 attempts, base 2 s, honouring Retry-After.
- **Pinned old-vs-main extract comparison** (scripts `37-scripts/`, `PYTHONPATH` pinned, see the footgun): passage text identical for 3,304 articles and 24,108 passages, so the LLM passage cache stays valid. 7 heading labels fixed: Okinawa `United States`, Quatre Bras `United Kingdom`, La Suffel, Lutetia, Poljana, Timor, Breda.
- **Overrides (user-approved):** ov-0035, ov-0105 and ov-0108 retired, because main's extract no longer produces "Main:" or "Ming:". ov-0126 is retargeted to `United Kingdom`, and ov-0534/0536 are confirmed. 464 entries load. Ulsan's "Ulsan castle" and "Reinforcements" are still commander mentions: a lead for the historiography reviewer.
- Main integration suite before these fixes: 374 passed, 0 failed, 0 skipped (`data/logs/main_integration_20261106.log`, scratch `general_war_endash_it`).

## 6. Footgun found

`pipeline` is pip-installed editable from the main checkout. `python path/script.py` imports main's code from any worktree, so pin it with `PYTHONPATH` (05-footguns.md). The earlier unpinned comparison was void and was re-run.

## 7. Housekeeping

- Worktrees removed: w-erarot, w-campx, w-endash, the wikilink worker's. Still there: w-c3seed (corpus3 data and the uncommitted reconcile work) and the extract worker's (locked by its finished agent; remove with `git worktree remove --force`, then `git branch -D worktree-agent-a980b5f84f241483e`).
- Branches deleted: all merged `wip/*` and `feat/offline-llm-and-resolve-eval`, `wip/cloud-extract-resolve-fixes`, and (force) `wip/noise-exp` 9d0e0d3 and `wip/reconcile-era-rot` eba92f3. `origin/wip/corpus3-seed` deleted on GitHub. **main is 18+ commits ahead of origin: the user pushes from GitHub Desktop.**
- Scratch DBs: new `general_war_c3_diag`. Older: `general_war_endash_it`, `general_war_c2_erarot`, `general_war_erarot_it`, `general_war_corpus3_recon`. Ask before dropping.
- Corpus3 data layout (user OK'd the direction): `data/corpus3/{raw,processed,llm_requests,llm_responses,logs,imputed}` plus a `GENERAL_WAR_DATA_ROOT` env var (default `data`), because every spec hard-codes `data/...`. Not built yet; do it after corpus3 reconcile passes, then remove w-c3seed.

## 8. A+B built (w-c3seed, uncommitted) and the 3-seed scratch check

- Implemented as designed:
  - model.py: `_battle_bin_projector`, `_marginal_side_mask`, `_row_of_side`. `_troop_side_prior` returns `(location, s_d, zd)`. `mu_side = side_loc + d_marg` on the marginal sides. The likelihood sd is `sqrt(var + s_d^2)` there. `row_eta`/`row_sd` stay conditional on d.
  - prior_check.py rebuilds the prior strength from m0, e, b, s_d and zd.
  - New tests: `tests/unit/test_reconcile_marginal_d.py` (10, including the quadrature check of B's exactness to 1e-5).
  - Two tests in `test_reconcile_p1_model.py` were edited because they encoded the uncentred b. The reviewer checks that the edits are legitimate.
- Verified: unit 2455 passed, ruff and mypy clean. **Not run:** model/recovery tests, integration.
- Scratch check (`general_war_c3_diag`; 4 x 1000 after 1000 tune; 0 divergences on every seed). Scripts and pickles in scratchpad `essdiag/abcheck/`. Baseline ESS was 9-40.

| | seed 20260923 | seed 1 | seed 2 |
|---|---|---|---|
| corpus_level ESS / rhat | 195 / 1.020 | 264 / 1.008 | 17 / 1.19 |
| u_c ESS | 52-307 | 285-583 | 8-40 |
| s_d | 74 / 1.053 | 228 / 1.009 | 181 / 1.216 |
| s_b | 67 / 1.081 | 164 / 1.023 | 32 / 1.12 |
| sigma_modern[ibx] | 54 / 1.067 | 268 / 1.007 | 20 / 1.16 |
| gate ratio | 1.475 | 1.460 | 1.526 |
| mu_side vs baseline, p99 / max abs dlog | 0.077 / 0.113 | 0.044 / 0.069 | 0.22 / 0.26 |

- `s_d^2 + sigma^2` = 0.479-0.481 on every seed.
- Reading: a large improvement on 2 of 3 seeds. Seed 2 still has chains out of step. `s_b` is the weakest parameter on every seed (reviewer MAJOR 3; option C, centred b, is the candidate fix). These fits used 1000 tune, while the stage uses 4000. The mu_side differences are measured against a baseline with ESS of about 10.
- Reviewer verdict on A+B and the next steps: section 9.

## 9. Reviewer verdict on A+B (Opus, read-only on the traces; scripts in scratchpad `essdiag/review2/`)

- **No BLOCKER, no MAJOR.**
  - B is exact; A, `row_eta`/`row_sd` and prior_check are correct.
  - The two edited p1 tests are legitimate (one is now stronger).
  - MINORs:
    1. Also assert `b == s_b*(zb - binmean(zb))`.
    2. The PPC redraws b iid, ignoring the centring (negligible).
    3. The explicit sides' zd bin mean is weakly duplicated (leave it).
    4. A comment cites the wrong docstring heading.
  - The design is now in the repo: `docs/handover-log/39-design-reconcile-A-B.md`.
- **Seed 2 was one frozen chain**, not a second mode: chain 2's marginalised zd (exact posterior N(0,1)) had sd 0.078 and lag-1 autocorrelation 0.996, so its step size collapsed in tuning. With that chain dropped, seed 2 passes (corpus_level 336/1.004, u_c 396-681). Option C would not prevent this. 1000 tune for about 3300 dimensions is short; the stage uses 4000.
- **mu_side differences are Monte Carlo noise:** two A+B seeds differ from each other (p99 0.084) more than either does from baseline. My earlier p99 < 0.02 / a_era-within-0.05 criteria were tighter than the noise floor, so they are replaced by "agrees with another A+B seed". corpus_level is not comparable with the baseline under A; the L_k agree.
- **Weak spot:** s_b ESS 67-164 per 4000 (corr with rms zb -0.86 to -0.89; MAJOR 3). That projects 540-1300 at 32000 draws. Option C (centred b) is plausible but only as a fallback.
- **Noise gate:** 1.46-1.48 on current data. With ov-0600..0606 applied, the projection is about 1.25-1.35 (sigma_ibx about 0.27-0.28, s_d about 0.64).

### Next steps (reviewer, adopted)

1. Add a frozen-chain diagnostic: the per-chain sd of the marginalised zd (expect 1, flag < 0.5) plus per-chain step size, and fail loudly. Fold in MINOR 1 and 4.
2. Scratch check at stage tuning on `general_war_c3_diag`: tune 4000, draws 2000, 4 chains, target_accept 0.99, seeds 2, 1 and 20260923, low-rank off. Accept on:
   - 0 divergences and no frozen chain;
   - rhat ≤ 1.02 on the globals;
   - s_b ESS ≥ about 0.0125 of draws;
   - L_k agree across seeds within 0.01.
3. Only if s_b misses, or a chain freezes again: C as an opt-in `battle_effect_centred` (default false), and/or low-rank on. Re-check on 3 seeds. C needs the recovery and contamination model tests.
4. Model tests one file at a time (recovery, p1_recovery, contamination), plus `tests/integration/test_reconcile_p1_gates.py` on a scratch DB, with one agent on the database.
5. With the user's OK: back up corpus3, then extract, resolve, classify, reconcile from w-c3seed (ov-0600..0606 apply at extract; LLM answers are cache-served). Read the gates (expect the noise ratio about 1.25-1.35); run bayesian-model-reviewer and `era_effect: none`.
