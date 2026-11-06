# Reconcile troop reparameterisation A + B (bayesian-model-reviewer design, 2026-11-06, user-approved)

Line numbers refer to pipeline/reconcilers/model.py in the w-c3seed worktree (same as essdiag/code copy).

## Diagnosis (from traces)
- Dated unlabelled singleton, bin k: eta = cl - W + u_k + b_j + s_d*zd_s + delta_l ; y ~ N(eta, sigma_type[ibx]) (ancient sigma if pre-500). g_unl, a_era cancel exactly.
- Identified: per-bin levels L_k = cl - W + u_k + mean_bin(b) + mean_bin(d) (ESS 117..1205); s_d^2 + sigma_ibx^2 = 0.480 +- 0.022 (ESS 155).
- BLOCKER 1: bin mean of non-centred Student-t zb duplicates e_k/u_k (prior-only ridge; corr(u_k, mean_bin zb) -0.70..-0.94). u_k ESS 9-40, corpus_level 16.
- BLOCKER 2: log(s_d/sigma) ESS 11, rhat 1.31: singletons inform only the sum; moving along the ridge needs all 2088 zd to rescale together.
- MAJOR 3: s_b vs rms(zb) corr -0.87 (non-centred mismatch).

## A. Centre zb within era bin (stated prior change: b is a deviation from the period mean; e carries the whole period mean)
In `_troop_side_prior`, replace lines ~1055-1056 with:
```python
zb = pm_module.StudentT("zb", nu=priors.side_prior_nu, mu=0.0, sigma=1.0, dims="battle")
if mode != "none":
    onehot, mean_w = _battle_bin_projector(design)
    zb_bar = pt_module.dot(zb, _data(pm_module, "data_battle_bin_mean", mean_w))
    zb = zb - pt_module.dot(_data(pm_module, "data_battle_bin_onehot", onehot), zb_bar)
battle_effect = pm_module.Deterministic("b", s_b * zb, dims="battle")
```
Helper:
```python
def _battle_bin_projector(design):
    n_bins = len(ERA_BIN_LABELS)
    bob = np.full(design.n_battles, n_bins); bob[design.battle_of_side] = design.era_bin_of_side
    onehot = np.zeros((design.n_battles, n_bins)); dated = bob < n_bins
    onehot[np.flatnonzero(dated), bob[dated]] = 1.0
    counts = onehot.sum(0)
    return onehot, onehot / np.where(counts > 0, counts, 1.0)
```
Undated battles stay uncentred. (Check actual attribute names: battle_of_side / era_bin_of_side may be named differently; `mode` is the era_effect mode.) Applies only when the shared battle effect exists (side_prior_independent false). Expected |delta median log mu_side| < 0.01.

## B. Marginalise d exactly on single-row point sides (troops, likelihood_family == "normal" only)
Mask: sides with rows_per_side == 1 and that row in point_rows (not a bound/censored row). All-false for casualties or non-normal family. ~1970 sides on corpus3. Bound and multi-row sides keep explicit zd.
- `_troop_side_prior` takes `marginal_side` mask; line ~1048 becomes `location = m0 + era_term + s_d * zd * _data(pm_module, "data_side_explicit_d", 1.0 - marginal_side)`; return `(location, s_d, zd)` instead of registering mu_side at ~1060.
- `build_model` ~798: call it with `marginal_side = _marginal_side_mask(design, priors)`; use `side_loc` in eta at ~925.
- After `var` (~966), replacing ~967:
```python
log_y = _data(pm, "data_log_y", design.log_y)          # moved up from ~969
marg = marginal_side.astype(float)
rs = _data(pm, "data_row_of_side", row_of_side)        # the side's single row; 0 elsewhere, masked
v_s = var[rs]; k = s_d**2 / (s_d**2 + v_s)
d_marg = (k * (log_y[rs] - eta[rs]) + pt.sqrt(k * v_s) * zd) * _data(pm, "data_side_marginal", marg)
mu_side = pm.Deterministic("mu_side", side_loc + d_marg, dims="side")
sd = pm.math.sqrt(var + s_d**2 * _data(pm, "data_marginal_row", marg[design.side_index]))
```
(Here eta[rs] must be the row's predictor WITHOUT d, i.e. built from side_loc.) Exactness: for marginalised sides zd has no likelihood term, so its posterior is iid N(0,1) and mu_side has the exact conditional d | theta, y ~ N(k*r, k*v). Joint posterior over (theta, mu_side) unchanged.
- `model.row_eta` / `row_sd` (~974-975): set to `mu_side[side_index] + (eta - side_loc[side_index])` and `sqrt(var)`, i.e. conditional on d, so summarise's `_row_eta_and_var` test still matches.
- Companion fix: `pipeline/reconcilers/prior_check.py:82,96` reads the prior draw of mu_side, which under B would depend on observed y. Rebuild it from the prior draws of m0, e, b, s_d and zd instead.
- Also check every other reader of mu_side / zd / row_eta / row_sd (summarise.py PPC ~506-531, recovery.py, likelihood_export.py, neutral fit) still gets correct quantities.

## Tests
Must still pass (unit): test_reconcile_model, test_reconcile_p1_model, test_reconcile_summarise, test_reconcile_prior_check (after fix), test_reconcile_neutral, test_reconcile_noise_prior, test_reconcile_likelihood_export, test_reconcile_manual_entry_pin, and the whole unit suite.
New unit tests:
- B: at fixed theta, compiled d_marg over zd draws has mean k*r and variance k*v.
- B: marginal log p equals the explicit model's integrated log p, by quadrature on a 3-side design.
- Mask excludes bound sides, multi-row sides and non-normal families.
- A: per-bin mean of b is 0 for dated battles; undated untouched.
Model/recovery (run later, one file per command, -m model): test_reconcile_recovery, test_reconcile_p1_recovery, test_reconcile_contamination. Integration: tests/integration/test_reconcile_p1_gates.py.

## Scratch-DB check (corpus3 copy general_war_c3_diag, 4 chains, 1000/1000, 3 seeds, ta 0.99) vs baseline (essdiag/code/post_base.pkl, mu_side_med_base.npy in essdiag/review)
0 divergences; ESS bulk >= ~150 per 4000 draws for corpus_level, u_c, s_d, s_b, sigma_type_modern[ibx], log(s_d/sigma), rhat <= 1.02; L_k within 0.01 of baseline; mu_side median |dlog| p99 < 0.02, max < 0.05; s_d^2+sigma^2 = 0.48 +- 0.02; g_unl and a_era means within 0.05; gate ratio agrees across seeds +-0.01 (expect ~1.49).
