## Stage 8: Evaluate (pipeline/stages/evaluate.py)
- [ ] Held-out prediction (5-fold CV on battles)
- [ ] Brier score / calibration plot
- [ ] Posterior predictive checks (simulate outcomes, compare to observed)
- [ ] Sensitivity analysis runner
  - [ ] g_unlabelled +/- 1 sd sensitivity on the neutral estimates (bounds the shared reconcile error impute does not propagate, see agents/impute.yaml)
  - [ ] Refit with each variant from agents/evaluate.yaml
  - [ ] New variants from the model review: exclude_extreme_ratio (|x_A - x_B| > 2),
        three_level_outcome (collapse decisive grades), realised_war ranking,
        equal_attribution consuming default_split sides at 1/n
- [ ] Sensitivity refits must not take MAX(run_id) away from the published run: filter on model_type
  - [ ] Compute rank correlation and top-20 overlap across variants
- [ ] Expert ranking comparison (load config/expert_rankings.yaml, compute overlap)
- [ ] Named sanity checks (Agrippa > Octavian, Belisarius > Justinian)
- [ ] Evaluation report generator (JSON + summary text)
- [ ] Diagnostic plots (rank stability, CI widths, skill vs n_battles)
- [ ] config/expert_rankings.yaml (curated from published lists)
