## Stage 6: Impute (pipeline/stages/impute.py)
Built and verified on fixtures; see handover.md §21.4.
- [x] config/imputation_priors.yaml (era bins on year_astronomical, priors, hyperparameter scales)
- [x] Joint Bayesian imputation model (replaces MICE/PMM; analytic marginalisation per battle)
- [x] Reconcile side prior divided out of the neutral posterior; weak sides treated as missing
- [x] Overimputation: 20 rounds, every side drawn from the conditional Gaussian
- [x] Rounds to data/imputed/{scale}/round_{n}/; only the neutral scale writes missing_data_log
- [x] Gates: completeness, ratio plausibility on imputed battles, range (published), KS warning, convergence
- [x] Stale draws file refused; bayesian-model-reviewer passes on design and code
- [ ] Run on a real corpus
- [ ] Propagate reconcile's shared measurement error: export the neutral fit's regime draws (g_unlabelled, a_era) in neutral_draws_run*.npz and carry them through the rounds, so battles in the mixed regime share their error instead of being drawn independently (beta_force uncertainty is understated until then)
