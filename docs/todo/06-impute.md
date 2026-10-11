## Stage 6: Impute (pipeline/stages/impute.py)
Built and verified on fixtures; see handover-log §21.4. A joint Bayesian model
replaces the MICE/PMM in the original spec. Only the neutral scale writes
missing_data_log.

- [ ] Run on a real corpus
- [ ] Propagate reconcile's shared measurement error: export the neutral fit's regime draws (g_unlabelled, a_era) in neutral_draws_run*.npz and carry them through the rounds, so battles in the mixed regime share their error instead of being drawn independently (beta_force uncertainty is understated until then)
- [ ] (pilot) gates read neutral/gate_report.json under scale=published
