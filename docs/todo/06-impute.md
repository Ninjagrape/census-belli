## Stage 6: Impute (pipeline/stages/impute.py)
Built and verified on fixtures; see handover-log §21.4. A joint Bayesian model
replaces the MICE/PMM in the original spec. Only the neutral scale writes
missing_data_log.

- [x] done 2026-10-13: Run on a real corpus (general_war_corpus2; max rhat 1.0092, thin margin)
- [x] done 2026-10-15, with m0 co-movement after review: Propagate reconcile's shared measurement error: export the neutral fit's regime draws (g_unlabelled, a_era) in neutral_draws_run*.npz and carry them through the rounds, so battles in the mixed regime share their error instead of being drawn independently (beta_force uncertainty is understated until then)
- [x] fixed 2026-10-12: (pilot) gates read neutral/gate_report.json under scale=published
- [x] done 2026-10-18 (user decision, 5a0d27a): run 85 failed ESS (302 < 400) because tau_p mixes slowly along a ridge with the Soviet Union polity offset (3 observed sides, corr 0.88); `mcmc_samples` 2000 to 4000, same model and seed; run 86 rhat 1.006, ESS 685. Centring the polity block or a Student-t effect would fix the mixing itself (handover-log §30 Q3)
- [ ] **Planned, workstream A**: the weak-side rule (`pipeline/imputers/likelihood.py:261-316`) treats 75 of 341 estimated sides as missing, including 7 of 10 Soviet sides, which are then filled from the Soviet Union offset (Stalingrad reconcile x_diff -0.36, imputed +1.29; §30 MAJOR 2). Check whether weakness should be judged on the residual (theta-regressed) precision rather than the raw covariance. Reviewer diagnoses, user approves
- [ ] **Planned** (next-sessions plan D): polity canonicalisation map. The polity key is the raw `battle_sides.polity` string (no side has a polity Wikidata id), so the Confederacy is 4 levels, Napoleonic France 7, the Habsburgs 4 (§29 M5). A reviewed `config/polity_map.yaml` applied in extract, plus a gate for unmapped strings. Model-meaning change: bayesian-model-reviewer
- [ ] **Planned, decide with A**: a 1914-1945 era bin or a theatre effect, so the polity effect stops carrying a total-war scale effect inside "modern" (Soviet Union +1.06, Israel -0.39, PLO -0.46; §30 MAJOR 2)
- [ ] Record the worst parameter's name and its tail ESS in `model_runs.diagnostics` (only the worst values are stored; diagnosing run 85 needed a re-fit)
