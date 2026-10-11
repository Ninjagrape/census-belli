## Stage 4: Reconcile (pipeline/stages/reconcile.py)
Built, reviewed, and writing estimates against live Postgres on fixtures;
see handover-log §19-§21. Estimates are written only after convergence.

- [ ] (pilot P4) on the real corpus every side's estimate is ~35k (log-sd 0.0117)
      whatever its reports (Kursk 2.5M -> 36,947). Diagnose before anything else;
      defaults also missed ESS 400 (359.6). unfillable_sides_are_logged 16 vs 3;
      ci_width_reasonable returns NULL
- [ ] est_troops_method counts lineages the cross-regime merge dropped (handover-log §21.7)
- [ ] Parameterise gate thresholds from the spec so integration can pass neutral_fit_converged (handover-log §21.5)
- [ ] Width gates on the identified scale; n_identifying_sides (handover-log §20.10)
- [ ] ExGaussian bound likelihood (handover-log §20.10)
- [ ] Prior predictive check; per-side casualty ratio; InverseGamma(2,1) on variance
- [ ] Run on a real corpus (needs crawl/extract/resolve on config/sources_seed_arsht.yaml first)
- [ ] (gap review A11) data_era is centred over all dated rows but applied only to
      unlabelled rows (design.py / model.py); centre within the unlabelled rows
- [ ] (gap review A12) a fit failing write thresholds leaves sides NULL with no
      missing_data_log row
- [ ] (gap review) dead spec params: bias_prior_mu, regime_deviation_prior_sd
