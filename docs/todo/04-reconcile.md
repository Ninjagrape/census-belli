## Stage 4: Reconcile (pipeline/stages/reconcile.py)
Built, reviewed, and writing estimates against live Postgres on fixtures;
see handover-log §19-§21. Estimates are written only after convergence.

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
