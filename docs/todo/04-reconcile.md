## Stage 4: Reconcile (pipeline/stages/reconcile.py)
Built, reviewed, and writing estimates against live Postgres on fixtures;
see handover-log §19-§21. Estimates are written only after convergence.

- [x] (pilot P4) fixed 2026-10-11: on the real corpus every side's estimate is ~35k (log-sd 0.0117)
      whatever its reports (Kursk 2.5M -> 36,947). Diagnose before anything else;
      defaults also missed ESS 400 (359.6). unfillable_sides_are_logged 16 vs 3;
      ci_width_reasonable returns NULL
- [x] fixed 2026-10-12: (pilot) ci_width_reasonable still returns NULL on the corpus after the P4
      fix (run 12); calibration_set_is_not_empty 0 (warning)
- [x] tried 2026-10-13: (pilot review) contamination component per row. Built as a symmetric
      two-Normal mixture (`likelihood_family: contaminated`), kept as an ablation only: on the
      corpus troops pi_contam was 0.02 and it did not fix the collapse. A one-sided component
      may still suit casualties (re-measure first, handover §7 step 3)
- [x] fixed 2026-10-13: (pilot review) a range "X-Y" entered as two half-line censorings and,
      on the corpus, drove infobox sigma to 1.4 and the collapse. Merged into one point row at
      sqrt(A·B) with range_var added (`merge_range_pairs`, handover §9.6)
- [ ] (pilot review S3) a key's own effect should require co-reporting (other keys
      on the same sides in >= k battles), not k battles alone. Latent today
- [ ] (pilot review S4) in a type mixing qualifying and pooled keys the zero-sum
      pins pooled keys to the qualifying mean; drop the zero-sum there. Latent
- [ ] (pilot review N1) floor of ~5 qualifying keys before tau_u goes hierarchical
- [ ] (pilot review N3) a_era (on unlabelled rows) learns that ancient armies were
      smaller and calls it inflation; needs a design before adding a period mean
- [x] fixed 2026-10-12: est_troops_method counts lineages the cross-regime merge dropped (handover-log §21.7)
- [x] fixed 2026-10-12: Parameterise gate thresholds from the spec so integration can pass neutral_fit_converged (handover-log §21.5)
- [ ] Width gates on the identified scale; n_identifying_sides (handover-log §20.10)
- [ ] ExGaussian bound likelihood (handover-log §20.10)
- [ ] Prior predictive check; per-side casualty ratio; variance prior InverseGamma alpha 3
      (currently 2, `precision_prior_alpha`)
- [x] done 2026-10-13: Run on a real corpus (general_war_corpus2, CLI)
- [x] fixed 2026-10-12: (gap review A11) data_era is centred over all dated rows but applied only to
      unlabelled rows (design.py / model.py); centre within the unlabelled rows
- [x] fixed 2026-10-12: (gap review A12) a fit failing write thresholds leaves sides NULL with no
      missing_data_log row
- [x] fixed 2026-10-12: (gap review) dead spec params: bias_prior_mu, regime_deviation_prior_sd
- [x] fixed 2026-10-13: (corpus run 2026-10-12) over-shrinkage: caused by range pairs (above);
      corpus run 7 spread_ratio 0.60, infobox sigma 0.52
- [x] confirmed 2026-10-13: neutral fit converges on the corpus (neutral_fit_converged passes)
- [ ] Audit the 6 low-precision sides and the scope-error candidates at extract (handover §7 step 3)
