## Stage 7: Model (pipeline/stages/model.py)
Built and verified on fixtures; see handover.md §23. Departures from the original
spec (no additive battle_type/era/terrain terms, no mu_era, xWAR) are explained there.
- [x] Ordinal Bradley-Terry in PyMC, symmetric cutpoints (orientation-invariant), non-centred skills
- [x] Attribution-weighted sides; default_split multi-commander and unknown sides as latents
- [x] Interval-censored outcomes when the two sides' grades disagree
- [x] Expected WAR (xWAR) against a per-draw 40th-percentile replacement; realised WAR in diagnostics
- [x] One fit per imputed round, mixture pooling, Rubin's T as the stored sd, widened intervals
- [x] Joint latent-x comparator (circularity check), on a per-battle x_diff prior
- [x] Writes model_runs, general_skill_estimates, battle_war_details; pooled row last
- [x] Gates: convergence (incl. diagnostics.converged), PPC, prior predictive, FMI, circularity,
      exclusion concentration, extreme ratios, career span, cross-component, exclusions
- [x] Recovery suite (tests/model/test_model_recovery.py) incl. the joint comparator
- [x] bayesian-model-reviewer passes on design and code
- [ ] Run on a real corpus
- [ ] CmdStanPy fallback (not needed so far; no convergence trouble on synthetic data)
- [ ] Merge 3+ side battles into two camps when outcomes split cleanly (excluded in v1)
- [ ] Force slope by battle type (naval/aerial share one beta_force and one skill population now)
- [ ] rhat/ESS on z_skill and centred skills rather than raw skill (reviewer LOW 8; optional)
- [ ] Recovery-test Spearman criterion (0.8) sits near the information limit: 0.797-0.922 across seeds
- [ ] (gap review A15) schema.sql comments: win_prob_* say P(win), they hold expected
      score; imputation_method lists mice_pmm
- [ ] (gap review A16) inspector Model page is a placeholder and queries `skill_estimates`
      (table is general_skill_estimates)
