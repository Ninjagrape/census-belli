## Stage 7: Model (pipeline/stages/model.py)
Built, reviewed and verified on fixtures; see handover-log §23 for the design
and its departures from the original spec (no additive battle_type/era/terrain
terms, no mu_era, xWAR against a per-draw 40th-percentile replacement).

- [ ] Run on a real corpus
- [ ] CmdStanPy fallback (not needed so far; no convergence trouble on synthetic data)
- [ ] Merge 3+ side battles into two camps when outcomes split cleanly (excluded in v1)
- [ ] Force slope by battle type (naval/aerial share one beta_force and one skill population now)
- [ ] rhat/ESS on z_skill and centred skills rather than raw skill (reviewer LOW 8; optional)
- [ ] Recovery-test Spearman criterion (0.8) sits near the information limit: 0.797-0.922 across seeds
- [ ] (pilot) real corpus admits 13 of 154 battles (108 no outcome, 33 no
      creditable side); model rejects scale=published
- [ ] (gap review A15) schema.sql comments: win_prob_* say P(win), they hold expected
      score; imputation_method lists mice_pmm
- [ ] (gap review A16) inspector Model page is a placeholder and queries `skill_estimates`
      (table is general_skill_estimates)
