## Stage 7: Model (pipeline/stages/model.py)
Built, reviewed and verified on fixtures; see handover-log §23 for the design
and its departures from the original spec (no additive battle_type/era/terrain
terms, no mu_era, xWAR against a per-draw 40th-percentile replacement).

- [x] done 2026-10-13: Run on a real corpus (run 54, 3000 draws, rhat 1.004)
- [x] dropped 2026-10-16: cmdstanpy removed as unused; nutpie converges: CmdStanPy fallback (not needed so far; no convergence trouble on synthetic data)
- [x] done 2026-10-16: clean outcome splits merge into two camps; unnamed share is latent: Merge 3+ side battles into two camps when outcomes split cleanly (excluded in v1)
- [x] done 2026-10-16: land slope plus a fixed water/air offset (a 2-group hierarchy was prior-dominated): Force slope by battle type (naval/aerial share one beta_force and one skill population now)
- [x] done 2026-10-15: rhat/ESS on z_skill and centred skills rather than raw skill (reviewer LOW 8; optional)
- [x] done 2026-10-15: corpus doubled, worst of 4 seeds 0.825 (margin still thin): Recovery-test Spearman criterion (0.8) sits near the information limit: 0.797-0.922 across seeds
- [x] measured 2026-10-18 (run 108): 195 of 210 in-scope battles admitted, 15 excluded (breakdown in `evaluation_report.json` counts); was 59 (2026-10-14) and 133 (run 79): Battles admitted on the real corpus
- [x] decided 2026-10-18 (user): keep the headline rule (`min_battles: 3`) for now. The Bayesian review (handover-log §30 MAJOR 1) found ranks 16-89 mostly prior-dominated subordinates whose WAR is their leader's side performance shared by weight (`war.py:182-184`); the user's view is that the top 15 are sound and many generals have few battles. Disclose it in the report (09-report.md)
- [ ] Optional, not planned: a per-commander WAR counterfactual (replace only w_i*s_i with w_i*s_rep) so WAR tracks a general's own skill (§30 MAJOR 1). Model-meaning change; user's call
- [x] fixed 2026-10-12: (gap review A15) schema.sql comments: win_prob_* say P(win), they hold expected
      score; imputation_method lists mice_pmm
- [x] fixed 2026-10-12: (gap review A16) inspector Model page is a placeholder and queries `skill_estimates`
      (table is general_skill_estimates)
