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
- [ ] Battles admitted on the real corpus: 59 of 210 (2026-10-14, duplicated
      sides), 133 (diagnostic run 79, 70 dropped for no outcome). Phase 1 cut
      the no-outcome count to 6 non-campaign battles; measure at the refit
- [x] fixed 2026-10-12: (gap review A15) schema.sql comments: win_prob_* say P(win), they hold expected
      score; imputation_method lists mice_pmm
- [x] fixed 2026-10-12: (gap review A16) inspector Model page is a placeholder and queries `skill_estimates`
      (table is general_skill_estimates)
