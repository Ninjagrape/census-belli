## Stage 9: Report (pipeline/stages/report.py)
- [x] BLOCKING before report is built: its gates select MAX(run_id) FROM model_runs; change to
      MAX(run_id) WHERE model_type='bradley_terry_hierarchical' AND diagnostics->>'converged'='true'
      (done in agents/report.yaml, 7d9ba5e; also requires estimates_written)
- [x] BLOCKING: treat the model's exclusion_concentration warning as an error (do not publish
      while top generals have most battles uncredited, i.e. before classify's LLM pass)
      (spec gate added, 7d9ba5e)
- [x] _check_model_gate honours params.run: evaluable, so report's exclusion_concentration reads
      the same run as the SQL gates (7d9ba5e; runs on the corpus)
- [x] (gap review A14) top_rankings_have_intervals and no_orphaned_war_details now filter to the
      evaluable run; evaluate's octavian check is now a named check by Q-id (3f01939)
- [x] Disclose: cross-era ranking rests on the shared skill prior; technology not modelled;
      total_war is volume-weighted; win_prob_* columns hold expected score, not P(win)
      (`pipeline/reporters/disclosures.py`, merged with evaluate's)
- [x] Rankings table generator (top N with CIs)
- [x] Interactive explorer (`explorer.html`, self-contained, plotly.js inlined)
  - [x] Searchable general lookup
  - [x] Per-general battle breakdown with WAR contributions
  - [x] Uncertainty visualisation (CI overlap between adjacent ranks)
  - [x] Sensitivity comparison view
- [x] Static report output (markdown + figures for the README/blog)
- [x] CSV/JSON export of final rankings
- [x] Integration test `tests/integration/test_report.py` (seeded run, no fit)
- [x] done 2026-10-16: war_ci_lo/hi and P(above next headline general) from pooled WAR draws: Rank overlap uses total_war +/- z*war_sd (normal approximation, exported as
      war_normal_approx_lo/hi, disclosed). Deferred by the user: model writes war_ci_lo/hi and
      P(rank i above next) from pooled WAR draws (handover §9.11); schema change + model re-run
- [x] Explorer polish: empty space under the overlap chart; low-contrast chart text in dark mode
- [x] done 2026-10-15: evaluate writes severity: Evaluate's report omits gate severity; report reads it from agents/evaluate.yaml. Cleaner:
      evaluate writes `severity` into each gate entry
- [ ] **Planned, workstream C** (handover §7 next-sessions plan): report.md lacks disclosures the ranking must carry (handover-log §29 M2): rule_single (138 sides at 1.0, McClellan absent at Glendale and Gaines's Mill), components (41 of the top 50 outside the largest), the unanswered passage pass, WAR as side performance shared by weight (user's 2026-10-18 call), corpus shape (no battles 700-1199, 1300-1499, the 1600s) and fame bias (73.6% vs 30.7%), troop provenance by era. Compute each from the database at report time
- [ ] **Planned, workstream C**: a "Data coverage and caveats" figure set (six panels: corpus timeline, comparison components, credit source, information vs rank, troop provenance, fame bias) in `figures.py` and an explorer tab; designed in the next-sessions plan
- [ ] **Planned, workstream C**: rank numbering runs over all 732 generals, so the 86th headline row reads "rank 723"; number headline rows 1..N among the headline
