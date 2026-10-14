## Stage 9: Report (pipeline/stages/report.py)
- [x] BLOCKING before report is built: its gates select MAX(run_id) FROM model_runs; change to
      MAX(run_id) WHERE model_type='bradley_terry_hierarchical' AND diagnostics->>'converged'='true'
      (done in agents/report.yaml, uncommitted; also requires estimates_written)
- [x] BLOCKING: treat the model's exclusion_concentration warning as an error (do not publish
      while top generals have most battles uncredited, i.e. before classify's LLM pass)
      (spec gate added, uncommitted)
- [x] _check_model_gate honours params.run: evaluable, so report's exclusion_concentration reads
      the same run as the SQL gates (uncommitted, unit-tested, not yet run on a database)
- [x] (gap review A14) top_rankings_have_intervals and no_orphaned_war_details now filter to the
      evaluable run; evaluate's octavian check is now a named check by Q-id (3f01939)
- [ ] Disclose: cross-era ranking rests on the shared skill prior; technology not modelled;
      total_war is volume-weighted; win_prob_* columns hold expected score, not P(win)
- [ ] Rankings table generator (top N with CIs)
- [ ] Interactive explorer (Plotly or HTML artifact)
  - [ ] Searchable general lookup
  - [ ] Per-general battle breakdown with WAR contributions
  - [ ] Uncertainty visualisation (CI overlap between adjacent ranks)
  - [ ] Sensitivity comparison view
- [ ] Static report output (markdown + figures for the README/blog)
- [ ] CSV/JSON export of final rankings
