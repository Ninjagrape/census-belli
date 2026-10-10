## Stage 9: Report (pipeline/stages/report.py)
- [ ] BLOCKING before report is built: its gates select MAX(run_id) FROM model_runs; change to
      MAX(run_id) WHERE model_type='bradley_terry_hierarchical' AND diagnostics->>'converged'='true'
- [ ] BLOCKING: treat the model's exclusion_concentration warning as an error (do not publish
      while top generals have most battles uncredited, i.e. before classify's LLM pass)
- [ ] (gap review A14) top_rankings_have_intervals, no_orphaned_war_details and evaluate's
      octavian_agrippa_check have no run filter at all; the LIKE '%Octavian%' match can hit
      several generals
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
