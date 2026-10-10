Finish building the General WAR pipeline. You are the orchestrating agent: delegate each task to a sub-agent (briefing and model routing per `CLAUDE.md`), verify the output, and proceed.

Phases 1-4 (infrastructure through the model stage) are built. Read `handover.md` for the current state and design decisions before starting; it supersedes anything below that conflicts with it.

## Build Sequence

After each phase, verify before moving on. If a step fails, diagnose, fix or re-delegate, and only proceed once it passes.

### Phase 5: Evaluation + Reporting
First fix the BLOCKING items in `docs/todo/09-report.md` (gates must filter on the published, converged model run).

**Task 5.1** — "Implement pipeline/stages/evaluate.py following agents/evaluate.yaml. Held-out 5-fold CV, Brier score, posterior predictive checks, sensitivity analysis (refit with each variant; refits must not displace the published run), expert ranking comparison, named sanity checks (Agrippa > Octavian, Belisarius > Justinian). Write evaluation_report.json and diagnostic plots. Register brier_score, jaccard_overlap and held_out_accuracy in METHOD_HANDLERS."

**Task 5.2** — "Implement pipeline/stages/report.py following agents/report.yaml. Generate: rankings table (top 100 with CIs), per-general battle breakdown, sensitivity comparison, and a static markdown report with figures. Also CSV and JSON exports. Carry the disclosures in handover.md §9.9."

After 5.x tasks:
- Verify: evaluation report is generated and all sanity checks pass
- Stop. Do not commit; report the phase complete and suggest the message "feat: evaluate and report stages" for the user to run.

### Phase 6: Integration
**Task 6.1** — "Write tests/integration/test_full_pipeline.py that runs the entire pipeline on a small fixture dataset (20 battles, 30 generals) and verifies: all tables are populated, quality checks pass for all stages, the model converges, rankings are produced with credible intervals, and the Agrippa > Octavian check passes."

**Task 6.2** — "Update README.md: architecture diagram (mermaid), methodology summary, how to interpret results."

Finally, stop and suggest the message "feat: full pipeline integration test" for the user to run. Do not commit.

## Orchestration Rules

- Never proceed to the next phase until the current phase's verification steps pass.
- If a sub-agent's output has lint or type errors, fix them before reporting the phase complete.
- If a test fails, diagnose whether it's a test issue or an implementation issue. Fix the root cause.
- After each phase, run the full test suite with `DATABASE_URL` exported (one suite at a time against the database).
- Keep the task list updated (`docs/todo/<stage>.md`; `TODO.md` is the index).
- Never run `git commit` or `git push` yourself. The user commits.
