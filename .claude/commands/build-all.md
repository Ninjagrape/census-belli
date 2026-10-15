Finish building the General WAR pipeline. You are the orchestrating agent: delegate each task to a sub-agent (briefing and model routing per `CLAUDE.md`), verify the output, and proceed.

Phases 1-5 (infrastructure through the report stage) are built and committed; Phase 6 remains. Read `handover.md` for the current state and design decisions before starting; it supersedes anything below that conflicts with it.

## Build Sequence

After each phase, verify before moving on. If a step fails, diagnose, fix or re-delegate, and only proceed once it passes.

### Phase 6: Integration
**Task 6.1**: "Write tests/integration/test_full_pipeline.py that runs the entire pipeline on a small fixture dataset (20 battles, 30 generals) and verifies: all tables are populated, quality checks pass for all stages, the model converges, rankings are produced with credible intervals, and the Agrippa > Octavian check passes."

**Task 6.2**: "Update README.md: architecture diagram (mermaid), methodology summary, how to interpret results."

Finally, stop and suggest the message "feat: full pipeline integration test" for the user to run. Do not commit.

## Orchestration Rules

- Never proceed to the next phase until the current phase's verification steps pass.
- If a sub-agent's output has lint or type errors, fix them before reporting the phase complete.
- If a test fails, diagnose whether it's a test issue or an implementation issue. Fix the root cause.
- After each phase, run the full test suite with `DATABASE_URL` exported (one suite at a time against the database).
- Keep the task list updated (`docs/todo/<stage>.md`; `TODO.md` is the index).
- Never run `git commit` or `git push` yourself. The user commits.
