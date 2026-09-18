Assess the current state of the project and recommend next steps.

1. Read TODO.md and identify what's checked off vs remaining.

2. Check which pipeline stages are implemented:
   ```
   ls pipeline/stages/*.py
   ```

3. Check which stages have tests:
   ```
   ls tests/unit/test_*.py tests/integration/test_*.py
   ```

4. Run the full lint and type check:
   ```
   ruff check pipeline/ scripts/ tests/
   mypy pipeline/
   ```

5. Run the full test suite:
   ```
   pytest tests/ -v --tb=short
   ```

6. If the DB is available, check data state:
   - How many battles in the DB
   - How many generals resolved
   - How many battle_commanders have command_role != 'unknown'
   - How many troop_reports exist
   - How many missing_data_log entries
   - Whether any model_runs exist

7. Produce a status report:
   - Infrastructure: done/partial/missing
   - Per-stage: not started / implemented / tested / passing quality checks
   - Blockers: anything preventing the next stage from running
   - Recommended next action: the single most impactful thing to do now

8. Do NOT make changes. This is a read-only assessment.
