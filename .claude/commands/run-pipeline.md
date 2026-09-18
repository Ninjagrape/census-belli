Run the full pipeline or a range of stages: $ARGUMENTS

$ARGUMENTS can be "all", a single stage name, or a comma-separated range like "crawl,extract,resolve".

1. Pre-flight checks:
   - Verify Postgres is running and the schema is applied
   - Verify ANTHROPIC_API_KEY is set in the environment
   - Verify all stage modules in the requested range exist and import cleanly
   - Estimate the scope: how many battles/records are in the DB, how many API calls each stage will make

2. Run the pipeline:
   ```
   python -m pipeline.orchestrator --stages $ARGUMENTS
   ```

3. Monitor progress via the structlog output. After each stage completes:
   - Report the stage result (success/failure, duration, retries used)
   - Report quality check results
   - If a stage fails and stop_on_error is true, diagnose the failure before deciding whether to fix and retry or halt

4. After the pipeline completes:
   - Generate a summary report: stages run, records processed per stage, quality check results, total duration, total API cost
   - If the model stage ran, report the top 20 generals with credible intervals
   - If the evaluate stage ran, report held-out accuracy, calibration, and sensitivity analysis highlights
   - Update TODO.md with any new issues discovered during the run

5. Do not commit any fixes made during the run; list them and suggest commit messages for the user to run.
