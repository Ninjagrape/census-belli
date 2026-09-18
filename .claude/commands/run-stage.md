Run the pipeline stage: $ARGUMENTS

1. Check that the stage's dependencies have been met:
   - Read agents/$ARGUMENTS.yaml and verify all input files/tables exist
   - If inputs are missing, report which prior stage needs to run first

2. Run the stage via the orchestrator:
   ```
   python -m pipeline.orchestrator --stages $ARGUMENTS
   ```

3. If the stage fails:
   - Read the error output and stack trace carefully
   - Check the structlog output for contextual information
   - If it's a quality check failure, query the relevant DB tables to understand why
   - If it's a code error, fix it, run the tests, and retry
   - If it's a data issue (missing inputs, bad data), report what's wrong and what needs to happen upstream

4. If the stage succeeds:
   - Run the quality checks manually to verify:
     ```
     python -m pipeline.quality_runner --stage $ARGUMENTS
     ```
   - Report the quality check results
   - Summarise the output: how many records were produced, any warnings, what the next stage should expect

5. If fixes were needed, do not commit; report them and suggest the message "fix: $ARGUMENTS stage — <brief description>" for the user to run.
