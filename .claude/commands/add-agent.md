Create or update an agent spec for: $ARGUMENTS

1. If agents/$ARGUMENTS.yaml already exists, read it first. Otherwise start fresh.

2. Read at least two existing agent specs in agents/ to match the established structure and conventions.

3. Create agents/$ARGUMENTS.yaml following this structure:
   - stage: name
   - description: what this stage does (one paragraph)
   - inputs: list of files/tables it reads
   - outputs: list of files/tables it writes
   - tools: list of tools/functions the agent can call
   - params: configuration parameters with sensible defaults
   - prompt: (if LLM-assisted) system prompt and user_template with {placeholder} variables, output_schema as JSON Schema
   - quality_checks: list of SQL-based assertions with name, check, threshold, severity
   - retry_policy: max_stage_retries, on_failure action

4. For the prompt (if applicable):
   - System prompt must be specific and directive, not vague
   - Include concrete examples of edge cases the LLM should handle
   - Specify the output format as a JSON schema
   - Temperature should be 0.0 for extraction/classification, up to 0.3 for generation

5. Quality checks must be concrete SQL queries that can run against the DB after the stage completes. Each needs a clear pass/fail threshold.

6. Stop. Do not commit; report what changed and suggest the message "feat: add agent spec for $ARGUMENTS stage" for the user to run.
