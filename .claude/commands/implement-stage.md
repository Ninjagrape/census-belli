Implement the pipeline stage: $ARGUMENTS

Follow this process exactly:

1. Read the agent spec at agents/$ARGUMENTS.yaml to understand the stage's inputs, outputs, tools, prompts, and quality checks.

2. Read pipeline/stages/base.py to understand the stage runner protocol.

3. Read any existing stage implementations in pipeline/stages/ for pattern consistency.

4. Implement pipeline/stages/$ARGUMENTS.py:
   - Import and conform to the stage runner protocol
   - Use pipeline/db.py for all database operations (SQLAlchemy Core, parameterised queries, never string interpolation)
   - Use pipeline/llm.py for any Anthropic API calls, passing temperature and max_tokens from the agent spec params
   - Use pipeline/config.py to load the agent spec
   - Use structlog for all logging, never bare print()
   - Type hints on all function signatures
   - Google-style docstrings on all public functions
   - Handle errors according to the retry_policy in the agent spec
   - For LLM-assisted stages: batch inputs, send with the prompt template from the spec, validate JSON output against the output_schema, log failures and mark for review rather than crashing

5. After implementation, write tests:
   - Unit tests in tests/unit/test_$ARGUMENTS.py for pure functions
   - Integration test in tests/integration/test_$ARGUMENTS.py using fixture data in tests/fixtures/
   - For LLM-assisted stages, mock the Anthropic client and test with canned responses

6. Run ruff check and mypy on the new files. Fix any issues.

7. Run the tests. Fix any failures.

8. Update TODO.md: check off all items completed for this stage.

9. Commit with message: "feat: implement $ARGUMENTS pipeline stage"

10. Summarise what you built, any design decisions you made, and flag anything that needs manual review or is blocked on another stage.
