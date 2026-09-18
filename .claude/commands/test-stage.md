Write or update tests for: $ARGUMENTS

1. Read the implementation at pipeline/stages/$ARGUMENTS.py to understand all code paths.

2. Read agents/$ARGUMENTS.yaml to understand the expected inputs, outputs, and quality checks.

3. Identify what needs testing:
   - Every public function needs a unit test
   - Every branching path (error handling, retries, edge cases) needs coverage
   - LLM-dependent code must be tested with mocked responses (both success and failure)
   - Database operations must be tested against a real test DB (not mocked)
   - Quality checks must be tested with data that passes and data that fails

4. Create or update:
   - tests/unit/test_$ARGUMENTS.py — pure function tests, mocked dependencies
   - tests/integration/test_$ARGUMENTS.py — end-to-end with test DB and fixture data
   - tests/fixtures/$ARGUMENTS/ — any fixture files needed (sample HTML, JSON responses, etc.)

5. For fixture data:
   - Use real data from known battles (Actium, Austerlitz, Cannae are good reference battles)
   - Include edge cases: battles with missing troop data, single commander, many commanders, draws
   - For LLM mock responses, create both well-formed and malformed JSON to test error handling

6. Run the tests:
   ```
   pytest tests/unit/test_$ARGUMENTS.py tests/integration/test_$ARGUMENTS.py -v
   ```

7. Fix any failures. Ensure all tests pass.

8. Run coverage check and report which code paths are not yet covered:
   ```
   pytest tests/ -v --cov=pipeline/stages/$ARGUMENTS --cov-report=term-missing
   ```

9. Stop. Do not commit; report what changed and suggest the message "test: add tests for $ARGUMENTS stage" for the user to run.
