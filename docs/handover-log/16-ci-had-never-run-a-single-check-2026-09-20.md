# Handover log §16. CI had never run a single check, 2026-09-20

Part of the session log; the index is `docs/handover-log/README.md`.

All three jobs failed on the first push that triggered them (`8c714de`), and
neither failure was in the code the jobs exist to check.

### 16.1 `pip install -e ".[dev]"` could not build the package

`pyproject.toml` had no `[build-system]` table and no package configuration, so
pip fell back to setuptools' flat-layout auto-discovery. Discovery found
`agents/`, `alembic/`, `config/`, `data/`, `models/` and `pipeline/` side by
side, refused to guess which was the package, and exited:

```
error: Multiple top-level packages discovered in a flat-layout:
['data', 'agents', 'config', 'models', 'alembic', 'pipeline'].
```

Every job dies at its Install step, which is why the lint job failed in eight
seconds -- too fast to have run ruff. Nothing downstream of the install had
ever executed in CI. The fix declares the backend and pins discovery to
`pipeline*`; `scripts/` and `models/` are empty placeholder packages run from a
checkout, not installed.

This never showed up locally because the editable install predates the
directories that broke discovery. `pip install -e . --dry-run` reproduces it in
a second, and is worth running after adding any top-level directory.

### 16.2 The baseline migration hit the `%` bug all over again

With the install fixed, `alembic upgrade head` failed on exactly the defect
§4.3 records against `apply_schema()`:

```
psycopg.ProgrammingError: incomplete placeholder: '%';
```

`upgrade()` applied `config/schema.sql` through `conn.exec_driver_sql()`, under
a comment asserting that exec_driver_sql "passes it through rather than
treating it as one parameterised statement". It does not -- it still hands
psycopg an empty parameter set, and the five `95% CI` comments in schema.sql
are parsed as placeholders. The same wrong belief was fixed in `pipeline/db.py`
and left standing in the migration.

The migration now uses a raw driver cursor, as `apply_schema()` does. It does
**not** commit the driver connection the way `apply_schema()` must: the cursor
shares Alembic's connection, so the DDL belongs to Alembic's transaction and is
committed with the version stamp. Committing here would split the two.

### 16.3 Verified

Against the live database on this machine, on two scratch databases created and
dropped for the purpose:

- `alembic upgrade head` -> `downgrade -1` -> `upgrade head` on an empty
  database, then confirmed 20 tables exist and `battles.year_astronomical` is
  present. A version stamp committed over rolled-back DDL is the §4.4 failure
  mode and would otherwise look identical to success.
- `python -m pipeline.db --apply-schema` on a second empty database: 19 tables.
- `ruff check`, `mypy --strict` (49 files), and `pytest -m "not live"` with
  DATABASE_URL set: 410 passed, 10 deselected, **0 skipped**, so the CI step
  that fails the run on any skip has nothing to trip over.

### 16.3b And then the lint job failed on an undeclared stub

Run 2 got past the install (47s of it) and died on:

```
pipeline/llm/parsing.py:100: error: Library stubs not installed for "jsonschema"
```

`types-jsonschema` was installed on this machine and was never declared. The
dev extras carried a comment naming "the YAML and jsonschema boundaries" and
then listed only `types-PyYAML`, so the comment described an intent the file
did not implement, and every local `mypy --strict` run passed on a stub CI
would never install. Now declared.

Same shape as §16.1: the development machine has something CI does not, and
nothing compared the two. An audit of every third-party import in `pipeline/`,
`tests/` and `alembic/` against the declared dependencies found no other gap,
so this class should now be closed for imports. Stub packages are the half that
is easy to miss, because they are invisible at runtime and only mypy wants
them.

### 16.4 What is still unverified, and the drift behind it

CI runs Python 3.11 and resolves `anthropic>=1.0` to 1.7.0. This machine runs
Python 3.14 with anthropic 0.111.0 installed, which does not satisfy the
project's own constraint. Local `mypy --strict` results are therefore results
for anthropic 0.x.

Run 2 settled the part of that which mattered: mypy checked all 49 files
against anthropic 1.7.0 and reported nothing in
`pipeline/llm/anthropic_client.py`. The module types clean under both majors.
What remains unsettled is *runtime* behaviour under 1.x -- no test exercises
the Anthropic provider against the installed SDK, and the live LLM tests are
still unrun (§15). The version drift on this machine is real and worth closing
the next time the environment is touched.
