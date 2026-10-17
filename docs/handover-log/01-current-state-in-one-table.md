# Handover log §1. Current state in one table

Part of the session log; the index is `docs/handover-log/README.md`.

| Stage | Spec | Implemented | Unit tested | Verified against a database |
|---|---|---|---|---|
| crawl | yes | yes | 41 tests | **yes**, plus 5 live against Wikipedia |
| extract | yes | yes | 79 tests | **yes** |
| resolve | yes | yes | 59 unit + 18 integration | **yes**, incl. an end-to-end run |
| reconcile | yes | **yes** (§19-§21), incl. the regime-neutral troops fit | unit + integration + model (§21.6) | **yes**, on fixtures; never run on a real corpus |
| classify | yes | **first slice** (§20), review fixes (§21.2) | unit + 11 integration | **yes**, on fixtures; LLM path offline/cache only, no billed call |
| impute | yes | **yes** (§21.4) | unit + 11 integration + 5 model | **yes**, on fixtures; never run on a real corpus |
| model | yes | **yes** (§23) | unit (`test_modelers_*`) + 12 integration + 6 model | **yes**, on fixtures; never run on a real corpus |
| evaluate | yes | **no** | — | — |
| report | yes | **no** | — | — |

Infrastructure is complete: `pipeline/db.py`, `config.py`, `quality.py`, `logging_config.py`, `stages/base.py`, `llm/` (9 modules), `orchestrator.py`, Alembic with `config/schema.sql` as baseline, `docker-compose.yml`, `Makefile`, `.env.example`, `.github/workflows/ci.yml`, `.pre-commit-config.yaml`, `README.md`.

**Verify the state yourself before trusting this file:**

```bash
export DATABASE_URL="postgresql+psycopg://general_war:general_war@127.0.0.1:5432/general_war"
ruff check pipeline/ tests/ scripts/ alembic/   # expect: All checks passed
python -m mypy pipeline/                         # expect: Success, 113 source files (strict)
python -m pytest tests/ -q -m "not live and not model"   # expect: 925 passed, 0 skipped (with DB)
python -m pytest tests/model -m model            # the recovery suite; slow, see §20.4
```

Without DATABASE_URL set the integration tests skip rather than fail, by
design, and a green run with skips looks almost identical to a green run
without. **Set it, or you are not testing the half of the project that
matters.**

`rapidfuzz` was installed on 2026-09-19 (`pip install "rapidfuzz>=3.6"`). It
was declared in `pyproject.toml` and had never actually been installed, so
importing `pipeline.resolvers` fails without it.

**Fifteen skips are now correct and expected**, and they are not the old
database skips. They are the opt-in live tests: five in
`tests/integration/test_crawl_live.py` (real Wikipedia), five in
`tests/integration/test_llm_live.py` (real, billed LLM calls), and five in
`tests/integration/test_resolve_live.py` (real Wikidata SPARQL, added §17.3).
All are gated on an environment variable so they cannot run by accident. A skip
count *above* fifteen means DATABASE_URL is unset and you are not running the
integration suite at all.

A warning about `asyncio_mode` is expected: `pytest-asyncio` is declared in
the dev extras and is still not installed, and the coroutine tests drive their
own event loop instead (§6).

---
