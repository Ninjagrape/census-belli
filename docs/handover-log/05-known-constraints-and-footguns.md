# Handover log §5. Known constraints and footguns

Part of the session log; the index is `docs/handover-log/README.md`.

### 5.1 BC dates cannot round-trip through Python -- use `year_astronomical`

`datetime.date` has `MINYEAR == 1`. No BC date can be constructed, bound as a
parameter, or **decoded from a result**. Postgres stores them fine.

```python
date(-30, 9, 2)   # ValueError: year must be in 1..9999
```

**Resolved 2026-09-19 by adding `battles.year_astronomical`**, a generated
STORED integer column. Read that instead of `date_start` whenever you need a
year in Python. It is generated, so it cannot drift from the date it mirrors,
and Postgres rejects any attempt to write it.

```sql
year_astronomical INT GENERATED ALWAYS AS (
    CASE WHEN EXTRACT(YEAR FROM date_start) < 0
         THEN EXTRACT(YEAR FROM date_start)::int + 1
         ELSE EXTRACT(YEAR FROM date_start)::int END
) STORED
```

Astronomical numbering, **not** Postgres's: 31 BC is -30 here and -31 from a
bare `EXTRACT`. See §4.5 for why that difference matters. NULL `date_start`
gives NULL year, so a battle known only by year must still be written as
`YYYY-01-01` with `date_precision='year'`.

It lives in **both** `config/schema.sql` and Alembic migration
`b2c3d4e5f6a7`. That looks redundant and is not: the integration fixtures
build their database with `apply_schema()`, which executes `schema.sql`
directly and never runs Alembic, so a migration-only change would be invisible
to every test. The migration is guarded with `IF NOT EXISTS` so both paths
converge. **Any future schema change needs the same treatment.**

Verified: migration applies to a database built from the original baseline,
downgrades, and re-upgrades with correct values on real BC rows. Pinned by
three tests in `tests/integration/test_db.py`.

`date_start` itself still cannot be read into Python, and the test pinning
that (`test_bc_dates_cannot_round_trip_through_python`) still stands.

### 5.1b The LLM step needs `google-genai`, now installed

`agents/resolve.yaml` routes the disambiguation step to Gemini.
**`google-genai` 2.24.0 was installed on 2026-09-19** and a real call has now
been made (§12.7).

Resolve builds its LLM client lazily (`LazyService` in
`pipeline/stages/resolve.py`), so a corpus where every commander resolves
deterministically runs fine with no package and no API key. The moment one
group is genuinely ambiguous, `LLMService.from_spec` raises `LLMConfigError`
and the stage stops. That is intended -- `retry_policy.on_failure` for resolve
is `pause_and_alert`, and quietly turning ambiguous commanders into new
entities would bury a configuration problem in the data -- but it means **a
real resolve run needs `pip install google-genai` and a `GEMINI_API_KEY`
exported**, and it will not tell you until it is part-way through.

To resolve without any LLM at all, set `params.candidate_source: local` and
accept that ambiguous groups end up `unresolved` and logged.

### 5.2 RESOLVED: Wikipedia rate limiting

**Fixed 2026-09-19 (§15.1).** `rate_limit_wikipedia` is now 2.0, and
`Fetcher` honours `Retry-After` on any retryable status, preferring it over
the computed backoff and capping it at `MAX_SERVER_BACKOFF_S` (120s). Both
header forms RFC 9110 allows are read: an integer seconds count and an
HTTP-date.

Measured live: ten consecutive article fetches at the new spacing, **zero
retries and zero 429s**. `tests/integration/test_crawl_live.py` asserts that
and fails if any fetch needs a second attempt, so the spec being too fast
again is a test failure rather than a corrupted crawl.

### 5.3 Windows encoding

`alembic ... --sql` dies with `UnicodeEncodeError` because `config/schema.sql` uses box-drawing characters in its section headers and the console is cp1252. Set `PYTHONIOENCODING=utf-8`. The Makefile targets already do.

### 5.4 Shell heredocs are unreliable here

Writing long Python files via `cat <<'EOF'` failed mid-file more than once. Use the file-writing tool for substantial source files; keep Bash for inspection, patching via a short Python script, and git.

---
