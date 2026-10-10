## Known constraints
- RESOLVED: Wikipedia rate limiting. `rate_limit_wikipedia` is now 2.0 and the
  fetcher honours `Retry-After` on every retryable status, preferring it over the
  computed backoff and capping it at 120s. Measured live on 2026-09-19: ten
  consecutive article fetches, zero retries, zero 429s.
  `tests/integration/test_crawl_live.py` fails if any fetch needs a second
  attempt, so a spec that is too fast again is a test failure rather than a
  corrupted crawl.
- `battle_type` must be inferred in classify (tracked as a task under Stage 5).
  The generic
  {{Infobox military conflict}} template carries no domain evidence, so extract
  now leaves battle_type unset for it, which is almost every battle. Nothing
  sets it yet, so the covariate is currently always NULL. `missing_data_log`
  now records this per battle rather than leaving it silent.
- BC dates cannot round-trip through Python. `datetime.date` has MINYEAR == 1,
  so no pre-1 AD date can be constructed, bound as a parameter, or decoded from
  a result. Postgres stores them fine.
  **Read `battles.year_astronomical` instead** — a generated integer column
  added for exactly this. Astronomical numbering, so 31 BC is -30, where a bare
  `EXTRACT(YEAR ...)` returns -31. Only the astronomical form subtracts
  correctly across the BC/AD boundary, and getting it wrong would put a silent
  off-by-one into every date-derived covariate.
  See tests/integration/test_db.py, which pins both forms.
- `alembic ... --sql` needs `PYTHONIOENCODING=utf-8` on Windows: schema.sql uses
  box-drawing characters that cp1252 cannot encode. The Makefile targets set it,
  but `make` is not installed on this machine, so run the raw commands.
- **`export DATABASE_URL` before running the test suite.** Without it the 28
  integration tests skip rather than fail, so a green run with 28 skips looks
  almost identical to a green run with none.
- Shell heredocs mangle backslashes in this environment. Writing a Windows path
  through one silently produced `C:\Program Files\PostgreSQL` + a literal
  backspace. Use the file-writing tool for anything containing backslashes.
