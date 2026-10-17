# Handover log §2. The database exists now

Part of the session log; the index is `docs/handover-log/README.md`.

**Resolved 2026-09-19.** PostgreSQL 15.19 is installed natively on this Windows
machine via `winget install PostgreSQL.PostgreSQL.15`. No Docker, no podman,
no `make` -- those are still absent and still not needed.

```
host 127.0.0.1  port 5432  db general_war  user general_war  password general_war
DATABASE_URL=postgresql+psycopg://general_war:general_war@127.0.0.1:5432/general_war
```

`psql` lives under `C:\Program Files\PostgreSQL\15\bin\` and is not on PATH.

**Nothing loads `.env`.** There is no `python-dotenv` anywhere in the project,
so `DATABASE_URL` must be exported in the shell. `.env` is read only by
docker-compose, which is unused here. This surprises people; it surprised this
session. The same applies to the API keys.

Do not bother re-checking pip-installable Postgres (`pgserver`,
`postgresql-wheel`): neither ships a Windows wheel. That avenue is closed.

All 28 formerly-skipped tests now execute and pass. Running them found **three
real bugs in one sitting** (§4.3, §4.4, §4.5), all in code that had never
touched a database, plus two broken quality gates (§11.4).

---
