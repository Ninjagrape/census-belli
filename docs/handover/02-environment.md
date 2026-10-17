# Handover §2. Environment

Part of the handover; the index is `handover.md` at the repo root.

- **Postgres 15.19, native** (winget). No Docker, podman or `make`.
  `host 127.0.0.1 port 5432 db/user/password general_war`. `psql` is under
  `C:\Program Files\PostgreSQL\15\bin\`, not on PATH. pip-installable Postgres
  (`pgserver`, `postgresql-wheel`) has no Windows wheel; do not retry it.
- **`.env`** is read by `pipeline.config.load_env()` (BOM sniffed: the file here
  is UTF-16). Exported variables win. The file holds only an unused Gemini
  key, so export `DATABASE_URL` yourself.
- **Python 3.14 locally, 3.11 in CI.** `pymc>=5.28,<6` is pinned because it
  installs on both; pymc 6 needs 3.12. Local anthropic is 0.111 although the
  project declares `>=1.0`; mypy is clean against both, runtime under 1.x is
  untested.
- **No C compiler**, so PyTensor's Python fallback is unusably slow. `nutpie`
  (numba) is the default `nuts_sampler`.
- CI installs only what is declared. Stub packages (`types-*`) are invisible at
  runtime and easy to forget. After adding a top-level directory, run
  `pip install -e . --dry-run` (package discovery is pinned to `pipeline*`).

---
