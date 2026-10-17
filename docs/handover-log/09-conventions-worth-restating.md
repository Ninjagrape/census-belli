# Handover log §9. Conventions worth restating

Part of the session log; the index is `docs/handover-log/README.md`.

From `CLAUDE.md`, plus what was actually enforced:

- Type hints on every signature; Google-style docstrings on every public function.
- `structlog` only, never `print()` in pipeline code.
- f-strings only.
- SQLAlchemy **Core**, parameterised. Never interpolate into SQL.
- `mypy --strict` must pass. It currently does, across 37 files. Keep it that way — it caught several `Any` leaks.
- Comment on **why**, not what.
- Never commit without the user asking. Attribution lines are disabled globally.
- Check `.env` is not staged before every commit. It exists, is gitignored, and contains real keys.

---
