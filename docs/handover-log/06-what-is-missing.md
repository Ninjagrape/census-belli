# Handover log §6. What is missing

Part of the session log; the index is `docs/handover-log/README.md`.

`LICENSE` now exists (GPL-3.0); the previous handover listed it as missing.

Two config files are referenced by agent specs but **do not exist**. Each blocks its stage:

| File | Blocks |
|---|---|
| `config/expert_rankings.yaml` | evaluate |
| `config/sensitivity_specs.yaml` | evaluate |

`README.md`, CI config, pre-commit hooks and the structlog configuration
were all written on 2026-09-19 (§15) and are no longer outstanding.

The resolve gold set now exists at `tests/fixtures/gold/resolve/` (§17.4),
agent-labelled and unreviewed. The extraction gold set (`tests/fixtures/gold/`
for `/extraction-eval`) is still outstanding (see §8).

`pytest-asyncio` is declared in `pyproject.toml` dev extras but **not
installed**, which is the `asyncio_mode` warning you will see. The crawl tests
work around it with a local `asyncio.run` decorator. Installing it would let
that go, and was deliberately not done in §15: `asyncio_mode = "auto"` would
change how those tests are collected, and that is not a change to make in a
session whose purpose was closing other gaps.

---
