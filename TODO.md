# General WAR: TODO (index)

The task list is split by area under `docs/todo/`. **Read only the file for
the stage you are working on**, not all of them. `handover.md` is the state of
the project and the traps (footguns are its §5); its §1 has the current verification commands and
expected counts.

All nine stages are built. Every code item that needed no user decision was
finished 2026-10-14..16 (handover-log §27). What is left is mostly the corpus
run (classify's LLM batch, then the refit), the end-to-end fixture test, and
decisions for the user (handover §7).
Items tagged `(gap review)` come from the 2026-10-10 audit, handover-log §24.

| File | Covers | Open items (2026-10-16) |
|---|---|---|
| [00-setup-infra.md](docs/todo/00-setup-infra.md) | Project setup, infrastructure, `.claude/` tooling, cross-cutting gaps | 7 |
| [01-crawl.md](docs/todo/01-crawl.md) | Stage 1 | 0 |
| [02-extract.md](docs/todo/02-extract.md) | Stage 2 | 2 |
| [03-resolve.md](docs/todo/03-resolve.md) | Stage 3, incl. the LLM quota blocker | 6 |
| [04-reconcile.md](docs/todo/04-reconcile.md) | Stage 4 | 1 |
| [05-classify.md](docs/todo/05-classify.md) | Stage 5 | 1 |
| [06-impute.md](docs/todo/06-impute.md) | Stage 6 | 0 |
| [07-model.md](docs/todo/07-model.md) | Stage 7 | 1 |
| [08-evaluate.md](docs/todo/08-evaluate.md) | Stage 8 | 1 |
| [09-report.md](docs/todo/09-report.md) | Stage 9 | 0 |
| [stretch-goals.md](docs/todo/stretch-goals.md) | Stretch goals | 7 |

When you tick or add an item, edit the stage file, and update the count here
only if it changes materially.
