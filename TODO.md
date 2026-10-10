# General WAR — TODO (index)

The task list is split by area under `docs/todo/`. **Read only the file for
the stage you are working on**, not all of them. `handover.md` is the state of
the project and the traps; its §1 has the current verification commands and
expected counts.

Stages 1-7 (crawl to model) are built; 1-3 are verified against a real
database, and 4-7 against fixtures only. Evaluate and report are unbuilt.
Items tagged `(gap review)` come from the 2026-10-10 audit, handover.md §24.

| File | Covers | Open items (2026-10-10) |
|---|---|---|
| [00-setup-infra.md](docs/todo/00-setup-infra.md) | Project setup, infrastructure, `.claude/` tooling, cross-cutting gaps | 10 |
| [01-crawl.md](docs/todo/01-crawl.md) | Stage 1 | 4 |
| [02-extract.md](docs/todo/02-extract.md) | Stage 2 | 6 |
| [03-resolve.md](docs/todo/03-resolve.md) | Stage 3, incl. the LLM quota blocker | 8 |
| [04-reconcile.md](docs/todo/04-reconcile.md) | Stage 4 | 8 |
| [05-classify.md](docs/todo/05-classify.md) | Stage 5 | 11 |
| [06-impute.md](docs/todo/06-impute.md) | Stage 6 | 2 |
| [07-model.md](docs/todo/07-model.md) | Stage 7 | 8 |
| [08-evaluate.md](docs/todo/08-evaluate.md) | Stage 8 (unbuilt) | 14 |
| [09-report.md](docs/todo/09-report.md) | Stage 9 (unbuilt), incl. two BLOCKING items | 12 |
| [known-constraints.md](docs/todo/known-constraints.md) | Footguns (also in handover.md §5) | — |
| [stretch-goals.md](docs/todo/stretch-goals.md) | Stretch goals | 7 |

When you tick or add an item, edit the stage file, and update the count here
only if it changes materially.
