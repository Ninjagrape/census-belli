# General WAR: TODO (index)

The task list is split by area under `docs/todo/`. **Read only the file for
the stage you are working on**, not all of them. `handover.md` is the state of
the project and the traps (footguns are its §5); its §1 has the current verification commands and
expected counts.

Stages 1-8 (crawl to evaluate) are built and pass their error gates on the
real corpus through the CLI, with every LLM request still deferred (handover
§1). Report is built and committed (6c91f1a); it runs on the corpus and marks
its output provisional. Work merged only on the unpushed `wip/outstanding`
branch (handover §0) is not ticked here until it reaches `main`.
Items tagged `(gap review)` come from the 2026-10-10 audit, handover-log §24.

| File | Covers | Open items (2026-10-15) |
|---|---|---|
| [00-setup-infra.md](docs/todo/00-setup-infra.md) | Project setup, infrastructure, `.claude/` tooling, cross-cutting gaps | 9 |
| [01-crawl.md](docs/todo/01-crawl.md) | Stage 1 | 2 |
| [02-extract.md](docs/todo/02-extract.md) | Stage 2 | 4 |
| [03-resolve.md](docs/todo/03-resolve.md) | Stage 3, incl. the LLM quota blocker | 6 |
| [04-reconcile.md](docs/todo/04-reconcile.md) | Stage 4 | 8 |
| [05-classify.md](docs/todo/05-classify.md) | Stage 5 | 10 |
| [06-impute.md](docs/todo/06-impute.md) | Stage 6 | 1 |
| [07-model.md](docs/todo/07-model.md) | Stage 7 | 6 |
| [08-evaluate.md](docs/todo/08-evaluate.md) | Stage 8 | 2 |
| [09-report.md](docs/todo/09-report.md) | Stage 9 | 2 |
| [stretch-goals.md](docs/todo/stretch-goals.md) | Stretch goals | 7 |

When you tick or add an item, edit the stage file, and update the count here
only if it changes materially.
