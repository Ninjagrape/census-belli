# General WAR: TODO (index)

The task list is split by area under `docs/todo/`. **Read only the file for
the stage you are working on**, not all of them. `handover.md` is the index to
the project's state (§1 has the verification commands and expected counts,
§7 the next steps); the footguns are `docs/handover/05-footguns.md`.

All nine stages are built. Phase 1 of the corpus repair (outcomes, resolve
lifespan guard and link hygiene, campaign typing) is on `main` and has run on
the real corpus through impute (2026-10-17). What is left: Phase 2 (the
`config/overrides.yaml` corrections file, design in
`docs/handover/07-phase2-overrides-design.md`), the refit, the end-to-end
fixture test, and decisions for the user (`docs/handover/07-record.md`).
Items tagged `(gap review)` come from the 2026-10-10 audit, handover-log §24.

| File | Covers | Open items (2026-10-17) |
|---|---|---|
| [00-setup-infra.md](docs/todo/00-setup-infra.md) | Project setup, infrastructure, `.claude/` tooling, cross-cutting gaps | 7 (most are user calls: `.env`, CI, pre-commit, live API test, gold set) |
| [01-crawl.md](docs/todo/01-crawl.md) | Stage 1 | 0 |
| [02-extract.md](docs/todo/02-extract.md) | Stage 2 | 2 (both user calls: deferred passages, web_secondary) |
| [03-resolve.md](docs/todo/03-resolve.md) | Stage 3 | 6 (incl. the death-only lifespan window, label search, threshold tuning) |
| [04-reconcile.md](docs/todo/04-reconcile.md) | Stage 4 | 1 |
| [05-classify.md](docs/todo/05-classify.md) | Stage 5 | 0 |
| [06-impute.md](docs/todo/06-impute.md) | Stage 6 | 0 |
| [07-model.md](docs/todo/07-model.md) | Stage 7 | 1 (measure battles admitted at the refit) |
| [08-evaluate.md](docs/todo/08-evaluate.md) | Stage 8 | 1 (user call: calibration gate's arm) |
| [09-report.md](docs/todo/09-report.md) | Stage 9 | 0 |
| [stretch-goals.md](docs/todo/stretch-goals.md) | Stretch goals | 7 |

When you tick or add an item, edit the stage file, and update the count here
only if it changes materially.
