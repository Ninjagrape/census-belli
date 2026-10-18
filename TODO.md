# General WAR: TODO (index)

The task list is split by area under `docs/todo/`. **Read only the file for
the stage you are working on**, not all of them. `handover.md` is the index to
the project's state (§1 has the verification commands and expected counts,
§7 the next steps); the footguns are `docs/handover/05-footguns.md`.

All nine stages are built. Phase 1 (corpus repair) and Phase 2 (the cited
`config/overrides.yaml` corrections) are on `main`, and the corpus has been
refit end to end (model run 108, 2026-10-18) and reviewed: no blocker,
internal reading only. The planned work is in
`docs/handover/07-next-sessions-plan.md`: **A** troop numbers (next),
**B** Phase 2b overrides (the user reads the 34 drafts first), **C** report
disclosures and a data-coverage view, **D** the remaining planned fixes.
Items marked **Planned** name their workstream. Items tagged `(gap review)`
come from the 2026-10-10 audit, handover-log §24.

Triaged 2026-10-18: R2d and the death-only window are done or decided, the
battles-admitted count is measured (195 of 210), and the two reviews' findings
are filed as planned items in the stage they belong to.

| File | Covers | Open items (2026-10-18) |
|---|---|---|
| [00-setup-infra.md](docs/todo/00-setup-infra.md) | Project setup, infrastructure, `.claude/` tooling, cross-cutting gaps | 8 (user calls: `.env`, CI, pre-commit, live API test, gold set, gate thresholds; planned: end-to-end test; the unregistered model-reviewer agent) |
| [01-crawl.md](docs/todo/01-crawl.md) | Stage 1 | 0 |
| [02-extract.md](docs/todo/02-extract.md) | Stage 2 | 6 (A: modern estimates, troop scope; Jena-Auerstedt design call; range parsing; two user cost calls) |
| [03-resolve.md](docs/todo/03-resolve.md) | Stage 3 | 6 (planned age-window heuristic; Wŏn Kyun and Francis I in 2b; extraction-eval, label search, threshold tuning, LLM quota) |
| [04-reconcile.md](docs/todo/04-reconcile.md) | Stage 4 | 2 (A: ratio compression, low-precision and scope audit) |
| [05-classify.md](docs/todo/05-classify.md) | Stage 5 | 0 |
| [06-impute.md](docs/todo/06-impute.md) | Stage 6 | 4 (A: weak-side rule; polity map; era bin; record the worst parameter) |
| [07-model.md](docs/todo/07-model.md) | Stage 7 | 1 (optional per-commander WAR counterfactual, user's call) |
| [08-evaluate.md](docs/todo/08-evaluate.md) | Stage 8 | 4 (calibration arm, user call; PPC pooled; circularity scale; replace a named check) |
| [09-report.md](docs/todo/09-report.md) | Stage 9 | 3 (C: disclosures, coverage figures, rank numbering) |
| [stretch-goals.md](docs/todo/stretch-goals.md) | Stretch goals | 7 |

When you tick or add an item, edit the stage file, and update the count here
only if it changes materially.
