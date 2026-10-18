# Handover log (index)

Session-by-session history, one file per section. Code comments cite it as
"handover-log §N": open the matching file below, never the whole log.
Two sections were both numbered 16; the second is §16b.

For the next session or agent picking this project up cold.

Read `CLAUDE.md` first for what the project *is*. This file covers what state it is **in**, what is verified versus merely written, and the traps that have already cost time.

`CLAUDE.md` points every session here, and asks you to update this file before you finish. Keep it current: a stale handover is worse than none, because the next agent will trust it. If you change the state of the project, change §1, §6 and §7 to match.

Last updated: 2026-10-10. §21 to §23 are committed (`a9abeba`, `401e1c7`, `3c6f6f8`, `463343a`). §24 is a gap review of every built stage; its findings are in `docs/todo/*.md` (indexed by TODO.md), tagged `(gap review)`, and nothing from it is fixed yet.

---

| Section | Title | File |
|---|---|---|
| §1 | Current state in one table | [01-current-state-in-one-table.md](01-current-state-in-one-table.md) |
| §2 | The database exists now | [02-the-database-exists-now.md](02-the-database-exists-now.md) |
| §3 | Read this before writing any code | [03-read-this-before-writing-any-code.md](03-read-this-before-writing-any-code.md) |
| §4 | Bugs found, and the lesson | [04-bugs-found-and-the-lesson.md](04-bugs-found-and-the-lesson.md) |
| §5 | Known constraints and footguns | [05-known-constraints-and-footguns.md](05-known-constraints-and-footguns.md) |
| §6 | What is missing | [06-what-is-missing.md](06-what-is-missing.md) |
| §7 | Next steps, in order | [07-next-steps-in-order.md](07-next-steps-in-order.md) |
| §8 | Project-specific tooling | [08-project-specific-tooling.md](08-project-specific-tooling.md) |
| §9 | Conventions worth restating | [09-conventions-worth-restating.md](09-conventions-worth-restating.md) |
| §10 | Honest status | [10-honest-status.md](10-honest-status.md) |
| §11 | Session of 2026-09-19: what changed, and what is still open | [11-session-of-2026-09-19-what-changed-and-what-is-s.md](11-session-of-2026-09-19-what-changed-and-what-is-s.md) |
| §12 | Session of 2026-09-19 (later): the resolve stage | [12-session-of-2026-09-19-later-the-resolve-stage.md](12-session-of-2026-09-19-later-the-resolve-stage.md) |
| §13 | Live verification of the resolve stage, 2026-09-19 | [13-live-verification-of-the-resolve-stage-2026-09-1.md](13-live-verification-of-the-resolve-stage-2026-09-1.md) |
| §14 | Chasing the misses from §13.4, 2026-09-19 | [14-chasing-the-misses-from-134-2026-09-19.md](14-chasing-the-misses-from-134-2026-09-19.md) |
| §15 | Session of 2026-09-19 (later still): closing out setup, infrastructure, stages 1 and 2 | [15-session-of-2026-09-19-later-still-closing-out-se.md](15-session-of-2026-09-19-later-still-closing-out-se.md) |
| §16 | CI had never run a single check, 2026-09-20 | [16-ci-had-never-run-a-single-check-2026-09-20.md](16-ci-had-never-run-a-single-check-2026-09-20.md) |
| §16b | Session of 2026-09-20: the `mul` label, and two false-merge paths | [16b-session-of-2026-09-20-the-mul-label-and-two-fals.md](16b-session-of-2026-09-20-the-mul-label-and-two-fals.md) |
| §17 | Session of 2026-09-21: offline processing, gold set, threshold sweep | [17-session-of-2026-09-21-offline-processing-gold-se.md](17-session-of-2026-09-21-offline-processing-gold-se.md) |
| §18 | Session of 2026-09-23: the Arsht reference set and a curated corpus | [18-session-of-2026-09-23-the-arsht-reference-set-an.md](18-session-of-2026-09-23-the-arsht-reference-set-an.md) |
| §19 | Session of 2026-09-24: the reconcile stage, first half | [19-session-of-2026-09-24-the-reconcile-stage-first-.md](19-session-of-2026-09-24-the-reconcile-stage-first-.md) |
| §20 | Sessions of 2026-09-29 to 2026-10-02: reconcile closed out, classify begun | [20-sessions-of-2026-09-29-to-2026-10-02-reconcile-c.md](20-sessions-of-2026-09-29-to-2026-10-02-reconcile-c.md) |
| §21 | Session of 2026-10-06/07: outstanding items closed, impute built | [21-session-of-2026-10-06-07-outstanding-items-close.md](21-session-of-2026-10-06-07-outstanding-items-close.md) |
| §22 | Session of 2026-10-07: the stage inspector | [22-session-of-2026-10-07-the-stage-inspector.md](22-session-of-2026-10-07-the-stage-inspector.md) |
| §23 | Session of 2026-10-09/10: the model stage | [23-session-of-2026-10-09-10-the-model-stage.md](23-session-of-2026-10-09-10-the-model-stage.md) |
| §24 | Gap review of every built stage, 2026-10-10 | [24-gap-review-of-every-built-stage-2026-10-10.md](24-gap-review-of-every-built-stage-2026-10-10.md) |
| §25 | Real-corpus pilot, 2026-10-11 (was handover §11) | [25-real-corpus-pilot-2026-10-11-was-handover-11.md](25-real-corpus-pilot-2026-10-11-was-handover-11.md) |
| §26 | Sweep of pilot and gap-review findings, 2026-10-12 (was handover §12) | [26-sweep-of-pilot-and-gap-review-findings-2026-10-1.md](26-sweep-of-pilot-and-gap-review-findings-2026-10-1.md) |
| §27 | "Fix all outstanding items", 2026-10-14 to 2026-10-16 (was handover §0) | [27-fix-all-outstanding-items-2026-10-14-to-2026-10-.md](27-fix-all-outstanding-items-2026-10-14-to-2026-10-.md) |
| §28 | Corpus repair and classify batch, 2026-10-16 (afternoon and evening) | [28-corpus-repair-and-classify-batch-2026-10-16-afte.md](28-corpus-repair-and-classify-batch-2026-10-16-afte.md) |
| §29 | Historiography review of the Phase 2 refit, with 34 drafted Phase 2b entries (ov-0055..ov-0088), 2026-10-18 | [29-phase2-refit-historiography-review-2026-10-18.md](29-phase2-refit-historiography-review-2026-10-18.md) |
| §30 | Bayesian review of the Phase 2 refit (runs 84, 86, 108), 2026-10-18 | [30-phase2-refit-bayesian-review-2026-10-18.md](30-phase2-refit-bayesian-review-2026-10-18.md) |
