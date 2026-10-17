# Handover log §28. Corpus repair and classify batch, 2026-10-16 (afternoon and evening)

Part of the session log; the index is `docs/handover-log/README.md`.

- Verified `main` 5b1788d: non-model 2040 passed / 0 skipped, all six model
  files green. The first run showed 61 failures and errors (UndefinedTable,
  DeadlockDetected): the previous session's cut-off verification chain was
  still running against `general_war`. Check for leftover pytest processes
  before trusting a red suite.
- Classify typed 0 battles (`battles_typed: 0`): extract re-runs delete
  classify's provenance notes. Fixed with `battles.battle_type_source` (A).
- Resolve's `no_duplicate_generals` led to stale sides: 69 of 210 battles had
  3-6 sides from label changes across three extract runs (B), which left
  orphan generals (C). The 2026-10-14 refit ran on the duplicated sides.
- A pre-merge review (code and database reviewers) found a HIGH: deleting
  stale sides during a deferred LLM pass would drop LLM-only sides. The
  suggested fix (skip on any deferral) would have disabled the repair on a
  corpus where every battle defers; the shipped rule keeps a stale side only
  if it holds `llm_extraction` rows.
- Ran classify without reconcile after extract, which logged ~100 false
  troop gaps (D). Workers answering the batch noticed ("troop total listed
  as known").
- Herdonia 210 BC dated 2015-12-19 and Dyrrhachium 48 BC dated 1948 came
  from DBpedia `dbp:date` (a Webarchive date, a lost era), not the infobox
  parser the brief first targeted (E). The worker stopped and reported
  rather than implementing the wrong fix.
- Classify batch: 400 requests, 27 Sonnet chunks, 399 valid; the session hit
  the usage limit mid-batch and resumed with `--skip-answered` intact. A
  cwd drift into the worktree broke relative chunk paths; brief workers with
  absolute paths.
- `historiography-reviewer` on the classified corpus: not fit for a refit
  (70 decided battles without outcomes, namesake misattributions, wars typed
  as battles). Plan in handover §7.
