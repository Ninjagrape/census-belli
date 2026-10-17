# Handover log §25. Real-corpus pilot, 2026-10-11 (was handover §11)

Part of the session log; the index is `docs/handover-log/README.md`.

The 211-battle Arsht seed run stage by stage via the CLI into database
`general_war_corpus` (left in place, with `data/raw`, `data/processed`,
`data/imputed` and three 204 MB extract batch files in `data/llm_requests`).
No LLM request was answered. Where the CLI wrote nothing, a scratch driver
passing `StageContext(db_conn=conn)` and wrapping each extract write in a
savepoint was used to reach later stages; nothing from it is in the repo.
P1-P4 were re-checked against the code and the database by the lead; the rest
are the worker's observations, with evidence, not yet re-checked.

**Blocking**

- **P1. [fixed 2026-10-11] The orchestrator never passed a `StageContext`.** Fix: `_call_stage` passes one and commits after `run` returns (also on the offline halt), rolls back if it raises; reconcile, impute and model set `OWNS_DB_CONNECTION = True` and keep their own connection so a failed fit's run row survives. Test: `test_orchestrator_persistence.py`. Original finding: `run_stage` calls
  `runner_module.run(spec)` (`orchestrator.py:169`). Extract logs
  `extract_database_write_skipped` and writes 0 rows; resolve spends ~6 min on
  SPARQL and writes 0 generals. Crawl, reconcile, classify, impute and model
  open their own connection and are unaffected. Extract's LLM service also has
  no connection, so no `llm_calls` audit trail.
- **P2. [fixed] One bad row aborted all of extract's writes.** Fix: `write_battle_isolated` (savepoint per battle; a failed battle leaves a bare `needs_review` row), and the same in resolve's per-identity loop. Classify had no such loop. Original finding: `write_battle` shares one
  transaction with no savepoint (`extract.py:686-693`): the first
  `InvalidDatetimeFormat` turns the remaining 207 battles into
  `InFailedSqlTransaction` and the commit rolls back. The comment there claims
  the opposite. Check resolve and classify write loops for the same pattern.
- **P3. [fixed] Unparsed date text reached `date_start`.** Fix: `pipeline/extractors/dates.py`, plus a guard in `store._upsert_battle`; 209 of 211 real infoboxes now parse (1982 Lebanon War has two incompatible spans, one infobox has no date). `c. 1200 BC` parses as year 1200 BC and loses the circa. Original finding: 56 of 211 battles (27%) fail
  to insert, mostly ancient or multi-day: `"212 BC"`, `"Spring 208 BC"`,
  `"21–22 May 1809"`, `"5–6 June 1967 ( 1967-06-05 – 1967-06-06 )"`,
  `"15 November [ O.S. 3 November ] 1812 – ..."`, `"January 635 AD (13 AH)"`.
  58 records have `date_precision` None. Losing them biases the corpus further
  toward the modern era (already 121 of 155 written battles are 17th-20th c.).
- **P4. [fixed; reviewed by `bayesian-model-reviewer`]** Cause:
  per-source bias and noise were nested in battles (every infobox covers one
  battle) and absorbed each battle's level; junk `total` rows inflated noise; a
  Normal side prior against a t likelihood called large armies' reports
  outliers. Fix: a source key gets its own effect only if it covers >= 3
  battles (`min_battles_per_source_effect`), else type-level bias and sigma
  (pooled keys write NULL bias); load-time plausibility filter
  (`reconcilers/plausibility.py`); `z_side` Student-t (`side_prior_nu: 4`);
  troop-unfillable sides judged on troop rows only and stale estimates
  cleared; new gates `estimates_not_collapsed` (spread_ratio >= 0.25) and
  `reports_honoured`. Corpus after the fix (run 12): rhat 1.007, ESS 699; log-sd
  of estimates 0.51 (was 0.0117), corr with max report 0.76 (was 0.31); Kursk
  Soviet 427,739 (25k-5.2M); stale and unfillable gates pass;
  `reports_honoured` 1 of 39 (Waterloo French, component rows); impute not
  re-run. Deferred review items are in `docs/todo/04-reconcile.md`.
  Original finding: **Reconcile's real-data estimates ignored the reports.** Converged fit
  (5000 draws; the default 1000 missed ESS 400 at 359.6), yet all 275 sides lie
  in 33,798-37,171 (log-sd 0.0117): Kursk Soviet max report 2,500,000 gives
  36,947; Actium Octavian (report 3,000) 35,069; Gaugamela Achaemenid
  (1,000,000) 35,440. Corr(ln est, ln max report) 0.31. Every source is
  `wikipedia_infobox` and 1728 of 1755 lineages are `regime_unlabelled`. The
  per-side posterior sd (~0.26) is wide while the medians are near-identical,
  which suggests side effects are shrunk to the corpus level or the likelihood
  is mis-indexed; undiagnosed. Impute then finds 275 weak sides and 0 battles
  to condition on (`impute_no_observed_sides`), so the model never runs.

**Wrong output**

- Extract stores sub-components as `branch='total'`: Cannae Rome has `Total:
  86,400` plus "25,000 Romans", "30,000 allies" as separate totals; Carthage
  likewise. Ranges become a lower and an upper bound row.
- Resolve: Hannibal (25 mentions), Julius Caesar (9) and Alexander (4) are
  `llm_deferred`, so Cannae and Gaugamela credit Hasdrubal or Parmenion. 13
  well-known people got "no candidate returned" (Schwarzenberg, MacDonald,
  Grouchy, Platov, Amr ibn al-As); Blücher failed the date gate. Luftwaffe,
  Stavka and Army Group B became generals. Austerlitz's "Russian Empire" side
  holds the Austrian commanders. Side polity strings like `"Octavian 's
  forces"` (stray space).
- Classify typed Marathon, Malvern Hill, Yom Kippur War and Operation Torch as
  naval; `field` is never produced (133 of 155 `unknown`).
- The seed includes wars, not battles ("1948 Arab–Israeli War", "Yom Kippur
  War"). Trafalgar is not in it.
- Model design on this data: 154 battles in scope, 108 with no outcome, 33
  with no creditable side, **13 included, 13 generals**. 216 of 308 side
  outcomes are blank, so outcome extraction is the next coverage limit after
  attribution.

**Gates and orchestration**

- Crawl `min_battles_crawled` counts every 200 URL including citations (1986
  here), so it measures citations, not battles. `wikidata_coverage` 0: a named
  corpus fetches no Wikidata entities, `data/raw/wikidata` is empty.
  Citation error rate 0.21 (doi.org 403 x114, 328 read timeouts).
- The orchestrator's stage retry reruns a fixed-seed sampler, so a
  non-converged reconcile fails identically twice and doubles the runtime.
- Reconcile `unfillable_sides_are_logged` 16 while the stage logged 3
  unfillable (naval-excluded reports likely); `ci_width_reasonable` returns
  NULL. Classify `missing_data_all_logged` 30 (error). Impute's gates read
  `data/imputed/neutral/gate_report.json` even under `scale=published`; model
  rejects `scale=published`.
- Extract: 11,337 `citation_line_malformed` warnings (cause not found); every
  re-run writes a new full batch file with identical hashes; 35,277 deferred
  calls dedupe to 22,591 requests, which is the real extract LLM bill for 211
  battles (~107 per battle, far above §7's estimate).

**Runtimes**: crawl 7,034 s; extract 128 s; resolve 353 s cold, 140 s with the
SPARQL cache; reconcile 206 s per attempt at defaults, ~310 s at 5000 draws;
impute (published) 82 s per attempt.
