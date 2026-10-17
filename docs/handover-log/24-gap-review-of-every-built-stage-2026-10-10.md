# Handover log §24. Gap review of every built stage, 2026-10-10

Part of the session log; the index is `docs/handover-log/README.md`.

Three read-only Sonnet audits compared each `agents/<stage>.yaml` with its
code (crawl to model, plus cross-cutting); Opus confirmed the highest-impact
findings by grep. Static reading only: no tests run, no database touched.
Full table: `~/.claude/plans/review-and-examine-what-spicy-wilkes.md`. Every
new item is in `docs/todo/<stage>.md` tagged `(gap review)`.

New findings, in rough order of what they cost a real run:

- **A1** `crawl.min_battles_crawled` is an ERROR gate at 3000 URLs. The
  211-battle arsht pilot (§7 step 0) halts on it.
- **A13** `--stages all` halts at evaluate: `pipeline/stages/evaluate.py`
  does not exist, the import fails, and `complete_with_warnings` does not
  stop the halt (only `continue_with_logging` does).
- **A4** Extract and resolve find battles by name (`LIMIT 1` / lowest id).
  Two battles sharing a name collapse into one, with commanders on the
  wrong battle.
- **A2** `extract.no_empty_extractions` reads `crawl_log.extraction_method`
  and `fields_extracted`, which nothing writes. It can never fail: §4.6's
  pattern again.
- **A5** `SqlCrawlLog.flush` swallows insert errors without rolling back. On
  Postgres every later flush then fails silently.
- **A8** Resolve counts mentions whose battle or side is missing from the DB
  but does not log them, so `resolution_rate` reads high. An extract re-run
  also clears resolve's denominator rows from `missing_data_log`.
- **A10** Classify reads troop presence from raw `troop_reports`. Reconcile's
  `unfillable` sides therefore look like they have troops, no missingness
  heuristic can fire, and impute's `mnar_delta` never reaches them.
- **A11** Reconcile centres `data_era` over all dated rows but applies it
  only to unlabelled rows, so `a_era` acts partly as a second intercept.
- Smaller: `battles.war_id`/`campaign_id` never populated (A3); crawl stores
  no wikitext, so extract's template path never runs (A6); `--limit`
  rewrites the processed JSONL with a subset (A7); `generals.born/died/era`
  never written (A9); report/evaluate gates with no run filter at all
  (A14); stale schema comments (A15); the inspector Model page is still a
  placeholder querying a wrong table name (A16); unused deps, dead spec
  params, missing `scripts.add_battle`/`quality_report` (A18).

Also confirmed: every item already listed as open for reconcile, classify,
impute and model is still open in code. None was quietly fixed.
