# Handover log §26. Sweep of pilot and gap-review findings, 2026-10-12 (was handover §12)

Part of the session log; the index is `docs/handover-log/README.md`.

Seven parallel workers (one worktree and one scratch database each,
`general_war_wt_*`), merged and committed. What a future session needs:

- **Orchestrator**: `--continue-past-deferred` runs later stages on
  deterministic data after an offline LLM deferral (exit code 3 at the end;
  the result is provisional). A gate failure retries only with
  `retry_policy.retry_on_gate_failure: true` (crawl only); reconcile, impute
  and model have `max_stage_retries: 0` (fixed seed; model retries
  non-convergence itself, reconcile and impute do not). `--stages all` skips
  unbuilt stages. Gate thresholds may say `{params.<name>}`; a missing param
  fails the check. `imputation_gate` reads the run's `params.scale`.
- **Crawl**: gates read a marker row `crawl-plan://battle-articles?targets=..&fetched=..&entities=..&floor=..`
  (http_status 0) written to crawl_log at the end of each run; error-rate SQL
  excludes it. Named corpora fetch each article's Wikidata entity via
  `Special:EntityData` (robots allows it), Q-id read from the article HTML.
- **Extract**: arm-of-service figures get their branch, contingents/nationalities
  `other`; a side with only components gets one derived total marked
  `is_lower_bound`, `page_or_section='infobox:derived_total'`. Victor matching
  uses demonym stems plus a small table, still exactly-one-side. Battles are
  keyed by `battles.source_slug` (the crawl filename stem), so same-name
  battles stay apart. Offline batch files with identical hash sets are reused.
  Circa dates are not flagged (`c. 1200 BC` reads as exact) - open.
- **Resolve**: non-person filter by name pattern and P31; title/particle
  stripping and a search fallback for zero-candidate names (fixture-tested
  only, not live); generals get born/died and `*_year_astronomical` columns.
- **Classify**: `battles.wikipedia_url` (Wikidata/DBpedia form) never matched
  crawl's hashed filenames, so no article had ever been read; lookup now uses
  `source_slug`, then title. Naval needs ship vocabulary to dominate. An
  existing non-`unknown` battle_type is never rewritten, so reset it before
  re-running classify on an old database. `is_defender` is set only for
  "Siege of X" where one side names X.
- **Reconcile**: `data_era` is centred over the unlabelled rows it applies to
  (reviewed, approved); a non-finite rhat/ESS fails the write gate; withheld
  sides are logged; `ci_width_reasonable` and the singleton gate measure
  multi-claim and single-claim sides by `est_troops_method`.
- **Model**: `scale=published` runs write model types with a `_published`
  suffix so the neutral ranking stays `MAX(run_id)` of the plain type.
- **Parallel workers**: give each its own database; three workers once picked
  the same Alembic revision id, so check `alembic heads` after any merge.
