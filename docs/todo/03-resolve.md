## Stage 3: Resolve (pipeline/stages/resolve.py)
Built and verified against live Wikidata and Postgres. Decisions to keep:
- Wikidata person labels are migrating to the `mul` language code; lookups ask
  in both `en` and `mul` (Nelson has no `en` label, handover-log §16.1).
- Matching: the lifespan date gate is the hard filter; polity is a tiebreak
  only, because polity labels are anachronistic more often than wrong.
- Battle deduplication reports to data/processed/duplicate_battles.jsonl and
  deliberately does not merge (handover-log §12.4).
- Live regression tests: tests/integration/test_resolve_live.py, gated on
  GENERAL_WAR_LIVE_CRAWL=1.

- [ ] **Blocker for any real run: the Gemini key is free tier, 20 requests per
      day.** 3,000 battles x 2-4 passages is 6,000-12,000 extract calls, so
      300-600 days. The cap is the problem, not cost (one measured call $0.0014,
      corpus order $50-200), so any paid tier removes it. handover-log §16.8.
      Workaround built: offline path (`pipeline/llm/offline.py`,
      `/process-llm-batch`, `scripts/llm_offline.py import`), correlated on
      `request_hash`.
      - [ ] ~50-battle cost pilot on both providers, reusing the extrapolation
            in `.claude/commands/extraction-eval.md` section 5
      - [ ] Run the ~20-call `/extraction-eval` on `claude-haiku-5-5` before
            relying on it as fallback: it cannot take temperature 0, so
            determinism rests on the structured-output constraint alone
- [ ] Label search (wbsearchentities / SPARQL CONTAINS): deferred, not dropped.
      It would fire only on zero-candidate names, where its hit is usually the
      sole candidate, so `matcher.py` skips the ambiguity-margin test and
      auto-links at ~0.62 confidence where evidence is weakest. Live probes are
      noisy. Size the residual class with a gold set first (handover-log §16.6).
- [ ] Tune fuzzy_threshold, ambiguity_margin, battle_duplicate_threshold against
      hand-labelled data. Spot check and `tests/fixtures/gold/resolve/` (15
      mentions, agent-labelled, unreviewed) both show 0 wrong links at defaults
      (85.0/6.0), but the gold set is too clear-cut to discriminate thresholds.
      Sweep: `scripts/resolve_sweep.py`, `/resolve-eval`.
- [ ] (gap review A8) mentions whose battle/side is missing from the DB are counted but
      not logged to missing_data_log, so resolution_rate reads high; an extract re-run
      also clears resolve's denominator rows
- [ ] (gap review A9) generals.born / died / era / highest_rank never written, though
      candidates carry birth and death years
- [ ] (gap review) `no_duplicate_generals` is vacuous (wikidata_id is UNIQUE); it cannot
      see duplicates among corpus-local generals
