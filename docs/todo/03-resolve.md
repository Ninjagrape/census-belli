## Stage 3: Resolve (pipeline/stages/resolve.py)
- [x] Wikidata entity lookup by name (SPARQL: find humans with matching label + military occupation)
      - verified against the live endpoint 2026-09-19. Doing so found that
        robots.txt had been silently disabling every SPARQL query in the
        project, in crawl as well as resolve. See handover.md 4.7.
- [x] Query coverage: entities matched by alias were being discarded after
      the query found them, which is why "Duke of Wellington" returned
      nothing. Fixed by selecting ?name. See handover.md 14.1.
- [x] Exact-literal lookup could not reach Horatio Nelson (Q83235). **The
      diagnosis in 14.7 was wrong** — he has no `en` label at all; the name
      lives on his **`mul`** label, Wikidata's multilingual code, onto which
      person labels have been migrating since 2024. The lookup asked only for
      `"..."@en`. Fixed by asking in both tags and setting the label service to
      `"en,mul"`; verified live 2026-09-20, and Nelson now links
      deterministically with no LLM call. See handover.md 16.1.
- [ ] Label search (wbsearchentities / SPARQL CONTAINS): **deferred, not
      dropped.** Its premise is refuted by the above, and it would fire only on
      zero-candidate names — where its hit is usually the *sole* candidate and
      `matcher.py` skips the ambiguity-margin test, auto-linking at ~0.62
      confidence exactly where evidence is weakest. Probed live it is noisy
      (paintings, pubs, a racehorse). The residual class is unmeasured; the
      gold set below should size it before anything is built. See handover.md 16.6.
- [ ] **Blocker for any real run: the Gemini key is free tier, 20 requests per
      day.** Stated properly: 3,000 battles x ~2-4 passages each is 6,000-12,000
      extract calls, so **300-600 days**. The project cannot run. The blocker is
      the *request cap*, not cost — the one measured call was $0.0014, putting
      the corpus at order $50-200 — so any paid tier removes it. See handover.md 16.8.
      - [x] **Offline processing path** replaces the batch API design:
            `pipeline/llm/offline.py` exports uncached requests to JSONL,
            the user processes them via Claude.ai upload or `/process-llm-batch`
            in a Claude Code session (Pro subscription, not billed API), and
            `scripts/llm_offline.py import` writes responses to `llm_calls`.
            Correlates on `request_hash`. See `.claude/commands/process-llm-batch.md`.
      - [ ] ~50-battle **cost pilot** on both providers to replace the estimate
            with a measurement, reusing the extrapolation in
            `.claude/commands/extraction-eval.md` section 5.
      - [x] Resolve the spec conflict: `CLAUDE.md` updated to reflect that
            extract/resolve use Gemini, classify uses Anthropic. Haiku 5.5
            (was 4.5) documented as fallback in both agent specs.
      - [ ] Run the ~20-call `/extraction-eval` on `claude-haiku-5-5` before
            relying on it as the fallback: it cannot take temperature 0, so
            determinism rests on the structured-output constraint alone.
- [x] Exact match linker (name -> Wikidata ID)
- [x] Fuzzy match with context (rapidfuzz + lifespan gate + polity tiebreak)
      - the date gate does the real work: a candidate who was not alive cannot
        have commanded. Era is the hard gate; polity is a tiebreak only,
        because polity labels are anachronistic more often than they are wrong.
- [x] LLM disambiguation for ambiguous cases (prompt from agents/resolve.yaml)
      - verified live 2026-09-19: google-genai 2.24.0 installed, one real
        gemini-3.8-flash call resolved the genuine Yi Sun-sin ambiguity
        correctly, wrote its llm_calls row, and the repeat call was served
        from the cache at zero cost. See handover.md 13.3.
- [x] General deduplication (merge records sharing a Wikidata ID)
- [x] Alias table builder
- [x] Battle deduplication (same event under different names)
      - detects and reports to data/processed/duplicate_battles.jsonl;
        deliberately does not merge. See handover.md 12.4.
- [x] Resolution audit log writer (data/processed/resolution_log.jsonl)
- [x] Unit tests for fuzzy matching edge cases (59 tests)
- [x] Integration test: two spellings of one commander resolve to one entity
      - 18 integration tests plus an end-to-end run of resolve.run() against
        live Postgres. Agrippa / Marcus Vipsanius Agrippa across two battles.
- [ ] Tune fuzzy_threshold, ambiguity_margin and battle_duplicate_threshold
      against hand-labelled data. Re-measured live 2026-09-20 after the mul fix
      on an 8-commander spot check: Nelson, Wellington, Napoleon, Agrippa and
      Scipio all link deterministically, Yi Sun-sin correctly defers to the LLM,
      and both Hannibal cases defer rather than link. **Zero wrong links**,
      which is the number that matters. Still a spot check, not a gold set.
      - [x] `tests/fixtures/gold/resolve/` — 15 mentions, 12 battles, synthetic
            cached candidates. Agent-labelled, unreviewed. See handover.md §17.4.
      - [x] `scripts/resolve_sweep.py` + `.claude/commands/resolve-eval.md`.
            Zero LLM quota. At defaults (85.0/6.0): 0 wrong links, 11 correct,
            1 ambiguous. Gold set too clear-cut to discriminate thresholds.
- [x] `tests/integration/test_resolve_live.py` — 5 live regression tests,
      gated on GENERAL_WAR_LIVE_CRAWL=1. Covers mul label, Q-id label shape,
      date gate separation, deterministic resolution. See handover.md §17.3.
- [ ] (gap review A8) mentions whose battle/side is missing from the DB are counted but
      not logged to missing_data_log, so resolution_rate reads high; an extract re-run
      also clears resolve's denominator rows
- [ ] (gap review A9) generals.born / died / era / highest_rank never written, though
      candidates carry birth and death years
- [ ] (gap review) `no_duplicate_generals` is vacuous (wikidata_id is UNIQUE); it cannot
      see duplicates among corpus-local generals
