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

- [ ] **LLM quota for a real run.** 3,000 battles x 2-4 passages is 6,000-12,000
      extract calls. Gemini is gone; every stage defaults to `llm_mode: offline`,
      answered by `/process-llm-batch` (Haiku workers, 15 requests per chunk)
      on the Claude subscription's weekly limits. The command is resumable
      (`split --skip-answered`), so a corpus spreads over several sessions.
      - [x] done 2026-10-16..17 for resolve, classify and extract_outcome on
            the 211-battle corpus (~550 requests, spot-checked); extract's
            14,705 passage requests remain the user's call (02-extract.md)
      - [ ] Run the ~20-call `/extraction-eval` on `claude-haiku-5-5` before
            trusting it on the corpus: it cannot take temperature 0, so
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
- [x] fixed 2026-10-12: (pilot) organisations become generals (Luftwaffe, Stavka, Army Group B);
      "no candidate returned" for Schwarzenberg, Grouchy, Platov, Amr ibn al-As;
      Blücher rejected by the date gate. Check write loop for the P2 pattern
- [x] fixed 2026-10-12: (gap review A8) mentions whose battle/side is missing from the DB are counted but
      not logged to missing_data_log, so resolution_rate reads high; an extract re-run
      also clears resolve's denominator rows
- [x] fixed 2026-10-12: (gap review A9) generals.born / died / era / highest_rank never written, though
      candidates carry birth and death years
- [x] fixed 2026-10-12: (gap review) `no_duplicate_generals` is vacuous (wikidata_id is UNIQUE); it cannot
      see duplicates among corpus-local generals
- [x] confirmed live 2026-10-17: Schwarzenberg Q163755, Macdonald Q298784, Grouchy Q333821, Platov Q963749, Amr ibn al-As Q314781, Blücher Q153522 all link on the corpus. Still unlinked or split: Moreau, Bülow, Wŏn Kyun, Lefebvre (one row of two), Bernadotte (mislinked to his brother Q15968159; Phase 2)
- [ ] The corpus-identity match path (`match_against_identities`) is not date-gated, and a lifespan with birth after death is trusted: A. P. Hill at Borodino, the younger Zieten at Hochkirch, Saxe-Weissenfels at Arcole (R2d, in progress 2026-10-17)
- [ ] A death-only or century-precision lifespan leaves the window ~100 years wide: Philippikos Bardanes (emperor 711-713) linked at Ajnadayn 634. Decide after R2d whether to tighten MAX_PLAUSIBLE_AGE_YEARS for death-only reckoning or leave such cases to overrides
