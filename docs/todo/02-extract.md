## Stage 2: Extract (pipeline/stages/extract.py)
Built; every SQL write verified against Postgres. Decisions to keep:
- The generic {{Infobox military conflict}} template does not assert
  battle_type (it covers naval battles too); classify infers it.
- DBpedia returns `sides=[]` deliberately: it flattens combatants and
  commanders into unordered lists, so splitting them into sides would be
  invention. They go to notes as "unassigned to sides" (handover-log §15.3).
  DBpedia writes BC years with three digits (31 BC is -031-09-02).

- [x] (pilot P2) fixed 2026-10-11: `write_battle` loop shares one transaction with no savepoint; one
      bad row aborts all later writes and the commit rolls back
- [x] (pilot P3) fixed 2026-10-11: unparsed date strings reach date_start (56/211 fail: "212 BC",
      "Spring 208 BC", "21–22 May 1809", O.S. brackets, AH dates)
- [x] fixed 2026-10-12: (pilot) **next up**: arm-of-service and component figures stored as
      branch='total' (Cannae "25,000 Romans", Waterloo "7,232 gunners"); the main
      error left in reconcile and the one reports_honoured violation
- [ ] (pilot) 22,591 requests for 211 battles; 11,337 citation_line_malformed
      warnings; each re-run rewrites an identical 204 MB batch file
- [ ] LLM extraction never run against a live key; fixture-tested only.
      Blocked on the free-tier quota (handover-log §15.7, 03-resolve.md).
- [x] fixed 2026-10-12: (gap review A2) `no_empty_extractions` reads crawl_log.extraction_method /
      fields_extracted, which nothing writes, so it can never fail. Rewrite it
- [x] fixed 2026-10-12: (gap review A4) battles are looked up by name with LIMIT 1 here and in resolve's
      `load_battle_index`; same-name battles collapse. Key on slug or source URL
- [ ] (gap review A3) `part_of` is extracted but no wars/campaigns rows are written,
      so battles.war_id / campaign_id are always NULL
- [x] fixed 2026-10-12: (gap review A7) a `--limit` run rewrites data/processed/*.jsonl with a subset
- [x] fixed 2026-10-12: (gap review) llm_calls.battle_id always NULL (battle_id not passed to the LLM pass);
      sources.url holds local file paths for infobox/wikidata/dbpedia sources
- [ ] Circa dates carry no flag: "c. 1200 BC" reads as exact; "1-3 May c. 1450 BC" gives a wrong range; "late 1250s BC", "1805 or 1806" return None
- [ ] 19,790 web_secondary citation passages (~94 per battle) remain after dedupe (projected 76 requests per battle); keep or cut is a cost decision
