## Stage 2: Extract (pipeline/stages/extract.py)
Built; every SQL write verified against Postgres. Decisions to keep:
- The generic {{Infobox military conflict}} template does not assert
  battle_type (it covers naval battles too); classify infers it.
- DBpedia returns `sides=[]` deliberately: it flattens combatants and
  commanders into unordered lists, so splitting them into sides would be
  invention. They go to notes as "unassigned to sides" (handover-log §15.3).
  DBpedia writes BC years with three digits (31 BC is -031-09-02).

- [ ] LLM extraction never run against a live key; fixture-tested only.
      Blocked on the free-tier quota (handover-log §15.7, 03-resolve.md).
- [ ] (gap review A2) `no_empty_extractions` reads crawl_log.extraction_method /
      fields_extracted, which nothing writes, so it can never fail. Rewrite it
- [ ] (gap review A4) battles are looked up by name with LIMIT 1 here and in resolve's
      `load_battle_index`; same-name battles collapse. Key on slug or source URL
- [ ] (gap review A3) `part_of` is extracted but no wars/campaigns rows are written,
      so battles.war_id / campaign_id are always NULL
- [ ] (gap review A7) a `--limit` run rewrites data/processed/*.jsonl with a subset
- [ ] (gap review) llm_calls.battle_id always NULL (battle_id not passed to the LLM pass);
      sources.url holds local file paths for infobox/wikidata/dbpedia sources
