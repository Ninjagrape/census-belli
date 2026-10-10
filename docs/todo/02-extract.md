## Stage 2: Extract (pipeline/stages/extract.py)
Implemented. 74 unit + 6 integration tests. Every SQL write is now verified
against Postgres, including the enum casts and the text-array column.
- [x] MediaWiki infobox parser using mwparserfromhell (infobox.py, infobox_html.py)
  - [x] Handle military conflict infobox template
  - [x] Handle campaignbox templates
  - [x] Handle variant infobox formats (naval, aerial, siege)
  - note: the generic template no longer asserts battle_type. It is used for
    essentially every battle including naval ones, so asserting "field" from it
    mislabelled every naval engagement. classify must infer it now.
- [x] Article body text cleaner (textnorm.py, article.py)
- [x] LLM extraction batch runner (article.py + pipeline/llm/)
  - [x] Chunk long articles into passages under context limit
  - [x] Send each passage with the extract prompt from agents/extract.yaml
  - [x] Parse and validate JSON responses against the output_schema
  - [x] Log every LLM call (input hash, output, tokens, cost)
  - [ ] never run against a live LLM key; all of this is fixture-tested only.
        Harness ready (tests/integration/test_llm_live.py); blocked on the
        free-tier quota. See handover.md 15.7.
- [x] Wikidata JSON mapper (wikidata_mapper.py)
- [x] DBpedia RDF mapper (pipeline/extractors/dbpedia_mapper.py)
      - contributes typed dates, coordinates, part_of, place and result. BC dates
        become Postgres literals: DBpedia writes 31 BC as -031-09-02, three year
        digits not four.
      - **returns sides=[] deliberately.** DBpedia flattens combatant1/combatant2
        and commander1/commander2 into unordered lists — Trafalgar reports one
        combatant ("Spain") and four commanders with no marker of which fleet
        anyone was in. Splitting that into sides would be invention. They go to
        notes marked "unassigned to sides". See handover.md 15.3.
- [x] Merge extractions from infobox + body + Wikidata + DBpedia (merger.py)
      - citations are merged via the LLM pass; DBpedia now enters at priority 35
- [x] Write to data/processed/*.jsonl and DB tables (store.py)
- [x] Unit tests for infobox parsing with fixture HTML
- [x] Integration test: extract known battle (Actium) and verify commander/troop fields
- [x] Log absent model covariates (battle_type, terrain, fortified) to missing_data_log
- [ ] (gap review A2) `no_empty_extractions` reads crawl_log.extraction_method /
      fields_extracted, which nothing writes, so it can never fail. Rewrite it
- [ ] (gap review A4) battles are looked up by name with LIMIT 1 here and in resolve's
      `load_battle_index`; same-name battles collapse. Key on slug or source URL
- [ ] (gap review A3) `part_of` is extracted but no wars/campaigns rows are written,
      so battles.war_id / campaign_id are always NULL
- [ ] (gap review A7) a `--limit` run rewrites data/processed/*.jsonl with a subset
- [ ] (gap review) llm_calls.battle_id always NULL (battle_id not passed to the LLM pass);
      sources.url holds local file paths for infobox/wikidata/dbpedia sources
