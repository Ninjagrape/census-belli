## Stage 1: Crawl (pipeline/stages/crawl.py)
Built and verified against a real database. The live test
(tests/integration/test_crawl_live.py, GENERAL_WAR_LIVE_CRAWL=1, never in CI)
builds its fetcher from agents/crawl.yaml and fails if any fetch needs a retry,
so a rate limit too fast for Wikipedia is a test failure. It also pins the 4.2
naval regression against live HTML.

- [x] fixed 2026-10-12: (pilot) min_battles_crawled counts citation URLs too; named corpora fetch
      no Wikidata entities (wikidata_coverage 0); seed holds wars, not battles
- [x] fixed 2026-10-12: (gap review A1) `min_battles_crawled` is an ERROR gate at 3000; the 211-battle
      arsht pilot halts on it. Parameterise per corpus or downgrade for named corpora
- [x] fixed 2026-10-12: (gap review A5) `SqlCrawlLog.flush` swallows insert errors with no rollback;
      one failure aborts every later flush on Postgres
- [ ] (gap review A6) crawl stores rendered HTML only, so extract's preferred
      `.wikitext` template path never runs on a real crawl
- [ ] (gap review) `external_datasets` (COW) in sources_seed.yaml never read;
      `citation_depth` > 1 behaves as 1
