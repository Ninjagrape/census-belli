## Stage 1: Crawl (pipeline/stages/crawl.py)
Implemented. 76 unit + 2 integration tests. The integration tests now run
against a real database.
- [x] Wikipedia battle list page parser (pipeline/crawlers/wikipedia.py)
- [x] Individual battle article fetcher with rate limiting and robots.txt (fetcher.py)
- [x] Citation URL extractor from Wikipedia article references (citations.py)
- [x] Citation fetcher (allow-listed domains only) (citations.py)
- [x] Wikidata SPARQL client using queries from sources_seed.yaml (wikidata.py, seeds.py)
- [x] DBpedia RDF fetcher (dbpedia.py)
- [x] Crawl state persistence (resume from interruption) (state.py)
- [x] Crawl log DB writer (log.py) — verified against Postgres
- [x] Deduplication of battle URLs across different list pages
- [x] Integration test with 5-10 known battle URLs
      - tests/integration/test_crawl_live.py, 10 real articles, opt-in via
        GENERAL_WAR_LIVE_CRAWL=1, never run in CI. **Run 2026-09-19: all 5 pass.**
        Builds its fetcher from the shipped agents/crawl.yaml, so a rate limit
        too fast for Wikipedia fails the test rather than being dodged by it.
      - it also pins the 4.2 naval regression against live HTML, and that
        robots.txt still permits /wiki/
- [x] 429-aware backoff before any real crawl
      - Fetcher honours Retry-After (both RFC 9110 forms), raising the wait but
        never lowering it, capped at 120s. rate_limit_wikipedia 1.0 -> 2.0.
      - measured live: 10 consecutive fetches, zero retries, zero 429s
- [ ] (gap review A1) `min_battles_crawled` is an ERROR gate at 3000; the 211-battle
      arsht pilot halts on it. Parameterise per corpus or downgrade for named corpora
- [ ] (gap review A5) `SqlCrawlLog.flush` swallows insert errors with no rollback;
      one failure aborts every later flush on Postgres
- [ ] (gap review A6) crawl stores rendered HTML only, so extract's preferred
      `.wikitext` template path never runs on a real crawl
- [ ] (gap review) `external_datasets` (COW) in sources_seed.yaml never read;
      `citation_depth` > 1 behaves as 1
