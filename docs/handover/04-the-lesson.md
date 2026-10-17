# Handover §4. The lesson

Part of the handover; the index is `handover.md` at the repo root.

Almost every serious defect in this project was invisible to a green suite and
found by pointing code at something real: live Wikipedia, live Wikidata, a real
Postgres, or a number a reader thought looked wrong. Treat "the tests pass" as a
floor. Recurring classes, with the log section that records each:

- **Fixtures encoding the code's own assumption.** Every naval battle labelled
  `field` passed 73 unit tests (log §4.2); a BC-year test asserted the wrong
  convention (§4.5); the circularity comparator passed its integration test with
  a fake force ratio in it (§23.2).
- **Gates whose SQL measures something the stage does not write**, or divides a
  number by itself: log §4.1, §4.6, §11.4, gap review A2.
- **robots.txt applied to an API.** `query.wikidata.org` disallows `/sparql`,
  and routing SPARQL through `Fetcher.fetch` silently disabled every query in
  the project (log §4.7). `www.wikidata.org/w/api.php` is also disallowed. API
  calls go through `fetch_raw`; ask before wrapping any new one in `fetch`.
- **Identifiers recalled from memory are wrong.** Q-ids (Nelson, Agrippa, Red
  Cliffs, Hannibal) and article titles have each been wrong when recalled.
  Look them up live.
- **Exists is not correct.** Disambiguation pages and redirects to campaign
  articles return 200 and pass title checks (log §18.3, §18.9).
- **Year-shaped non-years.** Arsht's `-5000` sentinel, Wikidata dates at
  precision below 9 (decade, century), and Wikidata's signed years (`+1850-...`;
  a fixture writing `1850-01-01` is rejected).
- **Clean diagnostics, prior-only posterior.** A parameter the data cannot
  identify samples at rhat 1.00 and returns its prior (log §19.8-19.9). Check
  prior-to-posterior contraction, not just rhat. The pilot's reconcile did this
  for every side at once (log §25, P4).
- **Tests that hand the stage what the CLI does not.** Integration tests pass a
  `StageContext` with a connection; the orchestrator passes none, so extract and
  resolve wrote nothing from the CLI while every test was green (log §25, P1).
  Verify a stage through `python -m pipeline.orchestrator`, not only pytest.

---
