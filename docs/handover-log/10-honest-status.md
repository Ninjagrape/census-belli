# Handover log §10. Honest status

Part of the session log; the index is `docs/handover-log/README.md`.

What is proven: the code lints, type-checks strictly, and 290 tests pass
**against a real PostgreSQL 15.19**. The schema applies, the Alembic migration
round-trips on real BC rows, and the database layer, crawl log and extract
writer have all now executed against Postgres rather than a fixture. Five real
bugs have been found and fixed.

What is not proven: any real LLM call, any real crawl at scale, and the whole
statistical half of the project, which does not exist yet. The quality gates
now execute correctly across all 9 specs, but 22 of them have only been run
against an empty database, so they have proven their SQL is valid, not that
their thresholds are right.

The lesson has now been paid for twice. §4.2 was invisible to 73 passing unit
tests. §4.3 and §4.4 were invisible to 255. In both cases the tests agreed
with the code because both had been written from the same assumption, and only
contact with the real thing -- live Wikipedia, live Postgres -- disagreed.
Treat "the tests pass" as a floor, not a finish line, and be specific about
which real thing you have actually touched.
