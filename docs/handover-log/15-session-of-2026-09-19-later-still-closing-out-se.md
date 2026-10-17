# Handover log §15. Session of 2026-09-19 (later still): closing out setup, infrastructure, stages 1 and 2

Part of the session log; the index is `docs/handover-log/README.md`.

The brief was to wrap up every outstanding item in project setup,
infrastructure, Stage 1 and Stage 2. All are now closed except the two that no
code change can close, recorded in §15.7.

Two decisions were the user's and were taken by them: the free-tier Gemini
blocker is documented rather than worked around, and the live crawl test was
built opt-in **and run**.

### 15.1 Wikipedia's 429, which the crawler had been ignoring

`Fetcher.fetch_raw` computed `backoff_base * 2**attempt` and ignored what the
server said. Against a rate limiter that answers "retry in 29s" that spends
three attempts inside ten seconds and gives up on a limit that was never going
to lift -- the same defect §14.4 found and fixed on the LLM side, still open on
the crawl side until now.

`retry_after_seconds()` reads both forms RFC 9110 allows, because Wikimedia
sends both depending on which layer refuses: an integer from the API limiter,
an HTTP-date from the CDN. A server-named delay **raises** the wait and never
lowers it -- the computed backoff stays a floor -- and is capped at
`MAX_SERVER_BACKOFF_S` (120s), because a crawl of thousands of pages cannot
block an hour on one of them. Four unit tests pin those behaviours separately.

`rate_limit_wikipedia` went 1.0 to 2.0. §5.2 had measured 1.5s drawing a 429,
and the spec was still set below the value that had already failed.

### 15.2 The live crawl test, and what running it found

`tests/integration/test_crawl_live.py`, five tests, gated on
`GENERAL_WAR_LIVE_CRAWL=1` and never run in CI. It builds its fetcher from the
**shipped `agents/crawl.yaml`**, not from test defaults, so a rate limit too
fast for Wikipedia fails the test rather than being dodged by it.

Run live, all five pass in about 30 seconds:

| Check | Result |
|---|---|
| Ten articles fetched at the configured spacing | **zero retries, zero 429s** |
| robots.txt still permits `/wiki/`, still forbids the edit path | yes |
| All ten infoboxes parse, each yielding two sides | yes |
| No naval battle labelled a land engagement (§4.2 regression) | yes, against live HTML |
| DBpedia still serving | yes |

**The first run failed, and the bug was mine.** `Battle_of_Hałycz` was in the
URL list and returns 404; I had written the title from memory. That is §14.6's
lesson -- "Look Q-ids up. Do not remember them." -- applying equally to article
titles. The replacement, `Battle_of_Łódź_(1914)`, was checked against the live
MediaWiki API before use, and covers the non-ASCII and parenthesised-title
cases at once. The comment above the list records this so it is not repeated.

### 15.3 The DBpedia mapper, and the one thing it refuses to do

`pipeline/extractors/dbpedia_mapper.py`. The crawler had been fetching DBpedia
since the crawl stage was written and nothing had ever mapped it, so the
`dbpedia` source_type, the `dbpedia_rdf` extraction_method and priority 35 in
`SOURCE_PRIORITY` were all reserved for a source that never arrived.

It contributes what DBpedia is good for: typed dates, numeric coordinates, the
conflict it was part of, a readable place name, and the result string. Dates
carry the same BC problem as Wikidata and are emitted as Postgres literals --
DBpedia writes 31 BC as `-031-09-02`, **three year digits, not four**, so a
four-digit ISO assumption would have silently rejected every ancient date.

**It returns `sides=[]`, deliberately.** Decided by inspecting the live
endpoint rather than by reading the ontology, and the evidence is worth
keeping:

- Actium's `dbo:combatant` is `["Ptolemaic Egypt", "Octavian's forces",
  "Antony's forces"]` -- three entries for two sides, in no order.
- Trafalgar's is `["Spain"]`, alone, for a battle where a British fleet fought
  a Franco-Spanish one. Its `dbo:commander` lists Villeneuve, Collingwood,
  Gravina and Nelson in one flat list with no marker of which fleet anyone was
  in.

There are **no numbered predicates**: DBpedia flattens the infobox's
`combatant1`/`combatant2` and `commander1`/`commander2`, losing the only thing
that assigned them. Splitting that back into sides would be invention, and a
commander on the wrong side is precisely the error the model cannot see -- it
fits one skill parameter to a career including battles the general fought
*against*. So belligerents, commanders, strengths and casualties go to `notes`
marked "unassigned to sides", exactly as the Wikidata mapper does with `P710`.
The infobox parser reads the numbered fields from wikitext directly and keeps
the assignment; that remains the only source sides come from.

Eleven unit tests, against a fixture captured from the live endpoint and
trimmed to the one resource the mapper reads.

### 15.4 The orchestrator could not reach the database

`main()` called `run_pipeline` without `db_conn`, so it defaulted to None and
**every SQL gate in every stage reported "No database connection available to
run this check"**. The gates were made real in §11.4 and the CLI had never been
wired to them, so a real pipeline run could not tell a passing stage from a
failing one. TODO.md filed this under Tooling; it is infrastructure, and it
blocked any real run.

`main()` now loads `.env`, configures logging, opens a connection and passes it
down. `--no-db` keeps the old behaviour explicitly and says in its help that
every SQL gate will then fail. A missing DATABASE_URL exits 2 immediately
rather than running nine stages and reporting nine identical gate failures.

Verified: `python -m pipeline.orchestrator --stages extract` now returns real
verdicts -- "Check returned NULL. This usually means the table is empty" --
which is the correct answer for an empty database, and a different answer from
the one it gave before.

### 15.5 `.env` was never read, and could not have been

TODO.md recorded "nothing actually loads .env". `pipeline.config.load_env()`
now does, and an already-exported variable wins by default, so a deliberate
export or a CI-injected secret is never overridden by a file in the tree.

**Running it immediately found that the `.env` on this machine is UTF-16-LE
with a BOM** -- the PowerShell redirect footgun. `python-dotenv` assumes UTF-8
and raised `UnicodeDecodeError` on the very first byte, so even once something
read the file, it read nothing. The loader now sniffs the BOM.

The first fix was wrong in an instructive way: decoding as `utf-16-le` keeps
the BOM as a character, producing a variable named `﻿GEMIINI_API_KEY`
that nothing looks up. Plain `utf-16` auto-detects endianness and strips it.
Both the crash and the invisible-BOM variant are pinned by tests.

An unreadable `.env` now degrades to a warning rather than killing the run: the
variables may well be exported already, and a traceback from inside dotenv says
nothing about which file or why.

**Two findings about the actual `.env`, left for the user:**

1. The key in it is spelled **`GEMIINI_API_KEY`** -- two I's. Nothing reads
   that name. The working Gemini calls in §13.3 came from a variable exported
   in the shell, not from this file.
2. It contains that one variable and nothing else: no `DATABASE_URL`, no
   `ANTHROPIC_API_KEY`.

It was deliberately not edited. It is a credentials file, and rewriting one
unasked is not a thing to do quietly.

### 15.6 structlog was never configured, and CI now needs a database

`pipeline/logging_config.py`. Every module calls `structlog.get_logger()` and
nothing had ever called `structlog.configure()`. That is not an error --
structlog falls back to a default -- which is the problem: the format, the
level and the filtering were accidents rather than decisions. Console renderer
on a terminal, JSON otherwise, `LOG_LEVEL` honoured, and httpx and
sqlalchemy.engine held at WARNING so a crawl is not one of our lines per twenty
of theirs. Named `logging_config` so it does not shadow the stdlib.

`.github/workflows/ci.yml` runs three jobs: lint plus `mypy --strict`; the
suite against a **real `postgres:15` service**; and an Alembic
upgrade/downgrade/upgrade round-trip. The test job asserts the database is
reachable *before* pytest, and then **fails the build if anything skipped** --
because this project's integration tests skip silently without a database, and
a green run full of skips is the exact blind spot that hid §4.3 and §4.4. The
`live` marker is excluded, so CI never touches Wikipedia.

`.pre-commit-config.yaml` runs ruff and mypy and refuses to commit a `.env`.
`ruff-format` is registered but staged `manual`: applying it would reformat the
whole tree, which belongs in its own commit rather than mixed into behaviour
changes.

`README.md` was a two-line stub and is now a real front page, written for
someone arriving at the repository cold. It states plainly that the statistical
half does not exist, that there is no published ranking, and lists the
limitations that bear on believing one.

#### A test-pollution bug worth remembering

The logging tests failed three *unrelated* tests in `test_quality.py` and
`test_resolve.py`, and only when the whole suite ran -- each module passed
alone. `configure()` is process-global, so a test that set the level to ERROR
silenced every later test asserting on captured output. An autouse fixture now
snapshots and restores `structlog.get_config()`. **Any future test that calls
`configure()` needs the same fixture.**

### 15.7 What is still open, and why no code closes it

**The Gemini key is free tier, 20 requests/day (§14.5).** The user chose to
document this rather than work around it. `tests/integration/test_llm_live.py`
is written and wired -- five tests covering a real call, the cache serving an
identical second call for free, a refusal reported as a refusal, a truncation
detected, and the `llm_calls` audit row -- gated on `GENERAL_WAR_LIVE_LLM=1`.
**It has never been run.** The refusal and truncation branches remain
unverified, and they are the two worth paying for: either one mistaken for a
success writes wrong data with a successful status beside it.

To close it: a paid tier, or route the bulk stages to Anthropic.

**Gate thresholds are still unmeasured.** The 22 SQL gates execute correctly
and have mostly run against an empty database. Their SQL is proven valid; their
thresholds are not. That needs a corpus, which needs a crawl, which needs the
LLM quota above.

### 15.8 Changed files

| File | Change |
|---|---|
| `pipeline/crawlers/fetcher.py` | `retry_after_seconds()`, `MAX_SERVER_BACKOFF_S`, Retry-After honoured in the retry loop |
| `agents/crawl.yaml` | `rate_limit_wikipedia` 1.0 to 2.0 |
| `pipeline/extractors/dbpedia_mapper.py` | **New.** The DBpedia RDF mapper |
| `pipeline/extractors/__init__.py` | Export `map_resource`, `dbpedia_date_to_literal` |
| `pipeline/stages/extract.py` | Discover `data/raw/dbpedia/`, map it, merge it |
| `pipeline/logging_config.py` | **New.** structlog configuration |
| `pipeline/config.py` | `load_env()` with BOM sniffing and graceful failure |
| `pipeline/orchestrator.py` | `main()` opens a real connection; `--no-db`, `--log-level` |
| `pyproject.toml` | `python-dotenv`, `pytest-cov`, `pre-commit`, `types-PyYAML`; `live` marker |
| `README.md` | Written |
| `.github/workflows/ci.yml` | **New.** Lint, tests against postgres:15, Alembic round-trip |
| `.pre-commit-config.yaml` | **New.** |
| `tests/unit/test_crawl.py` | 4 Retry-After tests |
| `tests/unit/test_extract.py` | 13 DBpedia tests: 11 mapper, 2 pinning the stage wiring |
| `tests/unit/test_infrastructure.py` | **New.** 14 tests, env loading and logging |
| `tests/integration/test_crawl_live.py` | **New.** 5 live Wikipedia tests, opt-in |
| `tests/integration/test_llm_live.py` | **New.** 5 live LLM tests, opt-in, never run |
| `tests/fixtures/extract/dbpedia_actium.json` | **New.** Captured from the live endpoint |
| `handover.md`, `TODO.md` | This |

### 15.9 Verification, stated honestly

**Verified against live PostgreSQL 15.19:** `ruff` clean, `mypy --strict` clean
across 49 files, **410 passed and 10 skipped**. The ten skips are the opt-in
live tests and are correct; a count above ten means DATABASE_URL is unset.

**Verified against live services:** ten Wikipedia articles fetched and parsed
at the new rate limit with zero retries; robots.txt read; three DBpedia
resources mapped, including two BC dates converting correctly
(`0031-09-02 BC`, `0216-08-02 BC`).

**Verified by running it:** the orchestrator CLI reaching the database and
reporting real gate verdicts, and `.env` loading from the genuine UTF-16 file
on this machine.

**Not verified:** every live LLM branch, for the quota reason in §15.7. The CI
workflow has not run -- there is no remote, and it cannot be proven correct
from here beyond its YAML parsing and its commands matching the ones that work
locally. `pre-commit install` has not been run either.

Suggested commit message:

```
feat: close out setup, infrastructure and stages 1-2

Honour Retry-After in the crawl fetcher and raise the Wikipedia rate limit to
2.0s. Blind exponential backoff answered "retry in 29s" with two seconds and
burned every attempt inside a window the limit could not lift in, the same
defect already fixed on the LLM side.

Add the DBpedia RDF mapper, which the crawler had been feeding since the crawl
stage was written with nothing to read it. It returns no sides: DBpedia
flattens the infobox's numbered combatant and commander fields into unordered
lists, so Trafalgar reports one combatant and four commanders with no marker of
which fleet anyone was in. Assigning those to sides would be invention.

Wire the orchestrator CLI to a real database connection. Every SQL quality gate
reported "No database connection available" from the CLI, so a real run could
not tell a passing stage from a failing one.

Read .env, which nothing had ever done, sniffing the BOM: the file on the
development machine is UTF-16 and python-dotenv failed on its first byte.

Configure structlog, which nothing had ever called, and add CI that runs the
suite against a real PostgreSQL and fails if any test skips.

Add opt-in live tests against Wikipedia and against an LLM provider. The crawl
ones pass; the LLM ones are unrun, blocked on a free-tier quota.
```

---
