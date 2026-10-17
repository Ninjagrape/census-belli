# Handover log §4. Bugs found, and the lesson

Part of the session log; the index is `docs/handover-log/README.md`.

Five systematic bugs have been found. **Not one was found by writing code.**
Every one surfaced from pointing the code at something real -- live
Wikipedia, or a live Postgres. 4.1 and 4.2 are from 2026-09-18; 4.3 to 4.5
are from 2026-09-19, the first session with a database.

### 4.1 The crawl gate could never pass

`agents/crawl.yaml`'s `min_battles_crawled` counted `DISTINCT battle_id` from `crawl_log`. But `battle_id` is a FK to `battles`, and `battles` has no rows until the **extract** stage. So during crawl it is necessarily NULL, the count is always 0, and as an ERROR gate with `pause_and_alert` it would have halted the pipeline after every crawl. Now counts `DISTINCT url`.

### 4.2 Every naval battle was labelled a land engagement

This is the important one.

`pipeline/extractors/infobox.py` mapped `{{Infobox military conflict}}` to a "land" variant that hardcoded `battle_type="field"`. But Wikipedia uses that generic template for **essentially every battle** — the naval-specific templates the code expected are rare. Live checks:

| Battle | Template | Extracted `battle_type` |
|---|---|---|
| Actium | Infobox military conflict | `field` |
| Trafalgar | Infobox military conflict | `field` |
| Midway | Infobox military conflict | `field` |

None carries a `ships=` or `vessels=` field to fall back on; Wikipedia puts ship counts in `strength` as free text.

`battle_type` is a covariate in `agents/model.yaml`, so the error fell precisely on the commanders whose ranking depends on it: Nelson, Yi Sun-sin, Themistocles, de Ruyter.

**This passed 73 unit tests.** The fixtures encoded the same wrong assumption as the code. It was caught only by running the parser against live Wikipedia.

> **Lesson for the next agent:** fixture-only testing validates that your code agrees with your assumptions. For a project whose entire output is a claim about the real world, periodically point the parsers at real data. It costs three HTTP requests.

Fixed: the generic template now leaves `battle_type` unset with a note, matching what the siege and rendered-HTML paths already did. **Consequence: nothing sets `battle_type` any more. `classify` must infer it** (from ship/fleet vocabulary, terrain, categories). Until then the covariate is always NULL.

### 4.3 `apply_schema()` could never apply the schema

`config/schema.sql` contains five `%` characters, all in comments like
`-- 95% CI lower bound`. psycopg reads `%` as a parameter placeholder, so the
whole file was rejected with `incomplete placeholder: '%'`.

The author had already anticipated the neighbouring problem -- `text()` chokes
on the `:` in casts -- and used `exec_driver_sql` to dodge it. But
`exec_driver_sql` still hands psycopg an empty parameter set, which is enough
to trigger placeholder parsing. Only a raw driver cursor called with no
parameter argument at all skips it.

**The schema had never once been applied by the code that exists to apply it.**

### 4.4 ...and then silently failed to commit it

With that fixed, `apply_schema(drop_existing=True)` still left an empty
database, and reported success.

The raw cursor's work is invisible to SQLAlchemy's transaction tracking. In
the `drop_existing` branch the DROP/CREATE is committed first, so by the time
the DDL runs there is no SQLAlchemy-level transaction open; `conn.commit()`
then finds nothing to commit, does nothing, and the DDL is rolled back on
close. The commit has to go to the driver connection that ran the statement.

That branch is what every integration fixture uses, so it affected all of
them. Both bugs are fixed in `pipeline/db.py`.

### 4.5 The BC-date test asserted a fact that was not true

`test_bc_date_is_stored_as_bc` asserted `EXTRACT(YEAR FROM date_start) == -30`
for 31 BC, commented "there is no year zero". Postgres returns **-31**:
`EXTRACT` negates the historical BC year, it does not use astronomical
numbering.

The comment's intent was right and the assertion was wrong, and the
distinction is not cosmetic. Only the astronomical form subtracts correctly
across the boundary -- 31 BC to 1 AD is 30.3 actual years, matching -30, not
-31. Left unexamined it would have put a silent off-by-one into every
date-derived covariate for the entire ancient corpus.

This is §4.2's pattern exactly: a fixture encoding an assumption, passing
because nothing had ever checked it against the thing it described.

### 4.6 Two more gates that could not measure what they claimed

Both found on 2026-09-19 while building resolve. Same family as §4.1, §11.4.

**`extract.commander_extraction_rate` counted a table extract does not
write.** §11.4 rewrote it from the non-existent `data.commanders_raw` to
`battle_commanders`, which does exist -- but `battle_commanders.general_id`
is NOT NULL, so the extract stage cannot write it and deliberately does not
(`pipeline/extractors/store.py` says so in its docstring: commander mentions
wait in `commanders_raw.jsonl` until resolve has canonical generals). The gate
therefore reported 0.0 for every extract run. It is now `sides_per_battle`,
measuring `battle_sides`, which extract does write; the commanders-per-battle
version moved to `agents/resolve.yaml` where it is true.

That is the third distinct rewrite of one gate. The pattern each time: the SQL
was written from the *spec's* idea of what the stage produces, and never run
against a database that had been through the stage.

**`resolve.resolution_rate` divided a number by itself.** As written it was:

```sql
SELECT COUNT(*) FILTER (WHERE general_id IS NOT NULL)::float / NULLIF(COUNT(*), 0)
FROM battle_commanders
```

`general_id` is NOT NULL. Every row counts in both the numerator and the
denominator, so the gate reads exactly 1.0 no matter how many commander
mentions the stage failed to resolve -- a permanently green light on the one
number that says whether the stage worked.

The fix needed a denominator that exists outside the table being measured.
Resolve now writes every mention it could not place to `missing_data_log`
under `field_name = 'commander_general_id'`, and the gate is written rows over
written rows plus logged failures. `test_the_resolution_rate_gate_can_actually_fail`
pins that it moves: one resolved mention and nine unresolved reads 0.1.

### 4.7 robots.txt silently disabled every SPARQL query in the project

Found on 2026-09-19 by pointing the resolve stage's candidate lookup at the
real endpoint for the first time. The sixth bug, and the sixth found by
contact with something real rather than by reading code.

`https://query.wikidata.org/robots.txt` is:

```
User-agent: *
Disallow: /sparql
Disallow: /bigdata
```

`Fetcher` honours robots by default, and `run_query` went through
`Fetcher.fetch`. So **every SPARQL query this project has ever issued returned
`robots_disallowed` without a request leaving the machine**, and both callers
degraded quietly:

- **crawl** -- `_crawl_wikidata` discovered no battles from Wikidata at all.
  The stage still reported success. §1 called crawl "verified against a
  database"; that was true of its article path and never true of this one.
- **resolve** -- every name would have found no candidate, every commander
  would have become a corpus-local entity, and the published ranking would
  have had no Wikidata identity in it anywhere.

The fix is one line in `pipeline/crawlers/wikidata.py`: queries go through
`fetch_raw`, the same robots exemption the code already makes for robots.txt
itself. robots.txt governs crawlers; WDQS is a documented public API whose own
policy asks for a descriptive user agent and considerate rates, and both are
still enforced -- the rate limiter, the retries and the user agent are
untouched. Only the crawler rule, which was never about API clients, no longer
applies.

**Two defences were added, because the silence was the real defect.**
`fetch_candidates` now raises `CandidateLookupError` when *every* batch fails,
rather than returning empty results: graceful degradation is for when part of
a system errors, and a total failure is a configuration problem that must stop
the run. And the stub fetcher in `tests/unit/test_resolve.py` implements only
`fetch_raw`, so a future change routing queries back through the robots gate
fails loudly instead of quietly returning nothing.

**If you add any other API call to this project, ask whether robots.txt
applies to it before wrapping it in `Fetcher.fetch`.**

---

---
