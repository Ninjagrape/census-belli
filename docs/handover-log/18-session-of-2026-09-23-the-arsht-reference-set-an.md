# Handover log §18. Session of 2026-09-23: the Arsht reference set and a curated corpus

Part of the session log; the index is `docs/handover-log/README.md`.

The question that started this session was how early a visualisable output is
possible. The answer, and the work that followed, both hinge on having a corpus
small enough to run and honest enough to check. This session built one.

### 18.1 What was built

| File | What it is |
|---|---|
| `scripts/arsht_reference.py` | harvester; decodes Arsht's published per-battle WAR, pulls his own infobox parse, writes the seed |
| `tests/fixtures/gold/arsht/arsht_war.jsonl` | 226 rows, one per (general, battle), with his WAR and running total |
| `tests/fixtures/gold/arsht/battles.jsonl` | 212 distinct battles, every title resolved against the live MediaWiki API, dated from Wikidata (§18.8) |
| `tests/fixtures/gold/arsht/career_totals.jsonl` | his career WAR for each of the 22 roster generals |
| `tests/fixtures/gold/arsht/arsht_infobox.jsonl` | 1,043 battle-commander rows from his own scrape: raw infobox strength text plus the numbers he parsed from it |
| `config/sources_seed_arsht.yaml` | 211 canonical article URLs: the pilot corpus |

Rebuild the lot with:

```bash
python -m scripts.arsht_reference --verify-titles --with-infobox --write-seed
```

The roster is 22 generals: the ten the article names, with the WAR scores it
prints, plus twelve chosen for the failure modes this file already records --
Yi Sun-sin and Themistocles for the naval `battle_type` bug (§4.2), Scipio and
Hannibal for BC dates (§5.1), Augustus and Agrippa for the Actium attribution
problem, Grant opposite Lee so relative skill is testable on a known pair.

### 18.2 This is a reference set, not a gold set

Say it in those words, because the distinction is the whole value. His numbers
are one model's output over one corpus, and this project exists because both
have problems. **Do not assert his values as facts.** The set is for three
things: a battle roster with real coverage, a correlation target for the model
stage, and a source of corpus-hygiene findings.

One value in it *is* checkable against an independent published source, and it
is asserted: Napoleon's career total decodes to 16.678282 against the 16.679
the article prints. That is the proof the harvest reads the columns it thinks
it reads. Every other general on his pages differs from the article's quoted
figure by up to 0.4 -- Rommel is -1.531 here against -1.953 in the text -- so
the published pages come from a later run than the article. Treat the article's
numbers and the page numbers as two runs, not one.

### 18.3 Three findings the reference set produced immediately

**His missing-year sentinel is -5000, and it is not rare.** 59 of 226 rows
carry it, covering *every* battle of seven generals: Caesar, Alexander,
Augustus, Agrippa, Themistocles, Belisarius, Khalid ibn al-Walid. Read as a
number it dates Khalid, who died in 642 AD, to five millennia BC. The harvester
records it as a null year with `year_missing: true` and keeps the raw value.
All seven are ancient or early medieval, so his era covariate was missing for
exactly the commanders whose era matters most. There is a unit test pinning
this.

**38 of 212 titles redirect**, and several of the redirects are unit errors
rather than spelling. `Battle of Gallipoli` and `Third Battle of Chattanooga`
resolve to *campaign* articles; `Battle of Perugia` resolves to `Perusine War`;
`Battle of Vicksburg` resolves to `Siege of Vicksburg`, which is a
`battle_type` this project models separately. A campaign entered as a battle is
precisely the scope conflation reconcile's troop numbers cannot survive.

**`First Battle of Philippi` and `Second Battle of Philippi` are the same
article.** His corpus counted one battle twice, once for Augustus and once for
Agrippa. The seed writer collapses them to one URL, which is why the corpus is
211 battles and not 212.

### 18.4 `arsht_infobox.jsonl` is the one thing extract can be diffed against

His `final_vd_fill.csv` covers 211 of the 212 roster battles with 1,043
battle-commander rows, carrying the **raw `strength` text from each side of the
infobox** next to the numbers he parsed out of it. Same source article, same
field, a different parser. Nothing else available gives extract a comparison
that does not require hand-labelling first.

His `Ships` column deserves particular attention. It is the only
machine-readable naval signal in the set, and §4.2 is this project having
labelled every naval battle a land engagement once already.

His CSVs carry mojibake (`GuantÃ¡namo`), repaired on the way in where the
round trip is exact and left alone where it is not.

### 18.5 Crawl now accepts a named corpus

`Seeds` gained `battle_articles`, fed by a new optional
`wikipedia_battle_articles:` key. Named articles are seeded into the crawl
state before any list page is parsed, so a run configured with articles and no
list pages has work to do, and a limited run spends its budget on the battles
it was asked for.

The change is additive: `wikipedia_battle_lists: []` on its own still raises,
as it did. The rule is now "at least one entry point", not "at least one list".

This matters beyond convenience. An evaluation corpus whose membership changes
because somebody edited a Wikipedia list page is not an evaluation corpus.
`config/sources_seed_arsht.yaml` therefore names no list pages and no SPARQL
queries at all.

### 18.6 Verification, stated honestly

`ruff` clean across `pipeline/ tests/ scripts/ alembic/`. `mypy --strict` clean
on 50 source files, and separately on `scripts/arsht_reference.py`.
**450 passed, 15 skipped** with `DATABASE_URL` exported -- up from 423, with 5
new crawl tests and 14 new harvester tests. The 15 skips are the expected
opt-in live tests.

Verified against something real: the harvest itself ran against the live GitHub
repository and the live MediaWiki API, and all 212 titles resolved. Napoleon's
total matches the published article to three decimal places.

**Not verified:** no battle in this corpus has been crawled, extracted or
resolved yet. The seed file loads and its 211 URLs are canonical and
title-verified, but the crawl has not been pointed at it. `data/processed/` is
still empty and the database still has 0 rows in every table.

**Not done, and the next real work:** there is still no hand-labelled gold set
saying what *we* expect extract, resolve and classify to produce for these
battles. `arsht_infobox.jsonl` gives a diff target, not a truth. The 12-15
battle subset worth hand-labelling is the one covering Actium, Trafalgar or
Noryang, Cannae, Zama, Alesia and Leipzig.

### 18.7 Changed files

- `scripts/arsht_reference.py` (new)
- `tests/unit/test_arsht_reference.py` (new, 14 tests)
- `tests/fixtures/gold/arsht/` (new: 4 data files plus `MANIFEST.md`)
- `config/sources_seed_arsht.yaml` (new)
- `pipeline/crawlers/seeds.py` (`battle_articles`, `_url_list`)
- `pipeline/stages/crawl.py` (seeds named articles; `_describe_plan` reports them)
- `tests/unit/test_crawl.py` (5 new tests)

### 18.8 Dates: the gap the user spotted, and the worse bug behind it

The user noticed Granicus sitting in `config/sources_seed_arsht.yaml` as "year
unknown" when it is plainly 334 BC. That was Arsht's `-5000` sentinel showing
through, faithfully. Fixing it turned up something more dangerous.

**The fix.** Each battle's Wikidata item is now reached through the article's
sitelink, and its date read from P585 (point in time), falling back to P580
(start time). 207 of 212 battles are dated from Wikidata, 4 keep Arsht's year,
**one is left undated** (`Siege of Gaza`, Q123406360, whose item carries no
date). The Q-ids are kept: resolve needs exactly these, and they came from the
sitelink rather than a name match.

**The bug behind it: two year conventions, silently one apart.**

| Source | Actium, 31 BC | Convention |
|---|---|---|
| Wikidata | `-0031` | historical |
| Arsht | `-31` | historical |
| `battles.year_astronomical` (§5.1) | **`-30`** | **astronomical** |

Every record now carries `year` (historical, as both sources write it) *and*
`year_astronomical` (this project's convention), plus `year_label` for humans
and `year_source` for provenance. **Join on `year_astronomical`.** Had the
original historical years been loaded against that column, all 60 BC battles in
the corpus would have sat one year off, silently, and the era covariate would
have been wrong for the ancient half of the set. This is §5.1 biting from a new
direction: the column was right, and the data arriving at it was not.

**A second bug, found by disagreement.** Two battles disagreed with Arsht's
year. One was mine: the 1948 Arab-Israeli War carries P585 = `+1940-00-00`
**at precision 8, a decade**, and I read the padding digits as a year 1940.
Wikidata's precision scale runs 11=day, 10=month, 9=year, 8=decade, 7=century;
anything below 9 is now rejected and the code falls through to P580, which
dates that item to the day. Same class of error as reading `-5000` as a year:
a non-year in a year-shaped field.

The one surviving disagreement is Suez Crisis, Arsht 1956 against Wikidata
1957, and it is recorded in the manifest rather than resolved. The crisis ran
from October 1956 into March 1957, and it is a campaign in a battle corpus
anyway, which §18.3 already flags.

Statement ranks are now honoured too: deprecated claims are skipped and
preferred ones win, rather than taking whichever statement happened to be
first.

`scripts/arsht_reference.py` also gained the retry-with-backoff the project's
conventions require. Wikidata answered 429 to a 50-id batch and lost a whole
run; entity reads now go 25 at a time and honour `Retry-After`.

Verified: ruff clean, `mypy --strict` clean on 51 files, **465 passed / 15
skipped** with `DATABASE_URL` set, up from 450. Granicus now reads
`# 334 BC; Alexander the Great` in the seed file, and a test pins it along with
the astronomical conversion for Granicus, Actium and the 1 BC/AD boundary.

### 18.9 Four disambiguation pages in the corpus, found by a spot-check

The user checked the one battle §18.8 left undated -- `Siege of Gaza` -- and
said it should be 332 BC. It was not a missing date. **The title resolves to a
disambiguation page**, and today that page offers Alexander's siege alongside
three 21st-century ones.

Sweeping all 211 titles for `pageprops.disambiguation` found **four**:

| Arsht's title | Article actually meant | General, year |
|---|---|---|
| Siege of Gaza | Siege of Gaza (332 BC) | Alexander, 332 BC |
| Battle of Arras | Battle of Arras (1940) | Rommel, 1940 |
| Battle of Bautzen | Battle of Bautzen (1813) | Napoleon, 1813 |
| Battle of Burkersdorf | Battle of Burkersdorf (1762) | Frederick, 1762 |

**Why every check passed them.** A disambiguation page exists, returns 200, and
does not redirect. Title verification (§18.5) asked only those three questions,
so all four sailed through. They have no infobox, no commanders and no date, so
crawled they would have produced four battles with nothing on either side, and
the only visible symptom would have been a slightly smaller corpus.

Arras is the one that would have hurt most quietly: the disambiguation page
lists battles at Arras in 1640, 1654, 1914, 1915, 1917 and 1918, so a
plausible-looking automatic pick could have credited Rommel with a First World
War engagement fought before he commanded anything.

**The fix, in three parts.**

1. `resolve_titles` now asks for `pageprops` in the same request it already
   makes and flags `is_disambiguation`. No extra API call.
2. `_DISAMBIGUATION_TARGETS` maps the four titles to the article meant, chosen
   by hand from each disambiguation page using the general and year already on
   the row, with the reason written beside each entry. **Deliberately a table,
   not a heuristic**: picking the nearest year would be right these four times
   and wrong silently somewhere else.
3. `write_seed_config` refuses any record still flagged, printing what it
   dropped. If Wikipedia turns another title into a disambiguation page, it
   leaves the corpus rather than entering it unnoticed.

**Result: every battle in the corpus now carries a year.** 211 from Wikidata,
1 from Arsht, none undated, where the set started with 59 rows on a `-5000`
sentinel.

Verified: ruff clean, `mypy --strict` clean on 51 files, **472 passed / 15
skipped** with `DATABASE_URL` set. Seven new tests, including one per
repointed title and one asserting no disambiguation page reaches the corpus.

The general lesson, which is the same one §4.2 taught: *exists* is not
*correct*. The crawl gate counts URLs that returned 200, and all four of these
would have counted.
