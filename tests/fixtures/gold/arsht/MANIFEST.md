# Arsht Reference Set

Decoded from the published output of Ethan Arsht's 2018 analysis,
`ethanarsht/military_rankings` on GitHub. Each general's page there embeds a
Bokeh `ColumnDataSource` carrying his battle list, per-battle WAR, running
cumulative WAR, outcome and year; this set is those columns, decoded.

## What this is not

**Not a gold set. Not ground truth.** These are one model's outputs over one
corpus, and this project exists because that corpus and that model have known
problems. Treat a disagreement as a question, not a failure.

Specifically, do not assert against these values in a unit test as though they
were facts. They are useful as:

1. a roster of battles and commanders with real coverage of hard cases
2. a correlation target for the model stage -- our WAR should track his and
   diverge in ways we can name
3. a source of corpus-hygiene findings: where his battle titles redirect to
   campaigns, his corpus counted a campaign as a battle

## Provenance

- Built: 2026-09-23
- Source: https://github.com/ethanarsht/military_rankings (per-general HTML pages)
- Article: https://towardsdatascience.com/napoleon-was-the-best-general-ever-and-the-math-proves-it-86efed303eeb/
- Rebuild with: `python -m scripts.arsht_reference --verify-titles`
- Reviewed by: **not yet reviewed by a human**

## Contents

| File | Rows | What it holds |
|---|---|---|
| `arsht_war.jsonl` | 226 | one row per (general, battle) with his WAR |
| `battles.jsonl` | 212 | deduplicated battles with Wikipedia URLs |
| `career_totals.jsonl` | 22 | his career WAR per general |

## Roster, and why each name is in it

| General | Battles | Arsht WAR | Why in the set |
|---|---|---|---|
| Napoleon | 43 | 16.678282 | article's headline result, WAR 16.679, 43 battles |
| Julius Caesar | 17 | 7.352571 | article's 2nd place, WAR 7.445 |
| Hannibal | 17 | 5.481862 | article's 6th place, WAR 5.519; Cannae is its worked example |
| Alexander the Great | 9 | 4.374161 | article's 10th place, WAR 4.391 |
| Robert E. Lee | 27 | -2.013387 | article's negative-WAR case, -1.89; 2nd most battles at 27 |
| Erwin Rommel | 9 | -1.531403 | article's negative-WAR case, -1.953 |
| George S. Patton | 2 | 0.998497 | article quotes WAR 0.9 |
| Pyrrhus of Epirus | 3 | -0.535262 | article quotes WAR -0.53; the eponymous victory problem |
| Moshe Dayan | 4 | 2.086226 | article quotes WAR 2.109, 60th; modern combined-arms |
| Ariel Sharon | 7 | 2.14132 | article quotes WAR 2.171, 58th |
| Yi Sun-sin | 5 | 2.703019 | naval-only career; battle_type must not default to field (§4.2) |
| Themistocles | 3 | 1.142147 | naval, BC dates, and a genuinely ambiguous Wikidata match |
| Scipio Africanus | 5 | 2.494654 | BC dates; Zama pairs him against Hannibal in one battle |
| Augustus | 7 | 3.378265 | Actium: nominal command with Agrippa commanding tactically |
| Marcus Vipsanius Agrippa | 4 | 2.006413 | the other half of the Octavian/Agrippa problem |
| Trajan | 3 | 1.447453 | BC/AD boundary era, small battle count, ancient troop inflation |
| Ulysses S. Grant | 16 | 5.021479 | Lee's opponent; the pair tests relative skill directly |
| Georgy Zhukov | 10 | 4.594388 | Rommel's era; vast troop numbers, Soviet source disagreement |
| Khalid ibn al-Walid | 14 | 5.626761 | non-European corpus; transliterated name variants |
| Subutai | 2 | 1.007151 | Mongol corpus; commanded under Genghis Khan, so hierarchy matters |
| Frederick the Great | 14 | 4.652316 | 18th century; head of state and field commander at once |
| Belisarius | 5 | 1.619888 | Byzantine; frequently outnumbered, so force ratio matters |

## Missing years in the source

59 of 226 rows carry Arsht's `-5000` missing-year sentinel, written here as a null year with `year_missing: true`.

It is not scattered noise. It covers every battle of these generals:

- Alexander the Great
- Augustus
- Belisarius
- Julius Caesar
- Khalid ibn al-Walid
- Marcus Vipsanius Agrippa
- Themistocles

All of them ancient or early medieval. Anything that read that column as a number would date Khalid ibn al-Walid, who died in 642 AD, to 5000 BC, and an era covariate built from it would be wrong for exactly the commanders whose era matters most. This project's own `missing_data_log` exists for this.


## Dates

Arsht's own years are kept as `arsht_year`, but the authoritative year comes
from the battle's Wikidata item (P585 point in time, falling back to P580
start time), reached through the article's sitelink. That fills every gap his
`-5000` sentinel left, including the whole of Alexander's career.

- From Wikidata: 211
- From Arsht, no Wikidata date: 1
- Still undated: 0

### Two conventions, and why `year_astronomical` exists

Wikidata and Arsht both write **historical** years: 31 BC is `-31`. This
project's `battles.year_astronomical` column writes **astronomical** years,
where a year zero exists and 31 BC is `-30`. Every record therefore carries
both. Join on `year_astronomical`; every BC battle in the corpus would
otherwise sit one year off the column it is compared against.

Statements whose Wikidata precision is coarser than a year are rejected rather
than read. The 1948 Arab-Israeli War carries P585 = `+1940-00-00` at decade
precision, and taken as a year that says 1940.

### Where the two sources disagree

| Battle | Arsht | Wikidata |
|---|---|---|
| Suez Crisis | 1956 | 1957 |

### Titles that landed on a disambiguation page

These exist and do not redirect, so title verification alone passes them,
but they carry no infobox, no commanders and no date. Each was repointed by
hand using the general and year on the row; the mapping is in
`_DISAMBIGUATION_TARGETS` in the harvester.

| Arsht's title | Article actually meant |
|---|---|
| Battle of Bautzen | Battle of Bautzen (1813) |
| Siege of Gaza | Siege of Gaza (332 BC) |
| Battle of Arras | Battle of Arras (1940) |
| Battle of Burkersdorf | Battle of Burkersdorf (1762) |


## Title verification

- Resolved: 212 of 212
- Redirected to a different title: 42
- Not found on Wikipedia: 0

Redirects matter. A battle name that redirects to a campaign article is a
campaign his corpus counted as a battle, which is exactly the kind of unit
error the reconcile stage's troop numbers cannot survive.

### Redirected

| Queried | Resolves to |
|---|---|
| Battle of 2nd Bull Run | Second Battle of Bull Run |
| Battle of Abu-Ageila (1967) | Battle of Abu-Ageila |
| Battle of Akraba | Battle of al-Yamama |
| Battle of Alam Halfa | Battle of Alam el Halfa |
| Battle of Appomattox Courthouse | Battle of Appomattox Court House |
| Battle of Arras | Battle of Arras (1940) |
| Battle of Aspern-Essling | Battle of Aspern–Essling |
| Battle of Bagbrades | Battle of the Great Plains |
| Battle of Bautzen | Battle of Bautzen (1813) |
| Battle of Burkersdorf | Battle of Burkersdorf (1762) |
| Battle of Busan (1592) | Battle of Busan |
| Battle of Cartagena (207 BC) | Battle of New Carthage |
| Battle of Champion's Hill | Battle of Champion Hill |
| Battle of Crotona | Battles of Kroton |
| Battle of Herdonia | Battle of Herdonia (212 BC) |
| Battle of Iron bridge | Battle of the Iron Bridge |
| Battle of Kalka River | Battle of the Kalka River |
| Battle of Khalkhin Gol | Battles of Khalkhin Gol |
| Battle of Marj-ud-Deebaj | Battle of Marj al-Dibaj |
| Battle of Noryang Point | Battle of Noryang |
| Battle of Perugia | Perusine War |
| Battle of Petersburg III | Third Battle of Petersburg |
| Battle of Prague (1757) | Battle of Štěrboholy |
| Battle of Rappahannock Station II | Second Battle of Rappahannock Station |
| Battle of Rhone Crossing | Battle of the Rhône Crossing |
| Battle of The Chinese Farm | Battle of the Chinese Farm |
| Battle of Vicksburg | Siege of Vicksburg |
| Battle of Yarmouk | Battle of the Yarmuk |
| Battle of Zela | Battle of Zela (47 BC) |
| Battle of the Bridge of Arcole | Battle of Arcole |
| Battle of the Kasserine Pass | Battle of Kasserine Pass |
| Battle of the Ticinus | Battle of Ticinus |
| First Battle of Philippi | Battle of Philippi |
| Kamenets-Podolsky pocket | Kamenets–Podolsky pocket |
| Korsun Pocket | Battle of Korsun–Cherkassy |
| Second Battle of Herdonia | Battle of Herdonia (210 BC) |
| Second Battle of Philippi | Battle of Philippi |
| Siege of Gaza | Siege of Gaza (332 BC) |
| Siege of Rome (537-538) | Siege of Rome (537–538) |
| Siege of Toulon | Siege of Toulon (1793) |
| Six Day War | Six-Day War |
| Third Battle of Chattanooga | Chattanooga campaign |

## How each stage uses this

| Stage | Use |
|---|---|
| crawl | `battles.jsonl` URLs are the pilot corpus; every one must fetch 200 |
| extract | commanders parsed from each infobox must include the roster general |
| resolve | roster generals must link to one Wikidata id each, not several |
| classify | Actium must not attribute Agrippa's tactical command to Augustus |
| reconcile | ancient battles here are where source inflation shows up |
| model | our career WAR should correlate with `career_totals.jsonl` |

