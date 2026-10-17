# Handover log §13. Live verification of the resolve stage, 2026-09-19

Part of the session log; the index is `docs/handover-log/README.md`.

The three gaps §12.5 listed were closed by pointing the stage at the real
services. The first check found a defect that reached the whole project.

### 13.1 Live Wikidata SPARQL: works now, did not before

See §4.7 for the defect: robots.txt disallows `/sparql`, `run_query` went
through the robots gate, and **every SPARQL query in the project silently
returned nothing** -- in crawl as well as resolve. Fixed by routing queries
through `fetch_raw`.

After the fix the query works: five names in one batch, about four seconds,
twelve candidates. The results are also the clearest possible argument for the
date gate.

| Mention | What the live query returns |
|---|---|
| "Napoleon" | Q517, **plus an American rapper (b. 1977), an Indian film actor (b. 1963), and a dateless NFT founder** |
| "Scipio Africanus" | Q2253 the general, **plus an 18th-century enslaved man of that name** |
| "Yi Sun-sin" | Q50184 the admiral **and Q12611867, a different Joseon officer of the same name, both alive in 1597** |

Name similarity alone would link any of these. The lifespan gate removes the
first two rows; the third is a genuine ambiguity and is what the LLM step is
for.

### 13.2 A second matcher fix the live data forced

Against real candidates, five of eight commanders were being deferred to the
LLM -- against a spec that expects "typically <5%".

The cause: **the date gate abstains on a dateless candidate, so every dateless
namesake survives it**, and Wikidata has many. Napoleon at Austerlitz was
going to the model because a dateless NFT founder matched his label exactly.

`match_group` now prefers, among *exact* matches, the candidates whose dates
positively confirm them. It concedes nothing: a candidate the dates contradict
was already dropped, and two confirmed candidates still go to the model. It
only breaks the tie between evidence and no evidence. Deterministic links went
from 2/8 to 4/8, with no new errors.

### 13.3 A real LLM call, and the cache

One live `gemini-3.8-flash` call on the genuine Yi Sun-sin ambiguity:

- picked **Q50184**, the correct admiral, and its reasoning named the thirteen
  ships and the Imjin War
- 363 in / 294 out, $0.0014, `llm_calls` row written with status `ok`
- **a second identical call was served from the cache**: zero tokens, zero
  cost. The stage is resumable without paying twice, which §3.1 claimed and
  nothing had demonstrated.

The credential's prefix is `AQ.A...`, which §3.1 previously said was the wrong
kind of key. It is not; that claim has been corrected.

### 13.4 The thresholds, measured for the first time

Eight commanders whose answer is not in doubt, run through the full stage path
including the LLM step. Not a gold set -- a spot check, and the smallest thing
that can say whether the defaults are roughly right.

| | |
|---|---|
| linked correctly | **5** (4 deterministic, 1 via the model) |
| **linked wrongly** | **0** |
| fell through to a new entity | 3 |
| LLM calls | 3, $0.0047 total |

**Zero wrong links is the result that matters**, because a false merge is the
error nothing downstream can detect. The split-preferring design holds up
against real data.

The three misses are worth separating, because only one is the matcher's:

1. **Horatio Nelson** -- the query never returned Q102462 at all. His label is
   "Horatio Nelson, 1st Viscount Nelson"; an exact literal match on the
   mention "Horatio Nelson" does not find it, and his English aliases did not
   cover the bare form either.
2. **Duke of Wellington** -- zero candidates. `name_keys` generates the key
   `wellington`, but the *query* only ever asks about raw surface forms, so
   the stripped form is never sent. The matcher is cleverer than the lookup.
3. **Hannibal** -- the right entity *was* offered and the model chose
   NEW_ENTITY. A conservative miss, in the direction the design prefers. Note
   the harness gave it thin context (`side_label="side"`, no role evidence)
   where a real mention carries the polity and the article's own words, so
   this is not a clean measurement of the prompt.

**The next piece of work on this stage is (1) and (2), which are one bug:**
the query asks about surface forms while the matcher reasons over generated
keys. Sending title-stripped and core-name variants -- case preserved, since
Wikidata literals are case-sensitive -- would likely recover both. A label
search, or `wbsearchentities`, would recover more. Until then, expect
commanders known by a title ("Duke of Wellington") or whose Wikidata label is
longer than their common name to resolve as corpus-local entities.

### 13.5 What this cost, and what it bought

Three live LLM calls (under a cent) and roughly twenty HTTP requests, against
a project whose statistical half does not exist yet. It found a defect that
had silently disabled a whole data source in two stages, corrected a false
claim about credentials in this file, and turned "the thresholds are reasoned"
into a number.

That is the same trade §4.2 recorded and it came out the same way. **Point the
next stage at something real before believing its tests.**
