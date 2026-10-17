# Handover log §14. Chasing the misses from §13.4, 2026-09-19

Part of the session log; the index is `docs/handover-log/README.md`.

§13.4 blamed Nelson and Wellington on "the query asks about surface forms
while the matcher reasons over generated keys". **That diagnosis was wrong.**
Chasing it properly found four defects, one of them mine from an hour earlier,
and an operational blocker that no code change can fix.

### 14.1 The real bug: every alias match was thrown away

`fetch_candidates` asked the endpoint for names, then decided which requested
name each returned entity answered to by **re-deriving the match from the
entity's label**:

```python
if fold(name) in set(candidate_keys(candidate)):   # label + aliases
```

But `_merge_rows` never selected the aliases, so `candidate_keys` was the
label alone. The query matches on `rdfs:label` **OR** `skos:altLabel`, and
"Duke of Wellington" is an *alias* of "Arthur Wellesley, 1st Duke of
Wellington". So the query found him, and the very next loop discarded him.

Measured directly: the query returned 15 rows and 1 person for "Duke of
Wellington"; `fetch_candidates` returned 0 candidates.

Every commander whose common name is not their Wikidata label was being lost
this way -- which is most of the people known by a title, and a great many
ancients.

**Fix:** the query already binds `?name` in its VALUES block, so it now
*selects* it, and the endpoint says which name matched. No re-derivation. The
matched names also become the candidate's `aliases`, which gives the matcher
an exact-match key for a commander whose label is longer than the mention.
`test_an_entity_matched_by_alias_is_kept` pins it. Wellington now resolves.

### 14.2 The query was thirty seconds, not four

While measuring the above, an eight-name batch took 29.7s, and the eval timed
out twice at eight minutes.

The cause was the three OPTIONAL multi-valued joins: one person yields
`|countries| x |occupations| x |articles|` rows. Timed on one batch:

| Query | Time | Rows | People |
|---|---|---|---|
| as written | 29.7s | 90 | 35 |
| without `?name` | 31.2s | 90 | 35 |
| aggregated with GROUP_CONCAT | **1.9s** | 36 | 35 |

Note the second row: **projecting `?name` cost nothing.** The cross-product
was the whole expense. The query now GROUPs and concatenates, and returns the
same people fifteen times faster. At corpus scale that is the difference
between usable and not.

### 14.3 One LLM call could hang a stage forever

The Gemini client built its SDK client with no request timeout, so a stalled
call blocked indefinitely -- and the retry loop never ran, because a request
that never returns never raises. Fixed: `GeminiClient` takes `timeout_s`
(default 90) and passes `http_options`, falling back with a warning on an SDK
that will not accept it.

### 14.4 The retry ignored the rate limiter telling it when to come back

A 429 said *"Please retry in 29s"*. The backoff slept 2.5s, failed, slept
4.1s, failed, slept 8.9s, failed, and gave up -- four attempts spent on a
limit that was never going to lift inside ten seconds.

`TransientLLMError` now carries an optional `retry_after`, `with_backoff`
prefers it over the computed delay (capped at 120s), and the Gemini client
reads it from a `Retry-After` header or from Gemini's prose. Seven tests cover
it. **This is the same defect §5.2 records for Wikipedia**, which is still
open on the crawl side.

### 14.5 The blocker: the Gemini key is free tier, 20 requests a day

```
429: Rate limit exceeded for model gemini-3.8-flash
     (limit: 20 requests per day on Free Tier)
```

This session's handful of calls exhausted the day's quota. **The resolve stage
cannot run over a real corpus on this key**, and neither can extract, which is
one LLM call per article passage. No code change fixes it: it needs a paid
tier, or a different provider for the bulk stages.

Worth knowing before planning any real run, and worth remembering when the
next stage's cost is estimated.

### 14.6 A correction to §13.4, and a warning about Q-ids

§13.4 said the query never returned Nelson's Q102462. **Q102462 is not
Nelson** -- its label is "James Stewart". That id was asserted from memory.
The real Nelson is **Q83235** ("British admiral (1758-1805)"), and separately
he genuinely has no English label or alias equal to "Horatio Nelson", so the
exact-literal lookup still cannot reach him from that mention.

> **Corrected 2026-09-20 -- see §16.1.** The second half of that paragraph is
> wrong, and wrong in the way this file keeps warning about. Q83235 carries
> "Horatio Nelson" as its **`mul`** label; it has no `en` label at all. The
> lookup could not reach him because it asked only for `"..."@en`, not because
> the name was absent. The recommendation that followed from it -- a label
> search -- was therefore solving the wrong problem, and is now deferred
> (§16.6). Two language tags fixed it.

That is the second time in one session that a Q-id recalled rather than looked
up was wrong; Agrippa is Q48174, not the Q167846 an earlier fixture used.
**Look Q-ids up. Do not remember them.**

### 14.7 What is still open

- **Exact-literal lookup misses people whose common name is neither label nor
  alias** (Nelson). A label *search* -- `wbsearchentities`, or a SPARQL
  `CONTAINS` -- would reach them, at the cost of more candidates to filter.
  This is the remaining half of §13.4's recommendation, and it is real, unlike
  the half that turned out to be §14.1.
- **Hannibal** still resolves to a new entity: the right candidate is offered
  and the model declines it. Worth re-testing with real mention context (the
  harness passed `side_label="side"` and no role evidence) once quota allows.
- The corpus-scale behaviour of the lookup is still unmeasured: 13 names, not
  13,000.
