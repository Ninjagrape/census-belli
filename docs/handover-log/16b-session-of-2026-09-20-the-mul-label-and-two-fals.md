# Handover log §16. Session of 2026-09-20: the `mul` label, and two false-merge paths

Part of the session log; the index is `docs/handover-log/README.md`.

Work on the three open Stage 3 items in `TODO.md`. The first turned out to
rest on a wrong diagnosis, for the second time on this stage.

### 16.1 §14.6 and §13.4 were both wrong about Nelson. Corrected.

§14.6 states that Nelson "genuinely has no English label or alias equal to
'Horatio Nelson'", and concluded the lookup needed a label *search*
(`wbsearchentities`). Looked up live on 2026-09-20:

```
Q83235 labels = {"en-gb": "Horatio Nelson, 1st Viscount Nelson",
                 "mul":   "Horatio Nelson"}
```

**He has no `en` label at all.** The string is there, under `mul` --
Wikidata's multilingual language code, introduced in 2024 for names spelled
the same in every language, onto which person labels have been migrating ever
since. The project asked only for `"..."@en` literals, so it could not see him.

This is systematic, not one awkward admiral: **every commander whose label has
migrated is invisible to an en-only lookup, and the class grows as the
migration proceeds.** It is the same shape of error as §14.1 -- a plausible fix
inferred from a misdiagnosis -- and the same lesson applies. Look it up.

**The fix is two lines, not a new HTTP surface.** `_NAME_LANGUAGE_TAGS =
("en", "mul")` now drives both the VALUES literals and the label service.
Measured live before and after, same batch:

| Query | Candidates for "Horatio Nelson" | Q83235 present? |
|---|---|---|
| `@en` only, label service `"en"` | 3 | **no** |
| `@en`+`@mul`, label service `"en,mul"` | 4 | **yes**, b=1758 d=1805 |

Nelson then links **deterministically**, no LLM call: four candidates match
exactly, the date gate rejects one and abstains on two, and the §13.2
date-confirmed rule leaves exactly one.

### 16.2 The Q-id that would have been published as a person's name

With the label service set to `"en"`, Q83235's `?personLabel` came back as the
literal string **`"Q83235"`** -- the service returns the bare id when it finds
no label in the language asked for. `_merge_rows` accepted it
(`row.get("personLabel","") or qid`), so it became a fuzzy matching key and,
on a link, `Decision.canonical_name`, which `store.py` writes into
`generals.canonical_name` **and** as the primary `general_aliases` row.

A general published as "Q83235", silently. Fixed in two places, deliberately:
`_preferred_name` in the lookup (shape-matched on `Q\d+`, falling back to the
longest matched name, logged as `sparql_candidate_label_was_a_qid`), and
`_publishable_name` at the store boundary, because this arrived from a
direction nobody predicted and the boundary before published data is worth
checking twice.

### 16.3 The worse bug, found by pointing the fixed code at real data

Running the changed lookup over eight real commanders, "Hannibal" at the
Battle of Lissa (1811) **linked to Q1576150 at 0.95 confidence** -- a
Carthaginian commander born about 300 BC.

`lifespan_verdict` bounded only the ends it had. Q1576150's death claim is an
explicit "no value", so `death_year` is None, the upper end was left open, and
the gate judged him alive in 1811 -- **and judged him positively**, so the
§13.2 rule *preferred* him over dateless namesakes. A half-dated ancient was
eligible for every later battle in history, and preferentially so.

That is a false merge, which `matcher.py` opens by saying is the one error
nothing downstream can detect. **It is pre-existing and not caused by the
`mul` change** (Q1576150 has an `en` label, so the old query returned it too),
but widening the lookup makes it more reachable.

Fixed: `MAX_PLAUSIBLE_AGE_YEARS = 100` bounds whichever end is missing. A
fully dateless candidate still abstains, which is correct -- absence of a
lifespan is not evidence against one. Both Hannibal cases now defer to the LLM
rather than linking, which is the designed-safe outcome.

**This is the third time this project has found a real defect within minutes
of pointing a stage at something real, and the second time in this file that
the defect was invisible to a fully green fixture suite.** §4.2, §13.5, and
now this.

### 16.4 What else changed

- The `mul` blind spot closed in the offline path too: `person_candidates`
  read `labels["en"]` and **skipped** an entity without one, so a `mul`-only
  person was absent from the local index rather than mislabelled. `_english`
  is now `_preferred_text`, and aliases are read from both languages.
- `wikidata_mapper._label` takes a language preference;
  `config/sources_seed.yaml`'s three label services became `"en,mul"` (nothing
  reads those labels yet, so it is pre-emptive).
- `LIMIT 400` -> 1000 with a `sparql_candidate_batch_truncated` warning.
  `GROUP BY ?name` groups by value *and* language tag, so an entity matching
  under both tags now takes two rows; truncation was silent and would have
  made a batch's candidate set depend on which rows came back.
- **`general_aliases` could carry two primary rows.** `_UPDATE_GENERAL`
  rewrites `canonical_name` unconditionally, so a re-run renames a general --
  which is exactly what the `mul` fix does to anyone stored under a Q-id -- and
  `_write_aliases` only ever inserted. Pre-existing; this change is the most
  likely thing ever to have fired it.
- Fixtures used **Q167846 for Agrippa**, the wrong id §14.6 itself records.
  Now Q48174, verified live.

### 16.5 The timing question §14.2 raised, answered

Doubling the literals does **not** reintroduce the 29.7s problem. Eight names,
en+mul, batch size 25: **2.2 seconds**. Consistent with §14.2's own finding
that the OPTIONAL cross-product was the whole expense and projecting `?name`
cost nothing. `sparql_batch_size` stays at 25.

### 16.6 The label search is deferred, on purpose

`TODO.md` asked for `wbsearchentities`. Not built, and the reasoning is
recorded so it is not re-litigated from memory:

1. Its premise (§14.6) is refuted, and the `mul` fix closes the named miss.
2. It fires only for names with **zero** candidates, so its hit is usually the
   sole candidate -- and `matcher.py` skips the ambiguity-margin test when
   `len(scored) == 1`. It would auto-link at ~0.62 confidence exactly where
   the evidence is weakest, inverting the stage's split-preferring design.
3. Probed live, it is noisy: "Horatio Nelson" returns five paintings and a
   racehorse; "Duke of Wellington" returns a peerage title and six pubs.
4. The residual class -- a name in neither `en` nor `mul`, label nor alias --
   is **unmeasured**. The gold set should measure it before anything is built.

Also worth knowing if it is ever built: `www.wikidata.org/robots.txt` is
`Disallow: /w/` with only `action=mobileview` allowed, so `/w/api.php` must go
through `fetch_raw`, not `fetch`. That is the §4.7 trap exactly.

### 16.7 State at the end of this session

Verified: ruff clean, `mypy --strict` clean on 49 files, **423 passed / 10
skipped** with `DATABASE_URL` exported (10 is correct -- the opt-in live
tests). `python -m pipeline.orchestrator --stages resolve` runs clean against
the live database and returns real gate verdicts.

**Not verified, and not claimable:** there is no corpus on disk
(`data/processed/` is empty), so the end-to-end run exercised the path and not
the data. Nothing in this session used LLM quota.

**Still open from the approved plan for this work** -- the large half, not
started:

- `tests/fixtures/gold/resolve/` and the threshold sweep
  (`scripts/resolve_sweep.py`, `.claude/commands/resolve-eval.md`). Needs no
  LLM quota; the design is in the plan file.
- `tests/integration/test_resolve_live.py` -- the live regression tests that
  would catch Wikidata migrating Nelson's label back.
- Batch-API submission in `pipeline/llm/` and the cost pilot (§16.8).

### 16.8 The quota blocker, stated properly

§7 and §15.7 call the free-tier quota "operational". That understates it.
`agents/crawl.yaml` targets **3,000 battles**, extract makes one call per
*passage*, so the corpus is roughly **6,000-12,000 calls**. At 20/day that is
**300-600 days.** The project cannot run.

The blocker is a **request cap, not cost**: the one measured call (§13.3) was
$0.0014, putting the corpus at order **$50-200** on a flash-class model. Any
paid tier removes the cap.

Two different things get called "batching". Cramming many battles into one
prompt stays rejected -- `llm_max_tokens: 4096` cannot hold fifty battles of
structured output, the schema is per-passage, and one failure would lose fifty
articles because the `request_hash` cache is per-request. The **provider batch
APIs** are the right fit and were never what was rejected: independent
requests, one async job, about half price, one call per passage preserved so
the cache and `llm_calls` logging are untouched. Correlate on `request_hash`
as the `custom_id` -- batch results return out of order and can partially
fail, and the cache key doubles as the correlation id.

**Resolved.** `CLAUDE.md` now documents the actual routing: Gemini 3.8 Flash
for extract and resolve (high volume, low judgement), Anthropic claude-sonnet-4-6
for classify (low volume, high judgement). Haiku 4.5 is documented as the API
fallback in both `agents/extract.yaml` and `agents/resolve.yaml`. The offline
processing path (§17.1) is the primary alternative to paid API for the bulk
stages.
