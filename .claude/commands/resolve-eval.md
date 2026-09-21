Evaluate the resolve stage's deterministic matching against the hand-labelled gold set: $ARGUMENTS

`$ARGUMENTS` may be `sweep` (run a threshold grid search), `check` (run at the current defaults), or empty for `check`.

The resolve stage's two tunable parameters — `fuzzy_threshold` and `ambiguity_margin` — decide whether a mention links to a Wikidata entity, goes to the LLM, or becomes a new corpus-local identity. A wrong link (false merge) is undetectable downstream: two commanders' records pool, the model fits one skill parameter to two careers, and nothing in the ranking can see it. A missed link (false split) is visible — two near-identical names — and costs only statistical power.

**This command spends zero LLM quota.** Groups that would go to the LLM are counted as `ambiguous` and reported as a projected cost column. The gold set includes cases labelled `ambiguous_ok` (Yi Sun-sin) that are correctly deferred; those are scored as correct when the matcher returns `ambiguous`.

## 1. Check the gold set exists

Ground truth lives in `tests/fixtures/gold/resolve/`:

```
tests/fixtures/gold/resolve/
  mentions.jsonl      mention + battle context + expected qid + verdict
  candidates.jsonl    cached candidates per surface form (offline, deterministic)
  battle_pairs.jsonl  duplicate battle pairs with expected verdicts
  MANIFEST.md         provenance (agent-labelled, unreviewed)
```

If missing or empty, stop and say so. Do not fabricate ground truth.

## 2. Run the sweep or check

### `check` (default)

Load the current defaults from `agents/resolve.yaml`:
- `params.fuzzy_threshold` (default 85.0)
- `params.ambiguity_margin` (default 6.0)

Run `scripts/resolve_sweep.py` at those values and report the single row.

### `sweep`

Run `scripts/resolve_sweep.py` with the full grid (threshold 70..95 step 5, margin 2..12 step 2). Display the results table.

## 3. How scoring works

Each gold mention carries an `expected_verdict`:

| Verdict | What it means | Scored correct when |
|---------|---------------|---------------------|
| `linked` | Should link to `expected_qid` | `decision.status == "linked"` and `decision.qid == expected_qid` |
| `new` | Should become a new entity | `decision.status == "new"` |
| `ambiguous_ok` | Genuine ambiguity, LLM is the right answer | `decision.status == "ambiguous"`, or linked to the correct qid |
| `placeholder` | "Unknown", "various", etc. | Handled before matching; always correct |

Error categories, in order of severity:

1. **Wrong link** (CRITICAL): linked to the wrong entity, or linked when expected new. This is a false merge. The number to minimise.
2. **Missed link**: expected linked, got new. A false split — visible and fixable.
3. **Unnecessary ambiguity**: expected linked, got ambiguous. Wastes LLM quota but produces the right answer if the model works.

## 4. Output

### Per-case results

For each mention, show: name, battle, expected verdict, actual status, actual qid, match/mismatch.

### Summary table (sweep mode)

```
threshold  margin  wrong  correct  missed  ambiguous  total
70.0       2.0     0      8        2       3          15
75.0       2.0     0      9        1       3          15
...
```

Highlight the recommended row: zero wrong links, then fewest ambiguous, then most correct links.

### Duplicate battle pairs

Run the battle pair gold cases and report pass/fail for each.

### Recommendation

State which threshold/margin combo to use and why. If the current defaults are optimal (or tied for optimal), say so. If they are not, suggest the change and explain what it buys.

## 5. Cost projection

Each `ambiguous` decision at the recommended settings is one LLM call. Project:

- At the gold set's ambiguity rate, how many calls per 1,000 commanders?
- At the crawl target of 3,000 battles and ~2 commanders per side per battle, how many total LLM calls?
- At the measured cost of $0.0014 per call (handover.md 13.3), what is the projected LLM spend for the resolve stage?

State the figure. It is the number that decides whether the disambiguation step is affordable.

## 6. Do not commit

Report the results. If `$ARGUMENTS` includes `--update-defaults` and the sweep found a better combo with zero wrong links, update `agents/resolve.yaml`'s `fuzzy_threshold` and `ambiguity_margin`. Suggest the commit message `test: tune resolve thresholds from gold set sweep` for the user to run.
