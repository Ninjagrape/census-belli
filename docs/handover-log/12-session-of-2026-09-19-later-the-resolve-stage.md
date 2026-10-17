# Handover log §12. Session of 2026-09-19 (later): the resolve stage

Part of the session log; the index is `docs/handover-log/README.md`.

### 12.1 What was built

`pipeline/resolvers/`, eight modules, plus `pipeline/stages/resolve.py`:

| Module | What it does |
|---|---|
| `records.py` | `Mention`, `MentionGroup`, `Candidate`, `Decision`, `Identity`, `ResolveCounts` |
| `names.py` | Surface form to match keys. Strips ranks and fate markers, keeps regnal numerals, recognises non-names |
| `mentions.py` | Reads `commanders_raw.jsonl` + `battles.jsonl`, groups mentions by exact shared key |
| `candidates.py` | `LocalCandidateSource` (offline, from `data/raw/wikidata`), `fetch_candidates` (live SPARQL), `NullCandidateSource` |
| `matcher.py` | Exact match, then fuzzy with a lifespan gate and an ambiguity margin; plus corpus-internal matching |
| `disambiguate.py` | The LLM tiebreak, prompt and schema loaded from the spec |
| `battles.py` | Suspected duplicate battles — **detected, never merged** (§12.4) |
| `store.py` | `generals`, `general_aliases`, `battle_commanders`, `missing_data_log` |

### 12.2 The one design decision everything else follows from

**Resolution errors are not symmetric, so the stage is built to prefer a
split.**

A *false merge* links two commanders to one entity. Their records pool, the
model fits one skill parameter to two careers, and nothing downstream can see
it: the merged record looks like an ordinary prolific commander. A *false
split* gives one commander two entities. Their record halves, both halves
shrink harder towards replacement level — a loss of power, and a visible one,
because two near-identical names appear in the ranking.

So a match must clear the fuzzy threshold **and** be date-compatible **and**
beat its runner-up by a margin. Anything else goes to the LLM or becomes a new
entity. The same asymmetry governs the LLM step (`UNCERTAIN` splits rather
than merges) and the duplicate-battle scan (§12.4).

**The date gate is what does the real work.** Name similarity alone links a
mention of "Hannibal" in an 1811 battle to Hannibal Barca, and "Scipio" to
whichever of the five the query service returned first. A candidate who was
not alive cannot have commanded, and `lifespan_verdict` returns True / False /
None — abstaining when either the battle or the life is undated, which is not
the same as passing.

Three more decisions worth knowing before changing anything:

- **"Unknown" never becomes a general.** It appears in thousands of infoboxes;
  resolving it would create one entity credited with every battle nobody
  recorded a commander for, and that entity would rank. Placeholder mentions
  go to `missing_data_log`.
- **A failed LLM call leaves the group unresolved**, not "new". An API failure
  is a known unknown; inventing an identity would put the outage into the data
  where nothing can see it.
- **Grouping is exact-key only.** "Agrippa" and "Marcus Vipsanius Agrippa" do
  not group on name; they are reunited later by the Q-id they both resolve to.
  A wrong merge at the grouping step is invisible, so it does no scoring.

### 12.3 Re-running resolve must not undo classify

`battle_commanders` is the one table two stages write. Resolve creates the
rows; classify refines `command_role`, `hierarchy_rank`, `reports_to_bc_id`
and `attribution_weight` on them.

The extract stage's pattern — delete the rows this stage owns, then rewrite —
would silently discard all of that. So resolve upserts on the
`UNIQUE (battle_id, side_id, general_id)` constraint and its `DO UPDATE` sets
only the columns it owns, leaving a `command_role` that has already moved off
`unknown`. `test_a_rerun_does_not_undo_the_classify_stage` pins it.

### 12.4 Duplicate battles are reported, not merged

`agents/resolve.yaml` asks the stage to "merge duplicate battle records".
`pipeline/resolvers/battles.py` finds them and deliberately does not merge
them, writing pairs to `data/processed/duplicate_battles.jsonl` instead.

Merging two battle rows means re-parenting their sides, and through those
their troop reports, casualty reports and commanders, then deleting one — it
is irreversible, invisible afterwards, and wrong more often than it looks.
"First" and "Second Battle of Bull Run" differ by one word; "Battle of
Panipat" names three engagements over two centuries. Battles also carry the
outcome data the whole model is fitted to, so a wrong merge costs more here
than anywhere else.

The scan therefore requires both names dated and within a year, refuses pairs
whose names carry different ordinals, and compares names only after the
generic "Battle of" prefix is removed. Five tests pin the false positives it
must refuse. **If a later session wants real merging, it needs its own
evidence and its own review step; do not bolt it onto this scan.**

### 12.5 Verification, stated honestly

**Verified against live PostgreSQL 15.19:**

- 18 integration tests: the writes, the re-run convergence, the classify
  interaction, the unresolved-mention log, all five quality gates executing,
  and the duplicate scan.
- **An end-to-end run of `resolve.run()`** driving the real stage over real
  processed files with a real database, checking nine behaviours: two
  spellings of Agrippa became one Wikidata-backed general across two battles;
  a same-name candidate from the wrong millennium was rejected by the date
  gate; "Unknown" became no general and one `missing_data_log` row; Mark
  Antony survived as a corpus-local identity with his dagger stripped;
  `years_active` came back as astronomical years spanning both battles.
- 59 unit tests, no database or network.

**All three of the gaps this section originally listed have since been
closed. See §13 for what the live checks found -- one of them was a
project-wide defect (§4.7).**

What remains unverified: the corpus-scale behaviour of the SPARQL source (it
has been run on 13 names, not 13,000), and the thresholds against a real gold
set, which still does not exist (§8).

One more thing the end-to-end run taught, at the cost of a confused half hour:
**Wikidata signs every year in a time value** (`+1850-01-01T00:00:00Z`). A
fixture that writes a bare `1850-01-01` is refused by
`wikidata_time_to_date`, the candidate comes back with no lifespan, the date
gate abstains, and the match looks ambiguous for no visible reason. The code
was right and the fixture was wrong — §4.2's pattern with the polarity
reversed, and worth remembering when writing any Wikidata fixture.

### 12.6 Changed files

| File | Change |
|---|---|
| `pipeline/resolvers/*` | **New**, 8 modules plus `__init__.py` |
| `pipeline/stages/resolve.py` | **New**, the stage runner |
| `pipeline/extractors/store.py` | `_source_id` → `source_id`, so resolve reuses it rather than copying it |
| `agents/resolve.yaml` | Added `prompt.output_schema`, the params the stage reads, and rewrote the gates (§4.6) |
| `agents/extract.yaml` | `commander_extraction_rate` → `sides_per_battle` (§4.6) |
| `tests/unit/test_resolve.py` | **New**, 59 tests |
| `tests/integration/test_resolve.py` | **New**, 18 tests |
| `handover.md`, `TODO.md` | This |

Suggested commit message:

```
feat: resolve stage — commander mentions to canonical generals

Group mentions by normalised name, look up Wikidata candidates, match
deterministically against label and lifespan, and ask an LLM only about what
is left. Writes generals, general_aliases and battle_commanders, which the
extract stage could not populate because general_id is NOT NULL.

Built to prefer a split over a merge: a match must clear the fuzzy threshold,
be compatible with the battle's date, and beat its runner-up by a margin. A
false merge pools two commanders' records and nothing downstream can see it.

Record unresolved mentions in missing_data_log rather than dropping them, and
rewrite resolution_rate against that denominator. The gate previously counted
battle_commanders.general_id IS NOT NULL, a NOT NULL column, so it read 1.0
however many mentions were lost.

Point extract's commander gate at battle_sides. It counted battle_commanders,
which the extract stage does not write, so it could only ever report 0.

Detect duplicate battles without merging them: re-parenting sides, troop
reports and commanders is irreversible, and ordinals and dates separate more
same-named battles than they join.
```
