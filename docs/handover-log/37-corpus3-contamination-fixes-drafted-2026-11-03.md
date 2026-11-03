# §37. Corpus3 contamination fixes: overrides approved, extract fix built (2026-11-03)

Step 1 of the §36 plan is complete: the fixes are drafted, approved and built, but nothing is applied yet. Nothing was written to any database, and nothing was committed.

## 1. Overrides approved by the user (48 entries, not yet in config/overrides.yaml)

The entries are in [37-corpus3-contamination-overrides-approved-2026-11-03.yaml](37-corpus3-contamination-overrides-approved-2026-11-03.yaml). Its indentation is ready to paste under `entries:`.

They have been validated against a scratch merge of w-c3seed's `config/overrides.yaml`: 419 + 48 = 467 entries, with no duplicate ids. Every target matched exactly one battle (and one side where a side is named) in `general_war_corpus3`.

- **ov-0500..0524, `battle_type: campaign`.** These are WWII strategic operations that were typed field, aerial or siege. The 14 the user named are 0500-0513. The other 11 (0514-0524: Białystok-Minsk, Baltic, Kiev 1943, Voronezh-Kastornoye, Bereznegovatoye, Crimea, Kamenets-Podolsky, Leningrad-Novgorod, East Pomerania, Romania, Scheldt) came from a sweep for rows of 300k or more.
  - **The type removes a battle from the WAR model, not only from reconcile** (`schema.sql` comment on `battle_type`).
  - The user therefore kept **Kursk and Berlin as battles**.
  - These retypes remove only about 12 contaminated multi-row sides. They are not the whole fix.
- **ov-0530..0549, `troop_strength` and `side_polity`.** These fix mixed-scope sides:
  - Fort Frontenac, Chaves (first siege only), Ticonderoga, Okinawa (183k-250k against 116k, polity United States), Croix d'Orade (both sides, plus polities) and Corbach 15k-20k.
  - Also Vyazma 1812, Genoa 1747, Budapest, Namwon, Plains of Abraham, Heilsberg (both sides) and New Carthage.
  - The user declined Ammunition Hill: Israel's 150 would only be inferred.
  - A `troop_strength` entry replaces every other total on its side and has its bias pinned to 0, so the side leaves the noise learning.
- **ov-0550, Seelow Heights, Soviet side.** The 1M is relabelled `theatre_strength`, because no Seelow-only figure exists. Reconcile models that scope as a level above engaged, with a learned offset (`design.py` SCOPE_ORDER, `model.py` `_level_offsets`), so the 1M now informs a smaller engaged estimate.
- **ov-0551 and ov-0552, Okinawa.** `battle_type: amphibious` (the user's choice) and `battle_date` 1945-04-01 to 1945-06-22, at day precision.
- **No override kind can rename a side label.** The "side 1"/"side 2" labels stay; only the polity is set.
- Drafted but not yet sourced: Dürenstein, Ratisbon, First siege of Badajoz and Siege of Odesa 1941 (phase or scope mixes).

## 2. Extract fix (branch `wip/extract-endash`, worktree `w-endash`, uncommitted)

The branch starts from `wip/corpus3-seed` at 288eee5. Diagnosis by a Sonnet worker:

1. **`{{endash}}` was missing from `_LITERAL_TEMPLATES`** (`textnorm.py`). `strip_code` dropped it, so `550{{endash}}800` became 550800.
   - It affects 13 fields in 8 US Civil War articles: 2 troop rows (Glasgow, Roan's Tan Yard) and 6 casualty rows (Fort Davidson 5001, McDowell, Mine Creek, Roan's Tan Yard).
   - The audit's 82 "candidates" were mostly false positives.
   - **Fix:** a new `_RANGE_DASH_TEMPLATES`, applied only through `strip_markup(..., infobox=True)` from `infobox.py:276`. The same cleaner builds LLM passage text, so a global change would have re-keyed the cache for up to 43 articles.
2. **"with <Name>" now reads as a detachment** (`_WITH_NAME_RE` in `classify_figure`), alongside `under <Name>`. Corbach's "400 to 500 with Luckner" was the only real case.
3. **Flag templates now skip `size=`/`NNpx`** (`_flag_country_name`). Tournai 1745 had a single side, "20px", holding both armies and all five commanders at weight 0.2.

**Verified** (by the worker; I read its raw output):

| Check | Result |
|---|---|
| Unit suite | 2424 passed, 0 failed; the new file `tests/unit/test_extract_endash_withname.py` has 26 tests, and 19 failed before the fix |
| ruff | clean |
| Passage text | identical old vs new for 3304 articles and 24,108 passages (sha256 of the passage; scripts in `37-scripts/`) |
| Infobox diff | only Corbach, Fort Davidson, Glasgow, Roan's Tan Yard, McDowell, Mine Creek and Tournai change, plus Marmiton River (values unchanged, bound flags added) |

**mypy fails** on `pipeline/reconcilers/model.py:765` ("float * None"). The failure predates this fix and is already on wip/corpus3-seed: it comes from 6eb2e10..288eee5. The handover's "mypy clean" holds for main, not for this branch.

Suggested commit message:

```
fix: keep en-dash templates in infobox ranges, read "with Name" as a detachment, skip flag size params
```

## 3. Other findings to file

- Derived totals for component lists sum the wrong subsets: Croix d'Orade, and Okinawa added 850 aircraft to its men. This needs a diagnosis and touches more than these two battles.
- Infobox headings are read as side labels ("Ground forces:", "Allies:", "Seventh Coalition:"; La Suffel, Quatre Bras, Timor, Breda). Find them with `side_label ~* ':$'`.
- The First Siege of Chaves infobox covers two sieges. A `battle_split` would restore the second siege as its own battle.
- The multi-row count depends on where it is taken: 412 sides with 2+ loaded reports, but 66 sides with 2+ rows in the model design (§36 audit). The 66 is the number that identifies the noise.

## 4. Cloud

The user tried cloud sessions. They cannot see local branches, so `wip/corpus3-seed` must be pushed first. The user pushes with GitHub Desktop, adding the worktree folder as a repository, since the CLI credential manager is not set up. The branch was not pushed this session, so no cloud work ran.
