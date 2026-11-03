# Cloud session: extract and resolve fixes from the corpus3 audit (2026-11-03)

Branch `wip/cloud-extract-resolve-fixes`, from `wip/corpus3-seed` at 2808819. These are the three §37 section 3 findings, as one commit per task with the tests first, plus this file. The local session numbers this section at merge. `handover.md` was not edited, to keep the merge simple; fold the state in when merging.

There is no corpus database, crawl data or Wikipedia access in the cloud container (the proxy refuses en.wikipedia.org). All fixtures are minimal reconstructions from the audit, not fetched articles.

## Commits

| Commit | Task |
|---|---|
| 083ddf3 | 1. Infobox headings read as side labels |
| 5a5b07d | 2. Derived totals summing the wrong subsets |
| 6fe50e0 | 3. Resolve identity from the infobox wikilink |

## Passage text guard (all tasks)

`tests/unit/test_passage_text_frozen.py` holds the sha256 of `build_passages` output for every `tests/fixtures/extract/*.wikitext`. The hashes were computed on unmodified code before each task's change, and the test fails if a fixture has no hash. It passed before and after every task.

No task touches `strip_markup`, `clean_article_text`, `split_list_items`, `parse_quantities`, the `_BRANCH_PATTERNS`, or `article.py`. The only `textnorm.py` change is to `is_equipment_count` and the new `is_personnel_count` and `is_support_arm`, and only `infobox.py` calls them.

## 1. Headings (083ddf3)

- **Rule:** an infobox list item that ends in `:` is a heading.
  - In combatant fields, the label is the first item that is not a heading, the other items stay aliases of the same side, and the polity is the label.
  - In commander fields, headings are skipped and `infobox_order` stays contiguous.
  - If a side has only headings, its label is the first heading without the colon.
- **Implementation:** `_is_heading` in `infobox.py`.
- **Fixtures:** `quatre_bras_headings`, `okinawa_headings`, `timor_headings`, `la_suffel_headings` and `tomaszow_headings`.
- **Tests:** `test_infobox_headings.py`. Before the fix, 16 failed and 12 passed; the 12 were the frozen-passage tests. One example: `assert 'Ground forces:' == 'United States'`.
- **Overrides retargeted** (the `side:` value only; each note says so):
  - ov-0126, Quatre Bras: now `United Kingdom of Great Britain and Ireland`.
  - ov-0534 and ov-0536, Okinawa: now `United States`.
  - These labels are what the minimal fixtures render. Confirm them on the next local extract; the `overrides_applied` gate fails loudly if they are wrong.
- **Action needed before the next local run.** ov-0035 (Noryang `"Main:"`), ov-0105 (Namwon `"Ming:"`) and ov-0108 (Ulsan `"Ming:"`) unlink heading mentions that no longer exist, so they will report unmatched and fail the gate.
  - They were not deleted, as instructed. The user should retire them, since the fix does what they did.
  - The people listed after those headings now stay on the same side as real mentions, so check Namwon and Ulsan for newly visible Ming commanders, such as Yang Yuan, who §36 noted was missing.

## 2. Derived totals (5a5b07d)

- **Diagnosis.** `_derived_total` sums a side's components when the side has no total. It takes the larger of two partitions, contingents (`other`) or arms of service.
  - Okinawa's Japanese side derived 116,850 because "850 kamikazes" was not an equipment word. `is_equipment_count` also read only the first three words of the noun, so "850 Japanese Navy and Army aircraft" slipped through.
  - Croix d'Orade derived 800 and 1,800. Engineers and sappers match no branch, so they fell into the contingent partition and lost to the arms partition, and "30 engineers" fell under the 100-man floor.
- **Rule chosen: a derived total sums only figures that count people.**
  - **Equipment.** The scan covers the whole noun phrase. The first personnel word (men, soldiers, troops, crews, sailors, marines, pilots, airmen, gunners, engineers, sappers, pioneers, conscripts, volunteers, regulars, guerrillas, ...) makes the figure personnel. The first equipment word makes it equipment.
    - Equipment words added: kamikaze(s), helicopters, gliders, seaplanes, airships, balloons, sorties, horses, mules, ponies, carts, APCs, (armoured) cars, machine-guns, rockets, missiles, craft, sloops, schooners, brigs, dhows, transports and batteries.
    - Singular "horse" is left out on purpose, because it means cavalry in older sources.
  - **Support arms.** Engineers, sappers, pioneers, miners, pontoniers and signallers join the arms partition under one key. Their stored branch is unchanged.
  - **The 100 floor** is waived when the figure's own words name personnel, or its branch is infantry, cavalry or irregular.
- **Fixtures:** `okinawa_strength` and `croix_dorade_strength`.
- **Tests:** `test_infobox_derived_personnel.py`. Before the fix, 9 failed and 5 passed; for example `116850.0 == 116000` and `800.0 == 950`.
- **Results.** Okinawa Japan is now 116,000, and Croix d'Orade is 950 and 1,830. These match the lower bounds of ov-0535, ov-0537 and ov-0538.
- **Effect on other fixtures.** A before/after run over every wikitext and HTML fixture showed no other derived-total change.
- **Known edges:**
  - "tank crews" now reads as equipment.
  - "men-of-war" reads as personnel, as it did before.
  - Excluding equipment can now promote a lone remaining contingent to the total, for example "50,000 French troops<br>120 horses".

## 3. Wikilink identity (6fe50e0)

- **Extract.** `CommanderMention.link_titles` holds the article titles linked inside each commander item. They are read from the raw wikitext (`pipeline/extractors/links.py`), written to `commanders_raw.jsonl` and read by `load_mentions`. Rendered-HTML infoboxes carry none, but extract prefers the wikitext sibling, which corpus3 has.
- **Resolve** (`pipeline/resolvers/wikilinks.py`):
  - A title maps to a Q-id through its Wikidata sitelink, by SPARQL through `fetch_raw`.
  - A miss, usually a redirect, falls back to a robots-checked `fetch` of `/wiki/<title>` and `qid_from_article_html`. The Action API is not used, because robots.txt disallows `/w/` and `/api/`.
  - Answers are cached in `data/processed/resolve/wikilink_qids.json`. The `local` and `none` candidate sources read the cache only.
- **Precedence per mention:**
  1. A manual override.
  2. A wikilink, when the mention's links name exactly one human (from the by-Q-id fetch) whose lifespan does not rule out the battle year. Method `infobox_wikilink`, confidence 0.95.
  3. The name matcher, unchanged.
  - Wikilinked mentions leave their name group the same way override links do, and never reach the LLM.
  - A rank link such as `[[Vice admiral]]` resolves to a non-human item and is ignored.
  - Battle lookup stays slug-first.
  - `"infobox_wikilink"` was added to `METHODS`. No schema change was needed.
- **Params** in `agents/resolve.yaml`: `use_infobox_wikilinks: true` and `wikilink_max_page_fetches: 2000`. Page fetches run at the 2 s rate limit, so 2000 of them take about an hour.
- **LLM cache.** A name group that loses mentions to the pre-pass renders a different disambiguation prompt, so only those groups re-key. Groups the pre-pass does not touch are byte-identical. Expect some new resolve requests on the next offline run.
- **Log line.** `resolve_wikilinks` gives the linked count and the reasons links were ignored. It also counts how many wikilink links agree or disagree with a name-only preview; compare that against the 242 from §36.
- **Tests:** `test_infobox_wikilinks.py` (9 tests) and `test_resolve_wikilinks.py` (70 tests). Before the fix, the infobox file had 8 failures and the resolve file failed to import (`ModuleNotFoundError: pipeline.resolvers.wikilinks`).
- **Regression sample.** 50 of ov-0300..ov-0474 (36 `general_merge`, 14 `commander_link`) pick the override's Q-id with overrides switched off, and a control run without the wikilink picks the wrong one.
  - The title-to-Q-id map in that test is taken from the overrides themselves, so it proves the precedence rule, not that live sitelinks return those Q-ids.
  - No override was deleted.

## Verification

| Check | Result |
|---|---|
| ruff, every commit | All checks passed |
| mypy | 159 source files after tasks 1 and 2, 161 after task 3, no issues |
| Unit suite | 2424 at base, then 2452, 2468 and 2547 after tasks 1, 2 and 3; 0 failed |
| Integration `-k "extract or resolve or override"`, scratch Postgres 16 `general_war_scratch` | 141 passed. The 7 skips are all `test_resolve_live.py` (live network) |

Not verified:
- Any real article: Wikipedia is unreachable from the cloud container, so the label strings and noun phrases come from reconstructions.
- Any real corpus3 extract or resolve run.
- The live sitelink and redirect path.
- The model test files and the integration tests outside the three areas above.

## Expected changes on the next local extract/resolve of corpus3

- **Task 1:**
  - Side labels and polities change for every side whose label ends in ':' (`side_label ~* ':$'`), including La Suffel, Quatre Bras, Timor, Breda and Okinawa.
  - Heading commanders disappear from Noryang, Namwon, Ulsan and Tomaszów (if its "First Phase"/"Second Phase" carry colons).
  - Retire ov-0035, ov-0105 and ov-0108 first, and confirm the retargets ov-0126, ov-0534 and ov-0536.
- **Task 2:**
  - Okinawa (Japan) and Croix d'Orade (both sides) get new `infobox:derived_total` rows. Their reconcile inputs do not move, because the `troop_strength` overrides replace those sides' totals.
  - Any other side whose components include kamikazes, horses, batteries, helicopters and the like, or engineers and sappers, also changes. Diff the `infobox:derived_total` rows before and after to list them.
- **Task 3:**
  - Many commander links change method to `infobox_wikilink`, and some change Q-id, which is the fix.
  - Many ov-0300..ov-0474 entries should become redundant but still match, because they target mentions, not outcomes.
  - Compare resolve's link changes against those overrides before trusting them, and send the result to the historiography reviewer.
