# Handover §7: Phase 2 overrides design

Part of the handover; the index is `handover.md` at the repo root.

**Built 2026-10-17/18** (merges 945ddcb loader, 0688491 resolve, b3cae8a
extract, 4f11350 classify, 74068f8 unlabelled-entity fix; entries 2e31821).
What a future session would otherwise get wrong:
- The gate re-hashes the overrides file and compares entry ids, so a stale or
  partial report fails; a dry run writes no report. `OverrideReport.write`
  raises unless every entry of the stage was reported exactly once.
- Removing an entry does not always revert: `side_polity` and `battle_date`
  stay when extract supplies nothing for the field (COALESCE in
  `pipeline/extractors/store.py`); a removed `battle_type` keeps
  `battle_type_source = 'manual'` until extract re-types the battle.
- A `commander_link` on one mention of a name group leaves the group's other
  mentions on automatic resolution. If those stay corpus-local, the second
  resolve run creates a twin of the linked Q-id (`no_duplicate_generals`
  fails). Link every mention of that person (ov-0053/ov-0054, Anderson).
- A manual link to a human with no English label (Yi Yeong-nam, Q12612045,
  Korean label only) takes the mention's name; automatic matching still
  rejects unlabelled entities.
- A `commander_link` to Q-old plus a `general_merge` Q-old to Q-new leaves the
  linked mention on Q-old (merges skip manual groups). The file has no merges.
- Any changed commander set resets that side's attribution, so classify
  issues fresh role requests for it (18 on the 2026-10-18 refit).
- Extract writes records carrying an override last, so a second record for
  the same battle cannot overwrite it.

**Design (lead, 2026-10-17).**
- One file, `config/overrides.yaml`, `version:` at the top, one list of
  entries. Every entry: `id` (stable, e.g. `ov-0001`), `kind`, the target
  keys for that kind, `value`, `source` (citation: a standard work with
  page, or a Wikipedia article and section; required, non-empty), `note`,
  `reviewed: {by, on}`. Loader `pipeline/overrides.py`: strict schema
  (unknown keys or kinds, a missing `source`, a duplicate `id` or two
  entries for one target fail at load, before any stage writes).
- Targets use keys that survive re-runs: battles by `battles.source_slug`,
  sides by `side_label` (extract-owned, stable within a crawl), generals by
  Wikidata Q-id, or by `canonical_name` for corpus-local rows. Never by
  database id.
- Each kind is applied by the stage that owns the field, on every run,
  after that stage's own logic, so a re-run never undoes it:
  - extract: `outcome` (every side's outcome for a battle, given
    explicitly, so nothing is mirrored implicitly), `battle_date`
    (start/end/precision), `side_polity` (writes `battle_sides.polity`;
    the label stays the key).
  - resolve: `commander_link` (battle + side + mention name to a Q-id, or
    `none` to unlink: the mention is logged unresolved with method
    `manual`), `general_merge` (corpus-local name or Q-id into a target
    Q-id). Applied before matching, so the guard and the prune see the
    final link set; a manual link bypasses the lifespan guard on purpose
    (that is what overrides are for) and is logged as such.
  - classify: `battle_type`, and `attribution` (battle + side: the full
    commander list with role, rank and weight; a whole side at a time;
    weights checked against the classify rules). Re-applied after resolve
    resets a side, which is why classify owns it and re-applies each run.
- Provenance: `extraction_method`/`attribution_method` = 'manual';
  resolution method 'manual'; `battle_type_source` gains 'manual' (one
  Alembic migration plus `config/schema.sql`, §3.4); outcome
  `outcome_source` points at a `sources` row of type `manual_entry` whose
  title is the citation.
- Rot guard: each owning stage gets an error gate `overrides_applied`
  that fails when an entry of its kinds matched no target (a renamed slug
  or label must not silently drop a correction), and logs a count per
  kind.
- Entries are drafted by `historiography-reviewer` with sources, read by
  the user, then built in; the reviewer re-checks after the refit.
- User calls 2026-10-17: per-battle cited outcomes for the seven
  unresolved battles; `side_polity` in scope.
