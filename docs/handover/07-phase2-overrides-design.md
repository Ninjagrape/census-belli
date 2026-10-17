# Handover §7: Phase 2 overrides design

Part of the handover; the index is `handover.md` at the repo root.

**Design (lead, 2026-10-17; build not started).**
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
