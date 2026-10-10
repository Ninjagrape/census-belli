## Stage 5: Classify (pipeline/stages/classify.py)
First slice built; see handover.md §20.3.
- [x] Command role classifier
  - [x] Deterministic rules (single commander = field_commander; consistent roles kept)
  - [x] LLM classification for multi-commander sides, offline cache path (no billed call yet)
  - [x] Hierarchy builder (reports_to_bc_id, hierarchy_rank)
  - [x] Attribution weights, normalised to sum to 1 per side
  - [x] A re-run never overwrites an LLM-classified side
- [x] `battle_type` inference (naval / siege / aerial / field only with an article read)
- [x] Missingness classifier, heuristics only
  - [ ] LLM classification for the rows the heuristics leave unclassified
- [x] Integration test: Actium with a cached answer (Agrippa field_commander, Octavian sovereign)
- [ ] Run the pending multi-commander sides through /process-llm-batch
- [ ] Per-side defender flag (agents/model.yaml needs it; nothing records it) -- §20.5
- [x] historiography-reviewer pass on the classify rules (2026-10-06, handover §21)
  - [x] Fixed: unmatched LLM commander no longer takes weight 1.0 (answer rejected);
        default split is 1/n, not 0.6/0.4 by listing order; co-field commanders;
        fixed non-field weights capped at 0.30; hierarchy validation; naval regex
  - [ ] Infobox commander order is lost (the reference file is alphabetical, bc_id is
        insert order); carry infobox and unresolved counts into SideGroup (#8)
  - [ ] Non-person "commanders" (Army Group B, 1st Ukrainian Front): filter in resolve (#9)
  - [ ] Few-shot prompt cases: Actium, Munda, Mutina, Chaeronea, Salamis, Mongols, Moltke (#10)
  - [ ] Siege per side (siege_defensive), and a campaign/operation type (#12)
  - [ ] Missingness: casualty-specific flags, post-1914 regime suppression rule (#13)
  - [x] Model stage must not consume `default_split` sides without an LLM or manual role
        (they enter as unattributed latents and credit nobody; handover §23)
  - [ ] Proportional unknown-count gate; extractor confidence into rule confidence (LOW)
- [ ] (gap review A10) troop presence is read from raw troop_reports, so reconcile's
      `unfillable` sides look like they have troops, no missingness heuristic fires,
      and impute's mnar_delta never reaches them
- [ ] (gap review) terrain never inferred; siege_defensive / combined / guerrilla never produced
