## Stage 5: Classify (pipeline/stages/classify.py)
First slice built (handover-log §20.3, §21). Rules to keep:
- Multi-commander sides without an LLM or manual role get a `default_split`
  of 1/n; the model stage treats them as unattributed latents and credits
  nobody (handover-log §23). Fixed non-field weights are capped at 0.30.
- An unmatched LLM commander rejects the answer; a re-run never overwrites an
  LLM-classified side.

- [x] done 2026-10-15 (offline path; not yet answered on the corpus): LLM missingness classification for rows the heuristics leave unclassified
- [ ] Run the pending multi-commander sides through /process-llm-batch
- [x] done 2026-10-15: the role LLM answers is_defender per side: Per-side defender flag beyond sieges: `classifiers/defender.py` sets `is_defender`
      only for "Siege of X" (handover-log §20.5)
- [x] done 2026-10-15: battle_commanders.infobox_order, in the prompt with unresolved counts: Infobox commander order is lost (the reference file is alphabetical, bc_id is
      insert order); carry infobox and unresolved counts into SideGroup (#8)
- [x] fixed 2026-10-12: Non-person "commanders" (Army Group B, 1st Ukrainian Front): filter in resolve (#9)
- [x] done 2026-10-15, corrected after historiography review: Few-shot prompt cases: Actium, Munda, Mutina, Chaeronea, Salamis, Mongols, Moltke (#10)
- [x] done 2026-10-15: is_defender replaces siege_defensive; campaign type added: Siege per side (siege_defensive), and a campaign/operation type (#12)
- [x] done 2026-10-15, list extended after historiography review: Missingness: casualty-specific flags, post-1914 regime suppression rule (#13)
- [x] done 2026-10-15 (no extractor confidence field exists to feed in): Proportional unknown-count gate; extractor confidence into rule confidence (LOW)
- [x] fixed 2026-10-12: (pilot) naval misfires: Marathon, Malvern Hill, Yom Kippur War, Operation
      Torch; `field` never produced; missing_data_all_logged 30
- [x] fixed 2026-10-12: (gap review A10) troop presence is read from raw troop_reports, so reconcile's
      `unfillable` sides look like they have troops, no missingness heuristic fires,
      and impute's mnar_delta never reaches them
- [x] done 2026-10-15: terrain, amphibious, guerrilla rules; combined still unproduced: (gap review) terrain never inferred; siege_defensive / combined / guerrilla never produced
- [x] done 2026-10-15: Siege-category rule types Operation Badr as siege ("Sieges involving Israel"); Fort Donelson naval from its category: review
