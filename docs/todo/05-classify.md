## Stage 5: Classify (pipeline/stages/classify.py)
First slice built (handover-log §20.3, §21). Rules to keep:
- Multi-commander sides without an LLM or manual role get a `default_split`
  of 1/n; the model stage treats them as unattributed latents and credits
  nobody (handover-log §23). Fixed non-field weights are capped at 0.30.
- An unmatched LLM commander rejects the answer; a re-run never overwrites an
  LLM-classified side.

- [ ] LLM missingness classification for rows the heuristics leave unclassified
- [ ] Run the pending multi-commander sides through /process-llm-batch
- [ ] Per-side defender flag (agents/model.yaml needs it; nothing records it) -- handover-log §20.5
- [ ] Infobox commander order is lost (the reference file is alphabetical, bc_id is
      insert order); carry infobox and unresolved counts into SideGroup (#8)
- [ ] Non-person "commanders" (Army Group B, 1st Ukrainian Front): filter in resolve (#9)
- [ ] Few-shot prompt cases: Actium, Munda, Mutina, Chaeronea, Salamis, Mongols, Moltke (#10)
- [ ] Siege per side (siege_defensive), and a campaign/operation type (#12)
- [ ] Missingness: casualty-specific flags, post-1914 regime suppression rule (#13)
- [ ] Proportional unknown-count gate; extractor confidence into rule confidence (LOW)
- [ ] (pilot) naval misfires: Marathon, Malvern Hill, Yom Kippur War, Operation
      Torch; `field` never produced; missing_data_all_logged 30
- [ ] (gap review A10) troop presence is read from raw troop_reports, so reconcile's
      `unfillable` sides look like they have troops, no missingness heuristic fires,
      and impute's mnar_delta never reaches them
- [ ] (gap review) terrain never inferred; siege_defensive / combined / guerrilla never produced
