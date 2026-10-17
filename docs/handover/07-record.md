# Handover §7: fixes, Phase 1 record and open items

Part of the handover; the index is `handover.md` at the repo root. Current next steps are in the index.

**2026-10-16 fixes, committed** (each reviewed, tested, suite green):
- **A. battle_type provenance**: `battles.battle_type_source`
  ('extract'|'classify', migration `c9d0e1f2a3b4`). Classify's "inferred"
  note used to live in `missing_data_log`, which extract rewrites per battle,
  so rule types froze. No backfill: a pre-existing rule type with NULL source
  is never revisited (reset it to 'unknown'). Populated covariate rows that
  extract re-logs are marked observed and kept out of missingness requests
  (stage and offline export share one helper).
- **B. stale `battle_sides`**: extract keyed sides by label and never deleted
  old ones (69 of 210 corpus battles had 3-6 sides; **the 2026-10-14 refit ran
  on this**). Extract now deletes a battle's sides it did not write this run,
  with dependents, only when it wrote >= 1 side; keeps a stale side holding
  `llm_extraction` rows while an LLM pass is deferred; never adopts a QID
  owned by another slug; skips cleanup for a battle written twice in one run.
  A relabel deletes resolve's links on that side: **re-run resolve onward
  after extract.**
- **C. orphan generals**: resolve prunes generals no row references, as its
  last write before the gates.
- **D. stage order**: **extract, resolve, reconcile, classify.** Classify run
  without a fresh reconcile logs `troop_total` "no reconciled troop estimate"
  on every side whose estimate went missing (impute selects targets from this
  log). Reconcile now deletes those classify rows for sides it estimates.
- **E. DBpedia dates**: `dbp:date` can be a citation's archive date
  (Herdonia 210 BC became 2015-12-19) or lose the era (Dyrrhachium 48 BC
  became 1948). The merger rejects a DBpedia date more than a year from the
  infobox's, and replaces any chosen date that contradicts an explicit year in
  the title.

Corpus snapshot from before the side repair:
`data/backups/general_war_corpus2_20261016_pre_side_repair.dump`. Classify
batch `classify_20261016T062311Z` (399/400) and the follow-ups are imported;
every stage answers from the cache.

**`historiography-reviewer` verdict on the classified corpus: not fit for a
refit.** Attribution logic is mostly sound; the data feeding it is not. The
user chose: root-cause fixes first (Phase 1), then a reviewed overrides file
for what remains (Phase 2), then refit.

**Phase 1 status (2026-10-17).** Merged to `main`:
- **R1** (extract): the merger took `victor` from DBpedia's raw, glued result
  literal ("Unionvictory") over the infobox's parsed one; victor and level now
  come from one source, DBpedia last and re-parsed. Compound victors take the
  unique side any part matches; qualifiers parse in any position ("Roman
  pyrrhic victory"); extract's side outcome is authoritative (NULL clears a
  stale one, except a battle written twice in one run). A deterministic
  coalition rule was **removed** after review: "Coalition victory" can be the
  side with fewer listed belligerents. Battles still without an outcome get one
  Sonnet `extract_outcome` request in their own batch file; the level never
  comes from the LLM; "indecisive" against a graded victory is rejected. A
  deferred or failed outcome request leaves NULL and drops the gap row (same
  contract as deferred passages; the pipeline halts, so nothing reads it).
  Corpus: undecided non-campaign battles 70 to 7 (Belmont, Bir Hakeim,
  Borodino, Kasserine, Maloyaroslavets, Reims 1814 answered "unclear"; Suez
  becomes a campaign at classify). Thebes is now correct.
- **R3** (classify): campaign if "Operation X" or "Crisis" title, a siege over
  730 days (user: two years, so Rome 537-538 at 375 days stays a siege;
  Leningrad 872 is a campaign), or date precision decade/century. Classify-typed
  campaigns are re-evaluated every run and revert when no rule fires. On the
  corpus clone: 17 campaigns (Moscow, Bagration, Barbarossa, Overlord, Torch,
  Leningrad, Suez, both Operation Badr). Tobruk and Corinth unchanged.
- **R2** (resolve): command window starts at birth + 15; mentions outside a
  linked candidate's window are detached and re-matched; a name group with a
  gap over 60 years is split; split-off parts never become corpus-local
  generals (they would re-merge by name). Two older bugs found on the way:
  **every SPARQL BC lifespan was one year late** (XSD 1.1 year 0 = 1 BC;
  Caesar was stored 99-43 BC), and **resolve never deleted a link it no longer
  made**, so every re-link left the old general credited too (Wagram carried
  both Bellegardes). Resolve now prunes those rows (not on sides with a pending
  LLM answer) and resets the side's attribution so classify recomputes it.
- **R2c** (resolve): Wikidata dates were read as exact years, but 55 linked
  generals have century/decade precision (Decebalus "born 100", Datis "born
  600 BC"); with birth + 15 Decebalus was rejected at 101/106. The SPARQL query
  now fetches `wikibase:timePrecision` (via `p:/psv:` with BestRank, same
  statements as `wdt:`); decade = +-10, century = +-100, millennium dropped; the
  window takes the widest reading; several statements fold deterministically
  (earliest birth, latest death). Stored born/died unchanged. Duplicate gates
  count only generals with a battle link: `remove_orphan_generals` keeps
  "zombies" that old model runs reference (Q28928, Q457386).
- Corpus after Phase 1 (2026-10-17): resolve 0 deferred, resolution 0.994,
  Wikidata links 0.81, probable_duplicate_generals warns 6 (UNCERTAIN splits);
  reconcile run 80 passes errors (warnings unchanged: prior_predictive 0.062,
  calibration set empty); classify 17 campaigns, 0 deferred; impute run 81
  rhat 1.004, ESS 1038; new warning `distributional_check` (early_modern,
  p=0.0016). Backup before Phase 1:
  `data/backups/general_war_corpus2_20261017_pre_phase1.dump`.
- Phase 2 list additions from this work: Lobositz (the outcome LLM followed
  the lead's strategic framing and credited Austria; usually a narrow Prussian
  tactical win), Borodino and the other five "unclear" battles, Rebilus at
  Munda (two Q-ids for one man), Scipio at Ilipa and Rowland Hill at Waterloo
  (UNCERTAIN), Charles John (Bernadotte) at Leipzig (NEW_ENTITY), Trebia
  "Sempronius Longus" and Yi Okki at Noryang (not catchable by dates), dates
  ending before they start (Salamis, Ulm, El Arish, Suez). Overrides design
  constraint: resolve resets a side's attribution when it prunes a link, so
  classify must re-apply manual attribution overrides on every run.
- **2026-10-17 `historiography-reviewer` on the post-Phase-1 corpus**: not yet
  fit for a refit; about 25 row-level corrections, all expressible as
  overrides (summary): Scipio merge (gid817 into Q2253), Charles of Lorraine
  Q3659993 at Chotusitz/Soor to Charles Alexander Q158232, Tuditanus at
  Ticinus, Bernadotte (linked to his brother Q15968159), Yi Okki at Noryang,
  Philippikos at Ajnadayn, Artaphernes at Marathon, Al-Qa'qa on the Christian
  side at Zumail, "Jena"/"Main" junk generals, Rebilus, Daun, Prince Henry,
  Actium (Agrippa 0.7, Octavian 0.1), Bulge (Eisenhower supreme_commander
  ~0.05), Cannae (Varro over Paullus), Asculum pyrrhic, Marathon and Fort
  Donelson not naval. Three wrong-person links (A. P. Hill at Borodino,
  Zieten at Hochkirch, Saxe-Weissenfels at Arcole) were a code gap: R2d.
- **User decisions 2026-10-17 for Phase 2**: the seven battles with no usable
  result (Borodino, Belmont, Bir Hakeim, Kasserine, Maloyaroslavets, Reims
  1814, plus Lobositz) get a **per-battle outcome with a citation**, not a
  uniform rule. **Polity relabelling is in Phase 2**: an override kind for
  side polity, fixing at least Rommel on "Kingdom of Italy" (Gazala, both El
  Alameins), Leipzig's coalition as "Russian Empire", the Bulge's Allies as
  "United States".
- Deferred caveat, not a blocker: rule_single gives the one infobox-named
  commander 1.0 even when absent (McClellan at Gaines's Mill), 140 rows;
  disclose with the ranking.

1. **Phase 1, root causes** (independent; parallel workers in their own
   worktrees from `main`, each with its own database; §5):
   - **R1 outcomes (largest)**: 70 of 204 decided non-campaign battles have no
     outcome on either side and are dropped (`battles_no_outcome`): Waterloo
     168, Leipzig 104, Borodino 38, Jena 94, Acre 216, Gettysburg 78,
     Chancellorsville 56, Fredericksburg 71, Shiloh 145, Issus 93, Granicus
     183, Chaeronea 50, Marathon 113, Artemisium 22, Second El Alamein 211.
     Napoleon loses 15 battles (5 defeats), Lee about 15. Cause: infobox
     result text ("Coalition victory", "Union victory") is not mapped onto
     side labels ("United States"). Decided: deterministic mapping (polity
     aliases, coalition/allied/union terms, named belligerents) in extract,
     then a targeted offline Sonnet request for battles still without an
     outcome (new request kind, a few dozen requests via
     `/process-llm-batch`). The user ruled out answering extract's 14,705
     deferred requests for this.
   - **R2 resolve lifespan guard**: reject a mention whose battle year is
     after the person's death or before birth+15, per mention, and split a
     name group whose years exceed one lifetime. Would catch 13 of 17 date
     conflicts: Munda 45 BC linked to Pompey the Great (d. 48 BC); Ticinus
     157 side 282 Scipio Nasica Q453860 (b. 228 BC) at 1.0 for P. Cornelius
     Scipio cos. 218; Trebia 195 side 351 Sempronius Tuditanus for Ti.
     Sempronius Longus; "Charles of Lorraine" Q3659993 (1684-1751) at
     Chotusitz 58, Soor 149, Prague 177 for Charles Alexander (1712-80);
     Bellegarde Q110721608 (b. 1798) at Aspern and Wagram; Prince Henry
     Q457386 (b. 1747) at Prague; Yi Okki at Noryang. Also the group-years
     issue in step 4 (Caesar's group includes 1948, Hannibal's 2015).
   - **R3 campaign typing**: Operation Overlord 207, Barbarossa 206,
     Bagration 205, Battle of Moscow 122 (precision 'decade'), Siege of
     Leningrad 223 (Zhukov 0.35 "victory"), Operation Torch 208, Suez Crisis
     234 are typed as battles; possibly Siege of Tobruk 228, Siege of Corinth
     219. Rules to add, each tested against this list: "Operation X" titles,
     a siege spanning over a year, decade-or-coarser date precision.
2. **Phase 2, `config/overrides.yaml`** (design: versioned, reviewed, applied
   by the owning stage with `extraction_method`/`attribution_method`
   'manual', survives re-runs, every entry cites a source), for what Phase 1
   leaves: inverted outcomes Thebes 156 (Alexander charged a loss) and
   Toulon 229 (Royalist victory); New Carthage 128 Mago Barca Q335604 (wrong
   Mago); merges Scipio Africanus (Q2253 holds only Baecula, no-QID
   "Publius Cornelius Scipio" holds New Carthage, Ilipa, Great Plains, Zama),
   Daun (3 rows), Charles Alexander; wrong side Al-Qa'qa at Zumail 175,
   Ariobarzanes at Issus, "Jena" as a general at Jena-Auerstedt; Artaphernes
   the elder at Marathon (0.45); attribution Actium side 11 (Agrippa 0.60 vs
   the 0.7 expert check, Octavian 0.2) and Bulge 180 (Eisenhower top at 0.28;
   should be supreme_commander at low weight). Re-check after Phase 1 which
   remain. (Re-checked 2026-10-17: Thebes, Toulon, Ariobarzanes fixed; the
   rest stand, plus the review's additions above.)

   (Design moved to `07-phase2-overrides-design.md`.)

3. **Refit** in stage order (extract, resolve, reconcile, classify with any
   new batch, impute, model, evaluate, report); `historiography-reviewer`
   again before reading the ranking; `bayesian-model-reviewer` on reconcile's
   prior_predictive warning and on whether shrinkage over-pools (step 4).
4. **Open, not decided** (each needs data or a user call):
   - From the 2026-10-16 historiography review, to document or decide:
     troop scope is effectively constant (1303 of 1305 reports 'engaged'; the
     context strings are infobox fragments); reconcile compresses force
     ratios (small sides x1.5-4.5, e.g. Uhud Muslims 754 to 2679; sides over
     150k halved, Berlin 2.3M to 454k; Marathon Persians 68k against a modern
     20-25k), which undercredits underdog wins; 405 of 553 credited generals
     have one decided battle and 12 have six or more; `polity` equals
     `side_label` on all 420 sides (107 unstandardised strings); parse errors
     ("2.6-2.9 million" read as 3, "40-45,000" as 40); 21 duplicate battle
     pairs are each a 0-side shell holding the QID plus the real row without
     one (one article crawled under two slug hashes): not double counted, but
     the real rows lost their QIDs (Arsht join, named checks); 17 sides stay
     default_split because answers broke the rank-0 rule (14) or dropped a
     duplicated infobox name (3).
   - Bayesian review leftovers: L8 (`z_skill` carries the common shift that
     raw skill was dropped from the convergence gate for); L-h (count sides
     whose weak classification differs between raw and residual covariance
     on the corpus).
   - Resolve: stray modern years in groups (Caesar's includes 1948,
     Hannibal's 2015, likely access dates) keep modern namesakes through the
     date gate; Lefebvre, Moreau, Bernadotte, Bülow and Wŏn Kyun are not
     returned by exact label lookup (label search, deferred by §9.3 until a
     gold set exists); `resolvers/mentions.py` uses `part_of[0]`, which can
     be a Q-id, as war context.
   - Model: the recovery Spearman margin is thin (worst 0.825 vs 0.8 over
     4 seeds); the joint-latent comparator's beta_force sits ~0.15 below truth
     at 400 battles per era.
   - The user's calls, unchanged: extract's 14,705 deferred requests and the
     web_secondary keep/cut decision, the calibration gate's arm, the gold
     sets, CI on the `origin` remote, the billed API smoke test.

Say explicitly what was and was not verified against a database.

---
