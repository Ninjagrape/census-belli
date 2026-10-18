# Refit review: general_war_corpus2, model run 108 (historiography, 2026-10-18)

Read-only. SELECT queries on general_war_corpus2, rankings.csv/report.md/evaluation_report.json read, Wikidata entity lookups, and a handful of Wikipedia checks. Nothing in the repo was written.

## Verdict

**You can show it to the user with caveats. Do not publish it.** Nothing here is a BLOCKER for an internal showing. All 54 overrides landed. The top 8 (Napoleon, Caesar, Hannibal, Khalid, Frederick, Lee, Alexander, Scipio) is a defensible set, and the six generals whose skill interval excludes zero are the six a specialist would expect. Two placements should be flagged as unreliable when it is shown: **Khalid #4** (and his top skill score, 3.84) and **Pyrrhus #14**. Both rest on troop figures where the pipeline threw away the infobox's own modern estimate and kept the inflated ancient one (M1). The report also lacks the three disclosures you named (M2). The lead can put those in its cover note, or report.md can be patched first.

## 1. Did the overrides land? Yes, 54/54

Each was checked against its row in the DB.

- **Outcomes (8):** Asculum Epirotes pyrrhic_victory / Rome defeat; Belmont both sides indecisive; Bir Hakeim Free France defeat / Germany victory; Borodino French pyrrhic_victory / Russia defeat; Kasserine US defeat / Germany victory; Lobositz Prussia victory / Austria defeat; Maloyaroslavets French victory; Reims 1814 France victory. All as written.
- **Dates (5):** Kasserine 1943-02-19 to 02-24 day; Salamis 0480-09-01 BC month, end null; Ulm 16-20 Oct 1805; El Arish 8-20 Feb 1799; Suez 29 Oct to 7 Nov 1956. All as written.
- **Polities (10):** Alam el Halfa, Bir Hakeim, Gazala, both El Alameins all Axis; Austerlitz Third Coalition; Bautzen and Leipzig Sixth Coalition; Waterloo Seventh Coalition; Bulge Allies. All as written.
- **Commander links (25):**
  - Vardan, "Jena:" (both sides), "Main:" and Rabi'a are all gone. Zumail's Arab Christian side is now unattributed, as intended.
  - Servilius Q554632, Charles Alexander Q158232 (Chotusitz, Soor, Prague), Daun Q93365 (Chotusitz, Leuthen), Rebilus Q1345437, Charles John Q52927 (Leipzig), Bernadotte Q52927 (Wagram), Hill Q336485, Sempronius Longus Q382276, Henry Q50993, Ticinus to Q3293507 (now the only battle of general 817), and Yi Yeong-nam Q12612045 (Noryang) are all in place.
  - The four Scipio battles (Ilipa, New Carthage, Zama, Great Plains) now sit on general 147 / Q2253 with Baecula, five battles in all.
  - Anderson is Q259769 (general 960) at both Kasserine and Torch.
- **Battle types (2):** Fort Donelson siege_offensive/manual; Marathon field/manual.
- **Attributions (4):**
  - Actium: Agrippa field 0.70, Octavian supreme 0.10, Arruntius 0.10, Lurius 0.10.
  - Cannae: Varro 0.50 rank 0, then Paullus 0.25, Servilius 0.15, Tuditanus 0.05, Octavius 0.05.
  - Bulge: Bradley 0.25, Montgomery 0.22, Patton 0.20, Hodges 0.18, Eisenhower supreme 0.05.
  - Kasserine: Fredendall 0.82, Anderson theatre rank 1 0.10, Juin 0.08.

  All are manual and as written. Agrippa sits at exactly 0.70, so a strict "exceeds 0.7" check would fail. The gated pairwise check passes (Agrippa 18, Augustus 34).

## 2. Classify's fresh attributions on the needs_review sides

| Side | Pipeline | Judgement |
|---|---|---|
| Wagram, French | Napoleon field 0.45; Davout 0.15, Masséna 0.08, Macdonald 0.06, Oudinot 0.06, four at 0.04, Wrede and Lasalle 0.02. Role evidence is only the infobox list (conf 0.4) | **Acceptable.** The order matches the standard account: Davout's turning attack on the Russbach and Macdonald's column decided it, and Masséna held the left. No entry needed. |
| Waterloo, coalition | Wellington 0.40 and Blücher 0.26, both rank 0, plus Prussian and Anglo-allied subordinates | **Acceptable.** Two independent army commanders on one recorded side is genuine co-command. One wrong link on this side, see M4 (Gneisenau, Zieten). |
| Noryang, Japan | Shimazu 0.57, Konishi 0.08, seven at 0.05 | **Acceptable.** Shimazu led the relief fleet. Konishi was blockaded at Suncheon and slipped out, so 0.08 is generous but small. |
| Jena-Auerstedt, Prussia | Brunswick 0.27 and Hohenlohe 0.27 co-field; FW III sovereign 0.07 | **Acceptable as attribution**: two separate battles, two field commanders. The real problems are battle identity and troop numbers (M3), and Hohenlohe and Kalckreuth are unlinked (M4). |
| Chotusitz, Austria | Charles Alexander 0.60, Batthyány 0.20, Daun 0.20 | **Acceptable.** |
| Suez | Equal 0.05 over 19 names, including Eden, Mollet, Coty and Ben-Gurion; campaign | **Acceptable** because it is excluded from the model. Cosmetic only. |
| Prague 1757, Prussia | Schwerin field rank 0 0.50; Frederick supreme_commander rank 1 0.40; Henry 0.10 | **Contested; an optional entry is drafted below.** The article backs a large Schwerin share. It says Frederick "seemingly deferred to Schwerin" and ascribes the victory to "Schwerin's leadership and the independent initiative of regimental commanders" (Wikipedia, "Battle of Prague (1757)"). The split is defensible. The role is not: supreme_commander means directing from above the field, and Frederick commanded the army in person and was on the field. Labelled this way, the field_commanders_only variant drops Frederick from a battle he commanded. The ranking effect is under 0.05 of one victory. |

## 3. The ranking

### BLOCKER
None for showing the user.

### MAJOR

**M1. Reconcile shrinks every troop figure toward a common mean, so it does not correct ancient inflation. The extractor also discards labelled modern estimates.** This is the finding most likely to be attacked.

- **Shrinkage, not de-biasing.** Geometric-mean ratio of the reconciled estimate to the reported "total" figures, by era and size:
  - Ancient sides reported above 50k: 0.83x (n=23).
  - Post-1800 sides above 50k: **0.76x** (n=80). These are figures from archival returns: Gettysburg 93.5-104k becomes 73.6k, Wagram 171.9k becomes 105.5k, Leipzig 257k becomes 130.7k, Kursk 1.91M becomes 533k.
  - Sides under 10k are inflated 2.3-3.9x in every era: Uhud 754 becomes 3,114, Mu'tah 3,000 becomes 9,465.

  So era-specific inflation correction is not visible in the output. Ancient inflated figures survive at about 0.8x, against the 3.2x pooled prior and the 10-50x of the worst cases. Every force ratio is compressed toward 1 (log ratio times about 0.65-0.75: Gettysburg 0.35 to 0.23, Leipzig 0.37 to 0.25, Kursk 0.90 to 0.67). That understates the advantage of generals who won with numbers (Grant 1864-65, the 1813 coalition, the Soviet fronts) and overstates the credit for winning "outnumbered". Pass this to bayesian-model-reviewer. It is a reconcile prior question, not a data row.
- **Modern estimates routed to branch `other` and dropped.** Where the infobox gives an ancient figure and a labelled modern estimate, extract tags the modern one `branch=other`, reconcile ignores it, and the ancient figure is the only total:

  | Battle | Side | Pipeline | Modern estimate (source) | Ratio |
  |---|---|---|---|---|
  | Hazir 637 | Byzantine | 57,856 | about 7,000; Crawford, *The War of the Three Gods* (2013) p.149, doubts Menas had "even a tenth" of the 70,000 | **8.3x** |
  | Chains 633 | Sasanian | 62,760 | 15-20k (Wikipedia infobox, "modern estimates") | **3.1-4.2x** |
  | Ullais 633 | Sasanian | 59,274 | about 30k (Crawford; the article says 70k "is the result of exaggeration") | **2.0x** |
  | Beneventum 275 BC | Epirus | 14,980 | about 40k (modern estimates; Plutarch's 20k was deflated to 15k) | **0.37x**: the ratio inverts. Rome is put at 28.7k vs modern 20-25k, so Pyrrhus appears outnumbered 1.9:1 when modern estimates have him outnumbering Rome |
  | Persian Gate 330 BC | Achaemenid | null (left to impute) | 700-2,000 modern vs Arrian's 40,700 | modern figure dropped. The Macedonian "17,000 picked fighters" is tagged `branch=air` |
  | Issus | Achaemenid | 53,168 | modern 108-158k tagged other | the estimate happens to be plausible |

  Khalid's 633-637 battles carry an opposing figure of 55-63k against his 18-26k at Yamama, Chains, Ullais, Hazir and Fahl. Those figures come from the Islamic tradition (al-Tabari, much of it the Sayf ibn Umar strand) that modern historians treat as inflated. Khalid's top skill score and #4 rank depend on them. Pyrrhus #14 (3 battles) depends on the inverted Beneventum ratio.

  The spot-checked battles with no labelled-modern conflict are fine: Gaugamela 70k Persian against a modern range of 50-250k, Cannae Rome 72k, Bibracte Helvetii 73k, Alesia Gauls 101k.

  **Fix (heuristic, extract):** read a "(modern estimate[s])" or "(modern est.)" label as branch total with the modern regime. When a modern-labelled and an ancient-labelled figure share a side, give reconcile the modern one at full weight. Re-run extract onward.

**M2. report.md lacks the three disclosures you named, plus the corpus composition a reader needs.** grep finds no mention of components, rule_single or the passage pass.

- **Components.** 41 of the top 50 generals by total WAR lie outside the largest connected component, component 0, which is Napoleon's. Among the 89 headline generals, 30 of the first 50 are outside it. Caesar (3), Hannibal and Scipio (6), Khalid (2), Frederick (9), Lee and Grant (4), Alexander (5) and Pyrrhus (13) all sit in separate components, so their order relative to Napoleon and to each other is identified only through the shared prior and the force slope.
- **rule_single.** 138 rows give the one infobox-named commander 1.0. I spot-checked all 138: almost all were present and commanding. The clear exceptions are McClellan at Glendale (absent aboard USS *Galena*, per the article and Sears, "dereliction of duty") and at Gaines's Mill (Porter fought it). See m1.
- **Passage pass.** Extract's 14,705 passage requests are unanswered. Every battle carries needs_review "LLM pass pending", so troop scope and the prose evidence come from infoboxes only.
- **Corpus shape:** 236 battles. **Zero battles from 700-1199, 1300-1499 and the whole 1600s.** That means no Crusades, no Hundred Years' War, no Thirty Years' War, no Marlborough, Gustavus, Genghis, Timur, Saladin, Suvorov, Moltke or any Chinese dynasty. 42 of the 89 headline generals are Revolutionary or Napoleonic and 16 are ancient. The American Civil War is 32 battles (14%). Wellington has 1 credited battle and Subutai 2.
- **Survivorship and fame.** Rank-0 commanders with 5 or more battles win 73.6% of their decided sides. Those with 1-2 battles win 30.7%. Sides carrying a top-12 general won 81%. The corpus is the famous generals' career lists, with opponents mostly appearing once. That is structural fame bias and should be stated in numbers, not as a sentence.

**M3. Scope is a default, not a judgement, and one major battle has summed partial figures wrongly.**

- `troop_reports.scope` is engaged in 1,303 rows and available in 2 (`INFOBOX_STRENGTH_DEFAULT_SCOPE`). Wording that should override the default does not:
  - Jena-Auerstedt French: "40,000 deployed to do battle, 96,000 total" are both engaged.
  - Bibracte Helvetii: 368,000, Caesar's figure for the whole migrating people including non-combatants, is engaged.
  - Leipzig: 257,000 is the coalition's initial strength (it grew to about 380k) and is engaged.
- **Jena-Auerstedt (battle_id 94).** Prussian partial figures ("46,000 (Jena)", "15,000 under Rüchel") are reconciled as competing reports of one total, giving Prussia **32,858** against France 57,537. The combined Prussian-Saxon force was about 110k: the side's own row "110,000", Hohenlohe plus Rüchel about 53k at Jena, Brunswick 64k at Auerstedt. Engaged numbers roughly favoured Prussia overall. The pipeline gives Napoleon a 1.75:1 edge. The single record also folds Davout's independent Auerstedt victory (27k against 64k) into Napoleon's credit: Napoleon 0.40, Davout 0.28.
- **Fix:** add "total", "deployed", "population", "including non-combatants" and "initial" to the wording rules in `textnorm.resolve_scope` (heuristic). For Jena-Auerstedt, split it into two battles or add a `side_troops` override kind. Neither is in the current override schema, so the lead has to make a design call.

**M4. Wrong-person links and split identities.** The resolver's date gate only checks that a candidate was alive. It passes undated stubs and implausible ages. Mostly low-weight subordinates, but two are headline rows and several are embarrassing in an explorer.

- Gneisenau → Q135401853, his son (P22 father = Q165519). He is also split three ways: the son's item, a corpus-local "Graf von Gneisenau" at Ligny, and the real Q165519 is absent.
- "Joseph I" at Somosierra → Q17631178, an Idumean nobleman, Herod's uncle. It should be Joseph Bonaparte, Q7726.
- "Dimitriadis" at Granicus (Persian side) → Q138847466, a Greek futures-studies consultant.
- "Louis Berthier" at Marengo → Q140078072, Louis de Berthier de Bizy. This splits the marshal from Q155734, whose label is also the Spanish "Luis Alejandro Berthier".
- Kleist at Bautzen and Vauchamps → Q58739712 (1746-1820). Kleist von Nollendorf is Q213784, already linked at Dresden, so he is split.
- "Ivan Dibich" at Dresden → Q15065850, the father (1738-1822). It should be Hans Karl von Diebitsch, Q703307.
- Uvarov at Borodino and Friedland → Q15068544 (1786-1845). It should be F. P. Uvarov, Q3072696, already linked at Krasnoi, so split.
- Pavel Belov at Kursk → Q89565157, a "researcher". It should be Q4082219, the Soviet general.
- "Jean de La Poype" at Toulon → Q108742379, a 15th-century man. It should be Q1685527.
- Sułkowski at Arcole → Q140763329, a stub. It should be Q946928.
- Schmettau at Jena → Q136281732 (1661-1732). It should be Q121684 (1743-1806, mortally wounded at Auerstedt).
- "Graf von Zieten" at Ligny and Waterloo is corpus-local. It should be Q67913, already linked at Dresden.
- Headline rows left unlinked: **Wŏn Kyun** (headline #77; should be Q379659) and **Francis I** (headline #303; should be Q130834).
- Also unlinked: Hohenlohe (Q696782), Kalckreuth (Q73253), Samuel Hood (Q332623), Carteaux (Q573519).
- Campaign-only, so no model effect but visible: "Farouk" → a Moroccan painter (should be Q213001), "Abdallah I" → a sultan of Anjouan (should be Q192780).

**Fix:** overrides (drafted below), plus a resolver heuristic: reject candidates with no dates and no sitelinks, and require age at the battle between 15 and 80 rather than just "alive".

**M5. Polity labels are not standardised.** No side has a `wikidata_id` (0 of 420), so impute's polity effect keys on raw strings:

- The Confederacy is 4 levels: "Confederate States", "CSA (Confederacy)", "Confederate States (Confederacy)", "CSA".
- The Union is 2.
- Napoleonic France is 7: France, First French Empire, French Empire, French First Empire, Imperial France, French Republic, French First Republic.
- The Habsburgs are 4, one a pure case variant: Austria, Austrian Empire, Habsburg monarchy, Habsburg Monarchy.
- Macedon is 3, Joseon 3, Rome 2 ("Rome" and "Roman Republic"), and Epirus is spelled three ways ("Epiru", "Epirotes", "Epirus").

Partial pooling then borrows from much thinner levels than intended. **Fix:** a canonicalisation map at extract (heuristic). It should not be done as dozens of overrides. Genuine mislabels are drafted below: Tobruk, Kasserine, Toulon, Heraclea.

**M6. The headline set will be read as "generals". Six are heads of state, and the lower half is mostly appearance count.**

- Hitler (#351), Stalin (#125), Alexander I, Frederick William III, Francis I and Cleopatra are headline rows on sovereign or supreme credit.
- 53 of the 89 headline generals have a skill sd of 1.9 or more, which is the prior. Their order is effectively their effective-battle count.
- Michel Ney is **#16** with skill −0.80 [−4.92, 2.96] yet total WAR +0.61, above Rommel, Agrippa, Zhukov and Davout. That puts 17 French marshals and 12 or more coalition Napoleonic commanders in the headline.
- Lee #6 and McClellan #13 (above Montgomery, Zhukov and Rommel) are volume effects of the 32-battle ACW block.

**Fix (report):** exclude or tag heads of state whose credit is all sovereign or supreme. Show skill beside WAR prominently (it is there). Consider marking generals whose posterior is about the prior. Whether WAR should be positive at below-mean skill is a model question; note it to bayesian-model-reviewer.

### MINOR
- **m1. Absent commanders under rule_single** (Glendale 79, Gaines's Mill 73: McClellan 1.0). This cannot be fixed with current override kinds. Glendale's article says McClellan "was not present during the battle and named no second-in-command", so the honest fix is an `unattribute_side` kind. Gaines's Mill needs Fitz John Porter (Q1421282) added, so an add-commander extension to `attribution`. Both need a design call.
- **m2. One man on both sides:** Darlan at Torch (US side 0.03 nominal plus Vichy theatre). Aegus and Roscillus on both sides at Dyrrhachium; they deserted Caesar for Pompey before the battle (Caesar, *BC* 3.59-61), so they belong on the Pompeian side only.
- **m3. Siege of Toulon:** Napoleon is subordinate 0.30, with Dugommier 0.40 and Carteaux 0.10, and Carteaux is co-rank 0 though he had been relieved in November. 0.30 for a chef de bataillon of artillery is generous even granting the l'Eguillette plan (Chandler, *Campaigns of Napoleon*, pp. 22-29). An optional entry is drafted.
- **m4. Range parsing:** "40–45,000" is read as 40 and 45,000 (Third Petersburg, Confederate); the bound_only estimate of 37.7k happens to be fine. "3,000 archers" is taken as Octavian's total at Actium. Heuristic.
- **m5. Calendars:** Roman dates before 45 BC are stored on the pre-Julian civil calendar as if Julian (Pharsalus "9 August 48 BC" is about 7 June Julian). Marathon is stored at day precision as 12 Sep 490 BC, Boeckh's conventional and disputed date; month precision would be the honest grade. No effect on era assignment. Calendar basis is unrecorded throughout.
- **m6. Hierarchy:** acyclic, no cross-side `reports_to`, weights sum to 1 on every side. But 34 rows have a parent whose `hierarchy_rank` is not less than the child's (e.g. Meade "field_commander" at rank 1). Cosmetic unless the rank is used.
- **m7. Pyrrhic victories:** only 3 in the corpus. Heraclea (280 BC) is often described as Pyrrhic too. Pyrrhic scores as a win, so there is no model effect.
- **m8. Shells and names:** the shell duplicates (Alexandria, Dyrrhachium, Chaeronea, Bautzen and others, 23 pairs) have no sides and are harmless. The Second Battle of Tapae is an undated shell, so Trajan's 101 AD victory is missing. The Kasserine battle row (slug Battle_of_Kasserine_Pass-72c07c66a7) is named "Battle of Kasserine Passe".
- **m9. Rank numbering:** the report numbers ranks among all 732 generals, so the 86th headline row reads "rank 723" (Sedgwick) and ranks 22-25 belong to unranked generals. Readers will be confused. Show the headline position, and give the global rank in the CSV only.

## 4. Draft Phase 2b override entries
Same schema as config/overrides.yaml. These are drafts for the user to read, not approved.

```yaml
  # --- commander_link ---
  - id: ov-0055
    kind: commander_link
    target: {battle: Battle_of_Waterloo-059b45c1f1, side: United Kingdom of Great Britain and Ireland, mention: August von Gneisenau}
    value: Q165519
    source: "Wikipedia, 'Battle of Waterloo', infobox; Wikidata Q165519 (August Neidhardt von Gneisenau, 1760-1831) and Q135401853 (P22 father = Q165519)"
    note: "Linked to Q135401853, the field marshal's son. Blucher's chief of staff is Q165519. Link every Gneisenau mention (ov-0056, ov-0057) so no twin appears."
    reviewed: {by: historiography-reviewer, "on": 2026-10-18}
  - id: ov-0056
    kind: commander_link
    target: {battle: Battle_of_Laon-5b24851299, side: Prussia, mention: August von Gneisenau}
    value: Q165519
    source: "Wikipedia, 'Battle of Laon', infobox; Wikidata Q165519"
    note: "Same wrong link (the son, Q135401853)."
    reviewed: {by: historiography-reviewer, "on": 2026-10-18}
  - id: ov-0057
    kind: commander_link
    target: {battle: Battle_of_Ligny-6140805ede, side: Prussia, mention: Graf von Gneisenau}
    value: Q165519
    source: "Wikipedia, 'Battle of Ligny', infobox; Wikidata Q165519"
    note: "Corpus-local twin of the same man."
    reviewed: {by: historiography-reviewer, "on": 2026-10-18}
  - id: ov-0058
    kind: commander_link
    target: {battle: Battle_of_Ligny-6140805ede, side: Prussia, mention: Graf von Zieten}
    value: Q67913
    source: "Wikipedia, 'Battle of Ligny', order of battle (I Corps, Zieten); Wikidata Q67913 (Hans Ernst Karl von Zieten, 1770-1848)"
    note: "Corpus-local; the same man is Q67913 at Dresden."
    reviewed: {by: historiography-reviewer, "on": 2026-10-18}
  - id: ov-0059
    kind: commander_link
    target: {battle: Battle_of_Waterloo-059b45c1f1, side: United Kingdom of Great Britain and Ireland, mention: Graf von Zieten}
    value: Q67913
    source: "Wikipedia, 'Battle of Waterloo', infobox (Prussian I Corps); Wikidata Q67913"
    note: "Companion to ov-0058."
    reviewed: {by: historiography-reviewer, "on": 2026-10-18}
  - id: ov-0060
    kind: commander_link
    target: {battle: Battle_of_Bautzen_1813-49d36886e7, side: Russian Empire, mention: Friedrich von Kleist}
    value: Q213784
    source: "Wikipedia, 'Battle of Bautzen (1813)', infobox; Wikidata Q213784 (Friedrich Graf Kleist von Nollendorf, 1762-1823)"
    note: "Linked to Q58739712 (1746-1820), a namesake. Kleist von Nollendorf is already Q213784 at Dresden."
    reviewed: {by: historiography-reviewer, "on": 2026-10-18}
  - id: ov-0061
    kind: commander_link
    target: {battle: Battle_of_Vauchamps-41896f739a, side: Prussia, mention: Friedrich von Kleist}
    value: Q213784
    source: "Wikipedia, 'Battle of Vauchamps', infobox; Wikidata Q213784"
    note: "Companion to ov-0060; Kleist's II Corps at Vauchamps."
    reviewed: {by: historiography-reviewer, "on": 2026-10-18}
  - id: ov-0062
    kind: commander_link
    target: {battle: Battle_of_Dresden-471cbcb35c, side: Austrian Empire, mention: Ivan Dibich}
    value: Q703307
    source: "Wikipedia, 'Battle of Dresden', infobox; Wikidata Q703307 (Hans Karl von Diebitsch, 1785-1831)"
    note: "Linked to Q15065850, his father Hans Ehrenfried (1738-1822). 'Ivan Dibich' is the Russian form of Hans Karl."
    reviewed: {by: historiography-reviewer, "on": 2026-10-18}
  - id: ov-0063
    kind: commander_link
    target: {battle: Battle_of_Borodino-8c0686ddbc, side: Russian Empire, mention: Fyodor Uvarov}
    value: Q3072696
    source: "Wikipedia, 'Battle of Borodino' (Uvarov's cavalry raid); Wikidata Q3072696 (Fyodor Petrovich Uvarov, general of cavalry)"
    note: "Linked to Q15068544 (1786-1845), a namesake; Q3072696 is already linked at Krasnoi."
    reviewed: {by: historiography-reviewer, "on": 2026-10-18}
  - id: ov-0064
    kind: commander_link
    target: {battle: Battle_of_Friedland-d4d7d40b1e, side: Russian Empire, mention: Fyodor Uvarov}
    value: Q3072696
    source: "Wikipedia, 'Battle of Friedland', infobox; Wikidata Q3072696"
    note: "Companion to ov-0063."
    reviewed: {by: historiography-reviewer, "on": 2026-10-18}
  - id: ov-0065
    kind: commander_link
    target: {battle: Battle_of_Marengo-811cf0c0d7, side: France, mention: Louis Berthier}
    value: Q155734
    source: "Wikipedia, 'Battle of Marengo', infobox (Berthier, nominal commander of the Army of the Reserve); Wikidata Q155734"
    note: "Linked to Q140078072 (Louis de Berthier de Bizy, an undated stub), splitting the marshal from his other four battles."
    reviewed: {by: historiography-reviewer, "on": 2026-10-18}
  - id: ov-0066
    kind: commander_link
    target: {battle: Battle_of_Somosierra-3d88fbab34, side: France, mention: Joseph I}
    value: Q7726
    source: "Wikipedia, 'Battle of Somosierra', infobox; Wikidata Q7726 (Joseph Bonaparte, King of Spain 1808-13)"
    note: "Linked to Q17631178, an Idumean nobleman (Herod's uncle); the item has no dates, so the date gate passed it."
    reviewed: {by: historiography-reviewer, "on": 2026-10-18}
  - id: ov-0067
    kind: commander_link
    target: {battle: Battle_of_the_Granicus-523fffa4fb, side: Achaemenid Empire, mention: Dimitriadis}
    value: none
    source: "Wikipedia, 'Battle of the Granicus', infobox (Persian commanders); Wikidata Q138847466 (a Greek futures-studies consultant)"
    note: "Not a Persian commander; almost certainly a citation author parsed as a name. Unlinked."
    reviewed: {by: historiography-reviewer, "on": 2026-10-18}
  - id: ov-0068
    kind: commander_link
    target: {battle: Siege_of_Toulon_1793-b5c04ccbb0, side: French First Republic, mention: Jean de La Poype}
    value: Q1685527
    source: "Wikipedia, 'Siege of Toulon (1793)', infobox; Wikidata Q1685527 (Jean Francois Cornu de La Poype, 1758-1851)"
    note: "Linked to Q108742379, a man who lived after 1450."
    reviewed: {by: historiography-reviewer, "on": 2026-10-18}
  - id: ov-0069
    kind: commander_link
    target: {battle: Siege_of_Toulon_1793-b5c04ccbb0, side: French First Republic, mention: Jean Carteaux}
    value: Q573519
    source: "Wikipedia, 'Jean Francois Carteaux' (painter turned general; commanded at Toulon until November 1793); Wikidata Q573519"
    note: "Corpus-local. The Wikidata description says 'painter', which is correct: Carteaux was a painter before the Revolution."
    reviewed: {by: historiography-reviewer, "on": 2026-10-18}
  - id: ov-0070
    kind: commander_link
    target: {battle: Siege_of_Toulon_1793-b5c04ccbb0, side: French Royalists, mention: Samuel Hood}
    value: Q332623
    source: "Wikipedia, 'Siege of Toulon (1793)', infobox; Wikidata Q332623 (Samuel Hood, 1st Viscount Hood)"
    note: "Corpus-local."
    reviewed: {by: historiography-reviewer, "on": 2026-10-18}
  - id: ov-0071
    kind: commander_link
    target: {battle: Battle_of_Arcole-794209f261, side: French Republic, mention: "Józef Sułkowski"}
    value: Q946928
    source: "Wikipedia, 'Józef Sułkowski' (Bonaparte's aide-de-camp at Arcole, 1773-1798); Wikidata Q946928"
    note: "Linked to Q140763329, an undated stub. Check the slug: the battle row is named 'Battle of the Bridge of Arcole'."
    reviewed: {by: historiography-reviewer, "on": 2026-10-18}
  - id: ov-0072
    kind: commander_link
    target: {battle: Battle_of_Jena_Auerstedt-7e1600f782, side: Prussia, mention: Friedrich Wilhelm von Schmettau}
    value: Q121684
    source: "Wikipedia, 'Battle of Jena-Auerstedt' (Schmettau's division at Auerstedt, mortally wounded); Wikidata Q121684 (1743-1806)"
    note: "Linked to Q136281732 (1661-1732)."
    reviewed: {by: historiography-reviewer, "on": 2026-10-18}
  - id: ov-0073
    kind: commander_link
    target: {battle: Battle_of_Jena_Auerstedt-7e1600f782, side: Prussia, mention: Prince of Hohenlohe}
    value: Q696782
    source: "Wikipedia, 'Battle of Jena-Auerstedt', infobox; Wikidata Q696782 (Friedrich Ludwig, Prince of Hohenlohe-Ingelfingen, 1746-1818)"
    note: "Corpus-local; he is a rank-0 field commander on this side."
    reviewed: {by: historiography-reviewer, "on": 2026-10-18}
  - id: ov-0074
    kind: commander_link
    target: {battle: Battle_of_Jena_Auerstedt-7e1600f782, side: Prussia, mention: Count of Kalckreuth}
    value: Q73253
    source: "Wikipedia, 'Battle of Jena-Auerstedt', infobox (Kalckreuth, reserve at Auerstedt); Wikidata Q73253 (Friedrich Adolf von Kalckreuth)"
    note: "Corpus-local."
    reviewed: {by: historiography-reviewer, "on": 2026-10-18}
  - id: ov-0075
    kind: commander_link
    target: {battle: Battle_of_Kursk-045b980c2f, side: Soviet Union, mention: Pavel Belov}
    value: Q4082219
    source: "Wikipedia, 'Battle of Kursk', infobox; Wikidata Q4082219 (Pavel Alexeyevich Belov, Soviet general, 1897-1962)"
    note: "Linked to Q89565157, a 'researcher'."
    reviewed: {by: historiography-reviewer, "on": 2026-10-18}
  - id: ov-0076
    kind: commander_link
    target: {battle: Battle_of_Okpo-4d21e18160, side: Kingdom of Joseon, mention: "Wŏn Kyun"}
    value: Q379659
    source: "Wikipedia, 'Battle of Okpo', infobox; Wikidata Q379659 (Won Gyun, 1540-1597)"
    note: "Headline general (#77) left corpus-local. Link all three mentions (ov-0077, ov-0078)."
    reviewed: {by: historiography-reviewer, "on": 2026-10-18}
  - id: ov-0077
    kind: commander_link
    target: {battle: Battle_of_Hansan_Island-bb4f6a36fc, side: Joseon navy, mention: "Wŏn Kyun"}
    value: Q379659
    source: "Wikipedia, 'Battle of Hansan Island', infobox; Wikidata Q379659"
    note: "Companion to ov-0076."
    reviewed: {by: historiography-reviewer, "on": 2026-10-18}
  - id: ov-0078
    kind: commander_link
    target: {battle: Battle_of_Busan-6dedfbac4a, side: Joseon navy, mention: "Wŏn Kyun"}
    value: Q379659
    source: "Wikipedia, 'Battle of Busan (1592)', infobox; Wikidata Q379659"
    note: "Companion to ov-0076."
    reviewed: {by: historiography-reviewer, "on": 2026-10-18}
  - id: ov-0079
    kind: commander_link
    target: {battle: Battle_of_Austerlitz-36474a0291, side: Russian Empire, mention: Francis I}
    value: Q130834
    source: "Wikipedia, 'Battle of Austerlitz', infobox; Wikidata Q130834 (Francis II, Holy Roman Emperor / Francis I of Austria)"
    note: "Headline general (#303) left corpus-local. Link all three mentions (ov-0080, ov-0081)."
    reviewed: {by: historiography-reviewer, "on": 2026-10-18}
  - id: ov-0080
    kind: commander_link
    target: {battle: Battle_of_Dresden-471cbcb35c, side: Austrian Empire, mention: Francis I}
    value: Q130834
    source: "Wikipedia, 'Battle of Dresden', infobox; Wikidata Q130834"
    note: "Companion to ov-0079."
    reviewed: {by: historiography-reviewer, "on": 2026-10-18}
  - id: ov-0081
    kind: commander_link
    target: {battle: Battle_of_Leipzig-d1490afa75, side: Russian Empire, mention: Francis I}
    value: Q130834
    source: "Wikipedia, 'Battle of Leipzig', infobox; Wikidata Q130834"
    note: "Companion to ov-0079."
    reviewed: {by: historiography-reviewer, "on": 2026-10-18}
  # Optional, campaign only (no model effect): 1948_Arab_Israeli_War-37bac64bef / Arab League:
  #   "Farouk" -> Q213001 (currently a Moroccan painter), "Abdallah I" -> Q192780 (currently a sultan of Anjouan).

  # --- side_polity ---
  - id: ov-0082
    kind: side_polity
    target: {battle: Siege_of_Tobruk-c774b6765b, side: Germany}
    value: Axis
    source: "Wikipedia, 'Siege of Tobruk', infobox (Belligerents: Germany, Italy; Commanders: Rommel, Gambara)"
    note: "The same Panzerarmee consistency as ov-0014/0017/0018/0022/0023; the side carries Gambara."
    reviewed: {by: historiography-reviewer, "on": 2026-10-18}
  - id: ov-0083
    kind: side_polity
    target: {battle: Siege_of_Tobruk-c774b6765b, side: Australia}
    value: United Kingdom
    source: "Wikipedia, 'Siege of Tobruk', infobox (garrison: 9th Australian Division, then British 70th Division under Scobie, Polish and Czechoslovak units, under Western Desert Force / Eighth Army)"
    note: "A Commonwealth garrison, labelled by its first unit. The corpus labels the Eighth Army 'United Kingdom' at Gazala and El Alamein."
    reviewed: {by: historiography-reviewer, "on": 2026-10-18}
  - id: ov-0084
    kind: side_polity
    target: {battle: Battle_of_Kasserine_Pass-72c07c66a7, side: Nazi Germany}
    value: Axis
    source: "Wikipedia, 'Battle of Kasserine Pass', infobox (Belligerents: Germany, Italy; Commanders include Calvi di Bergolo, 131st Centauro Armoured Division)"
    note: "Rommel's German-Italian force, consistent with his other North African sides."
    reviewed: {by: historiography-reviewer, "on": 2026-10-18}
  - id: ov-0085
    kind: side_polity
    target: {battle: Siege_of_Toulon_1793-b5c04ccbb0, side: French Royalists}
    value: First Coalition
    source: "Wikipedia, 'Siege of Toulon (1793)', infobox (Belligerents: Great Britain, Spain, Naples, Sardinia, French Royalists)"
    note: "All seven commanders on this side are British or Spanish (Hood, Langara, O'Hara, Mulgrave, Smith, Dundas, Gravina)."
    reviewed: {by: historiography-reviewer, "on": 2026-10-18}
  - id: ov-0086
    kind: side_polity
    target: {battle: Battle_of_Heraclea-3a35306e60, side: Epiru}
    value: Epirus
    source: "Wikipedia, 'Battle of Heraclea', infobox"
    note: "A truncated label. Pyrrhus' three sides read 'Epiru', 'Epirotes' and 'Epirus'; better fixed by the canonicalisation map (M5), in which case drop this entry."
    reviewed: {by: historiography-reviewer, "on": 2026-10-18}

  # --- attribution (optional; contested) ---
  - id: ov-0087
    kind: attribution
    target: {battle: Battle_of_t_rboholy-802739c6af, side: Prussia}
    value:
      - {general: Q33550, role: field_commander, rank: 0, weight: 0.45}
      - {general: Q70874, role: subordinate, rank: 1, weight: 0.45, reports_to: Q33550}
      - {general: Q50993, role: subordinate, rank: 1, weight: 0.10, reports_to: Q33550}
    source: "Wikipedia, 'Battle of Prague (1757)', sections 'Battle' and 'Aftermath' (Frederick 'seemingly deferred to Schwerin'; the victory owed to 'Schwerin's leadership and the independent initiative of regimental commanders'); Christopher Duffy, Frederick the Great: A Military Life (1985), ch. 5"
    note: "Classify made Frederick supreme_commander rank 1, but he commanded the army in person on the field. This keeps the near-equal split the article supports and fixes the role. The ranking effect is under 0.05 of one victory."
    reviewed: {by: historiography-reviewer, "on": 2026-10-18}
  - id: ov-0088
    kind: attribution
    target: {battle: Siege_of_Toulon_1793-b5c04ccbb0, side: French First Republic}
    value:
      - {general: Q723058, role: field_commander, rank: 0, weight: 0.55}
      - {general: Q1685527, role: subordinate, rank: 1, weight: 0.20, reports_to: Q723058}
      - {general: Q517, role: subordinate, rank: 1, weight: 0.15, reports_to: Q723058}
      - {general: Q573519, role: field_commander, rank: 1, weight: 0.10}
    source: "David G. Chandler, The Campaigns of Napoleon (1966), pp. 22-29; Wikipedia, 'Siege of Toulon (1793)', sections 'Siege' and 'Capture of Fort Mulgrave'"
    note: "Optional. Dugommier commanded the final assault (17 December). Carteaux was relieved in November, so he is not co-rank-0. Bonaparte commanded the artillery and proposed the l'Eguillette plan; 0.15 rather than 0.30. Depends on ov-0068/0069 landing first."
    reviewed: {by: historiography-reviewer, "on": 2026-10-18}
```

Not expressible in the current schema; these need a design call:

- Glendale, Union side: unattribute it. McClellan absent, "named no second-in-command", Wikipedia, 'Battle of Glendale'.
- Gaines's Mill, Union side: add Fitz John Porter, Q1421282, as field commander.
- Jena-Auerstedt: split it into two battles, or set side troop totals manually (Prussia about 110k).
- Hazir, Chains, Ullais, Beneventum: correct the troops. The extract heuristic in M1 fixes these at source and is preferable to per-row entries.

## 5. The one change that would most improve validity
Fix extract's handling of labelled modern estimates (M1), then have bayesian-model-reviewer look at reconcile's cross-era shrinkage. Today the pipeline both discards the modern correction and fails to de-bias the ancient figure. That falls hardest on the ancient and early-Islamic part of the ranking, which is exactly what critics will test.

## Sources
- [Wikipedia, Battle of Prague (1757)](https://en.wikipedia.org/wiki/Battle_of_Prague_(1757))
- [Wikipedia, Battle of Glendale](https://en.wikipedia.org/wiki/Battle_of_Glendale)
- [Wikipedia, Battle of Hazir](https://en.wikipedia.org/wiki/Battle_of_Hazir)
- [Wikipedia, Battle of Chains](https://en.wikipedia.org/wiki/Battle_of_Chains)
- [Wikipedia, Battle of Ullais](https://en.wikipedia.org/wiki/Battle_of_Ullais)
- [Wikipedia, Battle of Fahl](https://en.wikipedia.org/wiki/Battle_of_Fahl)
- [Wikipedia, Battle of Antietam](https://en.wikipedia.org/wiki/Battle_of_Antietam)
- [Wikipedia, Battle of Gaugamela](https://en.wikipedia.org/wiki/Battle_of_Gaugamela)
- [Wikipedia, Battle of the Yarmuk](https://en.wikipedia.org/wiki/Battle_of_the_Yarmuk)
- Wikidata entities Q135401853, Q138847466, Q140078072, Q17631178, Q58739712, Q15065850, Q15068544, Q89565157, Q108742379, Q136281732, Q140763329 (via Special:EntityData)
