# §42. Refit 7 reviews and the ACW-bias diagnosis (2026-11-10)

## 1. State

- Nothing committed since §41. main 9ee03a4, w-c3seed 5199c34, agent-a512e2a4e3d0264a4 (data root) 519e60d and agent-adf22892058598c5e ee1c9de all still hold uncommitted work.
- w-c3seed: records.py lone_branch docstring now says "per-side offset (``d_lone_branch``)" (the §41 nit; it was at line 162, not 465). ruff clean.
- Data root (agent-a512e2a4e3d0264a4) code-reviewed by Sonnet: no CRITICAL/HIGH. Unit 2565 (not 2578 as §41 claimed), mypy 162 clean, ruff errors only in `docs/handover-log/37-scripts/`. Follow-ups sent to a Haiku worker: extract.py:1099 hint names the root, `report_under_root` rejects `..`/absolute, comment that an explicit `report_path` outranks `report_under_root`, `UNIMPLEMENTED_METHODS` strings, tests for an absolute root and `report_under_root`. Result recorded in handover §000.
- Merge follow-up: w-c3seed's two new report gates (`disclosures_match_db`, `coverage_figures_written`) use `report: data/processed/report/manifest.json`; convert to `report_under_root: processed/report/manifest.json` after merging. Expected conflicts: agents/report.yaml (certain), agents/reconcile.yaml (possible).
- Reviewer scratch scripts (session scratchpad, ephemeral): `bmr/` (build.py, t1.py, t2_5.py, proposals.py) and `hist/` (acw.py, cwsac.py, nest.py, prop.py). Both read the corpus3 URL from refit7.sh and never print it.

## 2. bayesian-model-reviewer on reconcile run 29 / model run 52

No BLOCKER; estimates fit to write. Floor exemption (Fort Sumter 85 -> 100, 64-170), lone-branch promotion (62 sides, prior-only, contraction 0.09) and the B1 fix all behave as designed. Model run 52: rhat 1.005, ESS 1,382, 0 divergences, beta_force 0.66-0.90.

- M1 (carried): rule_single. 74% of ACW top-50 xWAR is rule_single (non-ACW 53%); every rule_single row is `field_commander`, so theatre-level sole commanders cannot be separated after the fact.
- M2 (carried): component skill levels are prior-set (means about 0 in every component; ACW is its own component, 1,308 of 1,350 entries).
- M3 (new): total_war turns corpus density into rank. Each component sits near 0, s_rep is one global value (-0.374) below every component centre, so a typical general earns positive xWAR on 80-89% of entries and total WAR scales with credited volume.
- MINOR: the lone-branch offset lifts Brice's Cross Roads CSA 3,500 cavalry to about 6,740 (wrong for an all-cavalry force; per-branch prior mean, cavalry-only near 0). reports_honoured checks 23 sides. FMI 0.33 on one top general.

## 3. historiography-reviewer on the ranking

Fix verification: Mesilla unlink, Fort Sumter 85, floor exemption, ov-0610..0614 campaign rows out, Fredericksburg date, Fort Wagner field all PASS. Brice's promotion PASS in mechanism, FAIL in value (offset, as above).

- Forrest (6): method artefact (74% of WAR from actions under 10k total, 94% rule_single, 81% outside CWSAC A/B; top rows include defeats).
- Beauregard (10): partly data error. Chester Station (fought by Ransom) and Swift Creek (Johnson/Hagood, reviewer from memory, unverified) credit him at weight 1; Charleston Harbor and Second Fort Sumter are department-level credit.
- Burnside (19): partly defensible; Fort Sanders double counts Knoxville; Antietam still missing.

MAJOR data issues: nested double counts (Siege of Petersburg + 8 sub-battles; Knoxville/Fort Sanders; Berlin/Seelow/Halbe; Bulge/Elsenborn/Heartbreak Crossroads; Hürtgen/Aachen; Gazala/Bir Hakeim/Tobruk; 55 parent/child pairs in all); Actium side 125 "3,000 archers" taken as total (infobox 16,000 marines + 3,000 archers; hurts Agrippa); about 20-25 campaign/operation rows still credited (Manassas Station Operations, Joint Expedition Against Franklin, Van Buren raid, Ambiorix's Revolt, Tikhvin, Barvenkovo-Lozovaya, Melitopol, Polesskoe, Lwów Uprising, Moonsund, Battle of Britain, St. Lawrence, Ruhr/Heiligenbeil/Kholm pockets). MINOR: 209 credited battles with the 1 January dummy date; Second Fort Sumter typed naval; "Capt. Cleaver" junk name; Thomas under-credited (rank 101).

## 4. The ACW finding (both reviewers agree)

- Not a Confederate bias: 11 of the 17 ACW top-50 generals are Union.
- Not outnumbered-win credit: ACW pays the least xWAR per outnumbered win (0.179 vs Napoleonic 0.266) and has the least imputation; its force slope matches the pooled beta_force.
- The model rates ACW generals lowest per battle (mean skill 0.21 vs Napoleonic 0.59). By skill_mean ACW is 11 of the top 50; Lee falls 12 -> 80.
- Mechanism: density x volume x concentrated credit x prior-centred components. ACW has 101-115 credited battles a year vs about 21-28 Napoleonic and WW2; median total engaged 8.9k (Napoleonic 24k, WW2 41k); 69% of ACW rows are below CWSAC A/B; 44% rule_single; 1.5 commanders per side vs 2.7 in WW2.
- WW2 (1-3 of top 50 under every option) is low because credit is split (mean weight 0.37, front commanders at 0.1-0.3), not because rows are operations. Aggregation cannot fix it.

## 5. Proposals (post-hoc reweights of run-52 xWAR, not refits)

| Option | ACW top 50 | WW2 | Forrest | Beauregard | Burnside | Lee | Grant | Zhukov | Montgomery | Agrippa |
|---|---|---|---|---|---|---|---|---|---|---|
| Current | 17 | 1 | 6 | 10 | 19 | 12 | 25 | 114 | 42 | 60 |
| BMR C+D: density weight min(1, sqrt(29/d)) in the WAR sum only + rule_single x0.5 | 9 | 3 | 18 | 21 | 37 | 26 | 34 | 65 | 29 | 56 |
| Hist E: size tiers (1 / 0.5 / 0.2 at 20k, 5k) + rule_single 0.6 + collapse sub-actions, likelihood and sum | 12 | 3 | 50 | 17 | 26 | 10 | 15 | 68 | 29 | 86 (Actium error) |
| Size weight alone | 16 | 1 | 14 | 13 | 18 | 10 | 26 | 95 | 38 | 74 |
| Headline on skill_mean | 11 | 2 | 23 (skill) | | | 80 | 59 | | | |

C+D side effect: the Napoleonic share of the top 50 rises to 46% (battle share 32%). E caveats: naval tier must be measured in ships (else Yi Sun-sin 15 -> 49); weighting the likelihood throws away evidence (the model reviewer's objection). Both say the cross-era claim is not identified (M2) and the honest headline is within-war/component rank, or a global table labelled as volume-weighted.

## 6. Decision

User decisions (2026-11-10), all four as recommended:

1. **Aggregation: density weighting.** Each battle's contribution to total_war is multiplied by `min(1, (d_ref / d) ** power)`, where `d` is the number of included battles per year within +-1 year of the battle, `d_ref` = 29 (75th percentile of d outside the ACW in run 52) and `power` = 0.5. The likelihood stays unweighted (a small battle is still evidence about skill); per-entry xWAR is unchanged and stored alongside its aggregation weight. Sensitivity variants: unweighted, d_ref and power +-. Code (BMR outline): `pipeline/modelers/design.py` (density, weight per battle), `pipeline/modelers/war.py` (weighted total), `config/model_default.yaml` (`war_aggregation: {density_ref, power, window}`), outputs/store plus an Alembic migration adding `battle_war_details.aggregation_weight` so `war_decomposition_reconciles` checks the weighted sum, `agents/model.yaml` WAR definition, report and disclosures, evaluate variants. Watch: Napoleonic share of the top 50 rises to about 46% (32% of battles); historiography pass on the new top 50.
2. **Sole-listed credit: classify fix plus variant.** Re-classify rule_single rows so weight is cut only where there is no evidence of tactical command (department/theatre commanders), sole field commanders keep 1.0. Needs a classify prompt/rule change and an LLM batch (about 1,000+ rows; ask usage % first). Until then, a blanket rule_single x0.5 evaluate variant.
3. **Report: global plus within-war.** Keep the global table (density-weighted, labelled) and add a within-war/within-component rank column and explorer view.
4. **Nesting: keep sub-battles.** Where a parent battle's sub-battles are credited, retype the parent as a campaign (out of the WAR model), as ov-0610..0614 did. Drafted as overrides for sign-off.
