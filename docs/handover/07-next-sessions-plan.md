# Handover §7: plan for the next sessions (written 2026-10-18)

Part of the handover; the index is `handover.md` at the repo root.

State this plan starts from: `main` has Phase 2 (54 overrides) and the refit
to model run 108, reviewed by both reviewers (handover-log §29 historiography,
§30 Bayesian): no blocker, internal reading only. Nothing below has been
started. Each workstream is sized for one fresh session.

User decisions, 2026-10-18:
- **Headline rule stays as it is for now** (`min_battles: 3` in
  `agents/report.yaml`). The reviewer's point (ranks 16-89 are mostly
  subordinates carrying their leader's side performance, §30 MAJOR 1) is
  recorded, not acted on: many generals have few battles and the top 15 look
  right. Disclose it instead (workstream C).
- Impute `mcmc_samples` 4000 (done, 5a0d27a).
- Workstream A (troop numbers) is next. B needs the user's reading first. C
  can run in parallel with A (different files).

Order and dependencies: A changes extract and reconcile, so it ends in a
refit from extract. B's entries also need a refit from extract (side_polity)
or resolve (links). **Do one refit for both**: finish A's code, fold in
whichever B entries the user has approved by then, refit once, review once.
C touches only the report stage and can merge any time; re-run report after.

---

## A. Troop numbers (code; next session)

Evidence: §29 M1 and M3, §30 MAJOR 2. Three separate defects:

1. **Extract drops labelled modern estimates.** `pipeline/extractors/infobox.py`
   (~lines 400-460, the branch classifier that returns `"other"`; ~575 where
   `other` reports become components) tags "N (modern estimate[s])" as
   `branch=other`; `pipeline/reconcilers/load.py:196` drops every non-total
   branch (`excluded_non_total_branch`: 731 reports on the corpus), so the
   ancient figure is the only total. Cases: Hazir 57,856 vs modern ~7k (8.3x),
   Chains 62,760 vs 15-20k, Ullais 59,274 vs ~30k, Beneventum Pyrrhus 14,980 vs
   ~40k (ratio inverts), Persian Gate modern 700-2,000 dropped (the Macedonian
   "17,000 picked fighters" tagged `branch=air`). Khalid #4 and Pyrrhus #14
   rest on these. The regime parser already knows `modern_scholarly`
   (`pipeline/reconcilers/claims.py:76-153`); the gap is the branch, not the
   regime.
2. **Scope is a default.** 1,303 of 1,305 `troop_reports.scope` are `engaged`
   (`INFOBOX_STRENGTH_DEFAULT_SCOPE`); "total", "deployed", "population",
   "including non-combatants", "initial" do not override it
   (`pipeline/extractors/textnorm.py` `resolve_scope`). Bibracte's 368,000 is
   Caesar's migrating population; Jena-Auerstedt's partial Prussian figures
   (46,000 at Jena, 15,000 under Rüchel) are reconciled as rival totals
   (32,858 vs a real ~110k).
3. **Reconcile compresses ratios toward 1 in every era**: totals above 50k
   come out at 0.76x (post-1800, archival: Wagram 172k to 105k, Leipzig 257k
   to 131k, Kursk 1.91M to 533k) and 0.83x (ancient); totals under 10k are
   inflated 2.3-3.9x (Uhud 754 to 3,114). Log ratios shrink by ~0.65-0.75.
   This is a prior question (the shared prior, `_UNLABELLED_WEIGHT = 0.5`,
   §9.6), for `bayesian-model-reviewer` to diagnose before any change.
4. **Eastern Front sides are mostly imputed** (§30 MAJOR 2): 7 of 10 Soviet
   sides have reconcile sd >= reconcile's s0, so impute's weak-side rule
   (`pipeline/imputers/likelihood.py:261-316`) treats them as missing and fills
   them from the Soviet Union polity offset (Stalingrad -0.36 to +1.29). 75 of
   341 estimated sides are discarded this way. Check whether weakness should be
   judged on the residual (theta-regressed) precision.

### A: approved design (2026-10-22)

Diagnosis (bayesian-model-reviewer via Opus, read-only refits of run 84 in
memory; full report in handover-log §31): one cause behind 3 and 4. The
observation noise (sigma ~0.51 under Student-t nu 3.4, effective ~0.8) is
identified only from 31 multi-report sides selected for disagreement, and is
applied to 309 singletons against one corpus-wide t(4) side prior (m0 ~40k,
s0 0.69): singleton slope 0.61, ratio slope 0.60. The corpus contradicts it
(observed between-battle log-ratio variance 0.60 < the 0.92 the noise alone
implies). g_unlabelled is pulled to ~0, so ancient "deflation" is shrinkage,
not inflation correction. Residualising shared error rescues only 12 weak
sides; weakness is an artefact of dividing out a t prior as if Normal
(corr(|report - m0|, sd) = 0.94), so the rule discards the largest and
smallest armies. Residual rule rejected.

User decisions, 2026-10-22:
1. **Extract** (wip/troop-extract): "(modern estimate[s]/est./estimation)"
   figures are `branch=total`; on a side with a usable `modern_scholarly`
   total, reconcile excludes every other total (`excluded_superseded_by_modern`,
   rows kept). `ancient_claim` also matches Muslim/Islamic/Arab/Byzantine/
   Roman/Greek/Persian source/estimate/account/tradition. Scope: "deployed" is
   engaged; "total" is available only beside a deployed/engaged figure in the
   same field; people/population/non-combatants/civilians are a new scope
   `population`, excluded by reconcile (`excluded_population_scope`); within
   a field of phase-dated or "initial" figures the largest is the total, the
   rest components. "N under <Name>" is a component; "picked fighters" are men.
2. **Reconcile noise** (approved as proposed, no explicit ancient term): an
   informative observation-scale prior from the gold set's faithful component
   (LogNormal(log 0.22, 0.3)) for post-1500 rows, a wider one for pre-1500 rows
   (LogNormal(log 0.40, 0.35), a judgement call; sensitivity below);
   sigma_round and sigma_est priors HalfNormal(0.1); Student-t nu floored at 3.
   Consequence accepted: published ancient head counts rise to near reported
   (~0.98x); force ratios are unaffected by that. Sensitivity variant with the
   modern scale centred at 0.35 must be run once and recorded.
   New gates: singleton slope >= 0.85 and spread_ratio >= 0.85 (error);
   within-battle log-ratio slope >= 0.8 (error); post-1800 sides reported
   > 50k geometric ratio in [0.85, 1.15] (warning); PPC of the between-battle
   reported log-ratio variance (warning).
3. **Impute likelihood** (approved): reconcile's neutral fit exports each
   side's likelihood directly (no Gaussian division of a t prior): per side
   with >= 1 point row, the likelihood mean and sd from its point rows at the
   fitted noise (Fisher-information sd for the Student-t), with loadings on the
   shared parameters (g_unlabelled, a_era, offset_bar), in the neutral npz.
   A side is missing only if it has no point row (bound-only or no reports);
   weak count logged by cause; no side with a point report may be weak.
   Gates: kept-pair impute ratio slope >= 0.8; imputed vs report slope >= 0.9.
4. **Jena-Auerstedt: split into two battles** (user's choice over a
   `side_troops` kind). A new extract override kind creates "Battle of Jena"
   and "Battle of Auerstedt" from the one article, routing commanders and
   strength figures by the infobox's "Jena:"/"Auerstedt:" headings and
   "(Jena)"/"(Auerstedt)" labels, with stable synthetic slugs, and retires the
   parent. Cost accepted: the Arsht reference row for this battle no longer
   joins; its 4 sides are re-resolved and re-classified.
5. Phase 2b overrides: none in this refit.

Verification after the refit: both reviewers; Khalid and Pyrrhus movement;
force-ratio compression before (reconcile ratio slope 0.60, impute kept-pair
0.43) and after; Soviet x_diff against reports for historiography-reviewer;
evaluate's `observed_only` variant and beta_force compared.

Jena-Auerstedt needs a design call either way (split into two battles, or a
new `side_troops` override kind); bring options to the user, do not choose.

### Prompt for the new session

> Resume the General WAR project. Read `handover.md` first (CLAUDE.md
> requires it), then `docs/handover/07-next-sessions-plan.md` workstream A,
> and handover-log §29 (M1, M3) and §30 (MAJOR 2). Verify `main` and the
> corpus against handover §1 first (check for leftover pytest processes).
>
> Goal: fix how troop numbers enter the model, in this order.
> 1. Extract: read a "(modern estimate[s])" / "(modern est.)" figure as
>    `branch=total` with the `modern_scholarly` regime, and when a
>    modern-labelled and an ancient-labelled figure share a side, give
>    reconcile the modern one at full weight. Extend `resolve_scope` so
>    "total", "deployed", "population", "including non-combatants" and
>    "initial" set scope. Build test cases from the real strings in §29 M1/M3
>    (Hazir, Chains, Ullais, Beneventum, Persian Gate, Issus, Bibracte,
>    Jena-Auerstedt, Leipzig) before changing code; a Sonnet worker in a
>    worktree (footguns: own data/processed copy, data/raw junction, own
>    `general_war_wt_*` database, remove the junction before
>    `git worktree remove`), then code and DB review.
> 2. Reconcile shrinkage: have `bayesian-model-reviewer` (if the agent type
>    is not registered, a general-purpose Opus agent told to follow
>    `.claude/agents/bayesian-model-reviewer.md`) diagnose why totals above
>    50k come out at 0.76x and under 10k at 2-4x in every era, using the
>    run-84 artefacts, and propose a fix. Show me the diagnosis and the
>    proposed fix before anything changes; it changes what the numbers mean.
> 3. Impute's weak-side rule: same reviewer, same rule: is weakness judged on
>    a covariance inflated by the propagated shared error? Show me before
>    changing.
> 4. Jena-Auerstedt: bring me the options (split the battle, or a
>    `side_troops` override kind) with their costs; do not choose.
> Then, with my approval of the fixes and of any Phase 2b entries I have
> signed off (plan workstream B), refit from extract in stage order, and run
> both reviewers before reading the ranking. Report whether Khalid and
> Pyrrhus move, and the force-ratio compression before and after.
>
> Rules: re-running a stage means re-running everything after it, in order.
> Subagents must not spawn subagents; when a command is backgrounded they end
> their turn and wait. No shell sleep/until loops. Brief llm-batch-workers with
> absolute paths; relaunch a stalled one. Before any /process-llm-batch over
> ~10 Sonnet chunks, ask me for my usage %. Never merge into main while a
> suite runs on the main checkout. The laptop may hibernate: re-run, don't
> trust a partial log. Commit to local branches and main as needed; do not
> push. Update handover.md before the session ends.

---

## B. Phase 2b overrides (user reads first)

The 34 drafted entries are in handover-log §29 section 4 (`ov-0055` to
`ov-0088`), in the `config/overrides.yaml` schema, each with a citation.

### For the user: how to review them

Open `docs/handover-log/29-phase2-refit-historiography-review-2026-10-18.md`,
section 4. For each entry, check three things: the target (battle, side,
mention) is the row you mean, the value is the right person or label, and the
`source` is a citation you accept. Groups:

| Ids | What | Your call |
|---|---|---|
| ov-0055..0057 | Gneisenau (three mentions, linked to his son or split) to Q165519 | Accept / reject |
| ov-0058..0059 | Zieten at Ligny and Waterloo to Q67913 | Accept / reject |
| ov-0060..0061 | Kleist at Bautzen and Vauchamps to Q213784 | Accept / reject |
| ov-0062 | Diebitsch at Dresden (linked to his father) to Q703307 | Accept / reject |
| ov-0063..0064 | Uvarov at Borodino and Friedland to Q3072696 | Accept / reject |
| ov-0065 | Berthier at Marengo to Q155734 | Accept / reject |
| ov-0066 | "Joseph I" at Somosierra (an Idumean noble) to Joseph Bonaparte Q7726 | Accept / reject |
| ov-0067 | "Dimitriadis" at Granicus (a futures consultant) unlinked | Accept / reject |
| ov-0068..0070 | Toulon 1793: La Poype, Carteaux, Hood | Accept / reject |
| ov-0071 | Sułkowski at Arcole to Q946928 | Accept / reject |
| ov-0072..0074 | Jena-Auerstedt Prussians: Schmettau, Hohenlohe, Kalckreuth | Accept / reject |
| ov-0075 | Belov at Kursk (linked to a researcher) to Q4082219 | Accept / reject |
| ov-0076..0078 | Wŏn Kyun (unlinked headline general) at Okpo, Hansan, Busan to Q379659 | Accept / reject |
| ov-0079..0081 | Francis I (unlinked headline general) at Austerlitz, Dresden, Leipzig to Q130834 | Accept / reject |
| ov-0082..0083 | Tobruk: Germany to "Axis", Australia to "United Kingdom" | **Decide**: is "United Kingdom" the right level for an Australian-led garrison? |
| ov-0084 | Kasserine: Nazi Germany to "Axis" | **Decide**: on 2026-10-17 the first draft kept Kasserine "Nazi Germany" (a German-led attack with one Italian division); this reverses it |
| ov-0085 | Toulon: French Royalists to "First Coalition" | Accept / reject |
| ov-0086 | Heraclea: typo "Epiru" to "Epirus" | Accept |
| ov-0087 | Prague 1757 Prussia: Frederick field_commander 0.45, Schwerin subordinate 0.45, Henry 0.10 (optional) | **Decide**: Frederick commanded in person; the LLM made Schwerin rank 0 |
| ov-0088 | Toulon: Napoleon lowered from 0.30 to 0.15 (optional) | **Decide** |

Tell the session which ids to accept, reject or change. Anything you have not
signed off stays out.

### For the session building them in

1. Copy only the approved entries into `config/overrides.yaml`; validate with
   `python -c "from pipeline.overrides import load_overrides; print(len(load_overrides(required=True)))"`.
   Keep `"on":` quoted in `reviewed`.
2. **Twin check before the refit** (07-phase2-overrides-design.md): for every
   person an entry links, list every mention of that person in
   `data/processed/commanders_raw.jsonl`. Any mention not covered stays on
   automatic resolution and, if corpus-local, becomes a twin of the Q-id on the
   second resolve run (`no_duplicate_generals` fails). Add a link entry per
   uncovered mention (with the user's OK) before refitting.
3. **Attribution sets**: an `attribution` entry must list exactly the side's
   commanders after the link entries apply (ov-0088 Toulon depends on
   ov-0068/69; ov-0087 Prague on the current links). Check against the DB the
   way the 2026-10-17 drafter's `check.py` did (its logic is described in
   handover-log §29 section 1).
4. Every side whose commander set changes is reset by resolve and re-requested
   by classify (about 20 sides here: two or three Sonnet chunks). Answer the
   batch, spot-check three answers, then continue the refit.
5. Not expressible in the current schema (design calls for the user):
   McClellan absent at Glendale and Gaines's Mill (a commander-absent kind);
   Darlan on both sides at Torch and Aegus/Roscillus at Dyrrhachium (a
   wrong-side unlink is a `commander_link: none`, but check the mention text);
   Jena-Auerstedt troops (workstream A).

---

## C. Report disclosures and a data-coverage view (code; can run beside A)

### Text the report must carry (missing from report.md, §29 M2)

Compute each from the database at report time, never hard-code:
- **rule_single**: count of sides where the one infobox-named commander gets
  1.0 (138 on run 108), and that a few were absent (McClellan, Glendale and
  Gaines's Mill).
- **Components**: the comparison graph's components; the share of the top 50
  outside the largest (41 of 50 on run 108); that cross-component order rests
  on the shared skill prior.
- **Unanswered passage pass**: troop scope and prose evidence come from
  infoboxes only (the 14,705 requests are the user's call, 2026-10-14).
- **WAR is shared side performance**: WAR splits a side's expected surplus by
  attribution weight, so a subordinate inherits the leader's result; show
  effective battles and skill-interval width beside rank (the user's
  2026-10-18 call: disclose, keep the headline rule).
- **Corpus shape and fame bias**: battles by century with empty spans named
  (700-1199, 1300-1499, the 1600s have none); share of headline generals by
  era (42 of 89 Revolutionary or Napoleonic); win rate of rank-0 commanders by
  battle count (73.6% at 5+ battles, 30.7% at 1-2).
- **Troop provenance**: share of model sides observed, imputed, or treated as
  weak (75 of 341 estimated sides), by era.
- **Rank numbering**: number headline rows 1..89 among headline generals, not
  positions among all 732 (the 86th headline row reads "rank 723"). Keep the
  global position as a separate column if useful.

### How to visualise them

Add one figure set, "Data coverage and caveats", built in
`pipeline/reporters/figures.py` (matplotlib PNGs for report.md, beside the
existing forest, calibration, CI-width, rank-stability and skill-vs-battles
plots) and a matching tab in the explorer (`pipeline/reporters/explorer.py`,
plotly, embedded data, same offline single file). Six panels, each from a
query the report already has access to:

1. **Corpus timeline**: battles per 25-year bin across -500..2000, bars
   coloured by era, empty spans shaded and labelled. Answers "what is
   missing" at a glance.
2. **Comparison components**: a bar per connected component (size =
   generals), the largest highlighted; each top-15 general drawn as a marker
   on its component's bar. Shows that cross-era order is assumed.
3. **Credit source**: for headline generals, a stacked bar per general of
   their credited weight by attribution method (llm, rule_single,
   default_split, manual), so a rank built on rule_single or default splits
   is visible.
4. **Information vs rank**: scatter of rank (x) against effective battles and
   skill contraction 1 - (sd/sigma)^2 (two y axes or two panels), with the
   reviewer's 0.25 contraction line. Shows where rank stops resting on a
   general's own data (rank 16 on run 108) without changing the headline rule.
5. **Troop provenance**: per era, stacked share of model sides observed /
   weak-as-missing / imputed; plus a small multiple of reported vs reconciled
   total (log-log, by era) so the compression in workstream A is visible
   before and after its fix.
6. **Fame bias**: win rate of rank-0 commanders by credited-battle count bin
   (1, 2, 3-4, 5-9, 10+), with counts on the bars.

Gate: a report check that each figure was written and that every number in
the disclosure text equals the query behind its figure (same pattern as
`export_matches_db`). Tests on the fixture database. This is a Sonnet task in
a worktree; the explorer must stay one offline file.

---

## D. TODO items (planned, not started)

From `docs/todo/` after the 2026-10-18 triage (details in each file):
- **User calls, unchanged**: `.env` holds only an unused key; CI never checked
  on `main`; `pre-commit install`; live `llm_mode: api` smoke test; the
  extract passage pass and web_secondary cost decisions; the calibration
  gate's arm; the gold sets.
- **Planned fixes** (each a session task, none started):
  - End-to-end fixture test crawl to model (00-setup-infra): one fixture
    battle set through `run_pipeline`, asserting each stage's gates.
  - Resolver age-window heuristic and undated-stub rejection (03-resolve):
    the date gate only checks a candidate was alive (Gneisenau's son, Joseph
    I, Dimitriadis, La Poype, §29 M4). Fixture cases from §29 M4.
  - Polity canonicalisation map (06-impute): the polity key is the raw string,
    so the Confederacy is 4 levels, Napoleonic France 7, the Habsburgs 4
    (§29 M5). A reviewed `config/polity_map.yaml` applied in extract, plus a
    gate for unmapped strings. Model-meaning change: bayesian-model-reviewer.
  - A 1914-1945 era bin or a theatre effect so the polity effect stops
    carrying a total-war scale effect (§30 MAJOR 2); decide with A's
    reconcile diagnosis.
  - PPC pooled over rounds with an orientation-free statistic (08-evaluate,
    §30 item 3); compare the circularity gap to the combined sd (§30 item 6);
    record the worst parameter and tail ESS in impute and model diagnostics
    (§30 Q3).
  - The `agrippa_above_augustus` named check is no longer independent (Actium
    is hand-set, §30 item 9): replace it with a check on an unoverridden
    battle.
  - Range parsing "40-45,000" read as 40 (§29 MINOR); Kasserine's name typo
    "Passe"; 34 rows where hierarchy_rank disagrees with reports_to.
- **Tooling**: `.claude/agents/bayesian-model-reviewer.md` was not registered
  as an agent type on 2026-10-18 (historiography-reviewer and llm-batch-worker
  were). Check its frontmatter; until fixed, run it through a general-purpose
  agent told to follow the file.
