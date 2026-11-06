# §38. Corpus3 reconcile run 3, cloud fixes, everything merged to main (2026-11-03 night)

Steps 1-5 of the §37 plan ran. Reconcile run 3 nearly passes: the contamination fixes worked. Two gates still fail, and the cause found is one bug plus weak identification. The model is unchanged.

## 1. Branches and commits (user asked for the commits)

| Commit | What |
|---|---|
| 63c86c3 | extract fix ({{endash}} infobox ranges, "with <Name>" detachment, flag `size=`), on wip/extract-endash |
| e973d14 | mypy narrowing at `pipeline/reconcilers/model.py:763`: `if rotated_era:` became `if era_levels is not None and pre_anchor is not None:`. These are None exactly when rotated_era is false, so behaviour is identical and the model reviewer was skipped |
| 35d9f3f | the 48 approved overrides ov-0500..0524, 0530..0552 (467 entries, they load, no duplicate ids) |
| 5879c07 | merge of wip/extract-endash into wip/corpus3-seed |
| 1e17806 | docs on main (handover, §36, §37, the checkpoint hook) |
| 2808819 | merge of main into wip/corpus3-seed; the user pushed this branch |
| 083ddf3, 5a5b07d, 6fe50e0, 40172bf | cloud session, branch wip/cloud-extract-resolve-fixes (from a bundle): headings, derived totals, wikilink identity, and its log `cloud-extract-resolve-fixes.md` |

**main was fast-forwarded to 40172bf (user: "merge everything into main").** Not pushed at the time of writing. Only wip/noise-exp (the rejected experiment) and wip/reconcile-era-rot (already in, as cherry-pick 288eee5, identical content) are unmerged.
- Verified on main: mypy 161 files clean, ruff clean, one Alembic head d0e1f2a3b4c5, unit 2547 passed 0 failed (`data/logs/main_unit_20261103.log`).
- Not verified on main: any integration test, any corpus run.

**wip/corpus3-seed (w-c3seed) stays at 2808819**, without the cloud commits. All corpus3 runs below used it.

## 2. Verification on wip/corpus3-seed (5879c07 / 2808819)

- Integration `-k "reconcile or prior_check"` on scratch `general_war_erarot_it`: 82 passed, 0 skipped (`w-c3seed/data/logs/c3_it_reconcile_20261103.log`).
- Extract integration (6 files) on new scratch `general_war_endash_it`: 51 passed, 0 skipped.

## 3. Corpus3 runs (DB general_war_corpus3, from w-c3seed)

Backups: `data/backups/general_war_corpus3_20261103_pre_extract3.dump` (before extract) and `..._20261103_pre_reconcile3.dump` (after classify, before reconcile).

1. **extract,resolve.** The extract override gate applied 68/68, including the 22 new troop_strength/side_polity/date entries. Tournai now has two sides (Kingdom of France and Dutch Republic). Outcome answers were 268/268 cached. The 45,250 passage requests stay deferred, as agreed. Resolve deferred 5 new Haiku requests (John Pope at Island No. 10, Felix Steiner at Narva, Minoru Ōta at Okinawa, Philomenus at Tarentum, Fritz Witt at Vevi). They were answered with `/process-llm-batch` and imported (5/5, spot-checked).
2. **resolve again, then classify, before reconcile (order change).** `battle_type` overrides belong to classify (`pipeline/overrides.py:141`), so the 26 retypes only land when classify runs. Reconcile's `exclude_campaigns` reads `battle_type`. Run 2 had the same order, with classify run first.
   - Resolve: 490/490 cached. Gates pass except probable_duplicate_generals 8 (the known warning).
   - Classify: 33/33 overrides. 25 battles are now campaign/manual and Okinawa is amphibious/manual; Kursk and Berlin stay field.
   - Classify deferred `data/llm_requests/classify_20261103T100314Z.jsonl`: 7 command_role requests and 2,488 missingness requests.
   - The 7 role requests were split into `classify_roles_20261103T100314Z.jsonl`, answered by Sonnet and imported 7/7.
   - Flagged needs_review on those 7: Port Royal (Elliott is not in the excerpt), New Mexico Campaign (Lynde is not in the excerpt), Tournai Dutch side (Van Dorth and Van Brackel commanded in sequence, so equal split).
   - **The 2,488 missingness requests are still unanswered.** They must wait until reconcile writes troop estimates.
3. **reconcile run 3** (log `w-c3seed/data/logs/corpus3_reconcile3_20261103.log`, 5005 s). It FAILED, and no troop estimates were written. Casualties converged (rhat 1.007, ESS 2369) and were written.

| Gate | Run 2 | Run 3 | Threshold |
|---|---|---|---|
| troop max_rhat | 1.58 | 1.041 | ≤ 1.05 (pass) |
| troop min_ess_bulk | n/a | **130** | ≥ 400 (FAIL) |
| neutral fit | n/a | rhat 1.031, ESS **116** | FAIL |
| noise_not_leaking | 3.30 | **1.514** | ≤ 1.5 (FAIL) |
| within_battle_ratio_recovers | 0.10 | 0.76 | ≥ 0.5 (pass) |
| wb_matches_reliability | n/a | 0.004 | ≤ 0.1 (pass) |

- Divergences were 0. `excluded_campaign` was 1110 reports (651 in run 2).
- `estimates_populated` 0 and `ci_width_reasonable` NULL follow from nothing being written.
- troop_span_warnings: 1 (Yonan, Joseon, 500 vs 1400).

## 4. Audit of run 3 (§36 script, read-only)

Output: session scratchpad `c3audit/audit_out_run3.txt`; run 2's is `audit_out_run2.txt`.

| | Run 2 | Run 3 |
|---|---|---|
| Troop rows | 2210 | 2123 |
| Sides with 2+ rows | 66 | 35 |
| Spans over 0.5 / 1 / 2 / 3 | 39 / 22 / 10 / 4 | 14 / 2 / 0 / 0 |

- **Bug: a troop_strength override does not replace a `modern_scholarly` row.** Kalka (Principality of Kiev) is the worst side in run 3, span 1.50. It still has the infobox "modern estimates" row (77,460, the geometric mean of 60k-100k) next to ov-0103's override row (17,321). Run 2 had the same. ov-0103 is `kind: troop_strength`, which by design replaces every other total on its side (§37). The likely place is the supersede/replace logic in `pipeline/reconcilers/load.py`. Only Kalka shows it in the multi-row list, but check every troop_strength side.
- **Next tier, drafted but never sourced (§37):** Odesa 1941 (both sides), Dürenstein (both), First Badajoz and Ratisbon. Also Ajnadayn 10k/20k and Yonan.
- Tarutino, Jena, Polotsk, Mariazell and Rovereto are available-vs-engaged pairs that the scope offsets model.
- **Reading:** with 35 identifying sides, the ESS shortfall looks like weak identification, not a structural leak. The Student-t fallback is not needed yet.

## 5. User decision (2026-11-03 night): plan for the next session

1. **(a)** Sonnet diagnoses and fixes the Kalka replacement bug, tests first, then bayesian-model-reviewer.
2. **(b)** Raise reconcile troop `tune`/`draws` for ESS. This is a sampler setting; it does not change what the model means.
3. **(c)** Back up, then re-run reconcile on corpus3. Afterwards: bayesian-model-reviewer, then an `era_effect: none` sensitivity run (pre-1800 shift).
4. Source the four drafted overrides only if (a)+(b) still miss the noise gate (historiography drafts, user approval).
5. Then §36 step 4: classify re-run and the 2,488 missingness requests (ask the user's usage % first).

Where to work: make (a) and (b) on wip/corpus3-seed in w-c3seed. Its reconcile code equals main's, so it merges cleanly later. Do not fast-forward w-c3seed to main before corpus3's reconcile passes: the cloud commits change extract/resolve, and they need their own corpus change (section 6).

## 6. Cloud branch: open items before main runs extract or resolve on any corpus

- The override gate will fail: ov-0035, ov-0105 and ov-0108 unlink heading mentions that no longer exist. Retiring them is the user's call.
- Confirm the retargeted side labels on ov-0126 (Quatre Bras UK), ov-0534 and ov-0536 (Okinawa "United States"). They come from reconstructed fixtures.
- Re-run the §37 passage-text check (`37-scripts/snap.py`, `cmp.py`) on the 3,304 real articles. The cloud guard covers fixtures only.
- Resolve now defaults to `use_infobox_wikilinks: true`: network lookups, up to about 2000 page fetches (around 1 h), and new resolve LLM requests. Send the link changes to the historiography reviewer.
- A Sonnet code review of 2808819..40172bf was recommended and has not run. The user has a cloud code-review prompt for main; check for `docs/handover-log/cloud-code-review.md` or a returned bundle.

## 7. Housekeeping

- **The user wants all four worktrees removed** (w-c3seed, w-erarot, w-campx, w-endash; keep the branches) **once corpus3 reconcile is done.**
  - w-erarot, w-campx and w-endash are clean and merged, so they can go any time.
  - w-c3seed holds corpus3's 6.2 GB `data/`. Do not merge it into the main clone's `data/`: extract reads every file under the raw dir, so corpus2 and corpus3 would mix. Give it its own folder (e.g. `data/corpus3/`, with path `--set`s) and show the user the layout first.
  - On Windows, junctions inside a worktree need `cmd /c rmdir` before `git worktree remove` (handover §0).
- Scratch DBs: `general_war_endash_it` is new this session. `general_war_c2_erarot`, `general_war_erarot_it` and `general_war_corpus3_recon` are older. Ask before dropping any.
- `cloud-extract-resolve-fixes.bundle` in the repo root (untracked) is redundant now and can be deleted.
- Footguns this session: `python -I` hides the user site-packages (sqlalchemy is there), so run own scripts without it. psql needs the URL via `-d`. `rtk` is not installed in Git Bash.
