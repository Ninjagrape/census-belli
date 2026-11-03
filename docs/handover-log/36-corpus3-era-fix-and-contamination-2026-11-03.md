# §36. Corpus3: rotated era effect, campaign exclusion, report contamination (2026-10-31 to 2026-11-03)

Branch `wip/corpus3-seed` (worktree `.claude/worktrees/w-c3seed`), not merged,
not pushed. Commits this session: 8f86dad, 6eb2e10, 8ae2b27, 288eee5.

## 1. ov-0102 Q-ids (8f86dad)

Waterloo's override names Uxbridge (Q335133) and Bülow (Q63023) by Q-id,
because the review merges changed their local names. 419 entries load.

## 2. The era-channel fault and the rotated fix (6eb2e10)

**Diagnosis.** Corpus3 troop reconcile run 1 failed with max rhat 2.54 and
chains in different modes. The fault is the P1 period mean `e[era_bin]`. Both
of its collinearities with the other terms are exact, because `design.era` is
a 0/1 flag for years before 500, and 500 is also the first era-bin edge. In
addition, the level anchor (1800) coincides with the 1800 bin edge.

- The contrast between pre-1800 and post-1800 bins in `e` duplicates
  `g_unlabelled`.
- The contrast between the before-500 bin and the two medieval bins
  duplicates `a_era`.
- Of `e`'s four degrees of freedom, the data identify only two: 500-1499
  against 1500-1799, and 1800-1899 against 1900 on. Priors set the other two.

On corpus2 the ridge stayed short. Corpus3 has about seven times the data,
which made the ridge long and off-axis, and a free `tau_e` added a funnel on
top.

**Check on the run-1 trace.** The per-bin unlabelled levels
`u_k = m0 + e_k + g_unl*pre_k + a_era*c_k` had rhat 1.12-1.63, against
1.9-2.5 for their components.

**Review.** `bayesian-model-reviewer` was not registered as an agent type this
session, so a general-purpose Opus agent ran with the spec file. It rejected
dropping `e` (the user's first proposal). Without `e`, the period gaps would
load onto `g_unl` and `a_era`. With 1653 battles the data would override their
gold-fitted priors and read the gaps as inflation, so published pre-1800 sizes
would shift toward the modern level. It also rejected fixing `a_era` at 0,
which leaves the first collinearity in place.

**Fix, chosen by the user.** `era_effect: rotated`, with `legacy` and `none`
kept for comparison.

- `e` keeps a ZeroSumNormal prior at a fixed `sigma_e` (no `tau_e`), applied
  as a Potential.
- The free zero-sum per-bin levels `u_c` are sampled directly, and
  `e = u_c + hbar - h`.
- `offset_bar` drops all unlabelled `g_unl` weight and adds `hbar`. Keeping
  the undated rows' weight would have broken the invariance through `m0`.
- `u_c_free` is a zero-logp CustomDist, because `pm.Flat` broke the prior
  check.
- Guards raise if the cutoff is not the first bin edge, or if the era column
  is not constant within a bin.
- Neutral-fit sensitivity slopes for `g_unlabelled` are not comparable with
  runs before this change.

**Validation on a corpus2 copy** (`general_war_c2_erarot`, reconcile run 133),
against run 132's reconcile (run 109):

| Measure | Result |
|---|---|
| Max rhat, min ESS, divergences | 1.008, 1271, 0 |
| wb gap | 0.045, as in run 132 |
| Spearman correlation | 1.000 |
| Median / 95th-percentile abs log change | 0.002 / 0.008 |
| Force-ratio change, median | 0.003 |

Every gate passed except `calibration_set_is_not_empty`, a warning at 10
against 20 that predates this change.

**Suites.**

- Unit: 2393 passed.
- Integration (`-k "reconcile or prior_check"`, scratch DB
  `general_war_erarot_it`): 82 passed.
- Model: two failures, both artifacts, fixed in 288eee5.
  - `identified_level` became an unweighted mean over the 5 bins, including
    empty ones. It is now row-weighted under `rotated`.
  - The era-contrast test compared against the generating `TRUE_E`. The
    realised contrast differs by 0.42 because battle effects are drawn at
    random. Legacy's learned `tau_e` had shrunk the contrast enough to hide
    this. The test now uses the realised contrast, with the tolerance
    unchanged.
  - After the fix: p1_recovery and recovery files 32 passed, 1 xfail.

## 3. Campaign exclusion (8ae2b27)

`exclude_campaigns: true` drops reports from battles typed `campaign`
(counter `excluded_campaign`). The sides of those battles are written to
`missing_data_log` as unfillable, with a note saying they are campaign-scale.

- Unit tests pass. The integration suite has not been re-run with this
  commit.
- On corpus3 it excluded 651 reports and logged 141 sides.
- Corpus3 has only 48 battles typed `campaign`, so many strategic operations
  are still typed `field`.
- Open question for impute and model: whether campaign battles belong in the
  model at all, since their sides are now logged as missing.

## 4. Corpus3 reconcile run 2: noise leakage

Run on 8ae2b27. Log `data/logs/corpus3_reconcile2_20261102.log`, backup taken
before it. The run failed and wrote no troop estimates. Casualties converged
(rhat 1.002).

| Measure | Corpus3 run 2 | Corpus2 (run 133) |
|---|---|---|
| Max rhat, ESS, divergences | 1.58, 7.5, 0 | 1.008, 1271, 0 |
| Worst variables | `mu_side`, `u_c_free` | – |
| Era contrast, pre vs post 1800 | +0.12 ± 0.23 (block fine) | -1.26 |
| Modern infobox noise sd (prior median 0.22) | **0.72** | 0.25 |
| Noise ratio | 3.30 | 1.13 |
| `s_d` | 0.24 | 0.43 |
| `s_b` | 1.06 | 0.68 |
| Within-battle slope | 0.10 | 0.63 |

**Audit** (read-only; scratchpad `c3audit/audit_out.txt`, script `audit.py`
built on the stage's own loader). Only 66 of 2136 troop sides have two or
more rows, and the noise sd is identified from those alone. Many of them are
contaminated:

1. **Strategic operations not typed `campaign`.** Battle of France (150k
   against 3.3M on one side), Dnieper, Caucasus, Odessa Offensive, Dukla Pass,
   Debrecen, Nikopol-Krivoi Rog, Rzhev, Belgorod-Kharkov, Donbass, Smolensk
   operation, Voronezh, Vyazemsky operation, Oder-Neisse. Possibly also Kursk
   and Berlin, which have 1.9M and 2.3M single rows.
2. **Extract dash-glue bug.** Roan's Tan Yard Union reads "c. 450500" and
   Glasgow Union "550800", most likely "450–500" and "550–800" with a template
   dash dropped. About 82 candidate rows by a loose regex. Single-report sides
   like these do not inflate the noise, but they corrupt force ratios.
3. **Sub-detachment and mixed-scope totals.**
   - Corbach: "400 to 500 with Luckner" was parsed as a total.
   - Fort Frontenac: 2635 against 135 regulars.
   - First Siege of Chaves: the infobox covers two sieges.
   - Tournai: the "20px" side mixes besiegers and garrison.
   - Ticonderoga, Okinawa "Ground forces:".
4. **Broken side labels.** Croix d'Orade ("side 1" and "side 2") and
   Ammunition Hill ("side 2").

User decision (2026-11-02): fix the data first, with no model change yet. A
Student-t troop likelihood is the fallback if clean data still leaks.

## 5. Classify command roles

Chunks 053-091 were answered by Sonnet workers. 573 answers are imported into
corpus3's `llm_calls`, with 780 already there. Two lines had corrupted
`request_hash` values (83972288404c in chunk 072; c23314579c21, Pirna, in
chunk 084), so those requests re-defer to the next run. Backup:
`data/backups/general_war_corpus3_20261102_pre_roles_import.dump`.

- Running `split --skip-answered` would renumber the chunks from 001 and
  overwrite response files that are already imported, so the existing chunks
  were used.
- A 5-hour usage limit hit at chunks 083 and 084. Both workers resumed with
  SendMessage.

**Leads for the classify historiography review** (mostly identity errors, the
same class as the wikilink follow-up):

- Valley of Tears: the Syrian side is linked to "Omar Bradley"; the article
  names Omar Abrash.
- Tula: the article names Vasilii S. Popov; the side is linked to Markian
  Popov.
- Odessa 1941: the Romanian side lists Manstein.
- Tomaszów: the German side lists "Nico Leonard Willem van Straten"; the
  Polish side lists the infobox headings "First Phase" and "Second Phase" as
  people.
- "Winfield Scott Hancock" for the article's "Dick Hancock" (hash
  0ad093943b2d).
- Budapest: Dezső László was weighted 0 as a diarist namesake.
- Overall commander missing from the listed set: Vitoria (Wellington),
  Aldeia da Ponte (Marmont), St Malo (Marlborough), Ullais (Jaban),
  Upperville (Stuart).
- Kulm: the excerpt does not mention Kleist or Eugen.
- Large equal splits: Atlantic, Voronezh (20 commanders), Yom Kippur.
- Clydebank Blitz is not a battle.
- Valutino: the article contradicts itself on Barclay and Tuchkov.
- Namwon: four unidentified commanders, and Yang Yuan is missing.

## 6. Footguns

- In Git Bash, `pg_dump -Fc db -f file` fails with "too many command-line
  arguments". The long options `--format=custom --file= --dbname=` work.
- The reconcile stage writes `data/processed/reconcile/neutral_draws_run<N>.npz`
  under the current directory. Run each corpus from its own worktree.
