# Handover log §19. Session of 2026-09-24: the reconcile stage, first half

Part of the session log; the index is `docs/handover-log/README.md`.

Stage 4 is now roughly half built. Schema, gates, the claim classifier, the
loader and the design matrix are written and verified against live PostgreSQL.
The model is written too and **converges** -- but only after the two rounds of
identifiability work in §19.8 and §19.9, and only on synthetic data.

**No estimate has ever been written.** `summarise.py`, `store.py`, `report.py`
and `pipeline/stages/reconcile.py` do not exist, and neither does the
parameter-recovery test. §19.6 is the honest boundary and §19.9 is the one to
read before trusting any number this stage eventually produces.

The session began by planning against `agents/reconcile.yaml` as written, and
two reviews changed the design before a line of code existed. Both are worth
reading before touching the statistics.

### 19.1 The spec's inflation prior was wrong by a factor of 4.6 on the log scale

`ancient_source_bias_prior_mu: 0.3` means 1.35x inflation, with a three-sigma
ceiling of 6x. A `historiography-reviewer` pass found documented cases at
4x-20x: Gaugamela 4-20x, Nicaea 10-20x, Caesar's Helvetii 7-18x, Xerxes 9-48x,
Agincourt 4-12x. The prior could not represent a single one of them.

This is not a tuning nicety. For most ancient battles no independent anchor
exists -- every surviving figure descends from one classical author -- so the
likelihood contributes nothing and **the posterior is the prior**. The number
chosen there *is* the estimate.

So it was measured. `tests/fixtures/gold/reconcile/` holds **45 cases** pairing
a source's figure against a named modern scholar's estimate for the same force,
across four traditions, built by four parallel agents and fitted by
`scripts/fit_inflation_prior.py` into `config/inflation_priors.yaml`:

| component | mu | sd | share | factor |
|---|---|---|---|---|
| faithful | +0.128 | 0.222 | 0.36 | 1.14x |
| rhetorical | +1.709 | 0.689 | 0.64 | 5.53x |

Pooled median log ratio **+1.150 (3.16x)**, against the spec's 0.3.

### 19.2 Bias belongs to the claim, not the document

The second review finding, and the one that changed the model's structure.
Wikipedia infoboxes carry both traditions in one field. Verbatim, live:

```
52,930-100,000 (modern estimates) 250,000-1,000,000 (ancient sources)
15,000-150,000 (modern estimates) 100,000-200,000 (primary Arab sources)
```

A single `source_type='wikipedia_infobox'` bias averages Herodotus and
Delbruck into a figure belonging to neither. So the inflation term is keyed on
a **claim regime** derived from `troop_reports.extracted_context`, not on the
battle's calendar year.

The gold set then supported that choice empirically, which was not guaranteed:

| regime | n | median log ratio | factor |
|---|---|---|---|
| chronicle | 28 | +1.386 | 4.00x |
| ancient_claim | 5 | +1.843 | 6.32x |
| administrative_partisan | 8 | +0.366 | 1.44x |
| staff_return | 4 | +0.068 | 1.07x |

That ordering was not imposed. It is the case for regime over calendar year.

**But measure the coverage before relying on it.** Against the 2,086 real
strength strings in `tests/fixtures/gold/arsht/arsht_infobox.jsonl`, only
**45 (2.2%)** carry a readable regime marker once multi-regime fields correctly
abstain. The era fallback therefore does about 98% of the work and is
load-bearing, not incidental. That number is pinned by
`test_the_real_infobox_corpus_classifies_without_error`.

### 19.3 Three defects found by running the code, not by reading it

**The `diagnostics_json` handler was registered but never dispatched.** An
agent added `METHOD_HANDLERS` and `_check_diagnostics_json` to
`pipeline/quality.py` and left `_run_method_check` reading only
`UNIMPLEMENTED_METHODS`. `ruff` passed, `mypy --strict` passed, all 472 tests
passed, and every call still returned "not implemented". Caught by driving the
gate against a real `model_runs` row rather than trusting the green suite.
Same family as §4.1 and §11.4.

**`classify_regime` labelled Herodotus' million `modern_scholarly`.** The first
implementation returned the first matching regime, so a field naming both
traditions resolved to whichever pattern was checked first. `modern_scholarly`
is the model's **anchor, pinned to exactly zero bias**, so a 1,000,000 Persian
figure would have entered as an unbiased modern estimate with no parameter free
to absorb it. Now a multi-regime field returns `unlabelled` unless the caller
passes the report's own value, in which case the nearest marker wins.

**A recalled Q-id was wrong, three times, in prompts written this session.**
The Red Cliffs id supplied from memory (Q207318) is *French Revolutionary
Wars*; correct is Q830059. A search hit for Chaeronea resolved to *Carrhae*,
and one for Issus to a different Issus five centuries later. Every one was
caught only because the agents were instructed to verify against live
Wikidata. §14.6's rule holds, and it applies to whoever writes the prompt as
much as to the agent reading it.

### 19.4 What was built

| File | What it is |
|---|---|
| `pipeline/reconcilers/records.py` | `SourceKey`, `Report`, `CLAIM_REGIMES`, `ReconcileCounts`, `SideEstimate`, `SourceBias` |
| `pipeline/reconcilers/claims.py` | `classify_regime`, `era_flag`, `roundness`, `assign_lineages` |
| `pipeline/reconcilers/load.py` | DB rows to `Report` records; branch/naval/non-positive exclusions; unfillable sides |
| `pipeline/reconcilers/design.py` | Index vectors, centred era covariate, censoring partitions |
| `scripts/fit_inflation_prior.py` | Two-component EM fit of the gold set to `config/inflation_priors.yaml` |
| `tests/fixtures/gold/reconcile/` | 45 cited cases across four traditions, plus MANIFEST |
| `alembic/versions/...c3d4e5f6a7b8...` | The reconcile provenance columns |

Schema additions, in **both** `config/schema.sql` and the migration (§5.1):
ten columns on `sources` (fitted sigma, casualty bias, `bias_run_id`,
`bias_n_*_reports`, `bias_updated_at`), ten on `battle_sides`
(`est_*_run_id` / `_n_reports` / `_n_sources` / `_method` / `_updated_at`),
three FKs to `model_runs`, three indexes. Also **`troop_reports.scope` default
changed from `'engaged'` to `'unknown'`**, which had disagreed with
`coerce_scope` in `pipeline/extractors/records.py` since the schema was
written; the permissive default asserted troops-on-the-field, which is the one
thing an unreadable scope is not evidence of.

`agents/reconcile.yaml` went from 3 gates to 8. Two were rewritten because they
could not measure what they claimed:

- **`estimates_populated`** counted every side with any `troop_reports` row,
  including sides whose only rows are a cavalry count -- structurally
  un-fillable, so the gate could never reach 0.99 for a reason that is not a
  defect. Now filtered to `branch='total' AND reported_value > 0`, and paired
  with `unfillable_sides_are_logged` so the narrowing cannot hide a loss.
- **`ci_width_reasonable`** measured `(hi-lo)/est`, a count-scale tail
  statistic, at a threshold no singleton side can meet. The only way to pass
  was to write dishonestly narrow intervals. It is now the median of
  `LN(hi/lo)` over multi-source sides, and its inverse
  `singleton_estimates_are_honestly_uncertain` **fails** when a one-report side
  is given an interval narrower than a factor of 2.2.

### 19.5 The environment question, settled

`pymc>=5.28,<6` and `arviz>=0.18,<1`. pymc 5.28.5 resolves pytensor 2.38.2,
which ships cp314 wheels and declares `requires-python >=3.11,<3.15`, so one
pin installs on the 3.14 development machine **and** on CI's 3.11. pymc 6.x
needs >=3.12 and would break CI outright.

**There is no C compiler on this machine**, so PyTensor falls back to its
Python op implementations: 42 seconds to sample a two-parameter toy model,
which puts the real model out of reach. `nutpie` compiles the logp through
numba instead, needs no toolchain, and fitted a 300-side hierarchical model in
**19 seconds**. It is declared in `pyproject.toml` and is the default
`nuts_sampler` in `agents/reconcile.yaml`. Both numbers were measured here, not
assumed.

A `model` pytest marker was added and CI gained a fourth job. Both existing
pytest invocations became `-m "not live and not model"`; deselection is not a
skip, so the "fail if anything skipped" guard still has nothing to trip over.

### 19.6 Verification, stated honestly

**Verified against live PostgreSQL 15.19:** `ruff` clean across
`pipeline/ tests/ scripts/ alembic/`; `mypy --strict` clean on 55 source files;
**532 passed, 15 skipped** (the 15 are the expected opt-in live tests). The
schema applies from `apply_schema()`; the migration upgrades, downgrades and
re-upgrades; and it was separately applied to a database built from the **old**
`config/schema.sql` via `git show HEAD:config/schema.sql`, which is the only
path that exercises the real ALTER rather than the `IF NOT EXISTS` no-op. All
eight reconcile gates execute with no SQL error.

**Verified by running it, not by testing it:** the `diagnostics_json` gate
driven against real `model_runs` rows through eight states (no run, incomplete
run, NULL diagnostics, converged, diverged, a rollup payload with no `worst`
key, no `model_type`, no connection). The censoring test was confirmed
non-vacuous by inverting the mapping in `design.py` and watching exactly three
tests fail, then restoring it.

**Not verified, and not claimable:**

- **No estimate has ever been written.** `model.py` exists, converges after
  the §19.9 fixes (max rhat 1.016, 0 divergences), and has been fitted only on
  synthetic data. `summarise.py`, `store.py`, `report.py` and
  `pipeline/stages/reconcile.py` do not exist, so nothing has ever written an
  `est_troops_total`. The parameter-recovery test does not exist either, and
  §19.9 records what it must assert on.
- The parameter-recovery test does not exist, so the specification is
  unvalidated in the only way that catches a mis-specified hierarchical model.
- The gold set is **agent-built and unreviewed by a human**. Its `MANIFEST.md`
  lists seven limitations; the sharpest are selection bias toward notorious
  cases, `ancient_claim` n=5 and `staff_return` n=4, and East Asia contributing
  no faithful case at all.
- `data/processed/` is still empty and the database still holds no corpus, so
  the loader has been exercised against seeded test rows, never against real
  extracted data.

### 19.7 Next steps for this stage, in order

1. `summarise.py` -- InferenceData to `SideEstimate` / `SourceBias` plus the
   diagnostics dict. `est_troops_total` is `exp(median(mu))`, **not**
   `mean(exp(mu))`, which inflates by `exp(sd^2/2)`, about +32% at sd 0.74.
2. **`tests/model/test_reconcile_recovery.py` before any write path exists.**
   If the model is mis-specified that must surface before estimates reach
   Postgres. The design is in the approved plan file, but assert on
   `m0 + offset_bar` rather than `m0` -- see §19.9 for why.
3. `store.py`, `report.py`, `pipeline/stages/reconcile.py`, and the
   stage-level integration tests.
4. Work through what §19.9 leaves outstanding, starting with
   `n_identifying_sides` and the prior-to-posterior contraction map in
   `model_runs.diagnostics`. That is the check that makes the rest visible from
   a run log rather than from a review.
5. **Run the `bayesian-model-reviewer` agent again before proposing a commit**
   (§7, §8). Its first pass found two critical defects this session's author
   had missed.

A side with exactly one report should come out with a 95% interval spanning
roughly 0.24x to 4.3x its point estimate. That is the correct answer and must
be allowed to be that wide; `singleton_estimates_are_honestly_uncertain` fails
if a later change quietly narrows it.

### 19.8 The model sampled and did not converge, and the reason mattered

*(Resolved in §19.9. Kept because the diagnosis is the reusable part.)*

`pipeline/reconcilers/model.py` is written, builds, and runs. On the corpus
shape that will actually occur it **does not converge**, and the cause is a
specification defect rather than a tuning problem.

Measured on synthetic data: 40 sides, 2 sources, 2 reports per side, log-normal
noise sd 0.3, true `m0` 9.2, two chains, 300 tune, 300 draws, nutpie.

| corpus | max rhat | ess(m0) | posterior m0 |
|---|---|---|---|
| every report `claim_regime="unlabelled"` | 1.378 | **5.3** | 8.680 |
| every report `claim_regime="modern_scholarly"` | 1.067 | 80.1 | 9.207 |

`g_regime` pins the anchor regime (`modern_scholarly`) to exactly zero and
gives every other regime a free scalar. When no report carries the anchor, the
`unlabelled` offset is added to *every* observation and is confounded with
`m0`. Both priors are proper, so the posterior is proper and the run completes
-- it simply cannot be sampled, and `ess(m0) = 5.3` is what that looks like.

**This is the normal case.** §19.2 measured 2.2% of real strength strings
carrying a readable regime marker, so in production essentially every report is
`unlabelled` and the anchor is absent from the data entirely. The second row of
that table is the reassuring one, and it is the row that will almost never
occur.

Three things worth carrying forward:

1. The anchor **does** work when it is present, so the mechanism is right and
   the problem is specifically what happens when the anchoring level has no
   observations. That is the same shape as the era-covariate trap the design
   already guards against by centring, arriving on a different term.
2. A fix must not quietly abandon the interpretive goal. Pinning `unlabelled`
   to zero instead restores identifiability in one line and silently redefines
   an estimate from "what a modern scholarly reconstruction would say" to "the
   average of whatever this corpus contains". That may be the right trade, but
   it has to be a decision someone wrote down, not a side effect.
3. **It was found by running the model, not by reading it.** `ruff`,
   `mypy --strict` and the whole suite were green over this file. That is the
   fifth time in this file's history (§4.2, §4.3, §13.1, §16.3, §19.3) and the
   second time in this session.

The `bayesian-model-reviewer` agent was dispatched on it, as §7 requires before
any Phase 4 commit. Do not summarise a posterior with ess 5: it would produce
confident nonsense, and `singleton_estimates_are_honestly_uncertain` would not
catch it, because the failure makes an interval too *wide*, not too narrow.

### 19.9 The fix, and two further defects the review found

The `bayesian-model-reviewer` agent confirmed §19.8's diagnosis and reproduced
the broken posterior mean by prior algebra alone -- predicted 8.7131 against a
measured 8.680, with `corr(m0, g_unlabelled) = -0.993`. A posterior recoverable
without any likelihood term is what perfect confounding looks like.

It also found **two critical defects that were not in §19.8**, both of which
sampled cleanly and would have shipped:

**A second exact ridge, in the source-bias nesting.** `z_source` was zero-sum
over *keys*, which leaves the type-level mean of the deviations free -- and
that mean is the same parameter as `b_type`. With one key in a type the two are
exactly confounded, measured at `corr = -0.985`. The reviewer's table showed
each fix alone still fails rhat; both are needed. `u_source` is now zero-sum
**within each source type**, and a type holding a single key gets no free
deviation at all, which is the correct answer.

**`a_era`'s prior mean had the wrong sign.** It was
`rhetorical_mu - faithful_mu = +1.581`, the gap between the two *mixture
components*, used as though it were the ancient-versus-later difference. Those
are different quantities. Measured from the gold set's own years:

```
mean log_ratio, year < 500  : 0.855  (n=16)
mean log_ratio, year >= 500 : 1.301  (n=29)
era contrast                : -0.446  (se 0.294, widened to 0.587)
```

The prior asserted +1.581 where the evidence says about -0.45, putting the
measured value at roughly -4 sigma, and it sampled at **ess 984, rhat 1.00**,
because with no era contrast it simply drew its prior. Clean diagnostics,
plausible output, wrong number -- the exact failure mode this file keeps
recording. `scripts/fit_inflation_prior.py` now fits the contrast and writes
`era.mu` / `era.sd` to `config/inflation_priors.yaml`, and the term is gated to
unlabelled rows so an ancient chronicle row is not deflated twice.

### What changed, and the measured result

Four fixes: the **rotation** (sample the corpus-average `level`, derive `m0`),
zero-sum **within type**, the measured and gated **era prior**, and
`casualty_type_offset_prior_sd` **wired through** -- it was declared in the spec
and never read, so casualty component offsets were shrunk at 0.4 instead of
0.7, inferring totals about 34% too high from a killed-only report. The troops
scope offsets are now genuinely **cumulative**, which is what the docstring had
claimed while the code drew independent half-normals that could place
`theatre_strength` below `available`. `nu`'s prior was `Gamma(2, 0.1)`, mean
**22**, where StudentT is visually Normal -- the robustness the comment claimed
was asserted and not delivered; it is now `Gamma(2, 0.5)`, mean 6.

On the corpus shape that actually occurs, at the spec's own sampler budget
(1000 draws, 4 chains):

| | before | after |
|---|---|---|
| max rhat | 1.378 | **1.016** |
| ess(m0) | 5.3 | 468 |
| min ess, any parameter | 15 | 343 |
| divergences | 0 | 0 |
| identified level `m0 + offset_bar` | 8.680 | **9.214** (true 9.2) |

### The thing to carry forward, which is not a bug

**The rotation makes the sampler honest; it does not make `m0` identified.** In
an all-unlabelled corpus nothing informs the split of the corpus level into
`m0` and the unlabelled offset, so the inflation adjustment is *entirely
prior*. The reviewer measured `g_unlabelled` coming back at its prior to two
decimals with textbook diagnostics. Three consequences:

1. **A recovery test must assert on `m0 + offset_bar`**, which the data
   identify, not on `m0`, which they do not. Both are in the trace.
2. `_UNLABELLED_WEIGHT = 0.5` in `model.py` is the most consequential constant
   in the stage and it is a judgement call, not a measurement: the gold set
   labels every case by construction, so it contains no unlabelled rows to fit
   against. It implies about 2.5x inflation applied to every unmarked figure in
   the corpus, Gettysburg included. Run the estimates both ways before
   publishing anything absolute.
3. **The damage is to published figures, not to the ranking.** A regime or era
   offset is a battle-level shift shared by both sides, so it cancels in the
   force-ratio covariate the model stage consumes. Absolute troop counts are
   wrong by the asserted factor; the WAR ranking is largely insulated. Worth
   knowing for triage, and worth writing down so nobody later assumes the
   reverse.

Also predicted, and not yet observed because no estimates have been written:
once intervals carry the ridge's irreducible common factor, `LN(hi/lo)` will
exceed 2.7 and **`ci_width_reasonable` (threshold < 2.2) will fail
structurally**. It is a warning, not an error. Raise the threshold with a
comment; do not narrow the intervals to meet it.

### Still outstanding from the review

Not done, in the reviewer's order of priority: `n_identifying_sides` per regime
and prior-to-posterior contraction in `model_runs.diagnostics` (the check that
would have made all of the above visible from a run log); a prior predictive
check, of which the repository has none anywhere; `tau_lineage` at 0.3 against
`sigma_source` near 0.8, so three mirrored Wikipedia/Wikidata/DBpedia copies
still buy roughly twice the precision of one report rather than counting once;
per-side variation in the casualty component ratio, currently asserted to be a
corpus constant, the same at Cannae as at the Somme; and `InverseGamma(2, 1)`
on the observation variance, which has infinite variance on a parameter that
sets every interval width.
