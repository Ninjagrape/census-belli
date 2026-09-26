# Reconcile Gold Set

45 cases pairing a pre-modern source's troop or casualty figure against a named modern scholar's estimate for the same force, across four historiographic traditions.

It exists because the reconcile stage's spec carried `ancient_source_bias_prior_mu: 0.3`, a 1.35x inflation factor whose three-sigma ceiling is 6x, while the documented cases (Gaugamela, Nicaea, the Helvetii, Xerxes) run 4x-20x. For most ancient battles no independent anchor survives; every figure descends from one classical author, the likelihood contributes nothing, and the posterior is the prior. The number this set produces is therefore not a modelling nicety, it is the estimate the reconcile model actually uses for ancient-source inflation. So it is measured against real cases rather than asserted from a round number.

## Provenance

- Created: 2026-09-24
- Built by: four agents working in parallel, one per tradition (classical_mediterranean, east_asia_steppe, islamic_india_byzantium, medieval_early_modern_europe)
- Reviewed by: **not reviewed by a human**
- Every Wikidata Q-id was fetched live from `Special:EntityData`, not recalled from training data. Two mismatches surfaced and were corrected during that verification: an initial hit for Chaeronea resolved to Battle of Carrhae, and an initial hit for Issus resolved to the unrelated 194 CE battle between Severus and Niger. Both are noted in the affected rows.

## Coverage

45 rows across four case files:

| Tradition | File | Rows |
|---|---|---|
| classical_mediterranean | `cases_classical.jsonl` | 15 |
| east_asia_steppe | `cases_east_asia.jsonl` | 5 |
| islamic_india_byzantium | `cases_islamic_india.jsonl` | 12 |
| medieval_early_modern_europe | `cases_medieval_europe.jsonl` | 13 |

By claim regime:

| Regime | n |
|---|---|
| ancient_claim | 5 |
| chronicle | 28 |
| administrative_partisan | 8 |
| staff_return | 4 |

By quantity: 42 troops rows, 3 casualties rows (Cannae, Trasimene, Towton). The cap on casualties rows was deliberate, to keep the set from being dominated by the messier and generally more contested business of counting the dead.

By side role: 21 `enemy_of_author`, 16 `own_side_of_author`, 8 `neutral`.

Faithful rows, defined here as `|log_ratio| < 0.3`: 11 of 45. Six are classical (Cannae troops, Raphia, Pharsalus, Chaeronea, Cynoscephalae, Trasimene casualties), four are 18th-century staff returns (Rossbach, Valmy, Austerlitz, Leuthen), and one is colonial-era (Omdurman). East Asia contributes none. Several rows sit just outside this band and are worth knowing about: Magnesia (0.336), Vienna 1683 (0.357), Naseby (0.375), and the Fall of Constantinople defenders (-0.342, discussed under limitations below) all cluster close to the 0.3 line, so the faithful count is sensitive to exactly where that line is drawn. It is a bucketing device for this manifest, not a boundary the fitted mixture itself uses; the mixture below assigns responsibility continuously.

## The measured result

Fitting `scripts/fit_inflation_prior.py` to all 45 cases produces a two-component mixture over `log(source figure / modern estimate)`:

| Component | mu | sd | share | factor |
|---|---|---|---|---|
| faithful | +0.128 | 0.222 | 0.36 | 1.14x |
| rhetorical | +1.709 | 0.689 | 0.64 | 5.53x |

A single Gaussian cannot represent this set: Cannae's troop figure and the Herodotean claims at Thermopylae are not two tails of one distribution, they are different generative processes, one with an administrative substrate and one without.

Per-regime weight on the rhetorical component:

| Regime | n | weight_rhetorical |
|---|---|---|
| staff_return | 4 | 0.189 |
| administrative_partisan | 8 | 0.401 |
| chronicle | 28 | 0.724 |
| ancient_claim | 5 | 0.857 |

The pooled median log ratio across all 45 cases is +1.150 (3.16x).

That ordering, staff_return below administrative_partisan below chronicle below ancient_claim, was not imposed on the data. It fell out of fitting the mixture to whichever regime label each case happened to carry. That ordering is the empirical case for keying the reconcile model's inflation term on claim regime rather than on the battle's calendar year: a 1757 Prussian staff return and a 1453 Byzantine civic muster are both administratively grounded and both land near the faithful end, while a 1302 Flemish friar's chronicle and a fifth-century-BC Persian ancient_claim are both narrative constructs and both land near the rhetorical end, regardless of which is centuries closer to the present.

## Limitations

This is not a gold standard, and should not be treated as one.

1. **Selection bias toward notorious cases.** Famous exaggerations (Thermopylae's 2.6 million, Gaugamela's million-man host) are far easier to source with a citable modern rebuttal than an ordinary, unremarkable troop count that nobody ever bothered to argue about. The rhetorical component's mu and share are therefore probably overstated relative to the true population of ancient claims, most of which never attracted a specialist's attention at all. The six faithful classical rows (Cannae, Raphia, Pharsalus, Chaeronea, Cynoscephalae, Trasimene) were added deliberately to push back against this. That is a correction applied by the set's builders, not a cure for the underlying bias.

2. **Not human-reviewed.** Every row was assembled by an agent from web sources under time pressure, in one pass, with no independent check by a person against the primary text.

3. **Thin cells.** ancient_claim has 5 cases, staff_return has 4, administrative_partisan has 8. A per-regime weight fitted on four or five points is not a stable estimate of anything; it is smoothed with a Beta(1, 1) prior in `regime_weights()` for exactly this reason, and the smoothed numbers above should be read as indicative, not precise. Adding or removing a single ancient_claim case would move that regime's weight substantially.

4. **Citation quality is uneven, and not uniformly across traditions.** The medieval agent reported that its post-1700 rows lean on secondary-literature attribution rather than direct citation of primary muster documents, and that two of those rows, Rossbach and Leuthen, both cite the same author, Christopher Duffy, on both sides of the comparison (as the ancient-claim source's modern interpreter and, functionally, as the only modern check on it). That is weaker than an independent pair, since a single historian's read of the archive is standing in for two things at once. The classical agent separately reported that several of its citations (Gaugamela, Issus, Thermopylae, Plataea) do not pin down an exact page for the specific claim under discussion, giving a chapter or a work rather than a locus. Neither problem invalidates the rows, but both mean the citations should not be treated as independently re-derivable without further digging.

5. **Africa and East Asia are under-represented, and not by symmetric accident.** The East Asia agent delivered 5 of a target 12 rows and, notably, no faithful case at all: Fei River, Legnica, Mohi, Gaixia, Sekigahara, Nagashino, and the Korean naval actions were all dropped for want of a citable modern figure with a specific, defensible number attached. Africa is a single row, Omdurman; Adwa and Isandlwana were both considered and dropped on citation quality. This means the mixture's low end is disproportionately European and Mediterranean, and it is an open question whether that reflects a genuine historiographic difference or simply which corpora are easiest to source in English.

6. **Scope gaps get mixed in with inflation, and they are not the same thing.** One case, the Fall of Constantinople's Byzantine defenders, has a negative log ratio (-0.342) because George Sphrantzes' civic muster counted only Byzantine subjects able to bear arms, while modern totals (Runciman) add the Genoese and Venetian mercenary companies operating under their own captains, who fell outside Sphrantzes' own jurisdiction to count. That is a scope difference between what two sources were counting, not chronicle-style exaggeration, and it happens to land on the opposite side of zero from every rhetorical case in the set. Pooling it into the same mixture as the inflation cases blurs two distinct phenomena that the model may eventually need to separate: sources disagreeing about magnitude, and sources disagreeing about what counts as part of the force.

7. **What this set cannot measure.** It cannot say whether a modern scholar's "reconstructed" figure has itself silently inherited some of the ancient source's inflation rather than independently rederiving it; the field only has one number per side per case, and there is no way to audit a modern historian's own workings from here. It cannot fit the own-side/enemy-side asymmetry with any confidence: the `side_role` field exists and is populated on every row, but with n this small, splitting the regimes further by side_role would leave most cells at one or two cases. And it cannot verify that any individual row's `ancient_claim_regime` label is the right one; several rows note genuine ambiguity in this call (Babur's Baburnama is a first-person memoir filed under `chronicle` for lack of a better bucket, Caesar's Helvetii tally is `administrative_partisan` because it claims to derive from a captured register but is transmitted through Caesar's own political memoir), and a different labelling agent might have split some of these differently.

## How to extend it

A stronger version of this set needs, in rough priority order: human review of every row against the cited primary and secondary texts; primary muster documents for the administrative_partisan rows currently resting on secondary-literature attribution (rather than accepting a historian's paraphrase of a return that was never itself inspected); more non-European faithful cases, since right now every faithful row outside the six classical anchors and Omdurman comes from 18th-century European staff returns; and enough rows per regime, particularly ancient_claim and staff_return, that the per-regime weights stop depending so heavily on the Beta(1, 1) smoothing to avoid asserting more than four or five points can support.

## How it is consumed

`python -m scripts.fit_inflation_prior` reads every `cases_*.jsonl` file in this directory, fits the mixture described above, and writes `config/inflation_priors.yaml`. That file carries a SHA-256 digest (`gold_set_digest`) over the exact bytes of every case file, so a fitted prior in production is traceable back to precisely the rows that produced it; changing a single `log_ratio` value and re-fitting produces a different digest. `agents/reconcile.yaml` reads the resulting file path via its `inflation_priors` param rather than hardcoding mu/sd values in the spec. Re-run the script after adding, removing, or correcting any case, and check the new digest lands in the committed `config/inflation_priors.yaml` before relying on it downstream.
