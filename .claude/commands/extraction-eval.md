Evaluate the LLM extraction and classification prompts against hand-labelled ground truth: $ARGUMENTS

`$ARGUMENTS` may be `extract`, `classify`, or empty for both. Add `--baseline` to overwrite the stored baseline after a deliberate improvement.

The extract stage will make tens of thousands of `claude-sonnet-4-6` calls. Without this harness the only way to know whether a prompt edit helped or hurt is to run the whole corpus and eyeball the result. This command makes a prompt change measurable in about twenty calls.

## 1. Check the gold set exists

Ground truth lives in `tests/fixtures/gold/`:

```
tests/fixtures/gold/
  battles/<slug>.html          raw article snapshot, so results are reproducible
  labels/<slug>.json           hand-labelled expected extraction
  baseline_scores.json         scores from the last accepted run
  MANIFEST.md                  what each fixture is for and who labelled it
```

If `tests/fixtures/gold/` is missing or empty, stop and build it first — see section 6. Do not fabricate ground truth to make the command run.

## 2. Run the prompts

Load the prompt from the spec rather than hardcoding it, so the harness always tests what the pipeline actually sends:

- `agents/extract.yaml` → `prompt.system`, `prompt.user_template`, `prompt.output_schema`
- `agents/classify.yaml` → `command_role_prompt` and `missingness_prompt`

Use `pipeline/llm.py` once it exists, so retry, logging, and cost accounting match production. Temperature 0 for both, per the spec params and the CLAUDE.md convention.

Run every fixture. Cache responses keyed by `(prompt_hash, fixture_slug)` in `tests/fixtures/gold/.cache/` so re-running after an unrelated code change costs nothing — a changed prompt changes the hash and forces a real call.

## 3. Score

Report per field, not just an aggregate. An aggregate hides the fact that commander names are near-perfect while scope is a coin flip.

**Extraction (`agents/extract.yaml`)**

| Field | Metric |
|---|---|
| `commanders[].name` | precision / recall, matched after normalising case, diacritics, and regnal numbers |
| `commanders[].apparent_role` | accuracy over correctly matched names, plus a confusion matrix |
| `troop_reports[].value` | share within ±10% of the label, and share within ±50% |
| `troop_reports[].scope` | accuracy, plus the predicted distribution — flag if `engaged` exceeds 80% |
| `troop_reports[].is_estimate` / `is_upper_bound` / `is_lower_bound` | precision / recall each |
| `casualty_reports[].value` | share within ±10% |
| `outcome.outcome_level` | accuracy, plus confusion matrix |
| `terrain`, `fortified` | precision / recall |
| JSON validity | share of responses parsing and validating against `output_schema` |

**Classification (`agents/classify.yaml`)**

| Field | Metric |
|---|---|
| `command_role` | accuracy, plus confusion matrix over the 7-value enum |
| `hierarchy_rank` | exact-match accuracy, and share with exactly one rank-0 per side |
| `reports_to` | accuracy over sides with more than one commander |
| `attribution_weight` | mean absolute error against the label |
| weights sum to ~1.0 | share of sides within 0.15, mirroring the `attribution_weights_sum` gate |
| `missingness_class` | accuracy over the 3-value enum |
| `needs_review` | precision / recall — over-flagging is as costly as under-flagging |

Report the named cases individually, since they are the ones the project is judged on: Actium (Agrippa `field_commander` with weight > 0.7, Octavian `sovereign`), Cannae (Varro and Paullus split), any D-Day fixture (Eisenhower `supreme_commander`).

## 4. Compare against baseline

Load `tests/fixtures/gold/baseline_scores.json` and diff every metric. Report improvements and regressions separately with the delta.

**Fail loudly on regression.** Any metric dropping more than 2 percentage points is a regression: name it, say which prompt changed, and recommend reverting unless the drop is a deliberate trade. Exit non-zero so this can gate CI.

With `--baseline`, overwrite the file — but only after showing the diff and stating what improved. Record the prompt hashes alongside the scores so a baseline can always be traced to the prompt that produced it.

## 5. Cost

Report for the fixture run: total input tokens, output tokens, call count, and estimated cost.

Then extrapolate. Read `SELECT COUNT(*) FROM battles` if the database is up, else use the crawl target of 3,000 from `agents/crawl.yaml`'s `min_battles_crawled`, and project the full-corpus cost at the observed per-battle token rate. Account for `agents/extract.yaml`'s `batch_size: 50` and for long articles being chunked into several passages — the per-battle call count is not 1.

State the figure plainly. It is the number that decides whether a prompt change is affordable to deploy.

## 6. Building the gold set

If the fixtures do not exist, this is the real work and should be done as its own task. Twenty battles chosen for coverage of the hard cases, not for being famous:

| Slug | Why it is in the set |
|---|---|
| `actium-31bc` | multi-commander, clear hierarchy, the project's canonical attribution test |
| `cannae-216bc` | alternating consular command; BC date; disputed numbers |
| `gaugamela-331bc` | ancient source inflation at its most extreme |
| `thermopylae-480bc` | Herodotus' 2.6M; tests whether inflation survives extraction |
| `talas-751` | non-European, sparse sourcing, transliterated names |
| `constantinople-1453` | siege; multi-month; attacker/defender asymmetry |
| `agincourt-1415` | disputed force ratio central to the battle's fame |
| `lepanto-1571` | naval; multi-national coalition on one side |
| `blenheim-1704` | genuine co-command (Marlborough and Eugene) |
| `austerlitz-1805` | well-documented; three polities; clean reference case |
| `waterloo-1815` | coalition command; Blücher's arrival mid-battle |
| `koniggratz-1866` | nominal royal command versus Moltke's actual direction |
| `antietam-1862` | dual naming (Sharpsburg); good casualty data |
| `gettysburg-1863` | multi-day; corps-level subordinates named |
| `tsushima-1905` | naval; non-European; decisive |
| `jutland-1916` | genuine draw; both sides claimed victory |
| `midway-1942` | air/naval branch split; carrier counts not troop counts |
| `stalingrad-1942` | months-long; army-group scale; `theatre_strength` versus `engaged` |
| `normandy-1944` | supreme versus field command; the Eisenhower case |
| `unknown-troops-case` | a real battle where one side's numbers are entirely absent, to test that missingness is logged rather than guessed |

For each: snapshot the article HTML so results stay reproducible when Wikipedia changes, then hand-label the expected output against the schemas in `agents/extract.yaml` and `agents/classify.yaml`. Record in `MANIFEST.md` who labelled it, when, and which sources they consulted — a gold set whose provenance is unknown is not gold.

Where scholarship genuinely disagrees, label the range and score a prediction inside it as correct. Do not encode a single contested number as truth.

Consider having the `historiography-reviewer` agent check the labels before they are accepted as baseline. Wrong ground truth is worse than no ground truth: it will silently train prompt edits in the wrong direction.

## 7. Output

- Field-level score table with the baseline delta.
- Named-case results, pass or fail each.
- Regressions, called out separately and prominently.
- Cost for the run and projected for the corpus.
- A recommendation: ship the prompt change, revert it, or investigate a specific field.

Commit only if `--baseline` was passed, with message: `test: update extraction eval baseline`.
