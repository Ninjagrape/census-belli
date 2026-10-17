# Handover log §17. Session of 2026-09-21: offline processing, gold set, threshold sweep

Part of the session log; the index is `docs/handover-log/README.md`.

### 17.1 Offline LLM processing

The Gemini free-tier quota (§16.8) makes the bulk stages unrunnable through the
API. Rather than a batch API design, the pipeline now has an offline processing
path that routes requests through the user's Claude Pro subscription:

| File | What it does |
|---|---|
| `pipeline/llm/offline.py` | `export_pending` serialises uncached requests to JSONL, `import_responses` writes completed responses back to `llm_calls` |
| `scripts/llm_offline.py` | CLI: `status --stage`, `export --stage`, `import --file`, `list` |
| `.claude/commands/process-llm-batch.md` | Slash command for Claude Code sessions: reads request JSONL, follows the system prompt, writes response JSONL. 25 requests per session. |
| `tests/unit/test_llm_offline.py` | 8 tests: export, import, caching, roundtrip |

The `request_hash` is the sole correlation key: a request exported with hash X
must come back with hash X, and the next pipeline run will find it in `llm_calls`
by that hash. The offline provider/model on the audit row defaults to
`"offline"` / `"claude-pro-subscription"`, distinguishing manual processing from
API calls.

New directories `data/llm_requests/` and `data/llm_responses/` are gitignored.

### 17.2 Spec conflict resolution

`CLAUDE.md` previously named claude-sonnet-4-6 as the sole LLM provider. Updated to
document the actual per-stage routing (Gemini for extract/resolve, Anthropic for
classify). Both `agents/extract.yaml` and `agents/resolve.yaml` now carry
`llm_fallback_provider: anthropic` and `llm_fallback_model: claude-haiku-4-5-20251001`.

### 17.3 Live resolve regression tests

`tests/integration/test_resolve_live.py`, 5 tests, gated on
`GENERAL_WAR_LIVE_CRAWL=1`. Tests the Wikidata SPARQL lookup against the live
endpoint for regressions the fixture suite cannot catch:

1. Four known commanders (Napoleon, Nelson, Wellington, Caesar) still have
   candidates
2. Nelson is reachable via the `mul` label (the §16.1 regression)
3. No candidate label is Q-id shaped (the §16.2 regression)
4. The date gate separates Hannibal Barca from an 1811 Hannibal
5. Napoleon and Nelson resolve deterministically with no LLM call

All Q-ids verified live, not recalled. Q47153 (from an earlier task) was
identified as a 2009 novel, not Hannibal Barca; Q36456 is correct.

### 17.4 Resolve gold set and threshold sweep

`tests/fixtures/gold/resolve/` contains 15 mentions across 12 battles, with
cached synthetic candidates. Covers exact match, alias match, `mul` label, date
gate separation, MAX_PLAUSIBLE_AGE, genuine ambiguity (Yi Sun-sin), placeholders,
two spellings of one person, and BC dates.

`scripts/resolve_sweep.py` sweeps `fuzzy_threshold` x `ambiguity_margin` over a
grid, scoring each combo against the gold set. Spends zero LLM quota: ambiguous
groups are counted, never resolved.

`.claude/commands/resolve-eval.md` is the `/resolve-eval` command that runs the
sweep (or a single check at the current defaults) and reports per-case results,
a summary table, and a cost projection.

**Result at the current defaults (85.0 / 6.0):** zero wrong links, 11 correct
links, 1 ambiguous (Yi Sun-sin, correctly deferred), 2 placeholders. Every combo
in the grid (threshold 70-95, margin 2-12) also achieves zero wrong links,
which means the gold set's 15 cases are too clear-cut to discriminate between
thresholds. The MANIFEST states this limitation: the set needs harder boundary
cases (commanders known only by a title, contested identities, same-era name
collisions) before it can tune thresholds on measurement rather than judgement.

### 17.5 Changed files

| File | Change |
|---|---|
| `pipeline/llm/offline.py` | **New.** Offline export/import for LLM requests. |
| `pipeline/llm/__init__.py` | Added offline exports to `__all__` |
| `scripts/llm_offline.py` | **New.** CLI for the offline workflow. |
| `scripts/resolve_sweep.py` | **New.** Threshold sweep against the gold set. |
| `.claude/commands/process-llm-batch.md` | **New.** Slash command for Pro subscription processing. |
| `.claude/commands/resolve-eval.md` | **New.** Slash command for resolve evaluation. |
| `tests/unit/test_llm_offline.py` | **New.** 8 tests for the offline module. |
| `tests/integration/test_resolve_live.py` | **New.** 5 live regression tests. |
| `tests/fixtures/gold/resolve/` | **New.** Gold set: mentions, candidates, battle pairs, manifest. |
| `agents/extract.yaml` | Added Haiku 4.5 fallback params. |
| `agents/resolve.yaml` | Added Haiku 4.5 fallback params. |
| `CLAUDE.md` | Updated tech stack to document per-stage LLM routing. |
| `TODO.md` | Marked offline processing and spec conflict as done. |
| `.gitignore` | Added `data/llm_requests/` and `data/llm_responses/`. |
| `handover.md` | This. |

### 17.6 Verification, stated honestly

`ruff` clean. `mypy --strict` clean on 50 source files. **376 passed, 70
skipped** without DATABASE_URL set: 70 skips = 55 database integration tests +
15 opt-in live tests (10 from §15 + 5 new from §17.3). The test count increased
from 410 (with DB) to at least 384 unit tests (376 passing + 8 new offline
tests).

The sweep runs end-to-end against the gold set and produces correct results.
The live resolve tests were not run this session (no `GENERAL_WAR_LIVE_CRAWL`
env flag); they were written and verified to skip correctly.

**Not verified:** the offline processing path has not been tested end-to-end
(exporting real requests, processing via Claude.ai, importing back). The gold
set candidates are synthetic, not captured from the live endpoint.

---
