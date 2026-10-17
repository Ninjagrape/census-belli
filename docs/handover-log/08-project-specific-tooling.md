# Handover log §8. Project-specific tooling

Part of the session log; the index is `docs/handover-log/README.md`.

Four additions exist that a generic agent will not know about.

**Agents** (`.claude/agents/`):

- **`bayesian-model-reviewer`** — inference correctness for the PyMC stages: Bradley-Terry identifiability (skills are identified only up to an additive constant — needs a sum-to-zero constraint), centred vs non-centred parameterisation, ordered cutpoints, Rubin's rules pooling, imputation circularity, replacement-level uncertainty. **Use before committing Phase 4.** A mis-specified model still samples, still converges, and still emits a plausible ranking; nothing downstream will catch it.
- **`historiography-reviewer`** — domain plausibility of extracted data: ancient source inflation, troop-number scope conflation, command misattribution, anachronistic polities, BC dates, corpus selection bias. Reviews data, not code. It would have caught §4.2.

**Commands** (`.claude/commands/`):

- **`/data-audit`** — runs every spec's quality gates against the live database plus coverage-by-era/region. Distinct from `/status`, which is code health.
- **`/extraction-eval`** — prompt regression harness against hand-labelled battles. **The gold set does not exist yet**; building it is the real work, and it also unblocks `/test-stage extract`.

Both agents and both commands are unexercised. They are written, not proven.

**Stage inspector** (§22): `streamlit run scripts/inspector_app.py`, a
read-only dashboard over every built stage. Use it to eyeball a stage's output
before trusting its gates.

---
