# Handover §8. Project tooling

Part of the handover; the index is `handover.md` at the repo root.

- **`bayesian-model-reviewer`** agent: gate after any change to reconcile,
  impute, model or WAR pooling. A mis-specified model still samples and still
  ranks; nothing downstream catches it.
- **`historiography-reviewer`** agent: domain plausibility of extracted data.
  Reviews data, not code.
- **`/data-audit`**: every spec's gates against the live DB plus coverage by
  era/region. **`/extraction-eval`**: prompt regression harness, waiting on the
  gold set. **`/resolve-eval`**: threshold sweep, zero LLM quota.
  **`/process-llm-batch`**: answers a deferred-request file with
  `llm-batch-worker` subagents (`.claude/agents/`, Read and Write only).
- **Stage inspector**: `streamlit run scripts/inspector_app.py`, read-only by
  construction (every connection sets `default_transaction_read_only`), safe to
  point at a corpus database. DB from `INSPECT_DATABASE_URL`, else
  `DATABASE_URL`.

---
