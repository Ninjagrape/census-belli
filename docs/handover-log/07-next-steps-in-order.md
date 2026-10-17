# Handover log §7. Next steps, in order

Part of the session log; the index is `docs/handover-log/README.md`.

Step 1 of the previous handover (stand up Postgres, run the 28 skipped tests)
is **done**. What follows is what is left.

00. **Fix the pre-pilot gaps from §24 first** (A1, A2, A4, A5, A8, A10, A13).
   Without A1 the pilot crawl halts on its own gate; without A4 same-name
   battles merge silently.

0. **Run the curated corpus through crawl, extract and resolve.** §18 built
   `config/sources_seed_arsht.yaml`: 211 title-verified battles covering the
   generals Arsht's article names. It has never been crawled. Doing so needs no
   LLM quota if extract's deterministic path is used, and it is what turns
   every gate threshold below from unmeasured into measured.
   **Use a separate database for it** (e.g. `general_war_corpus`): every
   integration test module drops and rebuilds the `public` schema of
   `DATABASE_URL`, so a corpus loaded into `general_war` is wiped by the next
   test run (§20.8).

1. **Model stage: built (§23).** What is left for it is a real corpus run
   (step 0) and the `docs/todo/07-model.md` items. Before any ranking is published,
   classify's second slice (step 2) must run: today every multi-commander side
   is `default_split`, enters the model as an unattributed latent, and credits
   nobody, so `exclusion_concentration` will fire on a real corpus.
2. **Classify, second slice.** Run the pending multi-commander sides through
   the offline LLM path (`python -m scripts.llm_offline export --stage
   classify`, then `/process-llm-batch`, then import); populate
   `is_defender`; LLM missingness classification for rows the heuristics
   leave `unclassified`; the deferred historiography findings in TODO.md.
3. **Reconcile, the remaining §20.10 items**: identified-scale width gates,
   `n_identifying_sides`, the `z_side` ridge, `max_rhat: 1.01`, era
   centring within unlabelled rows, the `b_type` reference-type note. Plus
   §21.7's open items (shared measurement error into impute, the
   `est_troops_method` lineage count, the neutral gate's hard-coded ESS).
4. **Phase 5 — evaluate and report.** `agents/report.yaml` already exists and
   is fairly prescriptive. Two BLOCKING report items are in TODO.md (§23.6):
   its gates read `MAX(run_id)` unfiltered, and `exclusion_concentration` must
   block publication.
5. **Phase 6 — full-pipeline integration test, README, CI.**

Both of the "smaller things" this section used to list are done: the
Wikipedia 429 backoff (§15.1) and the `google-genai` install (§5.1b).

**The one thing standing between this project and a real corpus is not code.**
It is the free-tier LLM quota in §15.7. Crawl, extract and resolve are built
and verified; extract makes one LLM call per article passage and resolve one
per ambiguous commander, and twenty requests a day runs neither. Settle that
before planning a real run, not during one.

Verification gates for phases 3–6 all require a database, and there now is
one. **`export DATABASE_URL` before running the suite or you are testing
nothing** — the integration tests skip silently without it, and a green run
with 28 skips looks almost identical to a green run with none. Say explicitly
what was and was not verified.
