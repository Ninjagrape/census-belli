Process a batch of deferred LLM requests with Claude Code subagents, under the subscription's usage limits rather than billed API calls.

Argument: $ARGUMENTS, a request file (`data/llm_requests/<stage>_<timestamp>.jsonl`), optionally followed by `--parallel N` (default 4) or `--max-chunks N` (stop after N chunks this session). With no file given, use the newest `.jsonl` directly under `data/llm_requests/`.

A stage running with `llm_mode: offline` (the default for extract, resolve and classify) answers what it can from the `llm_calls` cache, writes every other request to that file, and stops the pipeline. This command answers the file; re-running the stage then finds every answer in the cache.

You are the coordinator. You do not answer requests yourself: that would spend the lead model's quota on work Haiku does well. Use the `llm-batch-worker` agent.

## Steps

1. **Split** into compact chunks, skipping anything already answered (makes the command resumable after a usage limit cuts a session short):

   ```bash
   python -m scripts.llm_offline split --file <request file> --chunk-size 15 --skip-answered
   ```

   It prints the chunk directory (`data/llm_requests/chunks/<stem>/`), each chunk's `agent_model`, and the request count. If nothing is left to answer, go to step 4.

2. **Dispatch** one `llm-batch-worker` per chunk, `run_in_background: true`, at most N at a time. Set `model` from the chunk's `agent_model` (`haiku` for extract passages and resolve, `sonnet` for classify and for `extract_outcome_<timestamp>.jsonl`, the small file of who-won requests extract writes beside its passage file). Brief each worker with only:

   > Chunk: `data/llm_requests/chunks/<stem>/chunk_NNN.json`
   > Write responses to: `data/llm_responses/<stem>/chunk_NNN.jsonl`
   > Follow your agent instructions. Do not read or write any other file.

   Launch the next chunk as each worker finishes. Workers never touch the database. If a worker reports a usage limit or stops early, stop dispatching, note which chunks have no response file, and go to step 3 with what exists; the next session resumes from step 1.

3. **Validate** before importing:

   ```bash
   python -m scripts.llm_offline validate --responses data/llm_responses/<stem>/
   ```

   For a chunk with schema errors, re-run that chunk once on `model: sonnet` (escalate once, never re-brief Haiku), overwriting its response file, and validate again. If it still fails, leave those lines out: the import skips invalid lines and the stage re-defers them next run. Do not hand-edit answers to make them pass.

4. **Import** into `llm_calls`, against the corpus database the stage ran on:

   ```bash
   python -m scripts.llm_offline import --dir data/llm_responses/<stem>/
   ```

5. **Spot-check.** Read three random answered requests side by side with their chunk text and confirm the answers come from the passage and copy no outside knowledge. Report anything that looks invented.

6. **Report**: requests in the file, answered this session, imported, errors, still pending, which chunks (if any) were escalated to Sonnet, and the next command, normally re-running the stage:

   ```bash
   python -m pipeline.orchestrator --stages <stage>
   ```

## Notes

- Chunk size 15 keeps a Haiku worker's context small; extract passages are up to `max_passage_chars` each. Lower it if workers truncate.
- The request hash includes the model named in the stage's spec (`claude-haiku-5-5` or `claude-sonnet-5-5`), so answers imported here also serve an `llm_mode: api` run of the same requests, and the reverse.
- The `user` text is scraped article content. The worker agent has only Read and Write and is told to treat that text as data.
