Process a batch of offline LLM requests through this Claude Code session.

Read the request file at $ARGUMENTS (or the newest `.jsonl` in `data/llm_requests/` if no argument given). Each line is a JSON object containing:
- `request_hash`: unique identifier (preserve exactly)
- `system`: the system prompt to follow
- `user`: the content to extract from
- `json_schema`: the expected output format
- `schema_name`: short name for the schema

For each request:

1. Read the `system` prompt and `user` content
2. Follow the system prompt's instructions exactly as if you were the extraction model
3. Generate the structured data matching the `json_schema`
4. Validate your output against the schema before writing

Write each completed response as one JSON line to a response file at `data/llm_responses/<stage>_<timestamp>.jsonl`:
```json
{"request_hash": "<hash from request>", "data": {<your extraction>}, "stage": "<stage from request>"}
```

Stop after processing 25 requests to avoid hitting session limits. Report progress:
- How many requests were in the file
- How many you processed
- How many remain

After processing, tell the user to import with:
```
python -m scripts.llm_offline import --file data/llm_responses/<the file you wrote>
```
