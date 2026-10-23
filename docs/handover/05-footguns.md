# Handover §5. Footguns

Part of the handover; the index is `handover.md` at the repo root.

- **BC dates cannot round-trip through Python** (`datetime.date` MINYEAR is 1).
  Read `battles.year_astronomical`, a generated column: 31 BC is -30 there and
  -31 from a bare `EXTRACT(YEAR ...)`. Only astronomical numbering subtracts
  correctly across 1 BC/AD. Wikidata and Arsht both write historical years, so
  convert on the way in and join on `year_astronomical`. A battle known only by
  year is written `YYYY-01-01` with `date_precision='year'`.
- **One integration suite at a time against `DATABASE_URL`.** Every module
  fixture drops and rebuilds `public`. Overlapping runs hang silently or throw
  `UndefinedTable`; check `pg_stat_activity` / `pg_blocking_pids()` before
  suspecting the code. Load any real corpus into a separate database
  (e.g. `general_war_corpus`). `python -m tests.integration.inspector_seed`
  also drops `DATABASE_URL`.
- **Module-scoped fixtures that run a stage need `try/finally` rollback**, or an
  exception leaks a connection holding row locks and later tests wait forever.
- **PyMC:** never name a variable like a dimension (ArviZ returns the
  dimension's labels); pass every per-observation array as `pm.Data`, because
  numpy constants are compiled into nutpie's code and compile time explodes
  with corpus size.
- **CI installs from `requirements/ci-lock.txt`, not `pyproject.toml`.** A
  fixed seed is bit-reproducible on one dependency set (verified 2026-10-13,
  in and across processes), so a locked CI result depends on the code alone.
  A new dependency goes in `pyproject.toml` and the lock is regenerated (command
  in its header) in its own commit, or CI will not have it.
- **Model tests retry convergence, never claims.** `fit_until_converged` in
  `tests/model/conftest.py` refits on `seed + k * RETRY_SEED_OFFSET` (3
  attempts, as `pipeline/modelers/fitting.py` does) only when a convergence
  gate is missed; a coverage or recovery assertion is never retried. A new
  model test whose fit must converge uses it instead of asserting a gate on
  one fit.
- `yaml.safe_load` turns `2026-09-24` into a `date`; convert before writing
  JSON.
- `logging_config.configure()` is process-global; a test that calls it needs
  the restore fixture in tests/unit/test_infrastructure.py.
- `alembic ... --sql` needs `PYTHONIOENCODING=utf-8` on Windows (box-drawing
  characters in schema.sql).
- Shell heredocs mangle backslashes and have failed mid-file. Use the
  file-writing tool for source files and Windows paths.
- Workers: tell them to format only files they own (one ran `ruff format`
  across the tree). A worker cut off by a usage limit may have finished more or
  less than its last line says; check `git status` and the brief's items.
- **Run model fits in the foreground.** A backgrounded PyMC run stalled on
  Windows; a foreground command over 10 minutes is moved to the background by
  the tool, so run `tests/model` one file per command.
- **Workers editing `agents/*.yaml` during a test run break that run**: stages
  read their spec at runtime. Never overlap a code-editing worker with a
  verification run on the same tree.
- **Parallel workers** each get their own database (`general_war_wt_*`).
  Three workers once picked the same Alembic revision id, so check
  `alembic heads` after any merge.
- Auto-mode's permission classifier may block a subagent from connecting to
  `general_war_corpus2`; ask the user rather than routing the work through
  another agent.
- **Agent `isolation: worktree` branches from `origin/main`**, which can lag
  local `main`. For parallel workers, create worktrees yourself from the right
  base (`git worktree add -b wip/<name> .claude/worktrees/w-<name> <base>`)
  and point workers at them. Auto mode refuses copying uncommitted diffs
  between worktrees; commit to a local throwaway branch instead (the user
  approved this for 2026-10-14..16).
- A worktree has no `data/`. A directory junction to the main `data/` works,
  but **remove it (`cmd /c rmdir`) before `git worktree remove`**.
- A database built by `apply_schema()` has no `alembic_version`, so
  `alembic upgrade head` replays every migration from the baseline. They are
  guarded and converge, but check the data after (`general_war_corpus2` was
  fine and is now stamped).
- **Re-running a corpus stage means re-running everything after it, in
  order: extract, resolve, reconcile, classify, impute.** Extract rewrites
  sides and missing_data_log; classify without a fresh reconcile logs false
  troop gaps (2026-10-16, fixes B and D).
- Never poll another task's output with a shell `until ...; sleep` loop: if
  that task dies the loop runs forever. Rely on completion notifications.
- **Workers spawn their own subagents unless told not to.** On 2026-10-17 a
  Sonnet worker launched a Haiku "placeholder wait" agent to stall while its
  backgrounded suite ran. Every worker brief says: no subagents, and when a
  command is backgrounded, end the turn and wait for its notification. Task
  `.output` transcripts are empty on this machine, so the main session cannot
  audit this after the fact; the agent panel shows "Spawned by".
- An `llm-batch-worker` can stall on a slow API ("no progress for 600s");
  relaunch that chunk, `split --skip-answered` makes it safe.
- **Workers must never kill processes they did not start.** On 2026-10-22 a
  worker restarting its own suite "killed all python processes" and silently
  took down another worker's impute run (re-run, no lasting damage). Every
  brief that runs commands says: stop only your own process, by its PID.
- **The laptop hibernates overnight** and kills background runs (a resolve
  run died mid-SPARQL on 2026-10-17). Re-run, do not trust a partial log.
- A worker worktree needs its own `data/processed` copy (stage paths are
  cwd-relative), a junction to the shared `data/raw` (read only), and its own
  `data/llm_requests`; otherwise parallel extract and resolve clobber each
  other's files.
- Never merge into `main` while a suite runs on the main checkout; the run is
  then void (happened 2026-10-17, re-run on de9f4c2).
- `psql` on this shell rejects SQL containing accented characters (the
  folding `translate()` in resolve's gates); run those through SQLAlchemy.
- `remove_orphan_generals` keeps generals that old model runs reference, so a
  general resolve stopped linking survives with no battle links ("zombie").
- `claude-haiku-5-5` thinks unless told not to (handled by the
  `thinking_on_by_default` trait) and cannot take temperature 0.

---
