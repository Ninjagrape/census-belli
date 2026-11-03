"""PostToolUse hook: after handover.md is written, ask for a next-session prompt.

Updating handover.md is this project's checkpoint. Long sessions cost more per
turn, so at each checkpoint Claude hands the user a ready-to-paste prompt and
suggests /clear. Claude cannot run /clear itself; the user does.
"""

import json
import sys

REMINDER = (
    "Checkpoint: handover.md was just updated. Before ending this turn, if the "
    "current task step is finished (not mid-step), close your reply with a "
    "'Next session' section: one fenced code block holding a self-contained "
    "prompt for a fresh session (the §7 step to start, any file under "
    "docs/handover/ it needs, anything in flight such as background runs or "
    "unmerged branches), then one line telling the user to run /clear and "
    "paste it. Skip this if you will keep working on the same step this turn."
)


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return
    path = str(payload.get("tool_input", {}).get("file_path", ""))
    if path.replace("\\", "/").rsplit("/", 1)[-1] != "handover.md":
        return
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": REMINDER,
        }
    }))


if __name__ == "__main__":
    main()
