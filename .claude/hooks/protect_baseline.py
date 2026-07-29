#!/usr/bin/env python3
"""PreToolUse hook — refuse hand-edits to the committed scorecard.

`evals/baseline.json` is a generated artefact. Its whole value is that it was
produced by an actual eval run, so a diff against it means something. Editing it
by hand to make a regression go away is the single most damaging thing anyone
can do to this repo — it silently converts the regression gate into decoration.

Blocks Write/Edit to that file and points at the command that regenerates it.

Reads the hook payload on stdin, writes a PreToolUse deny decision on stdout when
the target is the baseline, and nothing at all otherwise.
"""

from __future__ import annotations

import json
import sys

PROTECTED_SUFFIX = "evals/baseline.json"
REGENERATE = ".venv/bin/python -m src.cli eval --set-baseline"


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0  # fail open — a malformed payload must not block all edits

    path = (payload.get("tool_input") or {}).get("file_path") or ""
    if not path.endswith(PROTECTED_SUFFIX):
        return 0

    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        "evals/baseline.json is generated, not hand-written — editing it "
                        "turns the regression gate into decoration. Regenerate it from a "
                        f"real run instead: {REGENERATE}"
                    ),
                }
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
