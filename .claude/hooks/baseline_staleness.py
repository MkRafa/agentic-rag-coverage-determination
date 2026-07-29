#!/usr/bin/env python3
"""PostToolUse hook — warn when a change invalidates the committed scorecard.

`evals/baseline.json` is only meaningful relative to the code that produced it.
Edit the gate, the pre-flight rules, the IO contracts, the retrieval stack or any
skill, and the committed numbers now describe a system that no longer exists.

This hook does not block anything. It prints one line reminding you to re-run the
eval and read the diff.

Reads the hook payload on stdin, writes a `systemMessage` JSON object on stdout
when a watched file changed, and nothing at all otherwise.
"""

from __future__ import annotations

import json
import sys

# Files whose behaviour the scorecard directly measures.
WATCHED_SUFFIXES = (
    "src/harness/gate.py",        # decides DETERMINE / REVIEW / REFUSE
    "src/harness/preflight.py",   # decides HALT, and what reaches the loop
    "src/harness/loop.py",        # iteration cap, clause filtering
    "src/contracts.py",           # structured-output schemas the agents fill
    "policy_corpus/retrieval.py", # ranking, fusion, effective-date filtering
)

RERUN = ".venv/bin/python -m src.cli eval --stub --diff"


def watched(path: str) -> bool:
    if any(path.endswith(suffix) for suffix in WATCHED_SUFFIXES):
        return True
    # Skills are versioned policy — a change there is a prompt change.
    return path.endswith("/SKILL.md") and "/.claude/skills/" in path


def label(path: str) -> str:
    """Last two path components — every skill file is called SKILL.md, so the
    bare filename would not say which one changed."""
    return "/".join(path.split("/")[-2:])


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0  # never break the turn over a malformed payload

    tool_input = payload.get("tool_input") or {}
    tool_response = payload.get("tool_response") or {}
    path = tool_input.get("file_path") or tool_response.get("filePath") or ""

    if not path or not watched(path):
        return 0

    print(
        json.dumps(
            {
                "systemMessage": (
                    f"{label(path)} changed — the committed eval baseline may now be "
                    f"stale. Re-run: {RERUN}"
                )
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
