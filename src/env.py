"""Minimal .env loader.

`.env` is gitignored and `.env.example` documents it, so the file has to
actually do something — a documented config file that is silently ignored is
worse than no file, because a missing key then looks like a mistyped key.

No python-dotenv dependency; this handles the subset that matters and nothing
more.

**The real environment always wins.** `export FOO=x` overrides `.env`, so a
stale file can never quietly shadow what you set on the command line.
"""

from __future__ import annotations

import os
from pathlib import Path

from .config import ROOT


def load_dotenv(path: Path | None = None, *, override: bool = False) -> list[str]:
    """Load KEY=VALUE lines. Returns the names loaded (never the values)."""
    env_path = path or ROOT / ".env"
    if not env_path.exists():
        return []

    loaded: list[str] = []
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue

        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()

        # Strip one layer of matching quotes, but only a matching pair.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]

        if not key:
            continue
        if key in os.environ and not override:
            continue  # a real env var beats the file

        os.environ[key] = value
        loaded.append(key)

    return loaded
