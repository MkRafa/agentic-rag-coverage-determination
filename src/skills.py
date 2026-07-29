"""Skill loader.

Skills are markdown files under `.claude/skills/<name>/SKILL.md`. They hold the
domain policy — citation shape, criteria logic, refusal rules — so that policy
is versioned in git and diffable, rather than buried in a prompt string. An
agent declares which skills it needs; this module assembles them into the
system prompt.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .config import SETTINGS


@lru_cache(maxsize=None)
def load_skill(name: str) -> str:
    path = SETTINGS.skills_dir / name / "SKILL.md"
    if not path.exists():
        raise FileNotFoundError(f"skill '{name}' not found at {path}")
    text = path.read_text()
    # Strip YAML frontmatter — the description there is for humans and tooling.
    if text.startswith("---"):
        _, _, rest = text.partition("---")
        _, _, body = rest.partition("---")
        text = body
    return text.strip()


def compose(*names: str) -> str:
    """Join the named skills into one block for a system prompt."""
    parts = []
    for name in names:
        parts.append(f"<skill name=\"{name}\">\n{load_skill(name)}\n</skill>")
    return "\n\n".join(parts)


def available() -> list[str]:
    if not SETTINGS.skills_dir.exists():
        return []
    return sorted(p.name for p in SETTINGS.skills_dir.iterdir() if (p / "SKILL.md").exists())


def fingerprint() -> dict[str, int]:
    """Byte length per skill — recorded in traces so a prompt-policy change is
    visible as a diff when a scorecard moves."""
    return {name: len(load_skill(name)) for name in available()}


__all__ = ["load_skill", "compose", "available", "fingerprint", "Path"]
