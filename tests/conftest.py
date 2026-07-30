"""Make a fresh clone work.

The corpus is a generated artifact and is gitignored — committing it would mean
reviewing diffs of generated JSON. But everything downstream depends on it, so
without this a fresh clone fails at import with a missing-file error before the
reader sees a single test pass.

Generation is deterministic and takes milliseconds, so building it on demand
costs nothing and removes a setup step from anyone evaluating the repo.
"""

from __future__ import annotations

import pytest

from src.config import SETTINGS


@pytest.fixture(scope="session", autouse=True)
def ensure_corpus() -> None:
    if SETTINGS.corpus_path.exists():
        return
    from corpus.generate import main as build

    build()
