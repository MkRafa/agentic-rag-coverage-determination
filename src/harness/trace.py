"""Per-run JSONL trace.

One line per event, appended as the run proceeds so a hung run still leaves
evidence. The trace is the only place the reasoning path is recorded — the
Verifier deliberately never sees it.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from ..config import SETTINGS
from .. import skills


class Trace:
    def __init__(self, run_id: str | None = None, directory: Path | None = None) -> None:
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.dir = directory or SETTINGS.traces_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"run-{self.run_id}.jsonl"
        self.started = time.monotonic()
        self.events: list[dict[str, Any]] = []
        self.event(
            "run_start",
            embedder=SETTINGS.embedder,
            reranker=SETTINGS.reranker,
            max_iterations=SETTINGS.max_iterations,
            skills=skills.fingerprint(),
        )

    def event(self, kind: str, **payload: Any) -> None:
        record = {
            "run_id": self.run_id,
            "t": round(time.monotonic() - self.started, 4),
            "kind": kind,
            **payload,
        }
        self.events.append(record)
        with self.path.open("a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def of_kind(self, kind: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["kind"] == kind]
