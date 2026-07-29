"""Configuration for the coverage-determination harness.

Every knob that changes cost, latency or behaviour lives here so an eval sweep
can move one thing at a time and attribute the scorecard delta to it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Anthropic's most capable widely-available model. Per-role overrides below are
# the cost lever: drop a role to a cheaper model, rerun `eval`, and read the
# scorecard diff rather than asserting the swap was safe.
DEFAULT_MODEL = "claude-opus-5"


@dataclass(frozen=True)
class RoleConfig:
    model: str = DEFAULT_MODEL
    effort: str = "high"          # low | medium | high | xhigh | max
    max_tokens: int = 8_000
    thinking: bool = True


@dataclass(frozen=True)
class Settings:
    # -- agentic loop ---------------------------------------------------
    max_iterations: int = 3       # hard cap on plan → retrieve → grade cycles
    retrieval_k: int = 8          # clauses per sub-query

    # -- gate thresholds ------------------------------------------------
    min_confidence: float = 0.70          # Synthesizer confidence floor
    min_field_confidence: float = 0.55    # pre-flight halt threshold

    # -- budget ---------------------------------------------------------
    max_total_tokens: int = 400_000       # hard stop across the whole run
    max_usd: float = 2.00                 # hard stop, estimated

    # -- roles ----------------------------------------------------------
    planner: RoleConfig = field(default_factory=lambda: RoleConfig(effort="medium", max_tokens=4_000))
    retriever: RoleConfig = field(default_factory=lambda: RoleConfig(effort="medium", max_tokens=8_000))
    grader: RoleConfig = field(default_factory=lambda: RoleConfig(effort="medium", max_tokens=6_000))
    synthesizer: RoleConfig = field(default_factory=lambda: RoleConfig(effort="high", max_tokens=8_000))
    verifier: RoleConfig = field(default_factory=lambda: RoleConfig(effort="high", max_tokens=6_000))
    adversary: RoleConfig = field(default_factory=lambda: RoleConfig(effort="high", max_tokens=16_000))
    judge: RoleConfig = field(default_factory=lambda: RoleConfig(effort="high", max_tokens=4_000))

    # -- paths ----------------------------------------------------------
    traces_dir: Path = ROOT / "traces"
    skills_dir: Path = ROOT / ".claude" / "skills"
    corpus_path: Path = ROOT / "corpus" / "generated" / "corpus.json"

    # -- retrieval variants (recorded in the trace so a diff is attributable)
    embedder: str = field(default_factory=lambda: os.environ.get("CDA_EMBEDDER", "tfidf"))
    reranker: str = field(default_factory=lambda: os.environ.get("CDA_RERANKER", "date_aware"))


# Rough $/MTok for cost estimation in the budget meter. Not billing-accurate;
# it exists so a runaway loop stops before it is expensive.
PRICING = {
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def estimate_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    inp, out = PRICING.get(model, PRICING[DEFAULT_MODEL])
    return (input_tokens / 1e6) * inp + (output_tokens / 1e6) * out


SETTINGS = Settings()
