"""Price a suite run before firing it.

Extrapolates from *measured* traces rather than guesses: run a couple of cases
live, then project the whole suite across models. The projection also shows what
prompt caching would do, because in this pipeline the system prompts — which
carry the skills — are 80% of input and are byte-identical on every case.

    cda cost                       # project from the most recent live traces
    cda cost --cases 152 --model claude-sonnet-5
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import ROOT

TRACES = ROOT / "traces"

# $ per million tokens: (input, output, cache_write_multiplier, cache_read_multiplier)
# Cache write is 1.25x input for the 5-minute TTL; reads are 0.1x.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-sonnet-5-intro": (2.00, 10.00),   # promotional, through 2026-08-31
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}
CACHE_WRITE = 1.25
CACHE_READ = 0.10


@dataclass
class Measured:
    cases: int = 0
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    per_role: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.cases > 0 and self.input_tokens > 0


def measure(trace_dir: Path | None = None, limit: int = 50) -> Measured:
    """Aggregate real usage from live traces. Stub traces are excluded — their
    token counts are fabricated and would make the projection fiction."""
    m = Measured()
    files = sorted((trace_dir or TRACES).glob("run-*.jsonl"))[-limit:]

    for path in files:
        events = [json.loads(line) for line in path.read_text().splitlines()]
        model_calls = [e for e in events if e.get("kind") == "model_call"]
        if not model_calls or any(e.get("model") == "stub" for e in model_calls):
            continue

        # The budget snapshot on run_end is the source of truth: it captures
        # multi-turn tool loops that a single model_call event can miss.
        end = next((e for e in reversed(events) if e.get("kind") == "run_end"), None)
        if end is None:
            continue

        m.cases += 1
        m.input_tokens += end.get("input_tokens", 0)
        m.output_tokens += end.get("output_tokens", 0)
        m.calls += end.get("calls", len(model_calls))
        for role, v in (end.get("per_role") or {}).items():
            slot = m.per_role.setdefault(role, {"calls": 0, "in": 0, "out": 0})
            slot["calls"] += int(v.get("calls", 0))
            slot["in"] += int(v.get("input_tokens", 0))
            slot["out"] += int(v.get("output_tokens", 0))
    return m


def project(
    m: Measured, *, cases: int, model: str, cached_system_tokens: int = 0
) -> dict[str, Any]:
    """Cost of `cases` runs at this model's rates.

    `cached_system_tokens` is the per-case system-prompt total. Anything cached
    is written once for the whole run and read thereafter, so it collapses from
    N x full price to 1 x 1.25 + (N-1) x 0.1.
    """
    inp, out = PRICES[model]
    in_per_case = m.input_tokens / m.cases
    out_per_case = m.output_tokens / m.cases

    naive_in = in_per_case * cases
    naive = (naive_in / 1e6) * inp + (out_per_case * cases / 1e6) * out

    cached = None
    if cached_system_tokens:
        volatile = max(0.0, in_per_case - cached_system_tokens)
        billed_in = (
            cached_system_tokens * CACHE_WRITE
            + cached_system_tokens * CACHE_READ * (cases - 1)
            + volatile * cases
        )
        cached = (billed_in / 1e6) * inp + (out_per_case * cases / 1e6) * out

    return {
        "model": model,
        "cases": cases,
        "calls": round(m.calls / m.cases * cases),
        "input_tokens": round(naive_in),
        "output_tokens": round(out_per_case * cases),
        "usd": round(naive, 2),
        "usd_cached": round(cached, 2) if cached is not None else None,
        "usd_per_case": round(naive / cases, 4),
    }


def render(m: Measured, cases: int, cached_system_tokens: int) -> str:
    if not m.ok:
        return (
            "no live traces found — projections need real usage.\n"
            "run a couple of cases first:  CDA_MODEL=claude-sonnet-5 cda eval --limit 2"
        )

    lines = [
        "",
        f"measured from {m.cases} live case(s): "
        f"{m.calls / m.cases:.1f} calls/case, "
        f"{m.input_tokens / m.cases:,.0f} in + {m.output_tokens / m.cases:,.0f} out per case",
        "",
        f"{'role':14s} {'calls/case':>11s} {'in/case':>10s} {'out/case':>10s}",
    ]
    for role, s in sorted(m.per_role.items(), key=lambda kv: -kv[1]["in"]):
        lines.append(
            f"{role:14s} {s['calls'] / m.cases:11.1f} "
            f"{s['in'] / m.cases:10,.0f} {s['out'] / m.cases:10,.0f}"
        )

    lines += [
        "",
        f"projected for {cases} cases:",
        "",
        f"  {'model':24s} {'calls':>7s} {'in (M)':>9s} {'out (M)':>9s} {'cost':>9s} {'+caching':>10s}",
    ]
    for model in (
        "claude-haiku-4-5",
        "claude-sonnet-5-intro",
        "claude-sonnet-5",
        "claude-opus-5",
    ):
        p = project(m, cases=cases, model=model, cached_system_tokens=cached_system_tokens)
        cached = f"${p['usd_cached']:.2f}" if p["usd_cached"] is not None else "—"
        lines.append(
            f"  {model:24s} {p['calls']:7d} {p['input_tokens'] / 1e6:9.2f} "
            f"{p['output_tokens'] / 1e6:9.2f} {'$' + format(p['usd'], '.2f'):>9s} {cached:>10s}"
        )

    lines += [
        "",
        f"  caching assumes {cached_system_tokens:,} system tokens/case are identical across",
        "  cases (they are — the skills are static), written once and read thereafter.",
        "  Requires cache_control on the system block, which is NOT yet wired.",
    ]
    return "\n".join(lines)
