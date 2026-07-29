"""Token and cost meter with a hard stop.

An unbounded agentic loop is a cost incident, not a bug report. The meter is
checked before every model call and raises rather than degrading quietly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import SETTINGS, estimate_usd


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    max_total_tokens: int = SETTINGS.max_total_tokens
    max_usd: float = SETTINGS.max_usd
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    calls: int = 0
    per_role: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def check(self) -> None:
        if self.total_tokens >= self.max_total_tokens:
            raise BudgetExceeded(
                f"token budget exhausted: {self.total_tokens} >= {self.max_total_tokens}"
            )
        if self.usd >= self.max_usd:
            raise BudgetExceeded(f"cost budget exhausted: ${self.usd:.4f} >= ${self.max_usd:.2f}")

    def record(self, role: str, model: str, input_tokens: int, output_tokens: int) -> None:
        cost = estimate_usd(model, input_tokens, output_tokens)
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.usd += cost
        self.calls += 1
        slot = self.per_role.setdefault(
            role, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0}
        )
        slot["calls"] += 1
        slot["input_tokens"] += input_tokens
        slot["output_tokens"] += output_tokens
        slot["usd"] += cost

    def snapshot(self) -> dict[str, object]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "usd": round(self.usd, 6),
            "per_role": {
                k: {**v, "usd": round(v["usd"], 6)} for k, v in sorted(self.per_role.items())
            },
        }
