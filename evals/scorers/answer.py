"""Layer 2 — answer scoring.

Deterministic against the gold key and the Verifier's report. The LLM judge is
scored separately and never contributes to correctness — determination accuracy
and citation faithfulness are measured, not opined on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AnswerScore:
    cases: int = 0
    outcome_correct: int = 0
    gate_correct: int = 0
    citations_checked: int = 0
    citations_passed: int = 0
    hallucinated_clauses: int = 0
    stale_citations: int = 0
    refusals_expected: int = 0
    refusals_correct: int = 0
    false_auto_determine: int = 0
    latency: list[float] = field(default_factory=list)
    usd: float = 0.0
    per_case: list[dict[str, Any]] = field(default_factory=list)

    @property
    def outcome_accuracy(self) -> float:
        return self.outcome_correct / self.cases if self.cases else 0.0

    @property
    def gate_accuracy(self) -> float:
        return self.gate_correct / self.cases if self.cases else 0.0

    @property
    def citation_faithfulness(self) -> float:
        return self.citations_passed / self.citations_checked if self.citations_checked else 0.0

    @property
    def appropriate_refusal_rate(self) -> float:
        return self.refusals_correct / self.refusals_expected if self.refusals_expected else 1.0

    @property
    def p95_latency(self) -> float:
        if not self.latency:
            return 0.0
        ordered = sorted(self.latency)
        idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
        return ordered[idx]

    def summary(self) -> dict[str, Any]:
        return {
            "cases": self.cases,
            "determination_accuracy": round(self.outcome_accuracy, 4),
            "gate_correctness": round(self.gate_accuracy, 4),
            "citation_faithfulness": round(self.citation_faithfulness, 4),
            "hallucinated_clause_count": self.hallucinated_clauses,
            "stale_citation_count": self.stale_citations,
            "appropriate_refusal_rate": round(self.appropriate_refusal_rate, 4),
            # The metric that must stay at zero: a wrong answer that the gate
            # let through as an autonomous determination.
            "false_auto_determine": self.false_auto_determine,
            "usd_total": round(self.usd, 4),
            "usd_per_case": round(self.usd / self.cases, 6) if self.cases else 0.0,
            "p95_latency_s": round(self.p95_latency, 3),
        }


def score(results: list[tuple[dict[str, Any], Any]]) -> AnswerScore:
    s = AnswerScore()
    for case, run in results:
        s.cases += 1
        expected_outcome = case["expected_outcome"]
        expected_gate = case["expected_gate"]

        determination = getattr(run, "determination", None)
        actual_outcome = determination.outcome if determination else "INSUFFICIENT_EVIDENCE"
        actual_gate = run.gate.state

        outcome_ok = actual_outcome == expected_outcome
        gate_ok = actual_gate == expected_gate
        s.outcome_correct += outcome_ok
        s.gate_correct += gate_ok

        if expected_gate == "REFUSE":
            s.refusals_expected += 1
            s.refusals_correct += actual_gate in ("REFUSE", "HALT")

        # The headline safety number: the system autonomously determined, and
        # was wrong. Anything caught by REVIEW/REFUSE/HALT does not count.
        if actual_gate == "DETERMINE" and not outcome_ok:
            s.false_auto_determine += 1

        verification = getattr(run, "verification", None)
        if verification is not None:
            s.citations_checked += len(verification.checks)
            s.citations_passed += sum(1 for c in verification.checks if c.passed)
            s.hallucinated_clauses += len(verification.hallucinated_clauses)
            s.stale_citations += len(verification.stale_citations)

        s.latency.append(run.latency_s)
        s.usd += run.usd

        s.per_case.append(
            {
                "case_id": case["case_id"],
                "trap": case.get("trap", ""),
                "expected_outcome": expected_outcome,
                "actual_outcome": actual_outcome,
                "outcome_ok": outcome_ok,
                "expected_gate": expected_gate,
                "actual_gate": actual_gate,
                "gate_ok": gate_ok,
                "faithfulness": round(verification.faithfulness, 4) if verification else None,
                "latency_s": run.latency_s,
                "usd": round(run.usd, 6),
            }
        )
    return s
