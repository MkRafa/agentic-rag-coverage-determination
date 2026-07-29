"""Layer 1 — retrieval scoring. Fully deterministic, no model in the path.

Answers "did we ever put the right clause in front of the writer?" separately
from "did the writer then say the right thing". Keeping the layers apart is what
makes a regression attributable: recall down and accuracy down is a retrieval
problem; recall flat and accuracy down is a synthesis problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from policy_corpus.retrieval import in_effect


@dataclass
class RetrievalScore:
    cases: int = 0
    scored_cases: int = 0          # cases with a non-empty gold key
    recall_hits: int = 0
    recall_total: int = 0
    stale_retrievals: int = 0
    total_retrieved: int = 0
    cases_with_any_stale: int = 0
    per_case: list[dict[str, Any]] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return self.recall_hits / self.recall_total if self.recall_total else 0.0

    @property
    def stale_rate(self) -> float:
        return self.stale_retrievals / self.total_retrieved if self.total_retrieved else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "cases": self.cases,
            "scored_cases": self.scored_cases,
            "gold_clause_recall": round(self.recall, 4),
            "stale_retrieval_rate": round(self.stale_rate, 4),
            "cases_with_any_stale_retrieval": self.cases_with_any_stale,
            "clauses_retrieved": self.total_retrieved,
        }


def score(
    results: list[tuple[dict[str, Any], Any]], store: Any
) -> RetrievalScore:
    """`results` is a list of (case, RunResult)."""
    s = RetrievalScore()
    for case, run in results:
        s.cases += 1
        retrieved = list(getattr(run, "retrieved_clause_ids", []) or [])
        gold = list(case.get("gold_clause_ids") or [])
        as_of = case.get("as_of_date") or ""

        stale = 0
        for cid in retrieved:
            clause = store.clauses.get(cid)
            if clause is not None and as_of and not in_effect(clause, as_of):
                stale += 1
        s.total_retrieved += len(retrieved)
        s.stale_retrievals += stale
        if stale:
            s.cases_with_any_stale += 1

        hits = 0
        if gold:
            s.scored_cases += 1
            hits = sum(1 for g in gold if g in retrieved)
            s.recall_hits += hits
            s.recall_total += len(gold)

        s.per_case.append(
            {
                "case_id": case["case_id"],
                "trap": case.get("trap", ""),
                "gold": gold,
                "retrieved_count": len(retrieved),
                "gold_hits": hits,
                "gold_recall": round(hits / len(gold), 4) if gold else None,
                "stale_retrieved": stale,
            }
        )
    return s
