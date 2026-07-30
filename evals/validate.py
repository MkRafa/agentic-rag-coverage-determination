"""Case validator — the gate on the answer key itself.

A generated case with a wrong label is worse than no case. It does not fail
loudly; it quietly changes what the eval measures, and the next `--set-baseline`
writes the mistake down as truth.

So nothing enters the suite unchecked. Every property that can be verified
against the corpus is verified here: clause ids must exist, gold clauses must
actually be in effect on the case's own as-of date, riders must belong to the
plan the case names, and the outcome/gate pair must be coherent.

What this cannot check is whether the clinical facts in the narrative genuinely
satisfy the criteria — that is the judgement the case exists to test. Those need
a human read. The point of this file is to make that human read short by
mechanically rejecting everything that does not need judgement.

    cda evals validate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from policy_corpus.retrieval import in_effect

VALID_OUTCOMES = {"COVERED", "NOT_COVERED", "INSUFFICIENT_EVIDENCE"}
VALID_GATES = {"DETERMINE", "REVIEW", "REFUSE", "HALT"}
REQUIRED = ("case_id", "as_of_date", "narrative", "question", "expected_outcome", "expected_gate")


@dataclass
class Problem:
    case_id: str
    field: str
    detail: str

    def __str__(self) -> str:
        return f"{self.case_id}: [{self.field}] {self.detail}"


@dataclass
class Report:
    problems: list[Problem] = field(default_factory=list)
    checked: int = 0

    @property
    def ok(self) -> bool:
        return not self.problems

    def add(self, case_id: str, field_: str, detail: str) -> None:
        self.problems.append(Problem(case_id, field_, detail))

    def render(self) -> str:
        if self.ok:
            return f"all {self.checked} cases valid"
        lines = [f"{len(self.problems)} problem(s) across {self.checked} cases:"]
        lines += [f"  - {p}" for p in self.problems]
        return "\n".join(lines)


def validate(cases: list[dict[str, Any]], store: Any) -> Report:
    report = Report()
    seen_ids: set[str] = set()
    known_codes = set(store.codes)

    for case in cases:
        report.checked += 1
        cid = case.get("case_id") or "<no case_id>"

        for key in REQUIRED:
            if not case.get(key) and case.get(key) != "":
                report.add(cid, key, "missing")
        if not case.get("case_id"):
            continue

        if cid in seen_ids:
            report.add(cid, "case_id", "duplicate id — later case would shadow the earlier one")
        seen_ids.add(cid)

        outcome = case.get("expected_outcome")
        gate = case.get("expected_gate")
        if outcome not in VALID_OUTCOMES:
            report.add(cid, "expected_outcome", f"{outcome!r} not in {sorted(VALID_OUTCOMES)}")
        if gate not in VALID_GATES:
            report.add(cid, "expected_gate", f"{gate!r} not in {sorted(VALID_GATES)}")

        # A refusal *is* an insufficient-evidence outcome; the two must agree or
        # the case is scoring two different things.
        if outcome == "INSUFFICIENT_EVIDENCE" and gate not in ("REFUSE", "HALT"):
            report.add(cid, "expected_gate", f"INSUFFICIENT_EVIDENCE implies REFUSE or HALT, got {gate}")
        if gate == "REFUSE" and outcome != "INSUFFICIENT_EVIDENCE":
            report.add(cid, "expected_outcome", f"REFUSE implies INSUFFICIENT_EVIDENCE, got {outcome}")

        as_of = case.get("as_of_date", "")
        try:
            date.fromisoformat(as_of)
        except (ValueError, TypeError):
            report.add(cid, "as_of_date", f"{as_of!r} is not an ISO date")
            continue

        payer = case.get("payer_id") or ""
        plan = case.get("plan_id")
        known_payers = {p["payer_id"] for p in store.payers}

        # An empty payer is legitimate only for a case that expects a pre-flight halt.
        if payer and payer not in known_payers:
            report.add(cid, "payer_id", f"{payer!r} is not a payer in the corpus")
        if not payer and gate != "HALT":
            report.add(cid, "payer_id", "empty payer is only valid for an expected HALT")

        if plan and plan not in store.plans:
            report.add(cid, "plan_id", f"{plan!r} is not a plan in the corpus")
        if plan and payer and plan in store.plans and store.plans[plan]["payer_id"] != payer:
            report.add(cid, "plan_id", f"{plan} belongs to {store.plans[plan]['payer_id']}, not {payer}")

        codes = case.get("procedure_codes") or []
        if not codes and gate != "HALT":
            report.add(cid, "procedure_codes", "empty outside an expected HALT")
        unknown = [c for c in codes if c.upper() not in known_codes]
        if unknown and gate != "HALT":
            report.add(cid, "procedure_codes", f"unrecognised codes {unknown}")

        _validate_gold(report, case, cid, store, as_of, payer, plan, gate)

    return report


def _validate_gold(report, case, cid, store, as_of, payer, plan, gate) -> None:
    gold = case.get("gold_clause_ids") or []

    if not gold and gate not in ("REFUSE", "HALT"):
        report.add(cid, "gold_clause_ids", "empty — a scorable case must name the clauses it rests on")

    for gid in gold:
        clause = store.clauses.get(gid)
        if clause is None:
            # The single most likely generated-case defect.
            report.add(cid, "gold_clause_ids", f"{gid} does not exist in the corpus")
            continue

        if not in_effect(clause, as_of):
            report.add(
                cid,
                "gold_clause_ids",
                f"{gid} was not in effect on {as_of} "
                f"({clause['effective_start']} to {clause['effective_end'] or 'present'}) — "
                "a gold key the system is correct to ignore",
            )

        if payer and clause["payer_id"] != payer:
            report.add(cid, "gold_clause_ids", f"{gid} belongs to {clause['payer_id']}, case is {payer}")

        if clause["source"] == "rider" and clause.get("plan_id") != plan:
            report.add(
                cid,
                "gold_clause_ids",
                f"{gid} is a rider on {clause.get('plan_id')}, case plan is {plan} — "
                "unreachable, so the case can never pass",
            )
