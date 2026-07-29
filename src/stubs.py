"""Deterministic stand-ins for the model, for offline testing.

These are not an attempt at a good agent. They exist so the machinery that has
to be correct regardless of the model — pre-flight, MCP retrieval, hydration,
the gate, budget accounting, traces, and the eval scorers — can be exercised and
asserted on with no API key and no network.

The stubs read the same rendered prompts the real model would, so the prompt
plumbing is under test too.

`CDA_STUB_FAULT` injects a specific failure so the gate's rejection paths are
covered rather than assumed:

    hallucinate   cite a clause id that does not exist
    paraphrase    cite a real clause but quote it non-verbatim
    stale         cite a version that was not in effect on the as-of date
    lowconf       return a determination below the confidence threshold
"""

from __future__ import annotations

import os
import re
from typing import Any

from .contracts import (
    AdversarialBatch,
    Citation,
    Claim,
    Determination,
    GradedClause,
    GradeReport,
    JudgeVerdict,
    Plan,
    RetrievalSelection,
    SubQuery,
    SupportJudgement,
    SupportVerdict,
)
from .harness.budget import Budget
from .llm import StubClient
from .mcp_client import PolicyCorpus

_FIELD = lambda name, text: (  # noqa: E731
    m.group(1).strip() if (m := re.search(rf"^{name}:\s*(.*)$", text, re.M)) else ""
)
_CLAUSE_BLOCK = re.compile(
    r'<clause id="([^"]+)"[^>]*section="([^"]*)"[^>]*on_[\d-]+="([^"]*)"[^>]*>\s*(.*?)\s*</clause>',
    re.S,
)
_EVIDENCE_BLOCK = re.compile(
    r'<clause id="([^"]+)"[^>]*section="([^"]*)"[^>]*>\s*(.*?)\s*</clause>', re.S
)
_PLAN_LINE = re.compile(r"^\d+\.\s+(.*?)(?:\s+\[codes:.*\])?$", re.M)
_ITEM = re.compile(r'<item index="(\d+)">', re.M)

NOT_COVERED_MARKERS = ("excluded", "investigational", "not medically necessary", "not covered")


def _request_from_prompt(text: str) -> dict[str, str]:
    return {
        "payer": _FIELD("payer", text),
        "plan": _FIELD("plan", text),
        "codes": _FIELD("procedure codes", text),
        "as_of": _FIELD("as-of date", text),
        "question": _FIELD("question", text),
    }


def _verbatim_span(clause_text: str, limit: int = 220) -> str:
    """A genuinely verbatim substring — the point is that verification passes on
    an honest citation, so a failure means the gate caught a real problem."""
    if len(clause_text) <= limit:
        return clause_text
    cut = clause_text[:limit]
    return cut[: cut.rfind(" ")] if " " in cut else cut


def make_stub_client(
    budget: Budget, trace: Any | None = None, corpus: PolicyCorpus | None = None
) -> StubClient:
    client = StubClient(budget, trace)
    fault = os.environ.get("CDA_STUB_FAULT", "").strip().lower()

    # -- planner ----------------------------------------------------------
    async def planner(_system: str, user: str) -> Plan:
        req = _request_from_prompt(user)
        codes = [c.strip() for c in req["codes"].split(",") if c.strip() and c != "(none)"]
        queries = [
            SubQuery(
                query=f"coverage criteria medically necessary {req['question'][:80]}",
                intent="find the base coverage criteria for the requested service",
                codes=codes,
            ),
            SubQuery(
                query="exclusions limitations investigational",
                intent="find any exclusion or limitation that would defeat coverage",
                codes=codes,
            ),
        ]
        if req["plan"] and req["plan"] != "(not supplied)":
            queries.append(
                SubQuery(
                    query="rider override waiver plan-specific amendment",
                    intent="find any plan rider that overrides the base policy",
                    codes=codes,
                )
            )
        return Plan(
            sub_queries=queries,
            required_facts=["diagnosis", "prior therapy history"],
            reasoning="stub decomposition: base criteria, exclusions, rider override",
        )

    # -- retriever: really calls MCP -------------------------------------
    async def retriever(_system: str, user: str) -> RetrievalSelection:
        assert corpus is not None, "stub retriever requires an MCP corpus session"
        req = _request_from_prompt(user)
        payer = req["payer"] or None
        plan_id = req["plan"] if req["plan"] and req["plan"] != "(not supplied)" else None
        as_of = req["as_of"] or None
        codes = [c.strip() for c in req["codes"].split(",") if c.strip() and c != "(none)"]

        queries = _PLAN_LINE.findall(user) or [req["question"]]
        clause_ids: list[str] = []
        ran: list[str] = []
        for q in queries:
            ran.append(q)
            for hit in await corpus.search(
                q, payer_id=payer, plan_id=plan_id, as_of_date=as_of, codes=codes, k=6
            ):
                if hit["clause_id"] not in clause_ids:
                    clause_ids.append(hit["clause_id"])

        if plan_id:
            riders = await corpus.plan_riders(plan_id)
            for rider in riders.get("riders", []):
                for cid in rider["clause_ids"]:
                    if cid not in clause_ids:
                        clause_ids.append(cid)

        return RetrievalSelection(
            clause_ids=clause_ids, queries_run=ran, notes="stub retriever (real MCP calls)"
        )

    # -- grader -----------------------------------------------------------
    async def grader(_system: str, user: str) -> GradeReport:
        graded: list[GradedClause] = []
        contradictions: list[str] = []
        relevant = 0
        for cid, section, in_effect, _text in _CLAUSE_BLOCK.findall(user):
            if in_effect == "NOT IN EFFECT":
                grade, reason = "STALE", "version not in effect on the as-of date"
            elif section in ("Coverage Criteria", "Exclusions", "Rider", "Step Therapy", "Limitations"):
                grade, reason = "RELEVANT", f"{section} clause bearing on the determination"
                relevant += 1
            elif "FAQ" in section or "Frequently" in section:
                grade, reason = "CONTRADICTORY", "secondary source; check against the bulletin"
                contradictions.append(f"{cid} is a provider FAQ that may conflict with the bulletin")
            else:
                grade, reason = "INSUFFICIENT", f"{section} clause does not bear on coverage"
            graded.append(GradedClause(clause_id=cid, grade=grade, reason=reason))

        return GradeReport(
            graded=graded,
            sufficient=relevant > 0,
            gap=None if relevant else "no in-effect coverage-criteria clause was retrieved",
            contradictions=contradictions,
            next_query=None if relevant else "policy bulletin coverage criteria for this code",
        )

    # -- synthesizer ------------------------------------------------------
    async def synthesizer(_system: str, user: str) -> Determination:
        blocks = _EVIDENCE_BLOCK.findall(user)
        if not blocks:
            return Determination(
                outcome="INSUFFICIENT_EVIDENCE",
                summary="No in-effect policy clause governing this service was retrieved.",
                claims=[],
                confidence=0.9,
                gap="no governing policy clause was found for this code and payer",
            )

        cid, _section, text = blocks[0]
        quote = _verbatim_span(text)
        outcome = (
            "NOT_COVERED"
            if any(m in text.lower() for m in NOT_COVERED_MARKERS)
            else "COVERED"
        )
        confidence = 0.86

        if fault == "hallucinate":
            cid = "MHP-MP-0000.v9.C9"
        elif fault == "paraphrase":
            quote = "the member is generally eligible for this benefit under the policy"
        elif fault == "stale":
            # Flip to the *other* version and quote that version's text, so the
            # verbatim check passes and the effective-date check is the only
            # thing that can fail. Quoting the original text here would trip
            # verbatim first and the date branch would never be exercised.
            flipped = re.sub(
                r"\.v(\d+)\.", lambda m: f".v{2 if m.group(1) == '1' else 1}.", cid
            )
            if corpus is not None and flipped != cid:
                other = await corpus.fetch_clause(flipped)
                if not other.get("error"):
                    cid, quote = flipped, _verbatim_span(other["text"])
        elif fault == "lowconf":
            confidence = 0.41

        return Determination(
            outcome=outcome,
            summary=f"Stub determination for {cid}: {outcome}.",
            claims=[Claim(text=f"The governing clause states: {quote[:120]}", citation=Citation(clause_id=cid, quote=quote))],
            confidence=confidence,
            gap=None,
            noted_conflicts=[],
        )

    # -- verifier (entailment judgement only) -----------------------------
    async def verifier(_system: str, user: str) -> SupportVerdict:
        return SupportVerdict(
            judgements=[
                SupportJudgement(
                    claim_index=int(idx),
                    supports=fault != "unsupported",
                    reason="stub entailment judgement",
                )
                for idx in _ITEM.findall(user)
            ]
        )

    async def judge(_system: str, _user: str) -> JudgeVerdict:
        return JudgeVerdict(
            reasoning_quality=3, clarity=3, appropriate_hedging=3, comments="stub verdict"
        )

    async def adversary(_system: str, _user: str) -> AdversarialBatch:
        return AdversarialBatch(cases=[])

    client.register("planner", planner)
    client.register("retriever", retriever)
    client.register("grader", grader)
    client.register("synthesizer", synthesizer)
    client.register("verifier", verifier)
    client.register("judge", judge)
    client.register("adversary", adversary)
    return client
