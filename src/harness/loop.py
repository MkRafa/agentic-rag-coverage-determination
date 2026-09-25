"""The agentic loop.

Deterministic control flow. Subagents reason; this module decides what happens
next, and the gate decides the outcome. Every iteration is capped, metered and
traced.

    pre-flight → [ plan → retrieve → grade ]×N → synthesize → verify → gate
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from ..agents import grader, planner, retriever, synthesizer, verifier
from ..config import SETTINGS
from ..contracts import (
    CoverageRequest,
    Determination,
    GateDecision,
    GradeReport,
    RunResult,
    VerificationReport,
)
from ..llm import CallTimeout, ModelCallError, ModelClient
from ..mcp_client import PolicyCorpus
from .budget import Budget, BudgetExceeded
from .gate import decide
from .preflight import run_preflight
from .trace import Trace


async def run(
    payload: dict[str, Any],
    client: ModelClient,
    corpus: PolicyCorpus,
    budget: Budget,
    trace: Trace,
    *,
    store: Any | None = None,
) -> RunResult:
    """Wall-clock-bounded entry point. Always returns a RunResult — a run that
    blows its time budget is a REVIEW, not an exception the caller has to guess
    at."""
    started = time.monotonic()
    try:
        return await asyncio.wait_for(
            _run(payload, client, corpus, budget, trace, store=store),
            timeout=SETTINGS.run_timeout_s,
        )
    except asyncio.TimeoutError:
        elapsed = time.monotonic() - started
        trace.event("timeout", phase="run", limit_s=SETTINGS.run_timeout_s,
                    elapsed_s=round(elapsed, 2), **budget.snapshot())
        gate = GateDecision(
            state="REVIEW",
            reasons=[f"run exceeded its {SETTINGS.run_timeout_s}s wall-clock budget"],
        )
        trace.event("gate", state=gate.state, reasons=gate.reasons)
        return RunResult(
            run_id=trace.run_id, gate=gate,
            latency_s=round(elapsed, 3), **_usage(budget),
        )


async def _run(
    payload: dict[str, Any],
    client: ModelClient,
    corpus: PolicyCorpus,
    budget: Budget,
    trace: Trace,
    *,
    store: Any | None = None,
) -> RunResult:
    started = time.monotonic()

    # ---- 0. pre-flight (no LLM) ------------------------------------------
    pre = run_preflight(payload, store=store)
    trace.event(
        "preflight",
        ok=pre.ok,
        mean_confidence=pre.mean_confidence,
        field_confidence=pre.field_confidence,
        redactions=pre.redactions,
        injection_flags=pre.injection_flags,
        halt_reason=pre.halt_reason,
    )
    if not pre.ok:
        gate = GateDecision(state="HALT", reasons=[pre.halt_reason or "pre-flight failed"])
        trace.event("gate", state=gate.state, reasons=gate.reasons)
        return RunResult(
            run_id=trace.run_id,
            gate=gate,
            latency_s=round(time.monotonic() - started, 3),
            **_usage(budget),
        )

    request = pre.request

    # ---- 1..N. plan → retrieve → grade -----------------------------------
    clauses: list[dict[str, Any]] = []
    report: GradeReport | None = None
    gap: str | None = None
    tried: list[str] = []
    iterations = 0
    exhausted = False

    try:
        for iteration in range(1, SETTINGS.max_iterations + 1):
            iterations = iteration

            plan = await planner.plan(client, request, previous_gap=gap, tried=tried)
            trace.event(
                "plan",
                iteration=iteration,
                sub_queries=[sq.model_dump() for sq in plan.sub_queries],
                required_facts=plan.required_facts,
                reasoning=plan.reasoning,
            )
            tried.extend(sq.query for sq in plan.sub_queries)

            selection = await retriever.retrieve(client, corpus, request, plan)
            trace.event(
                "retrieve",
                iteration=iteration,
                clause_ids=selection.clause_ids,
                queries_run=selection.queries_run,
                notes=selection.notes,
            )

            clauses = await _hydrate(corpus, selection.clause_ids, request.as_of_date)
            trace.event("hydrate", iteration=iteration, resolved=len(clauses),
                        requested=len(selection.clause_ids))

            report = await grader.grade(client, request, clauses, iteration=iteration)
            trace.event(
                "grade",
                iteration=iteration,
                sufficient=report.sufficient,
                gap=report.gap,
                contradictions=report.contradictions,
                grades={g.clause_id: g.grade for g in report.graded},
            )

            if report.sufficient:
                break

            gap = report.gap
            if report.next_query:
                tried.append(report.next_query)
            if iteration == SETTINGS.max_iterations:
                exhausted = True
                trace.event("iterations_exhausted", iteration=iteration, gap=gap)

        # ---- synthesize --------------------------------------------------
        relevant = _relevant_clauses(clauses, report)
        determination: Determination = await synthesizer.synthesize(
            client, request, relevant, report or GradeReport(graded=[], sufficient=False)
        )
        trace.event(
            "synthesize",
            outcome=determination.outcome,
            confidence=determination.confidence,
            claims=len(determination.claims),
            cited=[c.citation.clause_id for c in determination.claims],
            gap=determination.gap,
        )

        # ---- verify (cold fetch + entailment) ----------------------------
        verification: VerificationReport = await verifier.verify(
            client, corpus, request, determination
        )
        trace.event(
            "verify",
            faithfulness=round(verification.faithfulness, 4),
            hallucinated=verification.hallucinated_clauses,
            stale=verification.stale_citations,
            checks=[c.model_dump() for c in verification.checks],
        )

    except (BudgetExceeded, CallTimeout, ModelCallError) as exc:
        if isinstance(exc, BudgetExceeded):
            event_kind = "budget_exceeded"
        elif isinstance(exc, CallTimeout):
            event_kind = "call_timeout"
        else:
            event_kind = "model_error"
        trace.event(event_kind, detail=str(exc), **budget.snapshot())
        gate = GateDecision(state="REVIEW", reasons=[f"run halted: {exc}"])
        trace.event("gate", state=gate.state, reasons=gate.reasons)
        return RunResult(
            run_id=trace.run_id, gate=gate, iterations=iterations,
            latency_s=round(time.monotonic() - started, 3), **_usage(budget),
        )

    # ---- gate (no LLM) ---------------------------------------------------
    gate = decide(
        determination,
        verification,
        contradictions=(report.contradictions if report else None),
        iterations_exhausted=exhausted,
    )
    trace.event("gate", state=gate.state, reasons=gate.reasons)
    trace.event("run_end", **budget.snapshot(), latency_s=round(time.monotonic() - started, 3))

    return RunResult(
        run_id=trace.run_id,
        gate=gate,
        determination=determination,
        verification=verification,
        iterations=iterations,
        latency_s=round(time.monotonic() - started, 3),
        retrieved_clause_ids=[c["clause_id"] for c in clauses],
        **_usage(budget),
    )


async def _hydrate(
    corpus: PolicyCorpus, clause_ids: list[str], as_of: str
) -> list[dict[str, Any]]:
    """Resolve clause ids to full clause records. Ids the Retriever invented
    simply do not resolve — they are dropped here and never reach the Grader."""
    from policy_corpus.retrieval import in_effect  # local import; keeps src decoupled

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cid in clause_ids:
        if cid in seen:
            continue
        seen.add(cid)
        fetched = await corpus.fetch_clause(cid)
        if fetched.get("error"):
            continue
        fetched["in_effect_on_as_of"] = in_effect(fetched, as_of) if as_of else None
        out.append(fetched)
    return out


def _relevant_clauses(
    clauses: list[dict[str, Any]], report: GradeReport | None
) -> list[dict[str, Any]]:
    """Only RELEVANT clauses reach the Synthesizer. STALE clauses are withheld
    deliberately — handing the writer a superseded version is how stale
    citations happen.

    When nothing was graded RELEVANT the Synthesizer gets nothing, and the
    correct output is INSUFFICIENT_EVIDENCE. Falling back to every retrieved
    clause here would hand it exactly the STALE and off-point clauses the
    Grader just rejected."""
    if report is None:
        return clauses
    keep = {g.clause_id for g in report.graded if g.grade == "RELEVANT"}
    return [c for c in clauses if c["clause_id"] in keep]


def _usage(budget: Budget) -> dict[str, Any]:
    return {
        "input_tokens": budget.input_tokens,
        "output_tokens": budget.output_tokens,
        "usd": round(budget.usd, 6),
    }
