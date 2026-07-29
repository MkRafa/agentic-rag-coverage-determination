"""The gate. Pure Python, no model call.

This is the only place the outcome of a run is decided. Subagents reason; the
gate commits. Keeping it deterministic is what makes "faithfulness == 1.0" an
enforced invariant rather than an aspiration.
"""

from __future__ import annotations

from ..config import SETTINGS
from ..contracts import Determination, GateDecision, VerificationReport


def decide(
    determination: Determination | None,
    verification: VerificationReport | None,
    *,
    contradictions: list[str] | None = None,
    iterations_exhausted: bool = False,
) -> GateDecision:
    reasons: list[str] = []

    if determination is None:
        return GateDecision(
            state="REVIEW", reasons=["no determination was produced by the synthesizer"]
        )

    # An explicit insufficiency is a success state, not a failure — but it goes
    # to a human with the gap named, never straight back to a requester.
    if determination.outcome == "INSUFFICIENT_EVIDENCE":
        gap = determination.gap or "evidence insufficient; no specific gap was named"
        return GateDecision(state="REFUSE", reasons=[gap])

    if verification is None or not verification.checks:
        return GateDecision(
            state="REVIEW", reasons=["determination carries no verifiable citations"]
        )

    faithfulness = verification.faithfulness
    if faithfulness < 1.0:
        failed = [c for c in verification.checks if not c.passed]
        for c in failed:
            if not c.exists:
                reasons.append(f"cited clause {c.clause_id} does not exist in the corpus")
            elif not c.quote_verbatim:
                reasons.append(f"quote attributed to {c.clause_id} is not verbatim")
            elif not c.in_effect_on_as_of:
                reasons.append(f"clause {c.clause_id} was not in effect on the as-of date")
            elif not c.supports_claim:
                reasons.append(f"clause {c.clause_id} does not support the claim it is cited for")

    if contradictions:
        reasons.extend(f"unresolved contradiction: {c}" for c in contradictions)

    if determination.confidence < SETTINGS.min_confidence:
        reasons.append(
            f"stated confidence {determination.confidence:.2f} is below the "
            f"{SETTINGS.min_confidence:.2f} threshold"
        )

    if iterations_exhausted:
        reasons.append(
            f"iteration cap ({SETTINGS.max_iterations}) reached before evidence was sufficient"
        )

    if reasons:
        return GateDecision(state="REVIEW", reasons=reasons)

    return GateDecision(
        state="DETERMINE",
        reasons=[
            f"all {len(verification.checks)} citations verified against the corpus",
            "every cited clause was in effect on the as-of date",
            f"confidence {determination.confidence:.2f} at or above threshold",
        ],
    )
