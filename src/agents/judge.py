"""Judge — offline, contamination-isolated.

Sees the request and the determination. Never the trace, never the gold key,
never the retrieved clauses. It scores the qualities the deterministic scorers
cannot: is the reasoning sound, is it readable, is the hedging proportionate.

Correctness is *not* its job. Determination accuracy and citation faithfulness
are measured deterministically against the gold key and the Verifier; asking a
model to also grade those would launder a measurable number into an opinion.
"""

from __future__ import annotations

from ..contracts import CoverageRequest, Determination, JudgeVerdict
from ..llm import ModelClient

SYSTEM = """You are an independent reviewer scoring the quality of a coverage
determination. You are shown the request and the determination. You are not shown
the retrieval trace, the correct answer, or the policy corpus.

Score three things, 1-5 each:

  reasoning_quality    Does the determination work through the criteria rather
                       than asserting a conclusion? Are the cited quotes doing
                       real work in the argument?
  clarity              Could a utilisation-review nurse act on this without
                       going back to the policy documents themselves? Is the
                       outcome stated plainly up front?
  appropriate_hedging  Is the stated confidence proportionate to the evidence
                       shown? Over-confidence on thin evidence and hand-wringing
                       on clean evidence both score low. A well-named refusal is
                       a 5, not a 1.

Do not reward length. Do not penalise a determination for reaching an outcome
you would not have reached — you cannot see the corpus, so you cannot know.
Judge the argument as presented.
"""


def _prompt(request: CoverageRequest, determination: Determination) -> str:
    lines = [
        "<request>",
        f"payer: {request.payer_id}   plan: {request.plan_id or '(none)'}",
        f"procedure codes: {', '.join(request.procedure_codes) or '(none)'}",
        f"as-of date: {request.as_of_date}",
        f"question: {request.question}",
        "</request>",
        "",
        "<determination>",
        f"outcome: {determination.outcome}",
        f"confidence: {determination.confidence}",
        f"summary: {determination.summary}",
    ]
    for i, claim in enumerate(determination.claims, 1):
        lines += [
            f"  claim {i}: {claim.text}",
            f"    cites {claim.citation.clause_id}: \"{claim.citation.quote}\"",
        ]
    if determination.gap:
        lines.append(f"gap: {determination.gap}")
    if determination.noted_conflicts:
        lines += ["noted conflicts:", *(f"  - {c}" for c in determination.noted_conflicts)]
    lines.append("</determination>")
    return "\n".join(lines)


async def judge(
    client: ModelClient, request: CoverageRequest, determination: Determination
) -> JudgeVerdict:
    return await client.parse(
        role="judge",
        system=SYSTEM,
        user=_prompt(request, determination),
        output_format=JudgeVerdict,
    )
