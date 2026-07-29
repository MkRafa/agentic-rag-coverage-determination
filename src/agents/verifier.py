"""Verifier — closes the loop that most RAG systems leave open.

Split deliberately:

  deterministic (Python, cold MCP fetch)   does the clause exist? is the quote
                                           verbatim? was it in effect on the
                                           as-of date?
  model-mediated (this agent)              does the quote actually entail the
                                           claim it is cited for?

The model sees only the request, the claim, and the clause text. It never sees
the plan, the retrieval trace, or the Synthesizer's reasoning — a verifier that
can read the reasoning can be talked into agreeing with it.
"""

from __future__ import annotations

import re

from policy_corpus.retrieval import in_effect

from ..contracts import (
    CitationCheck,
    CoverageRequest,
    Determination,
    SupportVerdict,
    VerificationReport,
)
from ..llm import ModelClient
from ..mcp_client import PolicyCorpus

SYSTEM = """You are the Verifier in a coverage-determination pipeline.

For each numbered item you are given a claim and the verbatim text of the single
clause cited for it. Decide one thing per item: does the quoted text, on its own,
entail the claim?

You are not judging whether the claim is true, whether the determination is
sensible, or whether a better clause exists. You are judging entailment from the
quote alone. Apply it strictly:

  - A claim that is broader than the quote is not supported. "CGM is covered for
    diabetes" is not supported by a quote covering only Type 1.
  - A claim that adds a condition the quote does not state is not supported.
  - A claim that drops a condition the quote does state is not supported.
  - A quote that establishes scope or definitions does not support a conclusion
    about medical necessity.
  - Close paraphrase of what the quote says is fine. Inference beyond it is not.

You have no tools and no access to any other clause. If the quote is insufficient,
say so — a false pass here is the exact failure this pipeline exists to catch.
"""


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _quote_is_verbatim(quote: str, clause_text: str) -> bool:
    q, c = _normalise(quote), _normalise(clause_text)
    return bool(q) and q in c


def _prompt(request: CoverageRequest, items: list[tuple[int, str, str]]) -> str:
    lines = [
        "<context>",
        f"payer: {request.payer_id}   as-of date: {request.as_of_date}",
        f"question: {request.question}",
        "</context>",
        "",
    ]
    for idx, claim, clause_text in items:
        lines += [
            f"<item index=\"{idx}\">",
            f"  <claim>{claim}</claim>",
            f"  <cited_clause_text>{clause_text}</cited_clause_text>",
            "</item>",
            "",
        ]
    lines.append(
        "Return one judgement per item, using the same index. Judge entailment "
        "from the cited clause text alone."
    )
    return "\n".join(lines)


async def verify(
    client: ModelClient,
    corpus: PolicyCorpus,
    request: CoverageRequest,
    determination: Determination,
) -> VerificationReport:
    checks: list[CitationCheck] = []
    judgeable: list[tuple[int, str, str]] = []

    # -- deterministic pass: cold fetch, no model in the path ---------------
    for i, claim in enumerate(determination.claims):
        clause_id = claim.citation.clause_id
        fetched = await corpus.fetch_clause(clause_id)

        if fetched.get("error"):
            checks.append(
                CitationCheck(
                    claim_index=i,
                    clause_id=clause_id,
                    exists=False,
                    quote_verbatim=False,
                    in_effect_on_as_of=False,
                    supports_claim=False,
                    reason="clause id does not exist in the corpus",
                )
            )
            continue

        verbatim = _quote_is_verbatim(claim.citation.quote, fetched["text"])
        effective = in_effect(fetched, request.as_of_date) if request.as_of_date else False

        check = CitationCheck(
            claim_index=i,
            clause_id=clause_id,
            exists=True,
            quote_verbatim=verbatim,
            in_effect_on_as_of=effective,
            supports_claim=False,
            reason="",
        )
        checks.append(check)

        # Only spend a model call on citations that survived the mechanical
        # checks — a non-verbatim quote has already failed.
        if verbatim and effective:
            judgeable.append((i, claim.text, fetched["text"]))

    # -- model-mediated pass: entailment only -------------------------------
    if judgeable:
        verdict = await client.parse(
            role="verifier",
            system=SYSTEM,
            user=_prompt(request, judgeable),
            output_format=SupportVerdict,
        )
        by_index = {j.claim_index: j for j in verdict.judgements}
        for check in checks:
            j = by_index.get(check.claim_index)
            if j is not None:
                check.supports_claim = j.supports
                check.reason = j.reason
            elif check.exists and check.quote_verbatim and check.in_effect_on_as_of:
                # An unjudged item is a failure, not a pass. Silence is not consent.
                check.reason = "verifier returned no judgement for this citation"

    for check in checks:
        if not check.reason:
            if not check.quote_verbatim:
                check.reason = "quoted text is not a verbatim substring of the cited clause"
            elif not check.in_effect_on_as_of:
                check.reason = "cited clause version was not in effect on the as-of date"

    return VerificationReport(checks=checks)
