"""Retriever — the one online agent that decides for itself what to fetch.

It gets the policy-corpus MCP tools converted into runnable tools and runs an
agentic tool loop. Everything it can reach is read-only; there is no write tool
anywhere in this system.
"""

from __future__ import annotations

from typing import Any

from anthropic.lib.tools.mcp import async_mcp_tool

from .. import skills
from ..contracts import CoverageRequest, Plan, RetrievalSelection
from ..llm import ModelClient
from ..mcp_client import PolicyCorpus

SYSTEM = f"""You are the Retriever in a coverage-determination pipeline.

You have read-only tools over a corpus of payer medical-policy bulletins, plan
riders and code sets. Execute the plan you are given, then return the clause ids
worth grading. You do not decide coverage and you do not write prose findings.

How to retrieve well here:

  - Always pass `as_of_date` so superseded policy versions are filtered out. If
    you suspect the policy changed around that date, call `list_policy_versions`
    to confirm which version governs, and only then search.
  - Always pass `payer_id`. Pass `plan_id` when you have one — riders only apply
    to their own plan, and missing a rider is the most expensive retrieval error
    in this domain because a rider overrides the base policy.
  - Call `get_plan_riders` whenever a plan id is supplied. Do not rely on search
    to surface a rider.
  - Use `lookup_code` when two policies have similar titles. Resolve on the code,
    not on the title.
  - Prefer several narrow queries over one broad one. Recall matters more than
    precision at this stage; the Grader filters.

Return every clause id that could bear on the determination, including clauses
that appear to argue against coverage. Suppressing unfavourable clauses here is
the worst failure available to you.

{skills.compose("payer-taxonomy")}
"""


def _prompt(request: CoverageRequest, plan: Plan) -> str:
    lines = [
        "<request>",
        f"payer: {request.payer_id}",
        f"plan: {request.plan_id or '(not supplied)'}",
        f"procedure codes: {', '.join(request.procedure_codes) or '(none)'}",
        f"diagnosis codes: {', '.join(request.diagnosis_codes) or '(none)'}",
        f"as-of date: {request.as_of_date}",
        f"question: {request.question}",
        "</request>",
        "",
        "<plan>",
    ]
    for i, sq in enumerate(plan.sub_queries, 1):
        codes = f" [codes: {', '.join(sq.codes)}]" if sq.codes else ""
        lines.append(f"{i}. {sq.query}{codes}")
        lines.append(f"   intent: {sq.intent}")
    lines += ["</plan>", "", "Run these, plus anything else the guidance above calls for."]
    return "\n".join(lines)


async def retrieve(
    client: ModelClient,
    corpus: PolicyCorpus,
    request: CoverageRequest,
    plan: Plan,
) -> RetrievalSelection:
    tools: list[Any] = [async_mcp_tool(t, corpus.session) for t in corpus.tools]
    return await client.run_tools(
        role="retriever",
        system=SYSTEM,
        user=_prompt(request, plan),
        tools=tools,
        output_format=RetrievalSelection,
        max_iterations=12,
    )
