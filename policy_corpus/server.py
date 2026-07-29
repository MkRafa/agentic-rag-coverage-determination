"""policy-corpus MCP server.

The integration boundary. Every piece of corpus access the agents have goes
through these five tools; nothing else in the system reads the corpus files.
That is what makes the vector store swappable without touching agent logic —
change `CDA_EMBEDDER` / `CDA_RERANKER` (or the whole store implementation) and
the Planner, Retriever, Grader, Synthesizer and Verifier are untouched.

Read-only: there is no tool here that mutates anything.

Run standalone:  python -m policy_corpus.server
"""

from __future__ import annotations

from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from .store import get_store

server = MCPServer(
    name="policy-corpus",
    instructions=(
        "Read-only access to a synthetic corpus of payer medical-policy bulletins, "
        "plan riders and code sets. Clauses are the atomic citable unit. Policies are "
        "versioned with effective date ranges — always pass as_of_date so superseded "
        "versions are filtered out, and use get_clause_by_id to confirm any clause you "
        "intend to cite."
    ),
)


@server.tool(
    description=(
        "Hybrid lexical+dense search over policy and rider clauses. Filters to the "
        "versions in effect on as_of_date unless include_superseded is true."
    )
)
def search_policies(
    query: Annotated[str, Field(description="Natural-language sub-query to search for.")],
    payer_id: Annotated[str | None, Field(description="e.g. MHP, CSM, NBA.")] = None,
    plan_id: Annotated[
        str | None, Field(description="Restricts riders to those attached to this plan.")
    ] = None,
    as_of_date: Annotated[
        str | None, Field(description="ISO date (YYYY-MM-DD). Filters to versions then in effect.")
    ] = None,
    codes: Annotated[
        list[str] | None, Field(description="CPT/HCPCS/ICD codes to boost in ranking.")
    ] = None,
    k: Annotated[int, Field(description="Number of clauses to return.", ge=1, le=25)] = 8,
    include_superseded: Annotated[
        bool, Field(description="Return superseded versions too. Use to detect staleness.")
    ] = False,
) -> list[dict[str, Any]]:
    return get_store().search(
        query,
        payer_id=payer_id,
        plan_id=plan_id,
        as_of_date=as_of_date,
        codes=codes,
        k=k,
        include_superseded=include_superseded,
    )


@server.tool(
    description=(
        "Fetch one clause verbatim by its clause_id. This is the ground truth used to "
        "verify that a citation exists, is quoted accurately, and was in effect."
    )
)
def get_clause_by_id(
    clause_id: Annotated[str, Field(description="e.g. MHP-MP-0142.v1.C2")],
) -> dict[str, Any]:
    clause = get_store().clause(clause_id)
    if clause is None:
        return {"error": "clause_not_found", "clause_id": clause_id}
    return clause


@server.tool(
    description="List every version of a policy or rider with its effective date range."
)
def list_policy_versions(
    policy_id: Annotated[str, Field(description="e.g. MHP-MP-0142 or MHP-RID-STEP-WAIVE")],
) -> dict[str, Any]:
    versions = get_store().versions(policy_id)
    if versions is None:
        return {"error": "policy_not_found", "policy_id": policy_id}
    return {"policy_id": policy_id, "versions": versions}


@server.tool(description="Look up a CPT / HCPCS / ICD-10 code and the policies that govern it.")
def lookup_code(
    code: Annotated[str, Field(description="e.g. 95249, A9276, E11.9")],
) -> dict[str, Any]:
    entry = get_store().code(code)
    if entry is None:
        return {"error": "code_not_found", "code": code}
    return entry


@server.tool(
    description=(
        "List the riders attached to a plan and which base policies each overrides. "
        "Riders control where they conflict with a base policy."
    )
)
def get_plan_riders(
    plan_id: Annotated[str, Field(description="e.g. MHP-PPO-GOLD")],
) -> dict[str, Any]:
    riders = get_store().plan_riders(plan_id)
    if riders is None:
        return {"error": "plan_not_found", "plan_id": plan_id}
    return riders


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
