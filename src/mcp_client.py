"""Client for the policy-corpus MCP server.

Two consumers, deliberately different:

* The **Retriever** agent gets the tools converted into runnable tools and
  decides for itself what to call — genuinely agentic retrieval.
* The **Verifier** path calls `get_clause_by_id` through `fetch_clause()`
  directly from the harness. Ground truth must not be model-mediated: if the
  model could choose what to fetch, it could choose to fetch nothing and assert
  the citation was fine.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, AsyncIterator

from mcp import ClientSession, StdioServerParameters, stdio_client

from .config import ROOT


# The stdio transport does not inherit the parent environment, so retrieval
# knobs have to be forwarded explicitly. Without this the embedder/reranker
# sweep silently does nothing and every variant scores identically.
_FORWARDED_ENV = (
    "CDA_EMBEDDER",
    "CDA_RERANKER",
    "CDA_CORPUS",
    "CDA_VECTOR_BACKEND",
    "CDA_PINECONE_INDEX",
    "CDA_PINECONE_NAMESPACE",
    "CDA_PINECONE_MODEL",
    "CDA_PINECONE_REGION",
    "PINECONE_API_KEY",
    "PATH",
    "PYTHONPATH",
)


def _server_params() -> StdioServerParameters:
    env = {k: v for k in _FORWARDED_ENV if (v := os.environ.get(k)) is not None}
    env.setdefault("PYTHONPATH", str(ROOT))
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "policy_corpus.server"],
        cwd=str(ROOT),
        env=env,
    )


def _unwrap(result: Any) -> Any:
    """Pull the payload out of an MCP CallToolResult."""
    structured = getattr(result, "structuredContent", None) or getattr(
        result, "structured_content", None
    )
    if structured is not None:
        # MCPServer wraps non-dict returns under a "result" key.
        if isinstance(structured, dict) and set(structured) == {"result"}:
            return structured["result"]
        return structured
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
    return None


@dataclass
class PolicyCorpus:
    """Thin typed facade over the MCP session."""

    session: ClientSession
    tools: list[Any]

    async def _call(self, name: str, **kwargs: Any) -> Any:
        result = await self.session.call_tool(name=name, arguments=kwargs)
        return _unwrap(result)

    async def search(
        self,
        query: str,
        *,
        payer_id: str | None = None,
        plan_id: str | None = None,
        as_of_date: str | None = None,
        codes: list[str] | None = None,
        k: int = 8,
        include_superseded: bool = False,
    ) -> list[dict[str, Any]]:
        return await self._call(
            "search_policies",
            query=query,
            payer_id=payer_id,
            plan_id=plan_id,
            as_of_date=as_of_date,
            codes=codes,
            k=k,
            include_superseded=include_superseded,
        ) or []

    async def fetch_clause(self, clause_id: str) -> dict[str, Any]:
        return await self._call("get_clause_by_id", clause_id=clause_id) or {
            "error": "clause_not_found",
            "clause_id": clause_id,
        }

    async def versions(self, policy_id: str) -> dict[str, Any]:
        return await self._call("list_policy_versions", policy_id=policy_id) or {}

    async def code(self, code: str) -> dict[str, Any]:
        return await self._call("lookup_code", code=code) or {}

    async def plan_riders(self, plan_id: str) -> dict[str, Any]:
        return await self._call("get_plan_riders", plan_id=plan_id) or {}


@contextlib.asynccontextmanager
async def policy_corpus() -> AsyncIterator[PolicyCorpus]:
    """Spawn the policy-corpus server over stdio for the duration of the block."""
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            yield PolicyCorpus(session=session, tools=list(listed.tools))
