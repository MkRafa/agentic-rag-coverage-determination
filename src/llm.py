"""Model client.

One narrow surface — `parse()` returns a validated pydantic object — so every
subagent has a typed contract and the harness can meter, trace and cap each
call uniformly.

Two backends:

* `AnthropicClient` — the real thing, `messages.parse` with adaptive thinking
  and per-role effort.
* `StubClient`     — deterministic canned responses keyed by role. Lets the
  harness, gate, traces and eval scorers be exercised end to end with no API
  key and no network, which is how the deterministic half of this system is
  tested.
"""

from __future__ import annotations

import json
import os
from typing import Any, Awaitable, Callable, Protocol, Sequence, TypeVar

from pydantic import BaseModel

from .config import RoleConfig, SETTINGS
from .harness.budget import Budget

T = TypeVar("T", bound=BaseModel)


class ModelClient(Protocol):
    async def parse(
        self,
        *,
        role: str,
        system: str,
        user: str,
        output_format: type[T],
        config: RoleConfig | None = None,
    ) -> T: ...

    async def run_tools(
        self,
        *,
        role: str,
        system: str,
        user: str,
        tools: Sequence[Any],
        output_format: type[T],
        config: RoleConfig | None = None,
        max_iterations: int = 8,
    ) -> T: ...


def role_config(role: str) -> RoleConfig:
    return getattr(SETTINGS, role, RoleConfig())


class AnthropicClient:
    def __init__(self, budget: Budget, trace: Any | None = None) -> None:
        from anthropic import AsyncAnthropic  # imported lazily so stub mode needs no key

        self._client = AsyncAnthropic()
        self.budget = budget
        self.trace = trace

    async def parse(
        self,
        *,
        role: str,
        system: str,
        user: str,
        output_format: type[T],
        config: RoleConfig | None = None,
    ) -> T:
        cfg = config or role_config(role)
        self.budget.check()

        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "max_tokens": cfg.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_format": output_format,
            "output_config": {"effort": cfg.effort},
        }
        if cfg.thinking:
            kwargs["thinking"] = {"type": "adaptive"}

        response = await self._client.messages.parse(**kwargs)

        usage = response.usage
        self.budget.record(role, cfg.model, usage.input_tokens, usage.output_tokens)
        if self.trace is not None:
            self.trace.event(
                "model_call",
                role=role,
                model=cfg.model,
                effort=cfg.effort,
                stop_reason=response.stop_reason,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
            )

        # Refusals surface as a successful HTTP response — check before reading.
        if response.stop_reason == "refusal":
            raise RuntimeError(
                f"model declined the {role} request "
                f"(category={getattr(response.stop_details, 'category', None)})"
            )
        parsed = response.parsed_output
        if parsed is None:
            raise RuntimeError(f"{role} returned no parseable structured output")
        return parsed

    async def run_tools(
        self,
        *,
        role: str,
        system: str,
        user: str,
        tools: Sequence[Any],
        output_format: type[T],
        config: RoleConfig | None = None,
        max_iterations: int = 8,
    ) -> T:
        """Agentic tool-calling turn. Used by the Retriever, which is the only
        online agent that decides for itself what to fetch."""
        cfg = config or role_config(role)
        self.budget.check()

        runner = self._client.beta.messages.tool_runner(
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=list(tools),
            output_format=output_format,
            output_config={"effort": cfg.effort},
            thinking={"type": "adaptive"} if cfg.thinking else {"type": "disabled"},
            max_iterations=max_iterations,
        )

        final = await runner.until_done()

        for message in getattr(runner, "messages", []) or []:
            usage = getattr(message, "usage", None)
            if usage is not None:
                self.budget.record(role, cfg.model, usage.input_tokens, usage.output_tokens)
        if not getattr(runner, "messages", None) and getattr(final, "usage", None):
            self.budget.record(role, cfg.model, final.usage.input_tokens, final.usage.output_tokens)

        if self.trace is not None:
            self.trace.event(
                "model_call", role=role, model=cfg.model, effort=cfg.effort,
                stop_reason=getattr(final, "stop_reason", None), tool_loop=True,
            )

        if getattr(final, "stop_reason", None) == "refusal":
            raise RuntimeError(f"model declined the {role} tool-calling turn")

        return _parse_final_json(final, output_format, role)


def _parse_final_json(message: Any, output_format: type[T], role: str) -> T:
    """`tool_runner` returns a raw message; structured output arrives as JSON in
    the final text block, so validate it here."""
    for block in reversed(getattr(message, "content", []) or []):
        if getattr(block, "type", None) != "text":
            continue
        text = block.text.strip()
        try:
            return output_format.model_validate_json(text)
        except Exception:  # noqa: BLE001 — fall back to the first JSON object present
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end > start:
                try:
                    return output_format.model_validate(json.loads(text[start : end + 1]))
                except Exception:  # noqa: BLE001
                    continue
    raise RuntimeError(f"{role} tool loop produced no parseable {output_format.__name__}")


Handler = Callable[[str, str], Awaitable[BaseModel]]


class StubClient:
    """Deterministic backend for offline testing of the non-model machinery."""

    def __init__(self, budget: Budget, trace: Any | None = None) -> None:
        self.budget = budget
        self.trace = trace
        self._handlers: dict[str, Handler] = {}
        self.calls: list[tuple[str, str]] = []

    def register(self, role: str, handler: Handler) -> None:
        self._handlers[role] = handler

    async def parse(
        self,
        *,
        role: str,
        system: str,
        user: str,
        output_format: type[T],
        config: RoleConfig | None = None,
    ) -> T:
        cfg = config or role_config(role)
        self.budget.check()
        self.calls.append((role, user))
        handler = self._handlers.get(role)
        if handler is None:
            raise RuntimeError(f"stub client has no handler registered for role '{role}'")
        result = await handler(system, user)
        if not isinstance(result, output_format):
            raise TypeError(
                f"stub handler for '{role}' returned {type(result).__name__}, "
                f"expected {output_format.__name__}"
            )
        # Charge a nominal amount so budget plumbing is exercised.
        self.budget.record(role, cfg.model, 1_000, 200)
        if self.trace is not None:
            self.trace.event("model_call", role=role, model="stub", effort=cfg.effort,
                             input_tokens=1_000, output_tokens=200)
        return result

    async def run_tools(
        self,
        *,
        role: str,
        system: str,
        user: str,
        tools: Sequence[Any],
        output_format: type[T],
        config: RoleConfig | None = None,
        max_iterations: int = 8,
    ) -> T:
        # The stub's Retriever handler talks to the real MCP server, so the
        # tool-calling path is still exercised — only the model choosing which
        # tool to call is replaced.
        return await self.parse(
            role=role, system=system, user=user, output_format=output_format, config=config
        )


def build_client(
    budget: Budget,
    trace: Any | None = None,
    stub: bool = False,
    corpus: Any | None = None,
) -> ModelClient:
    if stub:
        from .stubs import make_stub_client  # local import to avoid a cycle

        return make_stub_client(budget, trace, corpus)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export a key, or run with --stub to exercise "
            "the deterministic harness (pre-flight, retrieval, gate, traces, scorers) "
            "without model calls."
        )
    return AnthropicClient(budget, trace)
