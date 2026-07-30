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

import asyncio
import json
import os
import time
from functools import lru_cache
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


class CallTimeout(RuntimeError):
    """A model call or tool loop exceeded its wall-clock budget."""


# Capability differences are per-model, not per-generation-you-remember. Adaptive
# thinking and `effort` are 4.6+ features: sending either to Haiku 4.5 is a 400,
# not a graceful ignore. Resolved live from the Models API once per process, with
# a conservative fallback so stub and offline paths never depend on the network.
_CAPS_FALLBACK = {"adaptive": False, "enabled": False, "effort": False}


@lru_cache(maxsize=16)
def _capabilities(model: str) -> dict[str, bool]:
    try:
        import anthropic  # noqa: PLC0415

        caps = anthropic.Anthropic().models.retrieve(model).capabilities
        caps = caps.model_dump() if hasattr(caps, "model_dump") else dict(caps)
        thinking = caps.get("thinking") or {}
        types = thinking.get("types") or {}
        return {
            "adaptive": bool((types.get("adaptive") or {}).get("supported")),
            "enabled": bool((types.get("enabled") or {}).get("supported")),
            "effort": bool((caps.get("effort") or {}).get("supported")),
        }
    except Exception:  # noqa: BLE001 — never let a capability probe break a run
        return dict(_CAPS_FALLBACK)


def build_request_options(cfg: RoleConfig) -> dict[str, Any]:
    """The thinking and effort parameters this model will actually accept."""
    caps = _capabilities(cfg.model)
    options: dict[str, Any] = {}

    if caps["effort"]:
        options["output_config"] = {"effort": cfg.effort}

    if not cfg.thinking:
        if caps["adaptive"] or caps["enabled"]:
            options["thinking"] = {"type": "disabled"}
    elif caps["adaptive"]:
        options["thinking"] = {"type": "adaptive"}
    elif caps["enabled"]:
        # Legacy models take a fixed budget, which must leave room for the answer.
        budget = max(1024, min(cfg.max_tokens // 4, 8_000))
        if budget < cfg.max_tokens:
            options["thinking"] = {"type": "enabled", "budget_tokens": budget}

    return options


class AnthropicClient:
    def __init__(self, budget: Budget, trace: Any | None = None) -> None:
        from anthropic import AsyncAnthropic  # imported lazily so stub mode needs no key

        # max_retries covers connection errors, 408/409/429 and 5xx with
        # exponential backoff. Per-request timeouts are set per role below.
        self._client = AsyncAnthropic(max_retries=SETTINGS.max_retries)
        self.budget = budget
        self.trace = trace

    def _for(self, cfg: RoleConfig) -> Any:
        return self._client.with_options(timeout=cfg.timeout_s)

    def _on_timeout(self, role: str, cfg: RoleConfig, phase: str, elapsed: float) -> CallTimeout:
        if self.trace is not None:
            self.trace.event(
                "timeout", role=role, phase=phase,
                limit_s=cfg.timeout_s, elapsed_s=round(elapsed, 2),
            )
        return CallTimeout(
            f"{role} {phase} exceeded its wall-clock budget after {elapsed:.1f}s "
            f"(limit {cfg.timeout_s}s x {SETTINGS.max_retries + 1} attempts)"
        )

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
            **build_request_options(cfg),
        }

        import anthropic  # noqa: PLC0415

        started = time.monotonic()
        try:
            response = await self._for(cfg).messages.parse(**kwargs)
        except anthropic.APITimeoutError as exc:
            raise self._on_timeout(role, cfg, "request", time.monotonic() - started) from exc
        except anthropic.RateLimitError as exc:
            # The SDK already retried this max_retries times.
            if self.trace is not None:
                self.trace.event("rate_limited", role=role, model=cfg.model)
            raise RuntimeError(
                f"{role} was rate limited after {SETTINGS.max_retries} retries"
            ) from exc

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

        runner = self._for(cfg).beta.messages.tool_runner(
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=list(tools),
            output_format=output_format,
            max_iterations=max_iterations,
            **build_request_options(cfg),
        )

        # `max_iterations` bounds the number of turns, not wall clock: a loop of
        # slow tool calls can still run for a very long time. The SDK has no
        # wall-clock cap on a tool loop at all, so impose one here.
        started = time.monotonic()
        try:
            final = await asyncio.wait_for(
                runner.until_done(), timeout=SETTINGS.tool_loop_timeout_s
            )
        except asyncio.TimeoutError as exc:
            elapsed = time.monotonic() - started
            if self.trace is not None:
                self.trace.event(
                    "timeout", role=role, phase="tool_loop",
                    limit_s=SETTINGS.tool_loop_timeout_s, elapsed_s=round(elapsed, 2),
                )
            raise CallTimeout(
                f"{role} tool loop exceeded {SETTINGS.tool_loop_timeout_s}s "
                f"(max_iterations={max_iterations})"
            ) from exc

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
