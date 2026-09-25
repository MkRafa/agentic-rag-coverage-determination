"""Model client.

One narrow surface — `parse()` returns a validated pydantic object — so every
subagent has a typed contract and the harness can meter, trace and cap each
call uniformly.

Three backends:

* `AnthropicClient` — the real thing, `messages.parse` with adaptive thinking
  and per-role effort.
* `OllamaClient`    — a local model through Ollama's native API. Free and
  offline; selected by a model id like `ollama/qwen2.5:7b`.
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

from .config import LOCAL_PREFIX, RoleConfig, SETTINGS, is_local
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
        session: Any,
        output_format: type[T],
        config: RoleConfig | None = None,
        max_iterations: int = 8,
    ) -> T: ...


def role_config(role: str) -> RoleConfig:
    return getattr(SETTINGS, role, RoleConfig())


class CallTimeout(RuntimeError):
    """A model call or tool loop exceeded its wall-clock budget."""


class ModelCallError(RuntimeError):
    """The model answered, but not with anything usable: a refusal, output that
    does not match the contract, or a transport failure after retries. The loop
    turns this into a REVIEW for that case rather than aborting the whole eval."""


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
            raise ModelCallError(
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
            raise ModelCallError(
                f"model declined the {role} request "
                f"(category={getattr(response.stop_details, 'category', None)})"
            )
        parsed = response.parsed_output
        if parsed is None:
            raise ModelCallError(f"{role} returned no parseable structured output")
        return parsed

    async def run_tools(
        self,
        *,
        role: str,
        system: str,
        user: str,
        tools: Sequence[Any],
        session: Any,
        output_format: type[T],
        config: RoleConfig | None = None,
        max_iterations: int = 8,
    ) -> T:
        """Agentic tool-calling turn. Used by the Retriever, which is the only
        online agent that decides for itself what to fetch. `tools` are MCP tool
        definitions; `session` executes them."""
        from anthropic.lib.tools.mcp import async_mcp_tool  # noqa: PLC0415

        cfg = config or role_config(role)
        self.budget.check()

        runner = self._for(cfg).beta.messages.tool_runner(
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[async_mcp_tool(t, session) for t in tools],
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

        # A tool loop is several model turns; sum them so the trace carries the
        # same token detail a single call does. Omitting them made the retriever
        # look free — it is in fact the largest input consumer in the pipeline,
        # because every turn resends the conversation so far.
        turns = list(getattr(runner, "messages", []) or [])
        loop_in = loop_out = 0
        for message in turns:
            usage = getattr(message, "usage", None)
            if usage is not None:
                loop_in += usage.input_tokens
                loop_out += usage.output_tokens
        if not turns and getattr(final, "usage", None):
            loop_in, loop_out = final.usage.input_tokens, final.usage.output_tokens
        if loop_in or loop_out:
            self.budget.record(role, cfg.model, loop_in, loop_out)

        if self.trace is not None:
            self.trace.event(
                "model_call", role=role, model=cfg.model, effort=cfg.effort,
                stop_reason=getattr(final, "stop_reason", None), tool_loop=True,
                turns=len(turns) or 1,
                input_tokens=loop_in, output_tokens=loop_out,
            )

        if getattr(final, "stop_reason", None) == "refusal":
            raise ModelCallError(f"model declined the {role} tool-calling turn")

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
    raise ModelCallError(f"{role} tool loop produced no parseable {output_format.__name__}")


# ---------------------------------------------------------------------------
# local models via Ollama
# ---------------------------------------------------------------------------


def _post_json(url: str, body: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    """Blocking JSON request. Standard library only, so the local path adds no
    dependency; callers run it in a worker thread."""
    import urllib.request  # noqa: PLC0415

    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _tool_result_text(result: Any) -> str:
    from .mcp_client import _unwrap  # noqa: PLC0415 — avoid a module-level cycle

    payload = _unwrap(result)
    return payload if isinstance(payload, str) else json.dumps(payload, default=str)


class OllamaClient:
    """A local model through Ollama's native `/api/chat`.

    The native API rather than the OpenAI-compatible one, because only the
    native API takes `num_ctx` per request. Ollama's default context window is
    far smaller than these prompts (the Retriever's alone is ~13k tokens), and
    it truncates silently — the answers then look like a weak model rather
    than a cut-off prompt.

    Structured output uses Ollama's `format`, which constrains decoding to the
    contract's JSON schema. There are no retries: a local server that fails
    once will fail again, and a ModelCallError becomes a REVIEW for that case.

    The tool loop is fenced more tightly than the API path, because small
    models degenerate in a way frontier models do not: on the first live run
    qwen2.5:7b spent 304s emitting 8,000 tokens of parallel tool calls, whose
    results then overflowed the context. Hence the caps below.
    """

    # A tool-calling turn needs ~100 tokens; only the final answer needs more.
    TOOL_TURN_MAX_TOKENS = 1_024
    MAX_TOOL_CALLS_PER_TURN = 4
    TOOL_RESULT_MAX_CHARS = 6_000

    def __init__(self, budget: Budget, trace: Any | None = None) -> None:
        self.base_url = os.environ.get("CDA_OLLAMA_URL", "http://localhost:11434").rstrip("/")
        self.num_ctx = int(os.environ.get("CDA_OLLAMA_NUM_CTX", "32768"))
        self.budget = budget
        self.trace = trace

    def ensure_model(self, model: str) -> None:
        """Fail at startup, once, rather than as a REVIEW on every case."""
        name = model.removeprefix(LOCAL_PREFIX)
        try:
            tags = _post_json(f"{self.base_url}/api/tags", None, timeout=5)
        except OSError as exc:
            raise RuntimeError(
                f"cannot reach Ollama at {self.base_url} ({exc}). Start it with `ollama serve`."
            ) from exc
        names = {m.get("name") for m in tags.get("models", [])}
        if name not in names and f"{name}:latest" not in names:
            raise RuntimeError(f"Ollama has no model '{name}'. Pull it with `ollama pull {name}`.")

    async def _chat(
        self,
        role: str,
        cfg: RoleConfig,
        messages: list[dict[str, Any]],
        *,
        schema: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        self.budget.check()
        max_tokens = max_tokens or cfg.max_tokens

        # Ollama truncates an over-long prompt without saying so. ~4 chars per
        # token is rough, but an overflow big enough to matter is caught.
        approx_tokens = sum(len(str(m.get("content") or "")) for m in messages) // 4
        if approx_tokens > self.num_ctx - max_tokens:
            raise ModelCallError(
                f"{role} prompt is ~{approx_tokens} tokens, which does not fit a "
                f"{self.num_ctx}-token context with {max_tokens} reserved for output; "
                "raise CDA_OLLAMA_NUM_CTX"
            )

        body: dict[str, Any] = {
            "model": cfg.model.removeprefix(LOCAL_PREFIX),
            "messages": messages,
            "stream": False,
            "options": {"num_ctx": self.num_ctx, "temperature": 0, "num_predict": max_tokens},
        }
        if schema is not None:
            body["format"] = schema
        if tools:
            body["tools"] = tools

        started = time.monotonic()
        try:
            data = await asyncio.wait_for(
                asyncio.to_thread(_post_json, f"{self.base_url}/api/chat", body, cfg.timeout_s),
                timeout=cfg.timeout_s,
            )
        except (asyncio.TimeoutError, TimeoutError) as exc:
            elapsed = time.monotonic() - started
            if self.trace is not None:
                self.trace.event("timeout", role=role, phase="request",
                                 limit_s=cfg.timeout_s, elapsed_s=round(elapsed, 2))
            raise CallTimeout(f"{role} request exceeded {cfg.timeout_s}s on the local model") from exc
        except (OSError, ValueError) as exc:  # URLError / HTTPError / reset / non-JSON body
            raise ModelCallError(f"{role} call to Ollama failed: {exc}") from exc

        # With prompt caching Ollama reports only the tokens it had to
        # evaluate, so input counts are a floor. They are free either way.
        input_tokens = int(data.get("prompt_eval_count") or 0)
        output_tokens = int(data.get("eval_count") or 0)
        self.budget.record(role, cfg.model, input_tokens, output_tokens)
        if self.trace is not None:
            self.trace.event(
                "model_call", role=role, model=cfg.model,
                stop_reason=data.get("done_reason"),
                input_tokens=input_tokens, output_tokens=output_tokens,
                elapsed_s=round(time.monotonic() - started, 2),
            )
        return data

    @staticmethod
    def _validate(content: str, output_format: type[T], role: str) -> T:
        try:
            return output_format.model_validate_json(content)
        except Exception as exc:  # noqa: BLE001 — pydantic ValidationError or bad JSON
            raise ModelCallError(
                f"{role} output does not match {output_format.__name__}: {str(exc)[:300]}"
            ) from exc

    @staticmethod
    def _schema_instruction(output_format: type[T]) -> str:
        # `format` constrains decoding; restating the schema in the prompt is
        # what Ollama recommends for answer quality, since the model otherwise
        # never sees the field descriptions.
        return (
            "Respond with only a JSON object conforming to this schema:\n"
            + json.dumps(output_format.model_json_schema())
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
        messages = [
            {"role": "system", "content": f"{system}\n\n{self._schema_instruction(output_format)}"},
            {"role": "user", "content": user},
        ]
        data = await self._chat(role, cfg, messages, schema=output_format.model_json_schema())
        return self._validate(data["message"].get("content") or "", output_format, role)

    async def run_tools(
        self,
        *,
        role: str,
        system: str,
        user: str,
        tools: Sequence[Any],
        session: Any,
        output_format: type[T],
        config: RoleConfig | None = None,
        max_iterations: int = 8,
    ) -> T:
        """A plain tool loop, then one schema-constrained turn for the answer.
        Asking for the schema *during* the loop stops small models calling tools
        at all — they go straight to writing JSON."""
        cfg = config or role_config(role)
        definitions = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    # mcp 2.x names it input_schema; 1.x used inputSchema.
                    "parameters": getattr(t, "input_schema", None)
                    or getattr(t, "inputSchema", None)
                    or {"type": "object"},
                },
            }
            for t in tools
        ]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        async def loop() -> T:
            for _ in range(max_iterations):
                data = await self._chat(
                    role, cfg, messages, tools=definitions, max_tokens=self.TOOL_TURN_MAX_TOKENS
                )
                message = {"role": "assistant", **data["message"]}
                calls = message.get("tool_calls") or []
                if not calls:
                    messages.append(message)
                    break
                skipped = calls[self.MAX_TOOL_CALLS_PER_TURN:]
                calls = calls[: self.MAX_TOOL_CALLS_PER_TURN]
                # Keep only the calls that run, so every tool_call in the
                # history is answered by exactly one tool message.
                messages.append({**message, "tool_calls": calls})
                if skipped and self.trace is not None:
                    self.trace.event("tool_calls_skipped", role=role, skipped=len(skipped))
                for call in calls:
                    fn = call.get("function", {})
                    args = fn.get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    try:
                        text = _tool_result_text(
                            await session.call_tool(name=fn.get("name", ""), arguments=args)
                        )
                    except Exception as exc:  # noqa: BLE001 — report to the model, keep going
                        text = json.dumps({"error": str(exc)})
                    if len(text) > self.TOOL_RESULT_MAX_CHARS:
                        text = text[: self.TOOL_RESULT_MAX_CHARS] + " …[truncated; narrow the query]"
                    if self.trace is not None:
                        self.trace.event("tool_call", role=role, tool=fn.get("name"),
                                         arguments=args, result_chars=len(text))
                    messages.append({"role": "tool", "content": text, "tool_name": fn.get("name", "")})
                if skipped:
                    messages.append({
                        "role": "user",
                        "content": (
                            f"{len(skipped)} further tool call(s) were not run: at most "
                            f"{self.MAX_TOOL_CALLS_PER_TURN} per turn. Call again if still needed."
                        ),
                    })

            messages.append({
                "role": "user",
                "content": "Tool use is finished. " + self._schema_instruction(output_format),
            })
            data = await self._chat(role, cfg, messages, schema=output_format.model_json_schema())
            return self._validate(data["message"].get("content") or "", output_format, role)

        started = time.monotonic()
        try:
            return await asyncio.wait_for(loop(), timeout=SETTINGS.tool_loop_timeout_s)
        except asyncio.TimeoutError as exc:
            elapsed = time.monotonic() - started
            if self.trace is not None:
                self.trace.event("timeout", role=role, phase="tool_loop",
                                 limit_s=SETTINGS.tool_loop_timeout_s, elapsed_s=round(elapsed, 2))
            raise CallTimeout(
                f"{role} tool loop exceeded {SETTINGS.tool_loop_timeout_s}s on the local model"
            ) from exc


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
        # Nominal tokens so budget plumbing is exercised; priced at zero.
        self.budget.record(role, "stub", 1_000, 200)
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
        session: Any,
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
    if is_local(SETTINGS.synthesizer.model):
        client = OllamaClient(budget, trace)
        client.ensure_model(SETTINGS.synthesizer.model)
        return client
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export a key, or run with --stub to exercise "
            "the deterministic harness (pre-flight, retrieval, gate, traces, scorers) "
            "without model calls."
        )
    return AnthropicClient(budget, trace)
