"""The local-model path, tested without a local model.

`_post_json` is the only thing that talks to Ollama, so replacing it with a
scripted fake exercises the client's real logic: schema-constrained parsing,
the tool loop against an MCP session, and turning bad output into a REVIEW
rather than an aborted eval.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from policy_corpus.store import get_store
from src import llm
from src.config import estimate_usd, is_local
from src.contracts import Plan, RetrievalSelection
from src.harness import loop as loop_module
from src.harness.budget import Budget
from src.harness.trace import Trace
from src.llm import ModelCallError, OllamaClient, StubClient

MODEL = "ollama/qwen2.5:7b"


class FakeOllama:
    """Replays scripted /api/chat responses and records every request body."""

    def __init__(self, *messages: dict) -> None:
        self.script = list(messages)
        self.requests: list[dict] = []

    def __call__(self, url: str, body: dict | None, timeout: float) -> dict:
        # Copy: the client keeps appending to the same messages list.
        self.requests.append(copy.deepcopy(body))
        message = self.script.pop(0)
        return {"message": message, "prompt_eval_count": 100, "eval_count": 20, "done_reason": "stop"}


@pytest.fixture
def client() -> OllamaClient:
    return OllamaClient(Budget(), Trace())


def _cfg():
    from src.config import RoleConfig

    return RoleConfig(model=MODEL, max_tokens=1_000, timeout_s=5.0)


async def test_parse_sends_schema_and_context_window(monkeypatch, client):
    fake = FakeOllama({"content": json.dumps({"clause_ids": ["A"], "queries_run": [], "notes": ""})})
    monkeypatch.setattr(llm, "_post_json", fake)

    result = await client.parse(
        role="retriever", system="s", user="u", output_format=RetrievalSelection, config=_cfg()
    )

    assert result.clause_ids == ["A"]
    body = fake.requests[0]
    assert body["model"] == "qwen2.5:7b", "the ollama/ prefix must be stripped"
    assert body["format"] == RetrievalSelection.model_json_schema()
    assert body["options"]["num_ctx"] == client.num_ctx, "Ollama's default window truncates silently"


async def test_output_that_breaks_the_contract_is_a_model_error(monkeypatch, client):
    monkeypatch.setattr(llm, "_post_json", FakeOllama({"content": '{"wrong": true}'}))
    with pytest.raises(ModelCallError, match="does not match RetrievalSelection"):
        await client.parse(
            role="retriever", system="s", user="u", output_format=RetrievalSelection, config=_cfg()
        )


async def test_prompt_that_cannot_fit_the_context_is_refused(monkeypatch, client):
    monkeypatch.setattr(llm, "_post_json", FakeOllama())  # must never be called
    huge = "x" * (client.num_ctx * 4)
    with pytest.raises(ModelCallError, match="CDA_OLLAMA_NUM_CTX"):
        await client.parse(
            role="planner", system="s", user=huge, output_format=Plan, config=_cfg()
        )


async def test_tool_loop_executes_calls_then_asks_for_the_schema(monkeypatch, client):
    fake = FakeOllama(
        {"content": "", "tool_calls": [
            {"function": {"name": "search_policies", "arguments": {"query": "cgm"}}}
        ]},
        {"content": "done searching"},
        {"content": json.dumps({"clause_ids": ["MHP-MP-0142.v1.C2"]})},
    )
    monkeypatch.setattr(llm, "_post_json", fake)

    calls = []

    class Session:
        async def call_tool(self, *, name, arguments):
            calls.append((name, arguments))
            return SimpleNamespace(structuredContent={"result": [{"clause_id": "MHP-MP-0142.v1.C2"}]})

    tool = SimpleNamespace(name="search_policies", description="search", input_schema={"type": "object"})
    result = await client.run_tools(
        role="retriever", system="s", user="u", tools=[tool], session=Session(),
        output_format=RetrievalSelection, config=_cfg(),
    )

    assert calls == [("search_policies", {"query": "cgm"})]
    assert result.clause_ids == ["MHP-MP-0142.v1.C2"]
    # Tools offered during the loop; schema only on the final turn.
    assert "tools" in fake.requests[0] and "format" not in fake.requests[0]
    assert fake.requests[-1]["format"] == RetrievalSelection.model_json_schema()
    tool_message = fake.requests[1]["messages"][-1]
    assert tool_message["role"] == "tool" and "MHP-MP-0142.v1.C2" in tool_message["content"]


async def test_a_model_error_mid_run_is_a_review_not_an_aborted_eval():
    budget, trace = Budget(), Trace()
    client = StubClient(budget, trace)

    async def broken_planner(_system: str, _user: str) -> Plan:
        raise ModelCallError("planner output does not match Plan")

    client.register("planner", broken_planner)
    result = await loop_module.run(
        {
            "payer_id": "MHP", "plan_id": "MHP-HMO-BASE", "procedure_codes": ["A9276"],
            "as_of_date": "2024-03-15", "question": "Is CGM covered?",
        },
        client, corpus=None, budget=budget, trace=trace, store=get_store(),
    )

    assert result.gate.state == "REVIEW"
    assert any("does not match Plan" in r for r in result.gate.reasons)
    assert trace.of_kind("model_error")


def test_local_models_cost_nothing_and_are_recognised():
    assert is_local(MODEL) and not is_local("claude-sonnet-5")
    assert estimate_usd(MODEL, 1_000_000, 1_000_000) == 0.0


def test_live_baselines_never_overwrite_the_stub_baseline(monkeypatch):
    from evals import runner

    assert runner.baseline_path(stub=True) == runner.BASELINE
    live = runner.baseline_path(stub=False)
    assert live.parent == runner.BASELINES_DIR and live != runner.BASELINE


async def test_runaway_parallel_tool_calls_are_capped(monkeypatch, client):
    """qwen2.5:7b once emitted dozens of parallel calls in one turn; their
    results overflowed the context. Only the first few run, and the model is
    told the rest were skipped."""
    burst = [{"function": {"name": "search_policies", "arguments": {"query": f"q{i}"}}} for i in range(12)]
    fake = FakeOllama(
        {"content": "", "tool_calls": burst},
        {"content": "enough"},
        {"content": json.dumps({"clause_ids": []})},
    )
    monkeypatch.setattr(llm, "_post_json", fake)
    ran = []

    class Session:
        async def call_tool(self, *, name, arguments):
            ran.append(arguments["query"])
            return SimpleNamespace(structuredContent={"result": "x" * 50_000})

    tool = SimpleNamespace(name="search_policies", description="", input_schema={"type": "object"})
    await client.run_tools(
        role="retriever", system="s", user="u", tools=[tool], session=Session(),
        output_format=RetrievalSelection, config=_cfg(),
    )

    assert ran == ["q0", "q1", "q2", "q3"]
    history = fake.requests[1]["messages"]
    assert len(history[2]["tool_calls"]) == OllamaClient.MAX_TOOL_CALLS_PER_TURN
    assert all(len(m["content"]) < 7_000 for m in history if m.get("role") == "tool")
    assert "not run" in history[-1]["content"]
    assert fake.requests[0]["options"]["num_predict"] == OllamaClient.TOOL_TURN_MAX_TOKENS
