"""Timeout and retry guardrails.

The SDK retries transport failures but does not bound total wall clock, and
bounds an agentic tool loop not at all. These tests cover the caps the harness
adds on top — a hung Retriever is a REVIEW with a stated reason, never a run
that hangs forever or an exception the caller has to interpret.
"""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

from policy_corpus.store import get_store
from src import config as config_module
from src.config import SETTINGS, RoleConfig
from src.contracts import Plan, SubQuery
from src.harness import loop as loop_module
from src.harness.budget import Budget
from src.harness.trace import Trace
from src.llm import CallTimeout, StubClient

PAYLOAD = {
    "payer_id": "MHP",
    "plan_id": "MHP-HMO-BASE",
    "procedure_codes": ["A9276"],
    "diagnosis_codes": ["E10.9"],
    "as_of_date": "2024-03-15",
    "narrative": "Type 1 diabetes.",
    "question": "Is personal-use CGM covered?",
}


@pytest.fixture
def short_run_timeout(monkeypatch):
    """SETTINGS is a frozen dataclass, so swap the whole object."""
    tightened = dataclasses.replace(SETTINGS, run_timeout_s=0.25)
    monkeypatch.setattr(loop_module, "SETTINGS", tightened)
    return tightened


async def test_run_timeout_returns_review_not_an_exception(short_run_timeout):
    budget = Budget()
    trace = Trace()
    client = StubClient(budget, trace)

    async def slow_planner(_system: str, _user: str) -> Plan:
        await asyncio.sleep(5)
        return Plan(sub_queries=[SubQuery(query="q", intent="i")], reasoning="r")

    client.register("planner", slow_planner)

    result = await loop_module.run(
        PAYLOAD, client, corpus=None, budget=budget, trace=trace, store=get_store()
    )

    assert result.gate.state == "REVIEW"
    assert any("wall-clock budget" in r for r in result.gate.reasons)
    assert result.determination is None
    # The cap must actually bite rather than the sleep completing.
    assert result.latency_s < 2.0
    assert trace.of_kind("timeout"), "the timeout must be recorded in the trace"


async def test_call_timeout_is_converted_to_review():
    """A per-call timeout raised mid-loop degrades to REVIEW, and the partial
    token spend is still reported."""
    budget = Budget()
    trace = Trace()
    client = StubClient(budget, trace)

    async def failing_planner(_system: str, _user: str) -> Plan:
        raise CallTimeout("planner request exceeded its wall-clock budget after 90.0s")

    client.register("planner", failing_planner)

    result = await loop_module.run(
        PAYLOAD, client, corpus=None, budget=budget, trace=trace, store=get_store()
    )

    assert result.gate.state == "REVIEW"
    assert any("run halted" in r for r in result.gate.reasons)
    assert trace.of_kind("call_timeout")


async def test_preflight_halt_is_reached_before_any_timeout_matters(short_run_timeout):
    """A malformed request must halt instantly, not burn the run budget."""
    budget = Budget()
    trace = Trace()
    client = StubClient(budget, trace)  # no handlers registered at all

    result = await loop_module.run(
        {**PAYLOAD, "payer_id": ""}, client, corpus=None,
        budget=budget, trace=trace, store=get_store(),
    )

    assert result.gate.state == "HALT"
    assert budget.calls == 0


# ---------------------------------------------------------------------------
# configuration invariants
# ---------------------------------------------------------------------------


def test_every_role_has_a_bounded_timeout():
    roles = ["planner", "retriever", "grader", "synthesizer", "verifier", "adversary", "judge"]
    for role in roles:
        cfg: RoleConfig = getattr(SETTINGS, role)
        assert cfg.timeout_s > 0, role
        assert cfg.timeout_s <= 600, f"{role} timeout is unreasonably long"


def test_tool_loop_cap_is_tighter_than_the_run_cap():
    """Otherwise the run cap fires first and the tool-loop message never shows,
    making a hung Retriever indistinguishable from a slow pipeline."""
    assert SETTINGS.tool_loop_timeout_s < SETTINGS.run_timeout_s


def test_worst_case_single_call_fits_inside_the_run_budget():
    """The SDK retries timeouts, so one logical call can cost
    timeout_s x (max_retries + 1). That must not exceed the whole-run cap."""
    attempts = SETTINGS.max_retries + 1
    for role in ["planner", "retriever", "grader", "synthesizer", "verifier"]:
        cfg: RoleConfig = getattr(SETTINGS, role)
        assert cfg.timeout_s * attempts <= SETTINGS.run_timeout_s, role


def test_retries_are_configured():
    assert SETTINGS.max_retries >= 1
    assert config_module.SETTINGS.max_retries == SETTINGS.max_retries
