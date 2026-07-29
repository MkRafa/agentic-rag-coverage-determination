"""Typed IO contracts for every subagent.

These double as the structured-output schemas passed to `messages.parse()`, so
the model is constrained to the contract rather than asked to follow it. Schema
constraints the API does not support (min/max, recursion) are deliberately
absent — validation of those lives in the harness.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# request (produced by pre-flight, consumed by the loop)
# ---------------------------------------------------------------------------


class CoverageRequest(BaseModel):
    payer_id: str
    plan_id: str | None = None
    procedure_codes: list[str] = Field(default_factory=list)
    diagnosis_codes: list[str] = Field(default_factory=list)
    as_of_date: str = ""
    narrative: str = ""
    question: str = ""


class PreflightResult(BaseModel):
    ok: bool
    request: CoverageRequest
    field_confidence: dict[str, float] = Field(default_factory=dict)
    mean_confidence: float = 0.0
    redactions: list[str] = Field(default_factory=list)
    injection_flags: list[str] = Field(default_factory=list)
    halt_reason: str | None = None


# ---------------------------------------------------------------------------
# planner
# ---------------------------------------------------------------------------


class SubQuery(BaseModel):
    query: str = Field(description="Natural-language query to run against the corpus.")
    intent: str = Field(description="What this sub-query is meant to establish.")
    codes: list[str] = Field(
        default_factory=list, description="Procedure/diagnosis codes to boost for this sub-query."
    )


class Plan(BaseModel):
    sub_queries: list[SubQuery]
    required_facts: list[str] = Field(
        default_factory=list,
        description="Facts about the member/service needed to evaluate the criteria.",
    )
    reasoning: str = Field(description="One or two sentences on how the question was decomposed.")


# ---------------------------------------------------------------------------
# retriever (returns clause ids it decided are worth grading)
# ---------------------------------------------------------------------------


class RetrievalSelection(BaseModel):
    clause_ids: list[str] = Field(
        description="Clause ids retrieved and judged worth passing to the Grader."
    )
    queries_run: list[str] = Field(default_factory=list)
    notes: str = ""


# ---------------------------------------------------------------------------
# grader
# ---------------------------------------------------------------------------

Grade = Literal["RELEVANT", "STALE", "CONTRADICTORY", "INSUFFICIENT"]


class GradedClause(BaseModel):
    clause_id: str
    grade: Grade
    reason: str


class GradeReport(BaseModel):
    graded: list[GradedClause]
    sufficient: bool = Field(
        description="True only if the RELEVANT clauses fully answer the question."
    )
    gap: str | None = Field(
        default=None,
        description="If not sufficient, the single most blocking missing piece, stated concretely.",
    )
    contradictions: list[str] = Field(
        default_factory=list,
        description="Descriptions of any unresolved conflicts between sources of equal authority.",
    )
    next_query: str | None = Field(
        default=None, description="A different query to try if evidence was insufficient."
    )


# ---------------------------------------------------------------------------
# synthesizer
# ---------------------------------------------------------------------------

Outcome = Literal["COVERED", "NOT_COVERED", "INSUFFICIENT_EVIDENCE"]


class Citation(BaseModel):
    clause_id: str = Field(description="Exact clause id, copied from a retrieved clause.")
    quote: str = Field(description="Verbatim contiguous substring of that clause's text.")


class Claim(BaseModel):
    text: str = Field(description="One factual claim, entailed by its citation's quote alone.")
    citation: Citation


class Determination(BaseModel):
    outcome: Outcome
    summary: str = Field(description="Two or three sentences stating the determination.")
    claims: list[Claim]
    confidence: float = Field(description="0.0-1.0 confidence in the outcome.")
    gap: str | None = Field(
        default=None, description="Named gap when outcome is INSUFFICIENT_EVIDENCE."
    )
    noted_conflicts: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# verifier
#
# The mechanical checks (exists / verbatim / in effect) are done in Python
# against a cold MCP fetch. Only the semantic judgement is model-mediated, and
# the model sees the answer and the clause — never the reasoning trace.
# ---------------------------------------------------------------------------


class SupportJudgement(BaseModel):
    claim_index: int
    supports: bool = Field(description="Does the quote alone entail the claim?")
    reason: str


class SupportVerdict(BaseModel):
    judgements: list[SupportJudgement]


class CitationCheck(BaseModel):
    claim_index: int
    clause_id: str
    exists: bool
    quote_verbatim: bool
    in_effect_on_as_of: bool
    supports_claim: bool
    reason: str = ""

    @property
    def passed(self) -> bool:
        return (
            self.exists
            and self.quote_verbatim
            and self.in_effect_on_as_of
            and self.supports_claim
        )


class VerificationReport(BaseModel):
    checks: list[CitationCheck] = Field(default_factory=list)

    @property
    def faithfulness(self) -> float:
        if not self.checks:
            return 0.0
        return sum(1 for c in self.checks if c.passed) / len(self.checks)

    @property
    def hallucinated_clauses(self) -> list[str]:
        return [c.clause_id for c in self.checks if not c.exists]

    @property
    def stale_citations(self) -> list[str]:
        return [c.clause_id for c in self.checks if c.exists and not c.in_effect_on_as_of]


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------

GateState = Literal["DETERMINE", "REVIEW", "REFUSE", "HALT"]


class GateDecision(BaseModel):
    state: GateState
    reasons: list[str] = Field(default_factory=list)


class RunResult(BaseModel):
    run_id: str
    gate: GateDecision
    determination: Determination | None = None
    verification: VerificationReport | None = None
    iterations: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    latency_s: float = 0.0
    retrieved_clause_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# offline agents
# ---------------------------------------------------------------------------


class AdversarialCase(BaseModel):
    case_id: str
    trap: str = Field(description="Which failure mode this case targets, e.g. T1 stale-version.")
    payer_id: str
    plan_id: str | None = None
    procedure_codes: list[str]
    diagnosis_codes: list[str] = Field(default_factory=list)
    as_of_date: str
    narrative: str
    question: str
    expected_outcome: Outcome
    expected_gate: GateState
    gold_clause_ids: list[str]
    rationale: str


class AdversarialBatch(BaseModel):
    cases: list[AdversarialCase]


class JudgeVerdict(BaseModel):
    reasoning_quality: int = Field(description="1-5. Is the determination well-argued?")
    clarity: int = Field(description="1-5. Would a reviewer understand and be able to act on it?")
    appropriate_hedging: int = Field(
        description="1-5. Is stated confidence proportionate to the evidence shown?"
    )
    comments: str
