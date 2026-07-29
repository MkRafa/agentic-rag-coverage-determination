"""Deterministic pre-flight. No LLM in this path.

Implements the `ocr-confidence-gate` skill: redact, extract, score confidence,
validate codes, normalise the date, neutralise injection attempts. A request
that fails here never reaches the Planner, and the halt names the specific
field rather than returning a generic error.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from ..config import SETTINGS
from ..contracts import CoverageRequest, PreflightResult

# ---------------------------------------------------------------------------
# PHI redaction — runs first so nothing downstream sees raw identifiers
# ---------------------------------------------------------------------------

PHI_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("MRN", re.compile(r"\bMRN[:# ]*\s*([A-Z0-9]{6,12})\b", re.I)),
    ("DOB", re.compile(r"\bDOB[:# ]*\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", re.I)),
    ("PHONE", re.compile(r"\b(?:\+1[ -]?)?\(?\d{3}\)?[ -]\d{3}[ -]\d{4}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")),
    ("NAME", re.compile(r"\b(?:member|patient)\s+name[:#]?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", re.I)),
]

# ---------------------------------------------------------------------------
# prompt-injection markers — narrative text is data, never instruction
# ---------------------------------------------------------------------------

INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("override_instruction", re.compile(r"\bignore (?:all |any )?(?:previous|prior|above) instructions?\b", re.I)),
    ("role_reassignment", re.compile(r"\byou are now\b|\bact as\b|\bnew system prompt\b", re.I)),
    ("forced_outcome", re.compile(r"\b(?:approve|authorize|deny) (?:this|the) (?:claim|request|determination)\b", re.I)),
    ("authority_claim", re.compile(r"\b(?:as the|per the) (?:medical director|plan administrator|anthropic)\b", re.I)),
    ("fabricate_citation", re.compile(r"\b(?:cite|use) (?:clause|policy)\s+\S+\s+(?:regardless|even if)\b", re.I)),
]

ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def redact(text: str) -> tuple[str, list[str]]:
    found: list[str] = []
    out = text
    for label, pattern in PHI_PATTERNS:
        if pattern.search(out):
            found.append(label)
            out = pattern.sub(f"[{label}]", out)
    return out, found


def scan_injection(text: str) -> list[str]:
    return [label for label, pattern in INJECTION_PATTERNS if pattern.search(text)]


def neutralise(text: str) -> str:
    """Defang instruction-shaped spans so they read as quoted data downstream."""
    out = text
    for _, pattern in INJECTION_PATTERNS:
        out = pattern.sub(lambda m: f"[REDACTED-INSTRUCTION: {m.group(0)}]", out)
    return out


def run_preflight(payload: dict[str, Any], store: Any | None = None) -> PreflightResult:
    """`store` is optional; when supplied, codes/payer/plan are validated against it."""
    narrative_raw = str(payload.get("narrative", "") or "")
    question_raw = str(payload.get("question", "") or "")

    narrative, red_a = redact(narrative_raw)
    question, red_b = redact(question_raw)
    redactions = sorted(set(red_a + red_b))

    flags = sorted(set(scan_injection(narrative) + scan_injection(question)))
    narrative = neutralise(narrative)
    question = neutralise(question)

    payer_id = (payload.get("payer_id") or "").strip().upper() or None
    plan_id = (payload.get("plan_id") or "").strip().upper() or None
    procedure_codes = [str(c).strip().upper() for c in payload.get("procedure_codes") or [] if str(c).strip()]
    diagnosis_codes = [str(c).strip().upper() for c in payload.get("diagnosis_codes") or [] if str(c).strip()]
    as_of = (payload.get("as_of_date") or "").strip()

    # -- confidence scoring -------------------------------------------------
    known_payers = {p["payer_id"] for p in store.payers} if store else set()
    known_plans = set(store.plans) if store else set()
    known_codes = set(store.codes) if store else set()

    def score(value: Any, valid: bool | None) -> float:
        if not value:
            return 0.0
        if valid is None:
            return 0.85          # present, not checkable here
        return 1.0 if valid else 0.60

    conf = {
        "payer_id": score(payer_id, payer_id in known_payers if store else None),
        "plan_id": score(plan_id, plan_id in known_plans if store else None),
        "procedure_codes": score(
            procedure_codes,
            all(c in known_codes for c in procedure_codes) if store and procedure_codes else None,
        ),
        "as_of_date": score(as_of, bool(ISO_DATE.match(as_of)) and _parseable(as_of)),
        "question": score(question, None),
    }
    mean_conf = sum(conf.values()) / len(conf)

    request = CoverageRequest(
        payer_id=payer_id or "",
        plan_id=plan_id,
        procedure_codes=procedure_codes,
        diagnosis_codes=diagnosis_codes,
        as_of_date=as_of,
        narrative=narrative,
        question=question,
    )

    halt = _halt_reason(payer_id, as_of, procedure_codes, known_codes, mean_conf, bool(store))

    return PreflightResult(
        ok=halt is None,
        request=request,
        field_confidence={k: round(v, 3) for k, v in conf.items()},
        mean_confidence=round(mean_conf, 3),
        redactions=redactions,
        injection_flags=flags,
        halt_reason=halt,
    )


def _parseable(value: str) -> bool:
    try:
        date.fromisoformat(value)
        return True
    except (ValueError, TypeError):
        return False


def _halt_reason(payer_id, as_of, procedure_codes, known_codes, mean_conf, has_store) -> str | None:
    if not payer_id:
        return "payer_id is missing; every policy is payer-scoped so retrieval cannot be filtered."
    if not as_of:
        return "as_of_date is missing; a determination is specific to a date and must not default to today."
    if not _parseable(as_of):
        return f"as_of_date '{as_of}' is not a parseable ISO date (YYYY-MM-DD)."
    if not procedure_codes:
        return "no procedure code supplied; there is nothing to determine coverage of."
    if has_store and not any(c in known_codes for c in procedure_codes):
        return (
            f"none of the supplied procedure codes {procedure_codes} are recognised; "
            "this is an intake problem, not a retrieval problem."
        )
    if mean_conf < SETTINGS.min_field_confidence:
        return (
            f"mean field confidence {mean_conf:.2f} is below the "
            f"{SETTINGS.min_field_confidence:.2f} threshold; the request is too "
            "underspecified to retrieve against."
        )
    return None
