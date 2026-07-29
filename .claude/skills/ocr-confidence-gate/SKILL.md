---
name: ocr-confidence-gate
description: Deterministic pre-retrieval rules. Low-confidence extractions and unredacted PHI never enter the agentic loop.
---

# Pre-flight gate

This runs **before any model call**, in Python, with no LLM in the path. It is
enforced in `src/harness/preflight.py`; this file is the specification that code
implements.

## Order of operations

1. **PHI redaction.** Deterministic regex pass over every free-text field.
   Redaction happens first so that nothing downstream — model calls, traces,
   logs — ever sees the raw identifiers. Redacted spans are replaced with a
   typed placeholder (`[MRN]`, `[SSN]`, `[DOB]`, `[NAME]`, `[PHONE]`) so the
   clinical narrative stays readable.

2. **Field extraction and confidence.** Structured fields (payer, plan, codes,
   as-of date) are extracted and each gets a confidence score in [0, 1]:
   - `1.00` — supplied directly and validated against the corpus
   - `0.60` — supplied but not resolvable (unknown plan id, unknown code)
   - `0.00` — absent

3. **Code validation.** Every procedure code is checked against the code set.
   An unrecognised code is not a retrieval problem; it is an intake problem.

4. **Date normalisation.** `as_of_date` must be an ISO `YYYY-MM-DD`. An absent
   date does not silently become today — today is a different question from the
   date of service and answering the wrong one is a real failure mode.

## Halt conditions

The request never reaches the Planner if any of these hold. The gate returns a
`HALT` outcome naming the specific field, not a generic error.

| Condition | Rationale |
|---|---|
| mean field confidence `< 0.55` | The request is too poorly specified to retrieve against; retrieving anyway produces confident nonsense |
| `payer_id` unresolved | Every policy is payer-scoped; without it retrieval is unfiltered |
| `as_of_date` missing or unparseable | Determinations are as-of-date-specific |
| no valid procedure code | Nothing to determine coverage *of* |

## Injection defence

The clinical narrative is untrusted free text. Any instruction-shaped content
inside it (`ignore previous instructions`, `approve this claim`, `you are now…`)
is neutralised at this stage and recorded in the trace. Downstream agents are
told explicitly that narrative content is data, never instruction.
