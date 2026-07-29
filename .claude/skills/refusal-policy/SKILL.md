---
name: refusal-policy
description: When "the corpus does not answer this" is the correct output, and how to name the specific gap so a human reviewer can act on it.
---

# Refusal policy

**Refusal is a success state.** A determination that says "insufficient
evidence, here is precisely what is missing" is a correct answer. A confident
determination built on evidence that was not actually retrieved is the failure
this whole system exists to prevent.

## Refuse when

- **No governing policy.** Nothing in the corpus governs the requested code for
  this payer. Note: a clause naming the code as *investigational* **is** a
  governing policy — that is a not-covered determination, not a refusal.
- **A required fact is missing from the request.** A criterion turns on a fact
  (A1c value, duration of prior therapy, presence of a comorbidity) that the
  request does not state. Name the fact.
- **An unresolved contradiction between sources of equal authority.** Two
  bulletins from the same payer that conflict on the same point. A
  bulletin-versus-FAQ conflict is *not* this — precedence resolves it (see
  `coverage-criteria-logic`); report the contradiction and proceed.
- **No version in effect on the as-of date.** The policy exists but every
  version starts after, or ended before, the date in question.
- **Plan not in corpus**, so rider applicability cannot be determined and the
  base policy is known to be rider-modifiable.

## Do not refuse when

- The answer is simply *not covered*. "Not medically necessary under clause X"
  is a determination, not a refusal.
- Retrieval was thin but the clauses you do have fully answer the question.
- You are uncertain about a judgement call that the clause text actually
  settles. Re-read the clause before reaching for a refusal.

## Naming the gap

A refusal must be actionable. State the single most blocking gap in one
sentence, concretely enough that a reviewer knows what to go and find:

- Good: "Determination requires the member's documented daily fingerstick
  frequency, which the request does not state."
- Good: "No Cascadia Mutual policy governs CPT 31241 as of 2024-08-01."
- Bad: "Insufficient information." / "Unable to determine." / "More context
  needed."

Cite whatever you *did* establish. A refusal with citations to the clauses that
set up the unanswered question is far more useful than a bare refusal.
