---
name: citation-format
description: The required output shape for a coverage determination. Every claim carries a clause_id and a verbatim quote so the Verifier can mechanically check it.
---

# Citation format

Every factual claim in a determination must be traceable to exactly one clause.
This is what makes faithfulness measurable rather than a matter of opinion — the
Verifier re-fetches each cited clause cold and checks the quote against it.

## Rules

1. **One clause per claim.** If a claim rests on two clauses, split it into two
   claims. A citation that "summarises" several clauses cannot be verified.
2. **`clause_id` must be copied exactly** from a retrieved clause. Never
   construct, abbreviate, or guess an id — `MHP-MP-0142.v1.C2` is valid,
   `MHP-MP-0142` (a policy id) and `MHP-MP-0142.C2` (missing version) are not.
3. **`quote` must be a contiguous verbatim substring** of that clause's text.
   Do not normalise whitespace differently, fix typography, or paraphrase.
   Quote the shortest span that actually supports the claim.
4. **The claim must be entailed by the quote alone.** If a reader with only the
   quote in front of them could not conclude the claim, the citation is wrong
   even if the clause is the right one.
5. **Cite the version in effect on the as-of date.** Where a policy has multiple
   versions, the clause id encodes the version — using a superseded version is a
   verification failure, not a stylistic one.

## Anti-patterns

- Citing a `Scope` clause to support a coverage conclusion. Scope says what the
  policy covers, not whether the service is medically necessary.
- Citing the base policy for a point the rider overrides.
- Attaching a citation to a sentence of your own reasoning rather than to the
  substantive claim.
- Inventing a plausible-looking `clause_id` when retrieval came back thin. If
  the evidence is not there, the correct output is a refusal — see
  `refusal-policy`.
