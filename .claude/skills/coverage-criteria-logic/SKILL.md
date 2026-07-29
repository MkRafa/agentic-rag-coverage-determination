---
name: coverage-criteria-logic
description: How to read AND/OR/UNLESS structures in policy criteria, and the precedence order between plan riders, base policy, and provider FAQs.
---

# Reading coverage criteria

## Compositional structure

Policy criteria are written as logical compositions, and the connectives are
load-bearing. Parse them literally.

- **AND** — every conjunct must be satisfied. A determination that satisfies
  three of four conjuncts is *not covered*, not "partially covered".
- **OR** — any one disjunct suffices.
- **UNLESS X** — X is an *exception that discharges the preceding requirement*.
  It does not add a requirement. "Requires A and B unless C" means:
  `(A and B) or C`. A member who meets C does not need A or B.
- **only when** — introduces a necessary condition, and is usually narrower than
  it looks. "Covered only when the member is on an intensive regimen" excludes
  everyone else.

Work through each conjunct explicitly against the facts you were given. If a
fact needed to evaluate a conjunct is absent from the request, that conjunct is
**unknown**, not satisfied — an unknown conjunct means the evidence is
insufficient (see `refusal-policy`), not that the criterion fails.

## Precedence

When sources conflict, this order controls:

1. **Plan rider** — a rider attached to the member's specific plan overrides the
   base policy on the points it addresses, and only those points. A rider that
   waives step therapy does not waive the diagnosis requirement.
2. **Medical policy bulletin** — the base policy for the payer.
3. **Provider FAQ and other secondary material** — informational. It never
   overrides a bulletin. Where an FAQ contradicts the bulletin, the bulletin
   controls **and the contradiction must be reported**, because the member or
   provider may have relied on the FAQ.

A rider only applies if the member is enrolled in the plan the rider is attached
to, and only if the rider was in effect on the as-of date. Check both.

## Effective dates

The determination is made **as of a specific date**, not as of today. Coverage
turns on the version of each policy in force on that date. When a policy has
changed, the fact that the current version would produce a different answer is
worth surfacing to the reviewer, but it does not change the determination.

## Codes

Match on the specific procedure code in the request. Two policies may have
near-identical titles and govern different codes — check `governed_by_policies`
for the code rather than matching on the policy title.
