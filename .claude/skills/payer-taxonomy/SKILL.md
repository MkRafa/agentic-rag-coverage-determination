---
name: payer-taxonomy
description: Glossary of the entities in this domain — payer, plan, rider, medical policy bulletin, provider FAQ, clause — and how they relate.
---

# Payer taxonomy

| Term | What it is |
|---|---|
| **Payer** | The insuring organisation (`MHP`, `CSM`, `NBA`). Policies are payer-scoped: a Meridian bulletin says nothing about a Cascadia member. |
| **Plan** | A specific product a member is enrolled in (`MHP-PPO-GOLD`). Belongs to one payer. Determines which riders apply. |
| **Medical policy bulletin** | The payer's published clinical coverage criteria for a service or class of services. Versioned, with effective date ranges. The primary authority. |
| **Plan rider** | An amendment attached to a specific plan that overrides the base policy on the points it addresses. Controls where it conflicts. |
| **Provider FAQ** | Secondary, informational material. Never overrides a bulletin, but a conflict with one is worth reporting. |
| **Clause** | The atomic unit: one numbered paragraph of one version of one document. The only thing that can be cited. |
| **Version** | A dated revision of a policy, with `effective_start` and `effective_end`. `effective_end: null` means currently in force. |
| **Section** | The heading a clause sits under — `Scope`, `Coverage Criteria`, `Limitations`, `Exclusions`, `Step Therapy`, `Rider`. Materially changes what the clause can support. |

## Identifier shapes

```
MHP-MP-0142              policy id      payer - doc type - number
MHP-MP-0142.v1.C2        clause id      policy id . version . clause number
MHP-RID-STEP-WAIVE       rider id
MHP-PPO-GOLD             plan id        payer - product type - tier
95249  A9276  E11.9      CPT / HCPCS / ICD-10
```

## Relationships that matter

- A **plan** has zero or more **riders**; each rider names the **policies** it
  overrides.
- A **policy** has one or more **versions**; each version has **clauses**.
- A **code** may be governed by several policies across different payers — and,
  within one payer, by policies with confusingly similar titles. Resolve on the
  code, not the title.
- "Not medically necessary", "excluded", and "investigational" are three
  distinct not-covered outcomes with different appeal paths. Preserve the
  distinction the policy actually draws.
