"""Corpus store — the data layer behind the MCP tools.

Read-only by construction: there is no write path anywhere in this package.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from .retrieval import HybridIndex, in_effect

DEFAULT_CORPUS = (
    Path(__file__).resolve().parent.parent / "corpus" / "generated" / "corpus.json"
)


class CorpusStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path(os.environ.get("CDA_CORPUS", DEFAULT_CORPUS))
        if not self.path.exists():
            raise FileNotFoundError(
                f"corpus not found at {self.path} — run `python -m corpus.generate` first"
            )
        raw = json.loads(self.path.read_text())
        self.payers = raw["payers"]
        self.plans = {p["plan_id"]: p for p in raw["plans"]}
        self.policies = {p["policy_id"]: p for p in raw["policies"]}
        self.riders = {r["rider_id"]: r for r in raw["riders"]}
        self.clauses = {c["clause_id"]: c for c in raw["clauses"]}
        self.codes = {c["code"].upper(): c for c in raw["codes"]}
        self.index = HybridIndex(raw["clauses"])

    # -- tool-backing operations ------------------------------------------

    def search(
        self,
        query: str,
        *,
        payer_id: str | None = None,
        plan_id: str | None = None,
        as_of_date: str | None = None,
        codes: list[str] | None = None,
        k: int = 8,
        include_superseded: bool = False,
    ) -> list[dict[str, Any]]:
        hits = self.index.search(
            query,
            k=k,
            payer_id=payer_id,
            plan_id=plan_id,
            as_of=as_of_date,
            include_superseded=include_superseded,
            codes=codes or (),
        )
        return [
            {
                "clause_id": h.clause["clause_id"],
                "policy_id": h.clause["policy_id"],
                "policy_title": h.clause["policy_title"],
                "doc_type": h.clause["doc_type"],
                "payer_id": h.clause["payer_id"],
                "plan_id": h.clause["plan_id"],
                "version": h.clause["version"],
                "section": h.clause["section"],
                "effective_start": h.clause["effective_start"],
                "effective_end": h.clause["effective_end"],
                "in_effect_on_as_of": in_effect(h.clause, as_of_date) if as_of_date else None,
                "text": h.clause["text"],
                "score": round(h.score, 6),
            }
            for h in hits
        ]

    def clause(self, clause_id: str) -> dict[str, Any] | None:
        c = self.clauses.get(clause_id)
        if c is None:
            return None
        return {
            "clause_id": c["clause_id"],
            "policy_id": c["policy_id"],
            "policy_title": c["policy_title"],
            "doc_type": c["doc_type"],
            "payer_id": c["payer_id"],
            "plan_id": c["plan_id"],
            "version": c["version"],
            "section": c["section"],
            "effective_start": c["effective_start"],
            "effective_end": c["effective_end"],
            "text": c["text"],
        }

    def versions(self, policy_id: str) -> list[dict[str, Any]] | None:
        p = self.policies.get(policy_id)
        if p:
            return [
                {
                    "policy_id": policy_id,
                    "version": v["version"],
                    "effective_start": v["effective_start"],
                    "effective_end": v["effective_end"],
                    "clause_ids": [c["clause_id"] for c in v["clauses"]],
                }
                for v in p["versions"]
            ]
        r = self.riders.get(policy_id)
        if r:
            return [
                {
                    "policy_id": policy_id,
                    "version": "v1",
                    "effective_start": r["effective_start"],
                    "effective_end": r["effective_end"],
                    "clause_ids": [c["clause_id"] for c in r["clauses"]],
                }
            ]
        return None

    def code(self, code: str) -> dict[str, Any] | None:
        entry = self.codes.get(code.upper())
        if entry is None:
            return None
        related = sorted(
            {
                p["policy_id"]
                for p in self.policies.values()
                if code.upper() in {c.upper() for c in p["codes"]}
            }
        )
        return {**entry, "governed_by_policies": related}

    def plan_riders(
        self, plan_id: str, as_of_date: str | None = None, include_superseded: bool = False
    ) -> dict[str, Any] | None:
        """Riders attached to a plan, filtered to those in force on `as_of_date`.

        Date filtering matters as much here as it does in search — arguably more.
        A rider overrides base policy, so handing back one that had not taken
        effect yet is how a determination gets inverted. `search_policies`
        filtered by date from the start; this did not, and 21 cases in the
        expanded suite were retrieving riders up to a year before they existed.
        """
        plan = self.plans.get(plan_id)
        if plan is None:
            return None

        riders = []
        for rid in plan["rider_ids"]:
            r = self.riders.get(rid)
            if r is None:
                continue
            effective = in_effect(r, as_of_date) if as_of_date else None
            if as_of_date and not include_superseded and not effective:
                continue
            riders.append(
                {
                    "rider_id": r["rider_id"],
                    "title": r["title"],
                    "overrides_policy_ids": r["overrides_policy_ids"],
                    "effective_start": r["effective_start"],
                    "effective_end": r["effective_end"],
                    "in_effect_on_as_of": effective,
                    "clause_ids": [c["clause_id"] for c in r["clauses"]],
                }
            )

        return {
            "plan_id": plan_id,
            "plan_name": plan["name"],
            "payer_id": plan["payer_id"],
            "riders": riders,
        }


@lru_cache(maxsize=1)
def get_store() -> CorpusStore:
    return CorpusStore()
