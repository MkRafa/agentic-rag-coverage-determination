"""The eval suite's own integrity.

The answer key is the thing everything else is measured against. If a gold
label is wrong, every downstream number is confidently wrong, and the next
`--set-baseline` writes the mistake down as truth. So the case set is tested
like code.
"""

from __future__ import annotations

import pytest

from evals.expand import PATTERNS, expand, is_plan_sensitive, is_version_sensitive
from evals.runner import case_set_fingerprint, load_cases
from evals.validate import validate
from policy_corpus.retrieval import in_effect
from policy_corpus.store import get_store


def test_every_committed_case_is_valid():
    """Gold clauses exist, are in effect on their own as-of date, belong to the
    case's payer, and riders belong to the case's plan."""
    report = validate(load_cases(), get_store())
    assert report.ok, "\n" + report.render()


def test_the_suite_is_big_enough_to_mean_something():
    assert len(load_cases()) >= 120


def test_both_answer_classes_are_well_represented():
    """A suite skewed to one outcome lets a trivial always-COVERED system look
    competent."""
    cases = load_cases()
    covered = sum(1 for c in cases if c["expected_outcome"] == "COVERED")
    not_covered = sum(1 for c in cases if c["expected_outcome"] == "NOT_COVERED")
    ratio = min(covered, not_covered) / max(covered, not_covered)
    assert ratio > 0.6, f"class imbalance: {covered} COVERED vs {not_covered} NOT_COVERED"


def test_every_trap_type_is_covered():
    traps = " ".join(c.get("trap", "") for c in load_cases())
    for marker in ("T1", "T2", "T3", "T4", "T5", "T6", "T7"):
        assert marker in traps, f"no case exercises {marker}"


def test_case_ids_are_unique():
    ids = [c["case_id"] for c in load_cases()]
    assert len(ids) == len(set(ids))


def test_fingerprint_changes_when_the_set_changes():
    """A baseline is only comparable against the same questions. The fingerprint
    is what makes a cross-set diff detectable rather than silently misleading."""
    a = [{"case_id": "x"}, {"case_id": "y"}]
    b = [{"case_id": "x"}, {"case_id": "y"}, {"case_id": "z"}]
    assert case_set_fingerprint(a)["hash"] != case_set_fingerprint(b)["hash"]
    # Order must not matter — only membership.
    assert case_set_fingerprint(a) == case_set_fingerprint(list(reversed(a)))


# ---------------------------------------------------------------------------
# expansion logic
# ---------------------------------------------------------------------------


def test_expansion_labels_are_derived_not_asserted():
    """Every generated gold clause must be in force on the date its own case
    uses. This is the property that lets labels be trusted without review."""
    store = get_store()
    import json

    corpus = json.loads(store.path.read_text())
    for case in expand(corpus):
        for gid in case["gold_clause_ids"]:
            clause = store.clauses[gid]
            assert in_effect(clause, case["as_of_date"]), (case["case_id"], gid)


def test_version_sensitive_patterns_produce_opposing_pairs():
    """The point of a boundary case is that the answer flips. A 'sensitive'
    pattern whose outcomes are all the same would be wasted cost."""
    sensitive = [p for pats in PATTERNS.values() for p in pats if is_version_sensitive(p)]
    assert sensitive
    for pat in sensitive:
        outcomes = {o for o, _ in pat.by_version.values()}
        assert len(outcomes) > 1, pat.key


def test_plan_sensitive_patterns_actually_differ_by_plan():
    """Instantiating both plans is only worth the cost if something changes.
    Either the outcome flips, or the clause a correct answer must rest on does —
    the second still matters: a rider that waives step therapy but not the
    diagnosis requirement should be cited for the part it does not waive."""
    plan_sensitive = [p for pats in PATTERNS.values() for p in pats if is_plan_sensitive(p)]
    assert plan_sensitive
    for pat in plan_sensitive:
        base = {(o, tuple(g)) for o, g in pat.by_version.values()}
        override = {(o, tuple(g)) for o, g in pat.by_plan.values()}
        assert base != override, f"{pat.key}: plan override changes neither outcome nor gold clause"


# ---------------------------------------------------------------------------
# the bug the expanded suite found
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "as_of,expected",
    [
        ("2023-06-01", {"MHP-RID-DME-EXPAND"}),                        # STEP-WAIVE not yet in force
        ("2024-06-01", {"MHP-RID-DME-EXPAND", "MHP-RID-STEP-WAIVE"}),
    ],
)
def test_plan_riders_are_filtered_by_effective_date(as_of, expected):
    """A rider overrides base policy, so returning one that had not taken effect
    inverts the determination. `search_policies` always filtered by date; this
    did not, and 21 cases in the expanded suite were retrieving a rider up to a
    year before it existed."""
    riders = get_store().plan_riders("MHP-PPO-GOLD", as_of_date=as_of)
    assert {r["rider_id"] for r in riders["riders"]} == expected


def test_plan_riders_unfiltered_without_a_date():
    riders = get_store().plan_riders("MHP-PPO-GOLD")
    assert len(riders["riders"]) == 2
