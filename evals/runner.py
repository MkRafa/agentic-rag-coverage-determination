"""Eval harness — replays every case through the full runtime path.

Same code path as `ask`. An eval that runs a shortcut version of the pipeline
measures the shortcut, not the system.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from policy_corpus.store import get_store
from src.config import SETTINGS
from src.harness import loop
from src.harness.budget import Budget
from src.harness.trace import Trace
from src.llm import build_client
from src.mcp_client import policy_corpus

from .scorers import answer as answer_scorer
from .scorers import retrieval as retrieval_scorer

ROOT = Path(__file__).resolve().parent
CASES_DIR = ROOT / "cases"
BASELINE = ROOT / "baseline.json"
RESULTS_DIR = ROOT / "results"


def load_cases(paths: list[Path] | None = None) -> list[dict[str, Any]]:
    files = paths or sorted(CASES_DIR.glob("*.json"))
    cases: list[dict[str, Any]] = []
    for f in files:
        payload = json.loads(f.read_text())
        cases.extend(payload["cases"] if isinstance(payload, dict) else payload)
    return cases


def _payload(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "payer_id": case.get("payer_id", ""),
        "plan_id": case.get("plan_id"),
        "procedure_codes": case.get("procedure_codes", []),
        "diagnosis_codes": case.get("diagnosis_codes", []),
        "as_of_date": case.get("as_of_date", ""),
        "narrative": case.get("narrative", ""),
        "question": case.get("question", ""),
    }


async def run_eval(
    *, stub: bool = False, limit: int | None = None, quiet: bool = False
) -> dict[str, Any]:
    cases = load_cases()
    if limit:
        cases = cases[:limit]

    store = get_store()
    results: list[tuple[dict[str, Any], Any]] = []

    async with policy_corpus() as corpus:
        for case in cases:
            budget = Budget()
            trace = Trace()
            client = build_client(budget, trace, stub=stub, corpus=corpus)
            run = await loop.run(_payload(case), client, corpus, budget, trace, store=store)
            results.append((case, run))
            if not quiet:
                outcome = run.determination.outcome if run.determination else "—"
                mark = "ok " if _case_ok(case, run) else "FAIL"
                print(
                    f"  [{mark}] {case['case_id']:<26} gate={run.gate.state:<10} "
                    f"outcome={outcome:<21} {case.get('trap', '')}"
                )

    layer1 = retrieval_scorer.score(results, store)
    layer2 = answer_scorer.score(results)

    scorecard = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "stub" if stub else "live",
        "config": {
            "model": SETTINGS.synthesizer.model,
            "embedder": SETTINGS.embedder,
            "reranker": SETTINGS.reranker,
            "max_iterations": SETTINGS.max_iterations,
            "min_confidence": SETTINGS.min_confidence,
        },
        "retrieval": layer1.summary(),
        "answer": layer2.summary(),
        "per_case": layer2.per_case,
        "per_case_retrieval": layer1.per_case,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (RESULTS_DIR / f"scorecard-{stamp}.json").write_text(json.dumps(scorecard, indent=2))
    return scorecard


def _case_ok(case: dict[str, Any], run: Any) -> bool:
    outcome = run.determination.outcome if run.determination else "INSUFFICIENT_EVIDENCE"
    return outcome == case["expected_outcome"] and run.gate.state == case["expected_gate"]


# ---------------------------------------------------------------------------
# scorecard rendering and diffing — the demo ends here, not on the happy path
# ---------------------------------------------------------------------------

HEADLINE = [
    ("retrieval", "gold_clause_recall", "gold-clause recall", "up"),
    ("retrieval", "stale_retrieval_rate", "stale retrieval rate", "down"),
    ("answer", "determination_accuracy", "determination accuracy", "up"),
    ("answer", "gate_correctness", "gate correctness", "up"),
    ("answer", "citation_faithfulness", "citation faithfulness", "up"),
    ("answer", "hallucinated_clause_count", "hallucinated clauses", "down"),
    ("answer", "stale_citation_count", "stale citations", "down"),
    ("answer", "appropriate_refusal_rate", "appropriate refusal rate", "up"),
    ("answer", "false_auto_determine", "FALSE AUTO-DETERMINE", "down"),
    ("answer", "usd_per_case", "cost per case (usd)", "down"),
    ("answer", "p95_latency_s", "p95 latency (s)", "down"),
]


def render(scorecard: dict[str, Any]) -> str:
    lines = [
        "",
        f"scorecard  mode={scorecard['mode']}  model={scorecard['config']['model']}  "
        f"embedder={scorecard['config']['embedder']}  reranker={scorecard['config']['reranker']}",
        "-" * 78,
    ]
    for section, key, label, _ in HEADLINE:
        value = scorecard[section][key]
        lines.append(f"  {label:<28} {_fmt(value):>12}")
    lines.append("-" * 78)
    return "\n".join(lines)


def diff(current: dict[str, Any], baseline: dict[str, Any]) -> str:
    lines = ["", "scorecard diff vs baseline", "-" * 78]
    regressed = False
    for section, key, label, direction in HEADLINE:
        now = current[section][key]
        was = baseline.get(section, {}).get(key)
        if was is None:
            lines.append(f"  {label:<28} {_fmt(now):>12}   (new)")
            continue
        delta = now - was
        better = (delta > 0) if direction == "up" else (delta < 0)
        if abs(delta) < 1e-9:
            mark = "  ="
        elif better:
            mark = "  +"
        else:
            mark = "  !"
            if key in ("false_auto_determine", "citation_faithfulness", "hallucinated_clause_count"):
                regressed = True
        lines.append(
            f"{mark} {label:<28} {_fmt(was):>12} -> {_fmt(now):>12}  ({_fmt(delta, signed=True)})"
        )
    lines.append("-" * 78)
    if regressed:
        lines.append("  ! a safety-critical metric regressed — this is a blocking diff")
    return "\n".join(lines)


def _fmt(value: Any, signed: bool = False) -> str:
    if isinstance(value, float):
        return f"{value:+.4f}" if signed else f"{value:.4f}"
    return f"{value:+d}" if signed and isinstance(value, int) else str(value)


def load_baseline() -> dict[str, Any] | None:
    return json.loads(BASELINE.read_text()) if BASELINE.exists() else None


def write_baseline(scorecard: dict[str, Any]) -> None:
    BASELINE.write_text(json.dumps(scorecard, indent=2))
