"""CLI.

    cda corpus build              regenerate the synthetic corpus
    cda ask ...                   run one determination end to end
    cda eval                      replay the eval set, print a scorecard
    cda eval --diff               compare against the committed baseline
    cda eval --set-baseline       freeze the current scorecard as the baseline
    cda trace <run_id>            pretty-print a run trace
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from .config import SETTINGS


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------


async def _ask(args: argparse.Namespace) -> int:
    from policy_corpus.store import get_store

    from .harness import loop
    from .harness.budget import Budget
    from .harness.trace import Trace
    from .llm import build_client
    from .mcp_client import policy_corpus

    payload = {
        "payer_id": args.payer,
        "plan_id": args.plan,
        "procedure_codes": args.code or [],
        "diagnosis_codes": args.dx or [],
        "as_of_date": args.as_of,
        "narrative": args.narrative or "",
        "question": args.question,
    }

    async with policy_corpus() as corpus:
        budget = Budget()
        trace = Trace()
        client = build_client(budget, trace, stub=args.stub, corpus=corpus)
        run = await loop.run(payload, client, corpus, budget, trace, store=get_store())

    _print_run(run, trace)
    return 0 if run.gate.state == "DETERMINE" else 1


def _print_run(run: Any, trace: Any) -> None:
    icon = {"DETERMINE": "✅", "REVIEW": "⚠️ ", "REFUSE": "🚫", "HALT": "⛔"}[run.gate.state]
    print()
    print(f"{icon} GATE: {run.gate.state}")
    for reason in run.gate.reasons:
        print(f"   · {reason}")

    if run.determination:
        d = run.determination
        print()
        print(f"OUTCOME: {d.outcome}   (confidence {d.confidence:.2f})")
        print(f"  {d.summary}")
        if d.gap:
            print(f"  GAP: {d.gap}")
        for conflict in d.noted_conflicts:
            print(f"  CONFLICT: {conflict}")
        if d.claims:
            print()
            print("CLAIMS")
            for i, claim in enumerate(d.claims):
                check = None
                if run.verification:
                    check = next(
                        (c for c in run.verification.checks if c.claim_index == i), None
                    )
                mark = "✓" if check and check.passed else "✗"
                print(f"  {mark} {claim.text}")
                print(f"      [{claim.citation.clause_id}] \"{claim.citation.quote[:110]}…\"")
                if check and not check.passed:
                    print(f"      └ {check.reason}")

    if run.verification:
        print()
        print(f"VERIFICATION: faithfulness {run.verification.faithfulness:.2f} "
              f"({sum(1 for c in run.verification.checks if c.passed)}"
              f"/{len(run.verification.checks)} citations)")

    print()
    print(
        f"iterations {run.iterations} · {run.input_tokens + run.output_tokens} tokens · "
        f"${run.usd:.4f} · {run.latency_s:.2f}s · trace {trace.path}"
    )


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


async def _eval(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(SETTINGS.traces_dir.parent))
    from evals import runner

    print(f"replaying eval set ({'stub' if args.stub else 'live'} mode)…")
    scorecard = await runner.run_eval(stub=args.stub, limit=args.limit)
    print(runner.render(scorecard))

    if args.set_baseline:
        runner.write_baseline(scorecard)
        print(f"baseline written to {runner.BASELINE}")
        return 0

    baseline = runner.load_baseline()
    if args.diff:
        if baseline is None:
            print("no baseline committed yet — run with --set-baseline to freeze one")
            return 1
        print(runner.diff(scorecard, baseline))

    return 0 if scorecard["answer"]["false_auto_determine"] == 0 else 1


# ---------------------------------------------------------------------------
# corpus / trace
# ---------------------------------------------------------------------------


def _corpus(args: argparse.Namespace) -> int:
    from corpus.generate import main as build

    build()
    return 0


def _trace(args: argparse.Namespace) -> int:
    path = Path(args.run_id)
    if not path.exists():
        matches = sorted(SETTINGS.traces_dir.glob(f"*{args.run_id}*.jsonl"))
        if not matches:
            print(f"no trace matching '{args.run_id}' in {SETTINGS.traces_dir}")
            return 1
        path = matches[-1]

    for line in path.read_text().splitlines():
        event = json.loads(line)
        kind = event.pop("kind")
        t = event.pop("t")
        event.pop("run_id", None)
        head = f"  {t:>7.3f}s  {kind:<20}"
        if args.verbose:
            print(head, json.dumps(event, indent=2, default=str))
        else:
            print(head, _brief(kind, event))
    return 0


def _brief(kind: str, event: dict[str, Any]) -> str:
    keys = {
        "preflight": ("ok", "mean_confidence", "injection_flags", "halt_reason"),
        "plan": ("iteration", "required_facts"),
        "retrieve": ("iteration", "clause_ids"),
        "grade": ("iteration", "sufficient", "gap"),
        "synthesize": ("outcome", "confidence", "cited"),
        "verify": ("faithfulness", "hallucinated", "stale"),
        "gate": ("state", "reasons"),
        "model_call": ("role", "model", "input_tokens", "output_tokens"),
        "run_end": ("usd", "latency_s"),
    }.get(kind)
    if keys is None:
        return json.dumps({k: v for k, v in event.items() if k != "checks"}, default=str)[:160]
    return " ".join(f"{k}={event.get(k)!r}" for k in keys if k in event)[:200]


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cda", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_corpus = sub.add_parser("corpus", help="corpus operations")
    p_corpus.add_argument("action", choices=["build"])
    p_corpus.set_defaults(fn=_corpus, is_async=False)

    p_ask = sub.add_parser("ask", help="run one coverage determination")
    p_ask.add_argument("--payer", required=True)
    p_ask.add_argument("--plan")
    p_ask.add_argument("--code", action="append", help="procedure code (repeatable)")
    p_ask.add_argument("--dx", action="append", help="diagnosis code (repeatable)")
    p_ask.add_argument("--as-of", dest="as_of", required=True, help="ISO date, YYYY-MM-DD")
    p_ask.add_argument("--narrative", default="")
    p_ask.add_argument("--question", required=True)
    p_ask.add_argument("--stub", action="store_true", help="run without model calls")
    p_ask.set_defaults(fn=_ask, is_async=True)

    p_eval = sub.add_parser("eval", help="replay the eval set and score it")
    p_eval.add_argument("--stub", action="store_true", help="run without model calls")
    p_eval.add_argument("--limit", type=int, help="only run the first N cases")
    p_eval.add_argument("--diff", action="store_true", help="diff against the committed baseline")
    p_eval.add_argument("--set-baseline", action="store_true", dest="set_baseline")
    p_eval.set_defaults(fn=_eval, is_async=True)

    p_trace = sub.add_parser("trace", help="pretty-print a run trace")
    p_trace.add_argument("run_id")
    p_trace.add_argument("-v", "--verbose", action="store_true")
    p_trace.set_defaults(fn=_trace, is_async=False)

    args = parser.parse_args(argv)
    try:
        return asyncio.run(args.fn(args)) if args.is_async else args.fn(args)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
