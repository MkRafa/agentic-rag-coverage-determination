"""CLI.

    cda corpus build              regenerate the synthetic corpus
    cda ask ...                   run one determination end to end
    cda eval                      replay the eval set, print a scorecard
    cda eval --diff               compare against the committed baseline
    cda eval --set-baseline       freeze the current scorecard as the baseline
                                  (stub -> evals/baseline.json; a live model ->
                                  evals/baselines/<model>.json)
    cda trace <run_id>            pretty-print a run trace
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from .env import load_dotenv

# Before anything imports config. Settings read os.environ at import time, so
# loading .env later (as this used to, inside main) meant CDA_MODEL and the
# other knobs set in .env were silently ignored. Real env vars still win.
load_dotenv()

from .config import SETTINGS  # noqa: E402


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
    from evals import runner

    mode = "stub" if args.stub else f"live, {SETTINGS.synthesizer.model}"
    print(f"replaying eval set ({mode})…", flush=True)
    scorecard = await runner.run_eval(stub=args.stub, limit=args.limit, suites=args.suite)
    print(runner.render(scorecard))

    if args.set_baseline:
        path = runner.write_baseline(scorecard)
        print(f"baseline written to {path}")
        return 0

    baseline = runner.load_baseline(stub=args.stub)
    failed = False
    if args.diff:
        if baseline is None:
            print("no baseline committed yet — run with --set-baseline to freeze one")
            return 1
        print(runner.diff(scorecard, baseline))
        failed = bool(runner.blocking_regressions(scorecard, baseline))

    if args.stub:
        # The stub's outcomes come from a keyword heuristic, so its accuracy and
        # false-auto-determine numbers describe the heuristic, not the system.
        # In stub mode the gate is "did the harness regress vs the baseline".
        print(
            "\nstub mode: outcome metrics reflect a keyword heuristic, not a model. "
            "Exit status gates on regressions vs the baseline (--diff) only."
        )
        return 1 if failed else 0

    # Live: an autonomous wrong answer fails the run outright.
    if scorecard["answer"]["false_auto_determine"] > 0:
        failed = True
    return 1 if failed else 0


# ---------------------------------------------------------------------------
# corpus / trace
# ---------------------------------------------------------------------------


def _corpus(args: argparse.Namespace) -> int:
    if args.action == "build":
        from corpus.generate import main as build

        build()
        return 0

    # `index` — push the corpus into a vector backend. A deploy step, not
    # something that runs on process start.
    import json as _json

    from policy_corpus.retrieval import HybridIndex, build_embedder
    from policy_corpus.vectorstore import (
        LocalVectorBackend,
        PineconeVectorBackend,
        clause_metadata,
    )

    corpus = _json.loads(SETTINGS.corpus_path.read_text())
    clauses = corpus["clauses"]

    if args.backend == "local":
        print("the local backend is built in-process on every start — nothing to index")
        return 0

    backend = PineconeVectorBackend()
    print(
        f"upserting {len(clauses)} clauses -> pinecone index "
        f"'{backend.index_name}' namespace '{backend.namespace}' "
        f"(model {backend.model})"
    )
    docs = [HybridIndex._doc_text(c) for c in clauses]
    backend.build(
        [c["clause_id"] for c in clauses], docs, [clause_metadata(c) for c in clauses]
    )
    print("done. run with CDA_VECTOR_BACKEND=pinecone to query it.")
    return 0


async def _evals(args: argparse.Namespace) -> int:
    import json as _json

    from policy_corpus.store import get_store

    from evals import runner
    from evals.validate import validate

    store = get_store()

    if args.action == "validate":
        report = validate(runner.load_cases(), store)
        print(report.render())
        return 0 if report.ok else 1

    if args.action == "expand":
        from evals.expand import main as expand_main

        return expand_main()

    # `generate` — the Adversary. A model proposes cases; nothing it produces is
    # trusted. Every case goes through the same validator the deterministic
    # expansion does, and anything that fails is dropped with the reason shown.
    from evals.expand import OUT as GENERATED

    from .agents import adversary
    from .harness.budget import Budget
    from .harness.trace import Trace
    from .llm import build_client

    existing = runner.load_cases()
    existing_ids = [c["case_id"] for c in existing]

    from dataclasses import replace as _replace

    from .config import SETTINGS

    cfg = SETTINGS.adversary
    if args.model:
        cfg = _replace(cfg, model=args.model)

    budget = Budget()
    trace = Trace()
    client = build_client(budget, trace, stub=args.stub)

    corpus = _json.loads(store.path.read_text())
    print(
        f"asking the Adversary ({cfg.model}) for {args.n} cases "
        f"(suite currently has {len(existing)})…"
    )
    batch = await adversary.generate(
        client, corpus, n=args.n, existing_ids=existing_ids, config=cfg
    )

    proposed = [c.model_dump() for c in batch.cases]
    for case in proposed:
        case["generated_by"] = f"adversary:{cfg.model}"

    seen = set(existing_ids)
    fresh = [c for c in proposed if c["case_id"] not in seen and not seen.add(c["case_id"])]
    duplicates = len(proposed) - len(fresh)

    kept: list[dict[str, Any]] = []
    rejected: list[tuple[str, str]] = []
    for case in fresh:
        report = validate([case], store)
        if report.ok:
            kept.append(case)
        else:
            rejected.append((case["case_id"], "; ".join(p.detail for p in report.problems)))

    print(f"\nproposed {len(proposed)} · duplicate ids {duplicates} · "
          f"rejected {len(rejected)} · kept {len(kept)}")
    for cid, why in rejected:
        print(f"  rejected {cid}: {why}")

    if not kept:
        print("\nnothing survived validation — not writing")
        return 1

    if args.stub:
        # The stub batch is a validator demo. Writing its "valid" case into the
        # committed suite would put a canned case into every future scorecard.
        print("\nstub mode — validator demo only, not writing to the case set")
        return 0

    out = GENERATED.parent / "adversarial.json"
    prior = _json.loads(out.read_text())["cases"] if out.exists() else []
    out.write_text(
        _json.dumps(
            {
                "description": (
                    "Adversary-proposed cases. Every case passed the deterministic "
                    "validator; the clinical judgement in each still wants a human read."
                ),
                "cases": prior + kept,
            },
            indent=2,
        )
    )
    print(f"\nwrote {len(prior) + len(kept)} cases -> {out}")
    print(f"tokens {budget.input_tokens + budget.output_tokens} · ${budget.usd:.4f}")
    print("\nreview the kept cases before trusting the labels, then re-baseline.")
    return 0


def _cost(args: argparse.Namespace) -> int:
    from evals import cost

    m = cost.measure()
    print(cost.render(m, cases=args.cases, cached_system_tokens=args.system_tokens))
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
    p_corpus.add_argument("action", choices=["build", "index"])
    p_corpus.add_argument(
        "--backend", choices=["local", "pinecone"], default="local",
        help="vector backend to index into (`index` action only)",
    )
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
    p_eval.add_argument(
        "--suite", action="append", choices=["seed", "generated", "adversarial"],
        help="case file(s) to run (repeatable); default is all of them",
    )
    p_eval.add_argument("--diff", action="store_true", help="diff against the committed baseline")
    p_eval.add_argument("--set-baseline", action="store_true", dest="set_baseline")
    p_eval.set_defaults(fn=_eval, is_async=True)

    p_evals = sub.add_parser("evals", help="build and check the eval case set")
    p_evals.add_argument(
        "action",
        choices=["expand", "validate", "generate"],
        help="expand: deterministic cases with derived labels; "
             "validate: check every case against the corpus; "
             "generate: ask the Adversary for new cases (needs ANTHROPIC_API_KEY)",
    )
    p_evals.add_argument("-n", type=int, default=20, help="cases to request (generate only)")
    p_evals.add_argument(
        "--model", help="override the Adversary model, e.g. claude-haiku-4-5 (generate only)"
    )
    p_evals.add_argument("--stub", action="store_true", help="run generate without model calls")
    p_evals.set_defaults(fn=_evals, is_async=True)

    p_cost = sub.add_parser("cost", help="project suite cost from measured live traces")
    p_cost.add_argument("--cases", type=int, default=152)
    p_cost.add_argument(
        "--system-tokens", type=int, default=8313,
        help="per-case system-prompt tokens eligible for prompt caching",
    )
    p_cost.set_defaults(fn=_cost, is_async=False)

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
