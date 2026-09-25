# Coverage Determination Agent

Agentic RAG over payer medical-policy documents. The goal is that **every claim
in the output can be mechanically traced to a clause that exists, was in effect
on the date in question, and actually says what it is cited for.** When that
can't be shown, the system sends the case to a human instead of answering.

The question it answers: *does payer P, under plan L, cover procedure C for a
patient with condition D as of date T, and which clause says so?*

> **The corpus is entirely synthetic.** Payers, plans, bulletins, riders and
> clause text are invented, modelled on the public format of medical-policy
> bulletins. No real payer document is used anywhere in this repository.

**In one paragraph:** a deterministic Python harness runs a
Planner → Retriever → Grader loop over an MCP policy server. A Synthesizer writes
a cited determination, and a Verifier re-fetches every cited clause cold and
checks it. A pure-Python gate then decides DETERMINE, REVIEW, REFUSE or HALT. The
models reason, but they never decide the outcome. A 152-case eval suite, most of
it labelled by construction from the corpus rather than by a model, measures
retrieval and answers separately.

---

## Status

| | |
|---|---|
| **Verified** | Harness, pre-flight, MCP server, hybrid retrieval with date filtering, the gate against five injected failure modes, budget and timeout caps, traces, both scorers, baseline diffing, the case validator, the local-model client. 93 deterministic tests. |
| **Run live** | Adversary case generation on Haiku 4.5 (48 proposed, 18 kept). Pinecone backend against a real serverless index. A two-case Sonnet 5 calibration run of the full pipeline, used for [cost measurement](docs/cost.md). |
| **Not yet run** | **The full eval suite against a real model.** The offline Judge (written and wired, never executed). |

So the committed baseline is a **stub-mode** scorecard. The stub is a
deterministic stand-in for the model: it performs real MCP retrieval, but its
outcomes come from a keyword heuristic. Read the numbers accordingly:

| metric | stub baseline | what it tells you |
|---|---:|---|
| gold-clause recall | 1.00 | Real hybrid retrieval over MCP finds every gold clause |
| stale retrieval rate | 0.00 | Date filtering holds across all 152 cases |
| citation faithfulness | 1.00 | Honest citations pass; the fault-injection table below shows dishonest ones fail |
| determination accuracy | 0.55 | **The heuristic, not a model.** Not a system result. |
| false auto-determine | 59 | **The heuristic, not a model.** The stub answers COVERED on 145 of 152 cases |

Producing the live numbers takes one command, `cda eval --set-baseline` with a
key. The measured projection is ~$21 and ~4 hours for the full suite on
Sonnet 5 ([docs/cost.md](docs/cost.md)).

---

## Why agentic rather than vanilla RAG

A single embedding lookup fails here, and each way it fails is built into the
corpus as a trap with labelled eval cases:

| | Failure | Example case |
|---|---|---|
| **T1** | Policies are **versioned**. The same facts give opposite answers either side of an effective date. | `T1-cgm-stale-02` vs `T1-cgm-current-03` |
| **T2** | A **plan rider overrides** its own base policy. Reading only the bulletin inverts the answer. | `T2-rider-waive-01` vs `T2-rider-absent-02` |
| **T3** | Two policies with **near-identical titles** govern different codes with different criteria. | `T3-lookalike-01` |
| **T4** | Criteria are **compositional**: `A and B unless C`. Partial satisfaction is not coverage. | `T4-sleep-comorbid-01` |
| **T5** | A provider FAQ **contradicts** the bulletin. Precedence resolves it; the conflict still has to be reported. | `T5-faq-conflict-01` |
| **T6** | The corpus **does not answer** the question, or a required member fact is missing. | `T6-missing-fact-02` |
| **T7** | **Instruction-shaped text** in the clinical narrative tries to force an outcome. | `T7-injection-01` |

So retrieval has to be planned, critiqued and re-run, and the answer has to be
verified against the corpus rather than trusted.

---

## Architecture

```
   request                ┌──────────────────────────────────────────────┐
   ─────────              │ HARNESS — deterministic Python, no LLM       │
   payer, plan,           │ owns control flow, budget, gate, traces      │
   codes, dx,             └──────────────────────────────────────────────┘
   as_of_date                             │
        │              ┌─────────────────▼──────────────────┐
        └────────────► │ 0. PRE-FLIGHT (no model)           │  skill: ocr-confidence-gate
                       │   PHI redaction → field extraction │
                       │   → confidence → code validation   │──── below threshold ──► ⛔ HALT
                       │   → injection defusal              │      (names the field)
                       └─────────────────┬──────────────────┘
                                         │
        ╔════════════════════════════════▼═══════════════════════════════╗
        ║  AGENTIC LOOP  (capped at 3 iterations, metered, traced)       ║
        ║                                                                ║
        ║   PLANNER ──► RETRIEVER ──► GRADER ──► sufficient? ─no─┐       ║
        ║   no tools    MCP tools     no tools                   │       ║
        ║                  │                              re-plan with   ║
        ║                  ▼                              the named gap  ║
        ║       ┌──────────────────────┐                          │      ║
        ║       │ MCP: policy-corpus   │◄─────────────────────────┘      ║
        ║       │ (read-only, 5 tools) │                                 ║
        ║       └──────────────────────┘                                 ║
        ╚════════════════════════════════┬═══════════════════════════════╝
                                         │ yes
                       ┌─────────────────▼──────────────────┐
                       │ SYNTHESIZER — no tools             │  skills: citation-format,
                       │ determination + [clause_id] cites  │  coverage-criteria-logic,
                       └─────────────────┬──────────────────┘  refusal-policy
                                         │
                       ┌─────────────────▼──────────────────┐
                       │ VERIFIER                           │
                       │  deterministic: cold MCP re-fetch  │  ◄── sees only the answer,
                       │    · clause exists?                │      never the reasoning
                       │    · quote verbatim?               │      trace
                       │    · in effect on as_of?           │
                       │  model-mediated:                   │
                       │    · does the quote entail it?     │
                       └─────────────────┬──────────────────┘
                                         │
                       ┌─────────────────▼──────────────────┐
                       │ GATE — pure Python, no model       │
                       │ faithfulness == 1.0                │
                       │ AND no unresolved contradiction    │
                       │ AND confidence ≥ threshold         │
                       └──┬──────────────┬──────────────┬───┘
                          │              │              │
                     ✅ DETERMINE    ⚠️ REVIEW      🚫 REFUSE
                     answer +        answer + the   "corpus does not
                     verified cites  gap named      answer this"
                                                    (a success state)
```

**The core idea:** the harness is deterministic and holds all control flow.
Subagents reason; they never decide the outcome. The Verifier runs cold. It is
given only the input and the final answer, so it can't be talked into agreeing
with reasoning it never saw. Refusal counts as a success state, and the eval
rewards it.

---

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q          # generates the corpus on first run
```

Everything below runs **without an API key** using `--stub`:

```bash
.venv/bin/cda eval --stub --diff       # 152 cases, scorecard vs committed baseline

# Watch the gate reject a citation to a policy version not yet in effect
CDA_STUB_FAULT=stale .venv/bin/cda ask --stub \
  --payer MHP --plan MHP-HMO-BASE --code A9276 --dx E10.9 --as-of 2024-03-15 \
  --question "Is personal-use CGM covered?"
```

```
⚠️  GATE: REVIEW
   · clause MHP-MP-0142.v2.C3 was not in effect on the as-of date
```

With `ANTHROPIC_API_KEY` exported (or in `.env`), drop `--stub` to run the real
subagents:

```bash
.venv/bin/cda ask \
  --payer MHP --plan MHP-HMO-BASE --code A9276 --dx E10.9 \
  --as-of 2024-03-15 \
  --narrative "Type 1 diabetes. Chart documents two fingerstick tests per day." \
  --question "Is personal-use CGM covered for this member?"
```

That is case `T1-cgm-stale-02`. The correct answer is **NOT_COVERED**, because
on 2024-03-15 the governing version required four daily fingersticks. Change
`--as-of` to `2025-01-20` and the same facts become **COVERED**, because the
requirement was removed on 2024-07-01.

### Free, on a local model

Any model id starting with `ollama/` runs every role on a local
[Ollama](https://ollama.com) server. No API key is needed and nothing leaves
the machine:

```bash
ollama pull qwen2.5:7b
export CDA_MODEL=ollama/qwen2.5:7b CDA_TIMEOUT_SCALE=4
.venv/bin/cda ask --payer MHP --plan MHP-HMO-BASE --code A9276 ...   # ~4 min on an M3 Pro
.venv/bin/cda eval --suite seed --set-baseline                       # 22 cases
```

The client uses Ollama's native API so it can set a 32k context window per
request. Ollama's default window is much smaller and truncates silently.
Small models also need a tighter tool loop: on its first run, qwen2.5:7b spent
five minutes emitting parallel tool calls. The local path therefore caps calls
per turn, output per tool turn, and tool-result size. A reply that breaks the
output contract becomes a REVIEW for that case rather than aborting the eval.
Each model's baseline is written to `evals/baselines/<model>.json`, so a live
run never overwrites the stub baseline that CI diffs against.

---

## Components

### MCP server: `policy-corpus`

This is the integration boundary. It is read-only, and nothing in the system
can write. All corpus access goes through these five tools, which is what makes
the retrieval stack swappable without touching agent logic.

| Tool | Returns |
|---|---|
| `search_policies` | Hybrid BM25 + dense hits, filtered to versions in effect on `as_of_date` |
| `get_clause_by_id` | One clause verbatim: the Verifier's ground truth |
| `list_policy_versions` | Version history with effective ranges |
| `lookup_code` | CPT / HCPCS / ICD-10 descriptor + the policies governing it |
| `get_plan_riders` | Riders on a plan in effect on `as_of_date`, and which base policies each overrides |

The swap is real and shows up in the scorecard. Changing only the reranker, on
the same 152 cases:

```bash
CDA_RERANKER=rrf .venv/bin/cda eval --stub --diff
```

```
  = gold-clause recall                 1.0000 ->       1.0000  (+0.0000)
  + determination accuracy             0.5526 ->       0.5987  (+0.0461)
  ! gate correctness                   0.8026 ->       0.5855  (-0.2171)
  + FALSE AUTO-DETERMINE                   59 ->           42  (-17)
```

Every knob is recorded in every trace, so a scorecard move can be traced to its
cause: `CDA_EMBEDDER=tfidf|sentence-transformers`,
`CDA_RERANKER=rrf|weighted|date_aware`, `CDA_VECTOR_BACKEND=local|pinecone`.

The dense half is pluggable: a local numpy backend by default, or Pinecone
serverless with hosted embeddings. Clause text is never written to the vector
index, and effective dates are filtered server-side before top-k rather than
after. [docs/vector-backends.md](docs/vector-backends.md) explains why both
matter, and gives the live Pinecone result: identical quality metrics, with p95
latency going from 13ms to 2.6s.

### Subagents

| Agent | Tools | Phase |
|---|---|---|
| Planner | none | online |
| Retriever | MCP (agentic tool loop) | online |
| Grader | none | online |
| Synthesizer | none | online |
| **Verifier** | MCP `get_clause_by_id` (harness-driven) | online |
| Adversary | corpus read | **offline**: generates near-miss eval cases |
| Judge | none | **offline**: scores reasoning blind to the trace and gold key. *Not yet run.* |

Per-role model, effort and token caps live in `src/config.py`, and `CDA_MODEL`
swaps every role at once. Moving a role to a cheaper model is the cost lever.
Make the change, rerun `eval --diff`, and read the scorecard instead of assuming
the swap was safe.

### Skills: `.claude/skills/`

Domain policy lives in versioned markdown, not in prompt strings, so a policy
change is a reviewable diff. Byte lengths are fingerprinted into every trace, so
when a scorecard moves you can tell whether a skill changed underneath it.

`citation-format` · `coverage-criteria-logic` · `ocr-confidence-gate` ·
`refusal-policy` · `payer-taxonomy`

### Harness: `src/harness/`

Iteration cap, token/cost budget with a hard stop, PHI redaction, injection
defusal, the gate, and JSONL trace emission. It makes no LLM calls. Anything
that costs money or needs an audit trail is decided here.

The SDK retries transient errors but does not bound total wall clock, and it
does not bound an agentic tool loop at all. So three caps sit on top: per
request (`timeout_s`), per Retriever tool loop (300s), and per run (900s). A
blown cap becomes a **REVIEW with a stated reason**, never an exception and
never a silent hang. A test asserts that the worst case with retries still fits
inside the run cap.

### Repo hooks: `.claude/`

Two project-scoped Claude Code hooks protect the committed baseline. That
baseline is what makes the regression gate meaningful.

| Hook | Event | Behaviour |
|---|---|---|
| [`baseline_staleness.py`](.claude/hooks/baseline_staleness.py) | PostToolUse | Editing the gate, pre-flight, loop, contracts, retrieval or any skill prints a reminder that the committed scorecard is now stale. Warns only. |
| [`protect_baseline.py`](.claude/hooks/protect_baseline.py) | PreToolUse | **Denies** hand-edits to `evals/baseline.json`. It may only change via `eval --set-baseline`. |

Both hooks fail open on a malformed payload, so a broken hook can't block all
edits.

---

## Evals: two layers, scored separately

Keeping the layers separate is what lets you tell which part regressed. If
recall and accuracy both drop, it's a retrieval problem. If recall is flat and
accuracy drops, it's a synthesis problem.

**Layer 1: retrieval** (deterministic). Gold-clause recall, stale-retrieval
rate, clauses retrieved per case.

**Layer 2: answer.** Determination accuracy, gate correctness, **citation
faithfulness** (the Verifier makes this measurable), hallucinated-clause count,
stale-citation count, appropriate-refusal rate, cost, p95 latency. Plus
**`false_auto_determine`**: the system decided on its own and was wrong. That
one must stay at zero. Anything the gate caught doesn't count against it.

The offline Judge scores reasoning quality, clarity and hedging, never
correctness. Determination accuracy and faithfulness are *measured*; having a
model also grade them would turn a measured number into an opinion.

```bash
.venv/bin/cda eval --stub                # scorecard
.venv/bin/cda eval --stub --diff         # vs committed baseline
.venv/bin/cda eval --stub --set-baseline # freeze a new baseline
```

**Exit status.** In live mode, `eval` fails if `false_auto_determine > 0`. In
stub mode the outcome numbers describe the heuristic, so `eval --stub --diff`
fails only when a safety-critical metric (false auto-determine, citation
faithfulness, hallucinated clauses) regresses against the baseline. That makes
it usable as a CI gate without an API key; see
[.github/workflows/ci.yml](.github/workflows/ci.yml).

### The case set: 152 cases, and where the labels come from

| Source | Count | Labels |
|---|---|---|
| Hand-written seeds (`cases/seed.json`) | 22 | Written with the corpus open |
| Deterministic expansion (`cases/generated.json`) | 112 | **Derived from corpus structure** |
| Adversary on Haiku 4.5 (`cases/adversarial.json`) | 18 | Model-proposed, checked by the validator, read by a human |

The split is the point. For the mechanical traps, a model has no business
writing the answer key: which policy version governs a date is arithmetic. So
[evals/expand.py](evals/expand.py) writes a dozen clinical fact-patterns once
and crosses them with every version, every effective-date boundary, and both a
rider-bearing and a rider-free plan. The labels are correct by construction. A
dimension is only crossed in where it changes the answer: the naive cross
product is 252 cases, and the informative subset is 112.

```bash
.venv/bin/cda evals expand      # regenerate derived cases
.venv/bin/cda evals validate    # check every label against the corpus
.venv/bin/cda evals generate    # Adversary (needs ANTHROPIC_API_KEY)
```

**Nothing enters the suite unchecked.** A case with a wrong label doesn't fail
loudly; it quietly changes what the eval measures. So
[evals/validate.py](evals/validate.py) checks mechanically:

- clause ids exist
- gold clauses are in effect on the case's own as-of date
- riders belong to the case's plan
- the cited policy governs the requested code
- an expected HALT is actually reachable (it runs pre-flight to check)

`evals generate --stub` shows it rejecting the defect classes a model really
produces, without writing to the suite.

### What the live Adversary run showed

48 cases proposed on Haiku 4.5 across two batches, **18 kept (a 37% yield)**:

| Outcome | Count | Caught by |
|---|---|---|
| Rejected: mechanical defect | 9 | validator |
| Rejected: wrong clinical label | 3 | human read |
| Rejected: duplicate coverage | 16 | `find_duplicates` |
| **Kept** | **18** | |

- **The first validator caught 1 of 6 defects.** The rest were things I hadn't
  thought to check: citing a `Scope` clause to support a determination, tagging
  a well-formed request `HALT` because its narrative contained an injection, and
  citing a clause whose policy doesn't govern the requested code. Three new
  rules now cover them. The 134 existing cases stayed clean, so there were no
  false positives.
- **16 of 37 survivors duplicated the deterministic expansion.** Duplicates are
  keyed on what a case actually tests, not on its wording. A model asked for
  adversarial cases mostly re-derives the mechanical traps that are already
  generated for free. Where it adds value is where structure can't reach:
  conflicts between sources, unusual fact patterns, missing-fact refusals.
- **Both injection cases got zero flags from the pre-flight detector.** Haiku
  invented `[SYSTEM DIRECTIVE]` framings that the blocklist doesn't match. That
  confirms the injection patterns are telemetry, not a barrier. The real defence
  is structural: narrative text is fenced as untrusted data, and the gate
  ignores what the model *says* about its citations.

### What expanding the suite caught

Going from 22 to 134 cases found a real bug within one run. `search_policies`
had always filtered by effective date, but `get_plan_riders` didn't. On a 2023
date the Retriever was handed a rider that took effect on 2024-01-01. **A rider
overrides base policy, so applying one before it exists inverts the
determination.** 21 cases were hitting this. `get_plan_riders` now takes
`as_of_date`, and a regression test covers both sides of the boundary.

The 22 hand-written cases had never picked a date before a rider existed,
because whoever wrote them already knew what the rider said.

**A baseline only compares against the same questions.** Every scorecard
carries a `case_set` fingerprint. `--diff` refuses to treat a comparison across
different case sets as a signal, and never fails the run on one.

### Fault injection

The gate's rejection paths are tested, not assumed. `CDA_STUB_FAULT` makes the
stub produce a specific bad citation:

| Fault | Gate | Caught by |
|---|---|---|
| `hallucinate` | REVIEW | cited clause does not exist |
| `paraphrase` | REVIEW | quote is not verbatim |
| `stale` | REVIEW | clause not in effect on the as-of date |
| `unsupported` | REVIEW | quote does not entail the claim |
| `lowconf` | REVIEW | confidence below threshold |
| *(none)* | DETERMINE | faithfulness 1.00 |

---

## Known limitations

- **Citation faithfulness is not outcome correctness.** The Verifier checks
  that each claim is supported by its quote. It does not check that the
  *outcome* follows from the claims. The stub shows this directly: for a Type 1
  diabetes member it cites the Type 2 CGM clause verbatim and returns
  DETERMINE / COVERED. Every check passes and the answer is wrong. That is why
  `false_auto_determine` is measured against the gold key rather than inferred
  from faithfulness. Closing this needs an outcome-consistency check: every
  criterion of the governing clause mapped to a stated fact.
- **Injection detection is a blocklist**, and the live Adversary beat it (see
  above). It's recorded as telemetry. The defences that matter are structural.
- **Evals run sequentially**, at ~98s per case live. Running cases concurrently
  and prompt-caching the static system prompts (8.3k tokens per case) are the
  two obvious next steps; see [docs/cost.md](docs/cost.md).
- **BM25 is in-process.** Moving the lexical half to Pinecone's hosted sparse
  model is the natural next step, but at 40 clauses it would be pure overhead.

---

## Layout

```
policy_corpus/      MCP server — server.py, store.py, retrieval.py, vectorstore.py
corpus/             synthetic corpus generator + generated/
src/
  config.py         per-role model/effort/caps, gate thresholds, budgets
  contracts.py      pydantic IO contracts, used as structured-output schemas
  skills.py         loads .claude/skills/*/SKILL.md into system prompts
  llm.py            model client (Anthropic SDK) + stub backend
  mcp_client.py     stdio session + typed facade over the MCP tools
  agents/           planner, retriever, grader, synthesizer, verifier,
                    adversary, judge
  harness/          preflight, loop, gate, budget, trace
  cli.py            ask / eval / evals / corpus / cost / trace
evals/              cases/, scorers/, runner.py, validate.py, expand.py,
                    cost.py, baseline.json
docs/               vector backends, cost measurement
tests/              deterministic tests — no model in the path
traces/             per-run JSONL (gitignored)
```
