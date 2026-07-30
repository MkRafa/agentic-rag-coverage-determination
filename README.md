# Coverage Determination Agent

Agentic RAG over payer medical-policy documents, built so that **every claim in
the output is mechanically traceable to a clause that exists, was in effect on
the date in question, and actually says what it is cited for.**

The question it answers: *does payer P, under plan L, cover procedure C for a
patient with condition D as of date T — and what clause says so?*

> **The corpus is entirely synthetic.** Payers, plans, bulletins, riders and
> clause text are invented, modelled on the public format of medical-policy
> bulletins. No real payer document is used anywhere in this repository.

---

## Why agentic rather than vanilla RAG

A single embedding lookup fails here for reasons that are demonstrable, not
theoretical. All four are baked into the corpus as traps with labelled eval
cases:

| | Failure | Case |
|---|---|---|
| **T1** | Policies are **versioned**. The same facts give opposite answers either side of an effective date. | `T1-cgm-stale-02` vs `T1-cgm-current-03` |
| **T2** | A **plan rider overrides** its own base policy. Reading only the bulletin inverts the answer. | `T2-rider-waive-01` vs `T2-rider-absent-02` |
| **T3** | Two policies with **near-identical titles** govern different codes with different criteria. | `T3-lookalike-01` |
| **T4** | Criteria are **compositional** — `A and B unless C`. Partial satisfaction is not coverage. | `T4-sleep-comorbid-01` |
| **T5** | A provider FAQ **contradicts** the bulletin. Precedence resolves it; the conflict still has to be reported. | `T5-faq-conflict-01` |

So retrieval has to be planned, critiqued, and re-run — and the answer has to be
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

**The load-bearing idea:** the harness is deterministic and holds all control
flow. Subagents reason; they never decide the outcome. The Verifier runs cold —
given only the input and the final answer — so it cannot be talked into agreeing
with reasoning it never saw. Refusal is a first-class success state and the eval
rewards it.

---

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/python -m corpus.generate
```

Everything below runs **without an API key** using `--stub` — a deterministic
stand-in for the model that still performs real MCP retrieval, so the harness,
gate, traces and scorers are genuinely exercised:

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m src.cli eval --stub --diff
```

With `ANTHROPIC_API_KEY` exported, drop `--stub` to run the real subagents:

```bash
.venv/bin/python -m src.cli ask \
  --payer MHP --plan MHP-HMO-BASE --code A9276 --dx E10.9 \
  --as-of 2024-03-15 \
  --narrative "Type 1 diabetes. Chart documents two fingerstick tests per day." \
  --question "Is personal-use CGM covered for this member?"
```

That is case `T1-cgm-stale-02`: the correct answer is **NOT_COVERED**, because on
2024-03-15 the governing version required four daily fingersticks. Change
`--as-of` to `2025-01-20` and the same facts become **COVERED** — the
requirement was removed on 2024-07-01.

---

## Components

### MCP server — `policy-corpus`

The integration boundary. Read-only; there is no write tool anywhere in this
system. Every piece of corpus access goes through these five tools, which is what
makes the retrieval stack swappable without touching agent logic.

| Tool | Returns |
|---|---|
| `search_policies` | Hybrid BM25 + dense hits, filtered to versions in effect on `as_of_date` |
| `get_clause_by_id` | One clause verbatim — the Verifier's ground truth |
| `list_policy_versions` | Version history with effective ranges |
| `lookup_code` | CPT / HCPCS / ICD-10 descriptor + the policies governing it |
| `get_plan_riders` | Riders on a plan and which base policies each overrides |

The swap is real and observable:

```bash
CDA_RERANKER=rrf .venv/bin/python -m src.cli eval --stub --diff
```

```
  = gold-clause recall              1.0000 ->   1.0000  (+0.0000)
  + determination accuracy          0.5455 ->   0.5909  (+0.0454)
  ! gate correctness                0.7727 ->   0.4091  (-0.3636)
  + FALSE AUTO-DETERMINE                 8 ->        4  (-4)
```

Knobs, all recorded in every trace so a scorecard move is attributable:
`CDA_EMBEDDER=tfidf|sentence-transformers`, `CDA_RERANKER=rrf|weighted|date_aware`,
`CDA_VECTOR_BACKEND=local|pinecone`.

### Vector backends — local and Pinecone

The dense half of retrieval sits behind a `VectorBackend`
([policy_corpus/vectorstore.py](policy_corpus/vectorstore.py)). Swapping it
touches nothing in `src/` — no agent, no prompt, no contract.

```bash
.venv/bin/pip install -e ".[pinecone]"

cp .env.example .env          # then set PINECONE_API_KEY — .env is gitignored
# or just: export PINECONE_API_KEY=pcsk_...

.venv/bin/python -m src.cli corpus index --backend pinecone   # deploy step, not startup
CDA_VECTOR_BACKEND=pinecone .venv/bin/python -m src.cli eval --diff
```

`.env` is read at CLI startup by [src/env.py](src/env.py) (no dependency) and
forwarded into the MCP subprocess. A real environment variable always beats the
file, so `export` still overrides a stale `.env`. Never put a key in
`.claude/settings.json` — that one is committed.

Three decisions in that backend are load-bearing:

**Clause text is never written to the index.** Search returns ids and scores;
the store hydrates text from the corpus. `get_clause_by_id` — the Verifier's
ground truth — must not be answerable by an index that could be stale, partially
upserted, or rebuilt with a different embedder. If the vector DB were the source
of clause text, a bad upsert would make the Verifier confirm a false citation
against the same wrong copy the Synthesizer used, and the one check that makes
this system trustworthy would quietly become a no-op. A test asserts no clause
text reaches metadata.

**Effective dates are filtered server-side**, encoded as `YYYYMMDD` integers with
`99991231` standing in for an open-ended version (Pinecone metadata cannot hold
null, and a missing key fails `$gte` rather than passing it). Filtering after
top-k is not equivalent: the superseded version consumes a slot among the k
nearest and the in-effect one silently falls off the end — which is exactly the
T1 trap. `build_filter()` is the single definition of eligibility, and the local
backend evaluates the *same* filter dict, so a scorecard difference between
backends is a ranking finding rather than a filter bug in disguise. A test checks
the filter against `in_effect()` for every clause on both sides of every version
boundary.

**The local backend returns top-k too**, rather than scoring the whole corpus.
Pinecone can only give you the k nearest; a local backend that scored everything
would fuse against a different candidate set, so the swap would change fusion
semantics and the diff would not be attributable to the vector store.

Currently dense-only: BM25 stays in-process. Pinecone hosts a sparse model
(`pinecone-sparse-english-v0`) and `query()` takes `sparse_vector`, so moving the
lexical half server-side is the natural next step — at 40 clauses it would be
pure overhead.

### Subagents

| Agent | Tools | Phase |
|---|---|---|
| Planner | none | online |
| Retriever | MCP (agentic tool loop) | online |
| Grader | none | online |
| Synthesizer | none | online |
| **Verifier** | MCP `get_clause_by_id` (harness-driven) | online |
| Adversary | corpus read | **offline** — generates near-miss eval cases |
| Judge | none | **offline** — scores blind to the trace and the gold key |

Per-role model, effort and token caps live in `src/config.py`. Dropping a role to
a cheaper model is the cost lever — make the change, rerun `eval --diff`, and
read the scorecard rather than asserting the swap was safe.

### Skills — `.claude/skills/`

Domain policy lives in versioned markdown, not in prompt strings, so a policy
change is a reviewable diff. Byte lengths are fingerprinted into every trace, so
when a scorecard moves you can tell whether a skill changed underneath it.

`citation-format` · `coverage-criteria-logic` · `ocr-confidence-gate` ·
`refusal-policy` · `payer-taxonomy`

### Harness — `src/harness/`

Iteration cap, token/cost budget with a hard stop, PHI redaction, injection
defusal, the gate, and JSONL trace emission. Zero LLM calls. If a decision costs
money or has to be auditable, it lives here.

### Timeouts and retries

The SDK retries connection errors, 408/409/429 and 5xx with exponential backoff,
but it does not bound total wall clock and does not bound an agentic tool loop at
all — a Retriever that keeps calling tools can run indefinitely inside a single
logical call. Three caps sit on top, all in `src/config.py`:

| Cap | Default | Bounds |
|---|---|---|
| `RoleConfig.timeout_s` | 90–600s per role | one HTTP request |
| `tool_loop_timeout_s` | 300s | the Retriever's whole tool loop |
| `run_timeout_s` | 900s | one end-to-end determination |

A blown cap is a **REVIEW with a stated reason**, never an exception the caller
has to interpret and never a silent hang. Because the SDK retries timeouts, one
logical call can cost `timeout_s × (max_retries + 1)`; a test asserts that worst
case still fits inside `run_timeout_s`, and that the tool-loop cap is tighter
than the run cap so a hung Retriever is distinguishable from a slow pipeline.

### Repo hooks — `.claude/`

Two project-scoped Claude Code hooks protect the thing that makes the regression
gate meaningful. Both are plain Python reading the hook payload on stdin, so they
have no `jq` dependency and are unit-testable.

| Hook | Event | Behaviour |
|---|---|---|
| [`baseline_staleness.py`](.claude/hooks/baseline_staleness.py) | PostToolUse | Editing the gate, pre-flight, loop, contracts, retrieval or any skill prints a reminder that the committed scorecard now describes a system that no longer exists. Warns only. |
| [`protect_baseline.py`](.claude/hooks/protect_baseline.py) | PreToolUse | **Denies** hand-edits to `evals/baseline.json`. Editing it to make a regression disappear silently converts the gate into decoration; it may only change via `eval --set-baseline`. |

Both fail open on a malformed payload — a broken hook must not block all edits.

---

## Evals — two layers, scored separately

Separating them is what makes a regression attributable: recall down *and*
accuracy down is a retrieval problem; recall flat and accuracy down is a
synthesis problem.

**Layer 1 — retrieval** (deterministic): gold-clause recall, stale-retrieval
rate, clauses retrieved per case.

**Layer 2 — answer**: determination accuracy, gate correctness, **citation
faithfulness** (the Verifier makes this measurable rather than a vibe),
hallucinated-clause count (target: zero), stale-citation count, appropriate-
refusal rate, cost per case, p95 latency — plus **`false_auto_determine`**, the
one that must stay at zero: the system autonomously determined, and was wrong.
Anything the gate caught does not count against it.

The offline Judge scores reasoning quality, clarity and hedging — never
correctness. Determination accuracy and faithfulness are *measured*; asking a
model to also grade those would launder a number into an opinion.

```bash
.venv/bin/python -m src.cli eval --stub                # scorecard
.venv/bin/python -m src.cli eval --stub --diff         # vs committed baseline
.venv/bin/python -m src.cli eval --stub --set-baseline # freeze a new baseline
```

`evals/baseline.json` is committed. `eval` exits non-zero when
`false_auto_determine > 0`, so it works as a CI gate.

### Fault injection

The gate's rejection paths are tested rather than assumed. `CDA_STUB_FAULT`
makes the stub produce a specific bad citation:

```bash
CDA_STUB_FAULT=hallucinate .venv/bin/python -m src.cli ask --stub ...
```

| Fault | Gate | Caught by |
|---|---|---|
| `hallucinate` | REVIEW | cited clause does not exist |
| `paraphrase` | REVIEW | quote is not verbatim |
| `stale` | REVIEW | clause not in effect on the as-of date |
| `unsupported` | REVIEW | quote does not entail the claim |
| `lowconf` | REVIEW | confidence below threshold |
| *(none)* | DETERMINE | faithfulness 1.00 |

---

## Layout

```
policy_corpus/      MCP server — server.py, store.py, retrieval.py
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
  cli.py            ask / eval / corpus / trace
evals/              cases/, scorers/, runner.py, baseline.json
tests/              deterministic tests — no model in the path
traces/             per-run JSONL
```

---

## Status

Verified working: corpus generation, the MCP server over stdio, hybrid retrieval
with effective-date filtering, pre-flight (redaction, injection detection, halt
conditions), the full loop, the gate against all five injected failure modes, the
budget meter, the timeout caps, traces, both eval scorers, the baseline-diff
workflow, the local vector backend, and both repo hooks (pipe-tested against
matching, non-matching and malformed payloads). 58 tests pass.

The hooks are written and validated but have **not been observed firing** — the
settings watcher only watches directories that already had a settings file when
the session started, and `.claude/settings.json` is new. Open `/hooks` once, or
restart Claude Code, and they go live.

**The Pinecone backend has also never been run against a live index** — no
`PINECONE_API_KEY` in the build environment. Its filter semantics, date encoding
and metadata shape are unit-tested against the same predicate the local backend
uses, and every SDK binding was checked against the installed `pinecone` 9.1.0
rather than written from memory. The network path itself is unverified.

**Not yet verified: the live model path.** No `ANTHROPIC_API_KEY` was available
in the build environment, so every real subagent call — planner, retriever tool
loop, grader, synthesizer, verifier entailment — is written against the SDK but
has never executed. The committed baseline is a **stub-mode** scorecard; its
reasoning-dependent numbers (determination accuracy 0.55, appropriate-refusal
rate 0.00) reflect the stub heuristic, not a model. Export a key, run
`eval --set-baseline`, and replace it before reading anything into those figures.
