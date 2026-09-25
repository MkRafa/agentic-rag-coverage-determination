# Pricing a run before firing it

```bash
CDA_MODEL=claude-sonnet-5 .venv/bin/cda eval --limit 2   # calibrate on real traces
.venv/bin/cda cost --cases 152                           # project
```

`cda cost` extrapolates from *measured* live traces, not guesses. Stub traces
are excluded, because their token counts are nominal. A two-case calibration
run on Sonnet 5 measured **25,786 input + 4,207 output tokens per case** over 5
calls:

| role | in/case | out/case |
|---|---:|---:|
| retriever | 13,474 | 762 |
| synthesizer | 4,252 | 1,300 |
| grader | 3,948 | 1,013 |
| planner | 2,812 | 849 |
| verifier | 1,301 | 282 |

Projected for the full 152-case suite:

| model | 152-case run | with prompt caching |
|---|---:|---:|
| Haiku 4.5 | $7.12 | $5.99 |
| Sonnet 5 | $21.35 | $17.97 |
| Opus 5 | $35.58 | $29.95 |

Rates are list prices from [evals/cost.py](../evals/cost.py). Check them before
relying on the projection.

Two things the measurement exposed:

- **The Retriever is 52% of all input.** Its tool loop resends the growing
  conversation every turn, so it costs more than the Synthesizer and Grader
  combined. This was invisible until a bug was fixed: the tool loop's tokens
  were charged to the budget but left out of the trace, so the Retriever looked
  free.
- **8,313 tokens per case are system prompts that never change** (the skills
  are static), which is what the caching column prices. `cache_control` is not
  yet wired.

Wall clock is the real constraint, not money: ~98s per case measured, so
**about 4 hours sequential** for 152 cases. The eval loop currently runs cases
one at a time.
