# Vector backends — local and Pinecone

The dense half of retrieval sits behind a `VectorBackend`
([policy_corpus/vectorstore.py](../policy_corpus/vectorstore.py)). Swapping it
touches nothing in `src/`: no agent, no prompt, no contract.

```bash
.venv/bin/pip install -e ".[pinecone]"

cp .env.example .env          # then set PINECONE_API_KEY — .env is gitignored
# or just: export PINECONE_API_KEY=pcsk_...

.venv/bin/cda corpus index --backend pinecone   # deploy step, not startup
CDA_VECTOR_BACKEND=pinecone .venv/bin/cda eval --stub --diff
```

`.env` is read at CLI startup by [src/env.py](../src/env.py) (no dependency) and
forwarded into the MCP subprocess. A real environment variable always beats the
file, so `export` still overrides a stale `.env`. Never put a key in
`.claude/settings.json` — that one is committed.

## Three load-bearing decisions

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

## Verified against a live index

40 clauses upserted to a serverless index (1024-dim, cosine, `us-east-1`,
`multilingual-e5-large`), then the eval replayed through it. This was the
22-case suite, before it was expanded, with the stub model: the point was to
exercise the retrieval backend, not the model.

```
scorecard  mode=stub
           vectors=pinecone  embedder=multilingual-e5-large  reranker=date_aware

  = gold-clause recall              1.0000 ->   1.0000  (+0.0000)
  = stale retrieval rate            0.0000 ->   0.0000  (+0.0000)
  = citation faithfulness           1.0000 ->   1.0000  (+0.0000)
  = determination accuracy          0.5455 ->   0.5455  (+0.0000)
  ! p95 latency (s)                 0.0130 ->   2.6310  (+2.6180)
```

Every quality metric identical; only latency moved — 13ms to 2.6s p95, because
each sub-query now costs two network round trips (hosted embedding, then ANN
query). That is the whole tradeoff, stated in numbers rather than asserted. The
T1 date trap and T2 rider scoping were both confirmed to hold server-side.

The live run also caught a bug the unit tests structurally could not:
`QueryResponse.matches` yields `ScoredVector` objects, which support attribute
and item access but are not `dict()`-coercible. Filter semantics are testable
offline; response shapes are not.
