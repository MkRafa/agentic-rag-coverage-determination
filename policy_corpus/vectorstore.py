"""Pluggable dense-vector backends.

Two implementations behind one interface:

  LocalVectorBackend      in-process numpy over TF-IDF/SVD vectors. Default.
                          No network, no key, deterministic — which is what the
                          committed eval baseline needs.
  PineconeVectorBackend   a Pinecone serverless index with Pinecone-hosted
                          embeddings. What an enterprise deployment actually
                          looks like.

Two rules make the swap honest.

**1. The filter is shared, not reimplemented.** Both backends take the same
Pinecone-syntax filter dict, built by `build_filter()`. The local backend
evaluates it with `matches()`. If the two diverged, a scorecard difference
between backends would be a filter bug masquerading as a retrieval finding.

**2. The local backend returns top-k, like the remote one does.** Pinecone can
only give you the k nearest; a local backend that scores every clause would fuse
against a different candidate set, so the swap would change fusion semantics and
the diff would not be attributable to the vector store.

What deliberately does NOT live in the vector store: clause text. Search returns
ids and scores; the caller hydrates from the corpus. `get_clause_by_id` — the
Verifier's ground truth — must not be answerable by an index that could be stale,
partially upserted, or silently rebuilt with a different embedder.
"""

from __future__ import annotations

import os
from datetime import date
from typing import Any, Iterable, Protocol, Sequence

import numpy as np

# Sentinel for "no end date" — an open-ended policy version is in effect
# forever. Pinecone metadata cannot hold null, and a missing key would fail a
# `$gte` comparison rather than pass it.
OPEN_ENDED = 99991231


# ---------------------------------------------------------------------------
# metadata encoding
# ---------------------------------------------------------------------------


def date_to_int(value: str | None, *, default: int) -> int:
    """ISO date -> YYYYMMDD int, so effective-date windows can be range-filtered
    server-side. Filtering client-side after top-k is not equivalent: the
    superseded version consumes a slot in the k nearest and the in-effect one
    silently drops off the end."""
    if not value:
        return default
    d = date.fromisoformat(value)
    return d.year * 10000 + d.month * 100 + d.day


def clause_metadata(clause: dict[str, Any]) -> dict[str, Any]:
    """Only what filtering needs. Clause text is deliberately excluded — see the
    module docstring."""
    meta: dict[str, Any] = {
        "payer_id": clause["payer_id"],
        "policy_id": clause["policy_id"],
        "version": clause["version"],
        "section": clause["section"],
        "source": clause["source"],
        "doc_type": clause["doc_type"],
        "effective_start_i": date_to_int(clause["effective_start"], default=0),
        "effective_end_i": date_to_int(clause.get("effective_end"), default=OPEN_ENDED),
    }
    # Pinecone rejects null metadata values; omit rather than send None.
    if clause.get("plan_id"):
        meta["plan_id"] = clause["plan_id"]
    if clause.get("codes"):
        meta["codes"] = list(clause["codes"])
    return meta


def build_filter(
    *,
    payer_id: str | None = None,
    plan_id: str | None = None,
    as_of: str | None = None,
    include_superseded: bool = False,
) -> dict[str, Any]:
    """Pinecone-syntax metadata filter. The single definition of 'which clauses
    are eligible', used by both backends."""
    clauses: list[dict[str, Any]] = []

    if payer_id:
        clauses.append({"payer_id": {"$eq": payer_id}})

    if as_of and not include_superseded:
        at = date_to_int(as_of, default=0)
        # Inclusive on both bounds, matching retrieval.in_effect().
        clauses.append({"effective_start_i": {"$lte": at}})
        clauses.append({"effective_end_i": {"$gte": at}})

    if plan_id:
        # A rider only applies to its own plan. Base policies carry no plan_id
        # and are plan-agnostic, so they pass via the first branch.
        clauses.append(
            {"$or": [{"source": {"$ne": "rider"}}, {"plan_id": {"$eq": plan_id}}]}
        )

    if not clauses:
        return {}
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


# ---------------------------------------------------------------------------
# local evaluation of that filter
# ---------------------------------------------------------------------------


def matches(meta: dict[str, Any], flt: dict[str, Any]) -> bool:
    """Evaluate the subset of Pinecone filter syntax `build_filter` emits."""
    if not flt:
        return True

    for key, cond in flt.items():
        if key == "$and":
            if not all(matches(meta, sub) for sub in cond):
                return False
        elif key == "$or":
            if not any(matches(meta, sub) for sub in cond):
                return False
        else:
            if not _matches_field(meta.get(key, _MISSING), cond):
                return False
    return True


class _Missing:
    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "<missing>"


_MISSING = _Missing()


def _matches_field(value: Any, cond: Any) -> bool:
    if not isinstance(cond, dict):
        cond = {"$eq": cond}
    for op, operand in cond.items():
        # A missing key never satisfies a positive operator, but does satisfy
        # `$ne` — same as Pinecone.
        if value is _MISSING:
            return op == "$ne"
        if op == "$eq" and value != operand:
            return False
        if op == "$ne" and value == operand:
            return False
        if op == "$lte" and not value <= operand:
            return False
        if op == "$gte" and not value >= operand:
            return False
        if op == "$lt" and not value < operand:
            return False
        if op == "$gt" and not value > operand:
            return False
        if op == "$in" and value not in operand:
            return False
    return True


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------


class VectorBackend(Protocol):
    name: str

    def build(self, clause_ids: Sequence[str], docs: Sequence[str],
              metadatas: Sequence[dict[str, Any]]) -> None: ...

    def search(self, query: str, *, k: int, flt: dict[str, Any]) -> list[tuple[str, float]]:
        """Return (clause_id, score) for the k nearest matching the filter."""
        ...


class LocalVectorBackend:
    name = "local"

    def __init__(self, embedder: Any) -> None:
        self.embedder = embedder
        self._ids: list[str] = []
        self._metas: list[dict[str, Any]] = []
        self._vectors: np.ndarray | None = None

    def build(self, clause_ids, docs, metadatas) -> None:
        self._ids = list(clause_ids)
        self._metas = list(metadatas)
        self._vectors = self.embedder.fit_transform(list(docs))

    def search(self, query: str, *, k: int, flt: dict[str, Any]) -> list[tuple[str, float]]:
        if self._vectors is None:
            raise RuntimeError("build() must be called before search()")
        eligible = [i for i, m in enumerate(self._metas) if matches(m, flt)]
        if not eligible:
            return []
        qvec = self.embedder.transform([query])[0]
        scores = self._vectors[eligible] @ qvec
        # Top-k only — see the module docstring on why this mirrors Pinecone.
        order = np.argsort(scores)[::-1][:k]
        return [(self._ids[eligible[j]], float(scores[j])) for j in order]


class PineconeVectorBackend:
    """Serverless Pinecone index with Pinecone-hosted embeddings.

    Uses hosted inference rather than the local TF-IDF/SVD embedder on purpose:
    TF-IDF is *fitted on the corpus*, so its vectors and even its dimensionality
    change whenever the corpus does. That is fine in-process and meaningless in a
    persistent remote index, where stored vectors must stay comparable to query
    vectors embedded weeks later.
    """

    name = "pinecone"

    def __init__(
        self,
        *,
        index_name: str | None = None,
        namespace: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        dimension: int = 1024,
    ) -> None:
        self.index_name = index_name or os.environ.get("CDA_PINECONE_INDEX", "policy-corpus")
        self.namespace = namespace or os.environ.get("CDA_PINECONE_NAMESPACE", "default")
        self.model = model or os.environ.get("CDA_PINECONE_MODEL", "multilingual-e5-large")
        self.dimension = dimension
        self._api_key = api_key or os.environ.get("PINECONE_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "PINECONE_API_KEY is not set. Export a key, or leave CDA_VECTOR_BACKEND "
                "unset to use the local backend."
            )
        self._pc: Any = None
        self._index: Any = None

    # -- connection -------------------------------------------------------

    def _client(self) -> Any:
        if self._pc is None:
            from pinecone import Pinecone  # imported lazily; optional dependency

            self._pc = Pinecone(api_key=self._api_key)
        return self._pc

    def connect(self, *, create: bool = False) -> Any:
        if self._index is not None:
            return self._index
        pc = self._client()
        if create and not pc.has_index(self.index_name):
            from pinecone import CloudProvider, Metric, ServerlessSpec  # noqa: PLC0415

            pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=Metric.COSINE,
                spec=ServerlessSpec(
                    cloud=CloudProvider.AWS,
                    region=os.environ.get("CDA_PINECONE_REGION", "us-east-1"),
                ),
            )
        self._index = pc.index(name=self.index_name)
        return self._index

    # -- embedding --------------------------------------------------------

    def _embed(self, texts: Sequence[str], *, input_type: str) -> list[list[float]]:
        result = self._client().inference.embed(
            model=self.model,
            inputs=list(texts),
            parameters={"input_type": input_type, "truncate": "END"},
        )
        return [list(e["values"]) for e in result]

    # -- write ------------------------------------------------------------

    def build(self, clause_ids, docs, metadatas) -> None:
        """Upsert the corpus. Called by `cda corpus index --backend pinecone`,
        not on every process start — this is a deploy step, not a hot path."""
        index = self.connect(create=True)
        ids, docs, metas = list(clause_ids), list(docs), list(metadatas)
        for start in range(0, len(ids), 96):
            chunk = slice(start, start + 96)
            values = self._embed(docs[chunk], input_type="passage")
            index.upsert(
                vectors=[
                    {"id": i, "values": v, "metadata": m}
                    for i, v, m in zip(ids[chunk], values, metas[chunk])
                ],
                namespace=self.namespace,
            )

    # -- read -------------------------------------------------------------

    def search(self, query: str, *, k: int, flt: dict[str, Any]) -> list[tuple[str, float]]:
        index = self.connect()
        qvec = self._embed([query], input_type="query")[0]
        response = index.query(
            # The caller already over-fetches (HybridIndex passes k x overfetch),
            # so multiplying again here only paid for matches that were discarded.
            top_k=k,
            vector=qvec,
            namespace=self.namespace,
            filter=flt or None,
            include_metadata=False,  # metadata is not the system of record
            include_values=False,
        )
        # QueryResponse exposes `.matches`; each is a ScoredVector, which
        # supports both attribute and item access but is NOT dict-coercible.
        raw = getattr(response, "matches", None)
        if raw is None and isinstance(response, dict):
            raw = response.get("matches", [])
        return [(m["id"], float(m["score"])) for m in (raw or [])][:k]


# ---------------------------------------------------------------------------


def build_backend(embedder: Any) -> VectorBackend:
    choice = os.environ.get("CDA_VECTOR_BACKEND", "local").lower()
    if choice == "pinecone":
        return PineconeVectorBackend()
    return LocalVectorBackend(embedder)
