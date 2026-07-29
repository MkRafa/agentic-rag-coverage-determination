"""Hybrid retrieval: lexical (BM25) + dense (TF-IDF→SVD) with pluggable fusion.

Two things here are deliberately swappable, because swapping them in front of a
customer without touching agent logic is the point of putting retrieval behind
MCP:

  EMBEDDER   default is TF-IDF + TruncatedSVD — no model download, no torch,
             deterministic, works offline. `SentenceTransformerEmbedder` is the
             drop-in upgrade when the extra dependency is acceptable.

  RERANKER   `rrf` (reciprocal rank fusion), `weighted` (score blend), or
             `date_aware` (RRF plus a boost for clauses in effect on the as-of
             date and for exact code matches).

Both are selected by environment variable so an eval run can sweep them:

    CDA_EMBEDDER=tfidf|sentence-transformers
    CDA_RERANKER=rrf|weighted|date_aware
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol, Sequence

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


# ---------------------------------------------------------------------------
# embedders
# ---------------------------------------------------------------------------


class Embedder(Protocol):
    name: str

    def fit_transform(self, docs: Sequence[str]) -> np.ndarray: ...

    def transform(self, queries: Sequence[str]) -> np.ndarray: ...


class TfidfSvdEmbedder:
    """Deterministic dense vectors with no external model download."""

    name = "tfidf-svd"

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim
        self._vec = TfidfVectorizer(
            lowercase=True, sublinear_tf=True, ngram_range=(1, 2), min_df=1
        )
        self._svd: TruncatedSVD | None = None

    def fit_transform(self, docs: Sequence[str]) -> np.ndarray:
        X = self._vec.fit_transform(docs)
        # SVD components must stay below the rank of the term-document matrix.
        n = min(self.dim, max(2, min(X.shape) - 1))
        self._svd = TruncatedSVD(n_components=n, random_state=0)
        return _l2(self._svd.fit_transform(X))

    def transform(self, queries: Sequence[str]) -> np.ndarray:
        assert self._svd is not None, "fit_transform must be called first"
        return _l2(self._svd.transform(self._vec.transform(queries)))


class SentenceTransformerEmbedder:
    """Optional upgrade path. Requires `pip install sentence-transformers`."""

    name = "sentence-transformers"

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self._m = SentenceTransformer(model)

    def fit_transform(self, docs: Sequence[str]) -> np.ndarray:
        return _l2(np.asarray(self._m.encode(list(docs))))

    def transform(self, queries: Sequence[str]) -> np.ndarray:
        return _l2(np.asarray(self._m.encode(list(queries))))


def _l2(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, 1e-9, None)


def build_embedder() -> Embedder:
    choice = os.environ.get("CDA_EMBEDDER", "tfidf").lower()
    if choice in ("st", "sentence-transformers"):
        return SentenceTransformerEmbedder()
    return TfidfSvdEmbedder()


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------


@dataclass
class Hit:
    clause: dict[str, Any]
    score: float
    lexical_rank: int | None = None
    dense_rank: int | None = None
    signals: dict[str, float] = field(default_factory=dict)


class HybridIndex:
    def __init__(self, clauses: list[dict[str, Any]], embedder: Embedder | None = None) -> None:
        self.clauses = clauses
        self.embedder = embedder or build_embedder()
        self._docs = [self._doc_text(c) for c in clauses]
        self._bm25 = BM25Okapi([tokenize(d) for d in self._docs])
        self._vectors = self.embedder.fit_transform(self._docs)
        self.reranker = os.environ.get("CDA_RERANKER", "date_aware").lower()

    @staticmethod
    def _doc_text(c: dict[str, Any]) -> str:
        # Title and section are part of the indexed text — "Coverage Criteria"
        # vs "Exclusions" is a strong retrieval signal in policy documents.
        codes = " ".join(c.get("codes") or [])
        return f"{c['policy_title']} | {c['section']} | {c['text']} | {codes}"

    # -- filtering ---------------------------------------------------------

    def _passes(self, c: dict[str, Any], payer_id, plan_id, as_of, include_superseded) -> bool:
        if payer_id and c["payer_id"] != payer_id:
            return False
        # A rider only applies to its own plan. Base policies apply plan-wide.
        if c["source"] == "rider" and plan_id and c["plan_id"] != plan_id:
            return False
        if as_of and not include_superseded and not in_effect(c, as_of):
            return False
        return True

    # -- search ------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        k: int = 8,
        payer_id: str | None = None,
        plan_id: str | None = None,
        as_of: str | None = None,
        include_superseded: bool = False,
        codes: Sequence[str] = (),
    ) -> list[Hit]:
        candidates = [
            i
            for i, c in enumerate(self.clauses)
            if self._passes(c, payer_id, plan_id, as_of, include_superseded)
        ]
        if not candidates:
            return []

        lex_scores = self._bm25.get_scores(tokenize(query))
        qvec = self.embedder.transform([query])[0]
        dense_scores = self._vectors @ qvec

        lex_rank = _ranks(candidates, lex_scores)
        dense_rank = _ranks(candidates, dense_scores)

        hits: list[Hit] = []
        for i in candidates:
            hits.append(
                Hit(
                    clause=self.clauses[i],
                    score=0.0,
                    lexical_rank=lex_rank[i],
                    dense_rank=dense_rank[i],
                    signals={
                        "lexical": float(lex_scores[i]),
                        "dense": float(dense_scores[i]),
                    },
                )
            )

        rerank(hits, strategy=self.reranker, as_of=as_of, codes=codes)
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]


def _ranks(candidates: list[int], scores: np.ndarray) -> dict[int, int]:
    order = sorted(candidates, key=lambda i: scores[i], reverse=True)
    return {i: r + 1 for r, i in enumerate(order)}


# ---------------------------------------------------------------------------
# rerankers — the swap-in-place demo
# ---------------------------------------------------------------------------


def rerank(hits: list[Hit], *, strategy: str, as_of: str | None, codes: Sequence[str]) -> None:
    codes = {c.upper() for c in codes}
    for h in hits:
        base = _rrf(h) if strategy != "weighted" else _weighted(h)
        if strategy == "date_aware":
            base += _date_and_code_boost(h, as_of, codes)
        h.score = base
        h.signals["reranker"] = {"rrf": 0.0, "weighted": 1.0, "date_aware": 2.0}.get(strategy, 0.0)


def _rrf(h: Hit, k: int = 60) -> float:
    return 1.0 / (k + (h.lexical_rank or 999)) + 1.0 / (k + (h.dense_rank or 999))


def _weighted(h: Hit) -> float:
    # BM25 is unbounded; squash it so the blend is not dominated by long docs.
    lex = h.signals["lexical"] / (1.0 + h.signals["lexical"])
    return 0.5 * lex + 0.5 * h.signals["dense"]


def _date_and_code_boost(h: Hit, as_of: str | None, codes: set[str]) -> float:
    boost = 0.0
    c = h.clause
    if as_of and in_effect(c, as_of):
        boost += 0.010
    if codes and codes.intersection({x.upper() for x in (c.get("codes") or [])}):
        boost += 0.008
    # Riders override base policy, so surface them above the policy they modify.
    if c["source"] == "rider":
        boost += 0.006
    if c["section"] == "Coverage Criteria":
        boost += 0.004
    h.signals.update(
        {"boost_in_effect": float(bool(as_of and in_effect(c, as_of))), "boost": boost}
    )
    return boost


# ---------------------------------------------------------------------------
# effective-date arithmetic — the single source of truth for staleness
# ---------------------------------------------------------------------------


def _d(value: str) -> date:
    return date.fromisoformat(value)


def in_effect(clause: dict[str, Any], as_of: str) -> bool:
    """True when `clause` was in force on `as_of` (inclusive of both bounds)."""
    at = _d(as_of)
    if at < _d(clause["effective_start"]):
        return False
    end = clause.get("effective_end")
    return end is None or at <= _d(end)
