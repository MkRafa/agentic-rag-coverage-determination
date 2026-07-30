"""Vector-backend filter semantics.

The whole point of putting retrieval behind an interface is that swapping the
vector store must not change *which clauses are eligible*. These tests pin that
down without needing a Pinecone key: the filter is built once, in one place, and
the local evaluator is checked against the same `in_effect()` predicate the rest
of the system trusts.

If these pass and a Pinecone run still differs, the difference is ranking — which
is a finding. If these failed, a difference would just be a filter bug wearing a
finding's clothes.
"""

from __future__ import annotations

import os
import pathlib

import pytest

from policy_corpus.retrieval import in_effect
from policy_corpus.store import get_store
from policy_corpus.vectorstore import (
    OPEN_ENDED,
    LocalVectorBackend,
    build_filter,
    clause_metadata,
    date_to_int,
    matches,
)


# ---------------------------------------------------------------------------
# date encoding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "iso,expected",
    [("2024-03-15", 20240315), ("2023-01-01", 20230101), ("2024-12-31", 20241231)],
)
def test_date_to_int(iso, expected):
    assert date_to_int(iso, default=0) == expected


def test_open_ended_end_date_becomes_a_far_future_sentinel():
    """Pinecone metadata cannot hold null, and a missing key fails `$gte`
    instead of passing it — an open-ended version would drop out of every
    date-filtered query."""
    assert date_to_int(None, default=OPEN_ENDED) == OPEN_ENDED
    assert OPEN_ENDED > 20991231


def test_int_encoding_preserves_date_ordering():
    """Range filters are only correct if the encoding is monotonic."""
    dates = ["2023-01-01", "2023-12-31", "2024-01-01", "2024-06-30", "2024-07-01"]
    encoded = [date_to_int(d, default=0) for d in dates]
    assert encoded == sorted(encoded)


# ---------------------------------------------------------------------------
# the equivalence that matters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "as_of",
    ["2022-01-01", "2023-06-15", "2024-06-30", "2024-07-01", "2025-06-01", "2099-01-01"],
)
def test_metadata_filter_agrees_with_in_effect_for_every_clause(as_of):
    """Server-side range filter must select exactly what the in-process
    effective-date predicate selects — on every clause, on both sides of every
    version boundary."""
    store = get_store()
    flt = build_filter(as_of=as_of)
    for clause in store.clauses.values():
        assert matches(clause_metadata(clause), flt) is in_effect(clause, as_of), (
            clause["clause_id"],
            as_of,
        )


def test_include_superseded_disables_the_date_filter():
    store = get_store()
    flt = build_filter(as_of="2024-03-15", include_superseded=True)
    v2 = store.clauses["MHP-MP-0142.v2.C2"]      # not in effect on that date
    assert not in_effect(v2, "2024-03-15")
    assert matches(clause_metadata(v2), flt)


# ---------------------------------------------------------------------------
# scoping
# ---------------------------------------------------------------------------


def test_payer_filter_excludes_other_payers():
    store = get_store()
    flt = build_filter(payer_id="MHP")
    assert matches(clause_metadata(store.clauses["MHP-MP-0142.v1.C2"]), flt)
    assert not matches(clause_metadata(store.clauses["CSM-MP-0331.v1.C2"]), flt)


def test_rider_is_scoped_to_its_own_plan():
    store = get_store()
    gold_rider = store.clauses["MHP-RID-STEP-WAIVE.C1"]   # attached to MHP-PPO-GOLD
    base_policy = store.clauses["MHP-MP-0210.v3.C2"]      # no plan_id

    on_gold = build_filter(payer_id="MHP", plan_id="MHP-PPO-GOLD")
    assert matches(clause_metadata(gold_rider), on_gold)
    assert matches(clause_metadata(base_policy), on_gold)

    on_base = build_filter(payer_id="MHP", plan_id="MHP-HMO-BASE")
    assert not matches(clause_metadata(gold_rider), on_base), (
        "a rider leaking onto the wrong plan inverts the T2 determination"
    )
    assert matches(clause_metadata(base_policy), on_base)


def test_riders_are_unscoped_when_no_plan_is_supplied():
    store = get_store()
    flt = build_filter(payer_id="MHP")
    assert matches(clause_metadata(store.clauses["MHP-RID-STEP-WAIVE.C1"]), flt)


def test_clause_text_is_never_written_to_vector_metadata():
    """Ground truth stays in the corpus. If clause text lived in the index, a
    stale upsert could make the Verifier confirm a bad citation against the same
    wrong copy the Synthesizer used."""
    store = get_store()
    for clause in store.clauses.values():
        meta = clause_metadata(clause)
        assert "text" not in meta
        assert clause["text"] not in str(meta)


def test_metadata_has_no_null_values():
    """Pinecone rejects null metadata values outright."""
    store = get_store()
    for clause in store.clauses.values():
        for key, value in clause_metadata(clause).items():
            assert value is not None, (clause["clause_id"], key)


# ---------------------------------------------------------------------------
# filter-evaluator edge cases
# ---------------------------------------------------------------------------


def test_missing_key_fails_positive_operators_but_satisfies_ne():
    assert not matches({}, {"plan_id": {"$eq": "X"}})
    assert matches({}, {"source": {"$ne": "rider"}})


def test_empty_filter_matches_everything():
    assert matches({"anything": 1}, {})
    assert build_filter() == {}


# ---------------------------------------------------------------------------
# local backend honours the filter
# ---------------------------------------------------------------------------


def test_local_backend_returns_only_filtered_clauses():
    from policy_corpus.retrieval import HybridIndex, build_embedder

    store = get_store()
    clauses = list(store.clauses.values())
    backend = LocalVectorBackend(build_embedder())
    backend.build(
        [c["clause_id"] for c in clauses],
        [HybridIndex._doc_text(c) for c in clauses],
        [clause_metadata(c) for c in clauses],
    )

    flt = build_filter(payer_id="MHP", as_of="2024-03-15")
    results = backend.search("coverage criteria", k=50, flt=flt)
    assert results
    for cid, _score in results:
        clause = store.clauses[cid]
        assert clause["payer_id"] == "MHP"
        assert in_effect(clause, "2024-03-15")


def test_local_backend_respects_k():
    from policy_corpus.retrieval import HybridIndex, build_embedder

    store = get_store()
    clauses = list(store.clauses.values())
    backend = LocalVectorBackend(build_embedder())
    backend.build(
        [c["clause_id"] for c in clauses],
        [HybridIndex._doc_text(c) for c in clauses],
        [clause_metadata(c) for c in clauses],
    )
    assert len(backend.search("coverage", k=3, flt={})) == 3


# ---------------------------------------------------------------------------
# subprocess environment
# ---------------------------------------------------------------------------


def test_every_config_env_var_is_forwarded_to_the_mcp_subprocess():
    """The MCP server runs over stdio with a scrubbed environment. A knob the
    server reads but the client does not forward is worse than no knob: the
    sweep runs, the scorecard does not move, and you conclude the change had no
    effect. This has already happened once, with CDA_RERANKER."""
    import re
    from pathlib import Path

    from src.mcp_client import _FORWARDED_ENV

    read_by_server: set[str] = set()
    for path in Path("policy_corpus").rglob("*.py"):
        read_by_server |= set(
            re.findall(r"environ\.get\(\s*[\"']((?:CDA|PINECONE)_[A-Z_]+)", path.read_text())
        )

    assert read_by_server, "expected the server package to read some CDA_/PINECONE_ vars"
    missing = read_by_server - set(_FORWARDED_ENV)
    assert not missing, f"read by the MCP server but never forwarded to it: {sorted(missing)}"


# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------


def test_dotenv_is_gitignored():
    """A key committed to the repo is the worst possible outcome here."""
    from pathlib import Path

    ignored = Path(".gitignore").read_text().splitlines()
    assert ".env" in [line.strip() for line in ignored]


def test_dotenv_parses_and_never_overrides_the_real_environment(tmp_path, monkeypatch):
    from src.env import load_dotenv

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# a comment",
                "",
                "PINECONE_API_KEY=pcsk_from_file",
                'CDA_PINECONE_INDEX="quoted-value"',
                "export CDA_PINECONE_NAMESPACE=exported",
                "CDA_ALREADY_SET=from_file",
                "malformed line without equals",
            ]
        )
    )
    monkeypatch.setenv("CDA_ALREADY_SET", "from_shell")
    for key in ("PINECONE_API_KEY", "CDA_PINECONE_INDEX", "CDA_PINECONE_NAMESPACE"):
        monkeypatch.delenv(key, raising=False)

    loaded = load_dotenv(env_file)

    assert os.environ["PINECONE_API_KEY"] == "pcsk_from_file"
    assert os.environ["CDA_PINECONE_INDEX"] == "quoted-value"       # quotes stripped
    assert os.environ["CDA_PINECONE_NAMESPACE"] == "exported"       # `export ` prefix handled
    assert os.environ["CDA_ALREADY_SET"] == "from_shell"            # shell wins
    assert "CDA_ALREADY_SET" not in loaded
    assert "malformed line without equals" not in loaded


def test_dotenv_returns_names_not_values():
    """The return value gets logged; it must never carry a secret."""
    from src.env import load_dotenv

    assert load_dotenv(pathlib.Path("/nonexistent/.env")) == []


def test_env_example_documents_every_pinecone_var():
    """A knob with no entry in .env.example is a knob nobody will find."""
    import pathlib as _pl
    import re

    example = _pl.Path(".env.example").read_text()
    from src.mcp_client import _FORWARDED_ENV

    for name in _FORWARDED_ENV:
        if name.startswith(("CDA_", "PINECONE_")):
            assert name in example, f"{name} is forwarded but undocumented in .env.example"
