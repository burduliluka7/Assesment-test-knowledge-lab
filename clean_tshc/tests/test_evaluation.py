from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from test_core import fixture
from tkh_abstraction.hierarchy import build, Merger
from tkh_abstraction.hypergraph import collapse
from tkh_abstraction.temporal import track, vi
from tkh_abstraction.labels import generate, split_members, evaluate_labels
from tkh_abstraction.retrieval import documents, search
from tkh_abstraction.evaluation import coherence, coherence_value
from tkh_abstraction.benchmark import evaluate_predictions, support_audit
from tkh_abstraction.benchmark import resolve_source
from tkh_abstraction.benchmark import align_claims


def config():
    return dict(
        knn_k=2,
        budgets=[2, 3, 4],
        alpha=0.5,
        seed=42,
        null_replicates=100,
        label_generation_fraction=0.7,
        label_max_words=6,
        label_evidence_max_chars=1800,
        context_cap=2,
        content_relations=["claims"],
        **{"lambda": 0.2},
    )


def test_source_resolution_tolerates_missing_doi():
    snap = dict(
        nodes=[
            dict(
                id="paper",
                type="article",
                surface_form="A title",
                provenance=dict(doi=None, articles=["p"]),
            )
        ],
        hyperedges=[],
    )
    assert resolve_source({"doi": "10.1/unknown"}, snap)["status"] == "UNMAPPED"
    assert resolve_source({"doi": "10.1/unknown", "title": "A title"}, snap)[
        "article_ids"
    ] == ["p"]


def test_reviewed_alignment_requires_identical_gold_text_and_source():
    class EncoderStub:
        def encode(self, texts):
            return np.ones((len(texts), 2))

    snap = dict(
        nodes=[
            dict(
                id="p",
                type="article",
                surface_form="Paper",
                provenance={"articles": [1]},
            ),
            dict(
                id="c",
                type="claim",
                surface_form="A model transfers to unseen systems.",
                provenance={"articles": [1]},
            ),
        ],
        hyperedges=[],
    )
    truth = {
        "Q": dict(
            type="B",
            required_claims=[dict(id="Q.C1", text="Transfer improves.", sources=["s"])],
            sources={"s": {"title": "Paper"}},
        )
    }
    ann = {
        "Q.C1": dict(
            claim_text="Transfer improves.",
            node_ids=["c"],
            source_article_ids=[1],
            rationale="Inspected paraphrase.",
        )
    }
    result = align_claims(truth, snap, EncoderStub(), ann)
    assert (
        result[0]["aligned_node_ids"] == ["c"]
        and result[0]["status"] == "REVIEWED_TEXT_AND_SOURCE"
    )
    ann["Q.C1"]["claim_text"] = "Different statement"
    with pytest.raises(AssertionError):
        align_claims(truth, snap, EncoderStub(), ann)


def test_heap_matches_exact_current_candidate_minimum():
    s, x = fixture()
    ids = [n["id"] for n in s["nodes"]]
    clusters = [[v] for v in ids]
    links = {(i, j) for i in range(8) for j in range(i + 1, 8)}
    e = Merger(
        clusters,
        x,
        {v: i for i, v in enumerate(ids)},
        collapse(s["hyperedges"], {v: v for v in ids}, 2022),
        links,
        {v: int(v) % 3 for v in ids[:-1]},
        0.5,
        0.2,
        8,
    )
    while len(e.members) > 1:
        candidates = sorted(
            (e.delta(a, b), a, b) for a in e.members for b in e.neighbors[a] if a < b
        )
        _, a, b = candidates[0]
        expected = set(e.members[a] + e.members[b])
        newid = e.next_id
        e.run(len(e.members) - 1)
        assert set(e.members[newid]) == expected


def test_disconnected_candidate_fallback_and_edge_mass():
    s, x = fixture()
    ids = [n["id"] for n in s["nodes"]]
    e = Merger(
        [[v] for v in ids], x, {v: i for i, v in enumerate(ids)}, [], set(), {}, 1, 0, 8
    )
    e.run(2)
    assert e.fallbacks == 6
    h = build(s, x, config(), "temporal")
    track(h, None)
    for l, edges in h["coarsened_hyperedges"].items():
        valid = {r["id"] for r in h["supernodes"] if r["level"] == int(l)}
        assert all(
            set(e["supernodes"]) <= valid
            and sum(e["multiplicity"].values()) == e["original_arity"]
            for e in edges
        )


def test_labels_reproducible_disjoint_and_visible(tmp_path):
    s, x = fixture()
    cfg = config()
    h = build(s, x, cfg, "temporal")
    track(h, None)
    a = generate(h, s, x, cfg, tmp_path)
    b = generate(h, s, x, cfg, tmp_path)
    assert a == b
    for p in a:
        ids = {n["id"] for n in p["generation_input"]["generation_members"]}
        assert not ids & set(p["held_out_ids"])
        assert (tmp_path / f"labels/inputs/temporal/{p['id']}.json").exists()
        assert set(p["generation_input"]["representative_ids"]) <= ids
    bad = deepcopy(s)
    bad["nodes"][0]["first_seen_year"] = 2030
    with pytest.raises(AssertionError):
        generate(h, bad, x, cfg, tmp_path)
    bad = deepcopy(s)
    bad["hyperedges"][0]["provenance"]["article_year"] = 2030
    with pytest.raises(AssertionError):
        generate(h, bad, x, cfg, tmp_path)


def test_nli_proxy_preserves_unevaluable_and_neutral():
    class Stub:
        def score(self, pairs):
            return [
                dict(
                    label="neutral",
                    probabilities=dict(neutral=0.8, entailment=0.1, contradiction=0.1),
                )
                for _ in pairs
            ]

    rows = evaluate_labels(
        [
            dict(
                evaluable=True,
                held_out_text="phonon prediction",
                gloss="A method.",
                label="phonon",
            ),
            dict(evaluable=False, held_out_text="", gloss="A method.", label="method"),
        ],
        Stub(),
    )
    assert rows[0]["overclaim_proxy"] == 1 and rows[0]["lexical_support"] == 1
    assert rows[1]["nli"] is None and rows[1]["overclaim_proxy"] is None


def test_all_leaf_beam_equals_flat_and_work_trace():
    s, x = fixture()
    cfg = config()
    h = build(s, x, cfg, "temporal")
    track(h, None)
    docs = documents(s, cfg)
    for d in docs:
        assert len(d["context_ids"]) <= 2 and d["node_id"] not in d["context_ids"]
    flat = search(h, docs, x, x[0])
    broad = search(h, docs, x, x[0], 100)
    narrow = search(h, docs, x, x[0], 1)
    assert broad["ranked"] == flat["ranked"]
    assert broad["comparisons"] == flat["comparisons"] + sum(
        len(h["partitions"][str(l)]) for l in range(3)
    )
    fs = {r["node_id"]: r["score"] for r in flat["ranked"]}
    assert all(r["score"] == fs[r["node_id"]] for r in narrow["ranked"])
    assert [r["comparison"] for r in narrow["trace"]] == list(
        range(1, narrow["comparisons"] + 1)
    )


def test_coherence_null_reproducible_and_all_clusters():
    s, x = fixture()
    cfg = config()
    h = build(s, x, cfg, "semantic")
    a = coherence(h, s, x, cfg)
    b = coherence(h, s, x, cfg)
    assert a == b and a["0"]["null"]["n"] == 100 and a["0"]["null_macro"]["n"] == 100
    assert coherence_value(np.arange(8), x)["size_weighted"] == pytest.approx(1)
    assert a["0"]["p_upper"] >= 1 / 101


def test_sparse_coherence_matches_original_indexed_sum_exactly():
    rng = np.random.default_rng(42)
    y = rng.normal(size=(300, 31)).astype("float32")
    y /= np.linalg.norm(y, axis=1)[:, None]
    labels = rng.integers(0, 12, len(y))
    sums = np.zeros((12, 31))
    np.add.at(sums, labels, y)
    sizes = np.bincount(labels, minlength=12)
    norm = np.linalg.norm(sums, axis=1)
    result = coherence_value(labels, y)
    assert result["macro"] == float((norm / sizes).mean()) and result[
        "size_weighted"
    ] == float(norm.sum() / sizes.sum())


def test_partial_not_promoted_to_complete_composite():
    docs = [dict(node_id="a", context_ids=["b"], article_ids=["paper"])]
    targets = [
        dict(
            question_id="Q",
            target="AB",
            status="COMPOSITE",
            components=[["a"], ["b"]],
            node_ids=["a", "b"],
        ),
        dict(
            question_id="Q",
            target="Partial",
            status="PARTIAL",
            components=[["b"]],
            node_ids=["b"],
        ),
    ]
    p = dict(
        question_id="Q",
        question_type="A",
        system="flat",
        beam=None,
        comparisons=1,
        leaf_comparisons=1,
        cluster_comparisons=0,
        ranked=[dict(node_id="a", score=0.5)],
        trace=[dict(kind="leaf", id="a", comparison=1)],
    )
    r = evaluate_predictions([p], docs, dict(records=targets), [], [10])[0]["at_k"][
        "10"
    ]
    assert (
        r["found_targets"] == ["AB"]
        and r["raw_recall"] == 0.5
        and r["supported_recall"] == 0.5
    )
    assert (
        r["partial_evidence_targets"] == ["Partial"]
        and r["strict_claim_recall"] is None
    )


@pytest.mark.parametrize("status", ["EXACT", "ALIAS"])
def test_identity_recovery_positive_control_and_context_exclusion(status):
    docs = [
        dict(node_id="claim", context_ids=["method"], article_ids=[]),
        dict(node_id="method", context_ids=[], article_ids=[]),
    ]
    target = dict(question_id="Q", target="Model", status=status,
                  components=[["method"]], node_ids=["method"])
    prediction = dict(question_id="Q", question_type="A", system="control",
                      beam=None, comparisons=2, leaf_comparisons=2,
                      cluster_comparisons=0,
                      ranked=[dict(node_id="claim", score=1.0),
                              dict(node_id="method", score=0.5)],
                      trace=[dict(kind="leaf", id="claim", comparison=1),
                             dict(kind="leaf", id="method", comparison=2)])
    row = evaluate_predictions([prediction], docs, dict(records=[target]), [], [1, 2])[0]
    assert row["at_k"]["1"]["raw_recall"] == 0
    assert row["at_k"]["2"]["raw_recall"] == 1
    assert row["at_k"]["2"]["direct_identity_targets"] == ["Model"]
    assert row["comparisons_to_first_direct_target"] == 2


def test_benchmark_mutation_does_not_change_predictions():
    s, x = fixture()
    cfg = config()
    h = build(s, x, cfg, "temporal")
    track(h, None)
    docs = documents(s, cfg)
    before = search(h, docs, x, x[0], 1)
    support_audit(s, {"Q": {"expected_methods": ["Model 1"]}}, {})
    support_audit(s, {"Other": {"expected_methods": ["Nonexistent target"]}}, {})
    assert search(h, docs, x, x[0], 1) == before
    root = Path(__file__).resolve().parents[1] / "src/tkh_abstraction"
    for name in ["labels.py", "retrieval.py"]:
        source = (root / name).read_text()
        assert (
            "ground_truth" not in source
            and "target_annotations" not in source
            and ".benchmark" not in source
        )
