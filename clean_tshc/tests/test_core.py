from collections import Counter
from pathlib import Path
import math
import numpy as np
import pytest
from tkh_abstraction.hypergraph import collapse, fragmentation, project
from tkh_abstraction.hierarchy import Merger, build, validate
from tkh_abstraction.temporal import vi, track
from tkh_abstraction.snapshots import snapshot


def fixture():
    nodes = [
        dict(
            id=str(i),
            type="method",
            surface_form=f"Model {i}",
            first_seen_year=2020,
            provenance={},
        )
        for i in range(8)
    ]
    edges = [
        dict(
            id="e",
            members=["0", "1", "2", "3", "4"],
            relation_type="claims",
            year=2020,
            provenance={"article_year": 2022},
        ),
        dict(
            id="f",
            members=["4", "5", "6"],
            relation_type="claims",
            year=2020,
            provenance={},
        ),
    ]
    x = np.random.default_rng(2).normal(size=(8, 5))
    x /= np.linalg.norm(x, axis=1)[:, None]
    return dict(snapshot=2022, nodes=nodes, hyperedges=edges), x


@pytest.mark.parametrize(
    "mapping,counts",
    [
        ({"0": "a", "1": "a", "2": "a", "3": "a", "4": "a"}, [5]),
        ({"0": "a", "1": "a", "2": "a", "3": "b", "4": "b"}, [2, 3]),
        ({"0": "a", "1": "a", "2": "a", "3": "b", "4": "c"}, [1, 1, 3]),
    ],
)
def test_collapse(mapping, counts):
    e = fixture()[0]["hyperedges"][:1]
    c = collapse(e, mapping, 2022)[0]
    assert (
        sorted(c["multiplicity"].values()) == counts
        and sum(counts) == c["original_arity"]
    )
    assert c["id"] == e[0]["id"] and c["provenance"] == e[0]["provenance"]
    assert c["internal"] == (len(counts) == 1)
    d = collapse([c], {k: "z" for k in c["supernodes"]}, 2022)[0]
    assert d["multiplicity"] == {"z": 5}


@pytest.mark.parametrize("pairwise", [False, True])
def test_delta_matches_full_objective(pairwise):
    snap, x = fixture()
    ids = [n["id"] for n in snap["nodes"]]
    index = {v: i for i, v in enumerate(ids)}
    clusters = [["0", "1"], ["2", "3"], ["4"], ["5"], ["6"], ["7"]]
    prev = {v: int(v) % 3 for v in ids[:-1]}
    edges = project(snap["hyperedges"])[0] if pairwise else snap["hyperedges"]
    labels = {v: str(i) for i, c in enumerate(clusters) for v in c}
    engine = Merger(
        clusters,
        x,
        index,
        collapse(edges, labels, 2022),
        set(),
        prev,
        0.5,
        0.2,
        len(ids),
    )

    def objective(cs):
        lab = {v: i for i, c in enumerate(cs) for v in c}
        s = sum(
            float(
                (
                    (x[[index[v] for v in c]] - x[[index[v] for v in c]].mean(axis=0))
                    ** 2
                ).sum()
            )
            for c in cs
        ) / (4 * len(ids))
        shared = sorted(prev)
        t = vi([lab[v] for v in shared], [prev[v] for v in shared]) / (
            2 * math.log(len(shared))
        )
        return 0.5 * s + 0.5 * fragmentation(edges, lab, pairwise) + 0.2 * t

    for a in range(len(clusters)):
        for b in range(a + 1, len(clusters)):
            merged = [c for i, c in enumerate(clusters) if i not in [a, b]] + [
                clusters[a] + clusters[b]
            ]
            assert engine.delta(a, b) == pytest.approx(
                objective(merged) - objective(clusters), abs=1e-12
            )


def test_temporal_and_laminar_reproduction():
    s, x = fixture()
    cfg = dict(knn_k=2, budgets=[2, 3, 4], alpha=0.5, **{"lambda": 0.2})
    h = build(s, x, cfg, "temporal")
    b = build(s, x, cfg, "temporal")
    assert h["partitions"] == b["partitions"]
    track(h, None)
    later = dict(s, snapshot=2024)
    g = build(later, x, cfg, "temporal", h)
    events = track(g, h)
    validate(g, set(str(i) for i in range(8)), cfg["budgets"])
    assert all(e["from_snapshot"] == 2022 for e in events)
    assert all(v["lambda_used"] == 0.2 for v in g["stages"])


def test_snapshot_defers_article():
    s, _ = fixture()
    assert len(snapshot(s, 2020)["hyperedges"]) == 1
    assert len(snapshot(s, 2022)["hyperedges"]) == 2


def test_benchmark_isolation_source():
    folder = Path(__file__).resolve().parents[1] / "src/tkh_abstraction"
    for name in [
        "hierarchy.py",
        "hypergraph.py",
        "temporal.py",
        "snapshots.py",
        "embeddings.py",
    ]:
        text = (folder / name).read_text()
        assert (
            "ground_truth" not in text
            and "questions.csv" not in text
            and "benchmark import" not in text
        )
