"""Native multiplicity-preserving collapse and one normalized pairwise baseline."""

from collections import Counter, defaultdict
from itertools import combinations
import math


def collapse(edges, membership, year):
    result = []
    for e in edges:
        counts = Counter()
        for v, m in e.get("multiplicity", {v: 1 for v in e.get("members", [])}).items():
            counts[membership[v]] += m
        arity = e.get("original_arity", len(e.get("members", [])))
        assert sum(counts.values()) == arity
        result.append(
            dict(
                id=e["id"],
                relation_type=e["relation_type"],
                provenance=e["provenance"],
                original_arity=arity,
                supernodes=sorted(counts),
                multiplicity=dict(counts),
                internal=len(counts) == 1,
                snapshot=year,
                weight=e.get("weight", 1.0),
            )
        )
    return result


def project(edges):
    weights = defaultdict(float)
    contributors = defaultdict(list)
    instances = 0
    for e in edges:
        pairs = list(combinations(sorted(e["members"]), 2))
        instances += len(pairs)
        for p in pairs:
            weights[p] += 1 / len(pairs)
            contributors[p].append(e["id"])
    projected = [
        dict(
            id=f"pair_{i}",
            members=list(p),
            relation_type="projected",
            provenance={},
            weight=w,
        )
        for i, (p, w) in enumerate(sorted(weights.items()))
    ]
    collisions = [
        dict(pair=list(p), edges=es) for p, es in contributors.items() if len(es) > 1
    ]
    return projected, dict(
        original_hyperedges=len(edges),
        pair_instances=instances,
        unique_pairs=len(weights),
        repeated_pair_instances=instances - len(weights),
        collision_pairs=len(collisions),
        collision_fraction=len(collisions) / len(weights) if weights else 0,
        collision_examples=collisions[:10],
        lost="Aggregated pair weights do not identify source hyperedges, original arities, or endpoint grouping.",
    )


def fragmentation(edges, labels, pairwise=False):
    total = sum(e.get("weight", 1) for e in edges)
    if not total:
        return 0.0
    value = 0.0
    for e in edges:
        c = Counter(labels[v] for v in e["members"])
        r = len(e["members"])
        f = (
            (len(c) > 1)
            if pairwise
            else -sum(m / r * math.log(m / r) for m in c.values()) / math.log(r)
        )
        value += e.get("weight", 1) * f
    return value / total
