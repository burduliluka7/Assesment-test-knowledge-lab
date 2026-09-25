"""Restricted-candidate Ward + entropy fragmentation + level-specific VI merges.

This module accepts only graph data, construction vectors and a previous hierarchy.
No benchmark or evaluation encoder is accessible here.
"""

import heapq
import math
import time
from collections import Counter
from itertools import combinations
import numpy as np
from .hypergraph import collapse, project


def xlog(x):
    return x * math.log(x) if x else 0.0


def semantic_links(X, k):
    sim = X @ X.T
    np.fill_diagonal(sim, -np.inf)
    n = len(X)
    links = set()
    # Stable full ordering fixes tie behavior across fresh executions.
    for i in range(n):
        for j in np.argsort(-sim[i], kind="stable")[: min(k, n - 1)]:
            links.add((min(i, int(j)), max(i, int(j))))
    return links


class Merger:
    def __init__(self, clusters, X, base_index, edges, links, previous, alpha, lam, n):
        self.members = {i: list(c) for i, c in enumerate(clusters)}
        self.next_id = len(clusters)
        self.count = {i: len(c) for i, c in self.members.items()}
        self.sums = {
            i: X[[base_index[v] for v in c]].sum(axis=0, dtype=np.float64)
            for i, c in self.members.items()
        }
        self.alpha = alpha
        self.lam = lam
        self.n = n
        self.heap = []
        self.merges = 0
        self.fallbacks = 0
        self.previous = {
            i: Counter(previous[v] for v in c if v in previous)
            for i, c in self.members.items()
        }
        self.shared = sum(sum(c.values()) for c in self.previous.values())
        self.temporal_den = 2 * math.log(self.shared) if self.shared > 1 else 1
        self.inc = {i: {} for i in self.members}
        self.coeff = {}
        total = sum(e.get("weight", 1) for e in edges)
        for ei, e in enumerate(edges):
            assert (
                e["original_arity"] > 1
            ), "Input hyperedges must have original arity > 1; collapse retains that arity."
            r = e["original_arity"]
            self.coeff[ei] = (
                e.get("weight", 1) / (total * r * math.log(r)) if total else 0
            )
            for key, m in e["multiplicity"].items():
                self.inc[int(key)][ei] = m
        self.neighbors = {i: set() for i in self.members}
        for a, b in sorted(links):
            self.neighbors[a].add(b)
            self.neighbors[b].add(a)
            self.push(a, b)
        self.initial_pairs = len(links)

    def delta(self, a, b):
        na, nb = self.count[a], self.count[b]
        diff = self.sums[a] / na - self.sums[b] / nb
        ds = na * nb / (na + nb) * float(diff @ diff) / (4 * self.n)
        ia, ib = self.inc[a], self.inc[b]
        if len(ia) > len(ib):
            ia, ib = ib, ia
        df = sum(
            self.coeff[e] * (xlog(m) + xlog(ib[e]) - xlog(m + ib[e]))
            for e, m in ia.items()
            if e in ib
        )
        dt = 0.0
        if self.lam and self.shared > 1:
            ca, cb = self.previous[a], self.previous[b]
            sa, sb = sum(ca.values()), sum(cb.values())
            dh = -(xlog(sa + sb) - xlog(sa) - xlog(sb)) / self.shared
            if len(ca) > len(cb):
                ca, cb = cb, ca
            dj = (
                -sum(
                    xlog(m + cb[k]) - xlog(m) - xlog(cb[k])
                    for k, m in ca.items()
                    if k in cb
                )
                / self.shared
            )
            dt = (2 * dj - dh) / self.temporal_den
        return self.alpha * ds + (1 - self.alpha) * df + self.lam * dt

    def push(self, a, b):
        if a > b:
            a, b = b, a
        heapq.heappush(self.heap, (self.delta(a, b), a, b))

    def run(self, budget):
        while len(self.members) > budget:
            while self.heap:
                delta, a, b = heapq.heappop(self.heap)
                if a in self.members and b in self.members:
                    break
            else:
                ids = sorted(self.members)
                cent = np.array([self.sums[i] / self.count[i] for i in ids])
                best = None
                for j, a in enumerate(ids[:-1]):
                    dd = ((cent[j + 1 :] - cent[j]) ** 2).sum(axis=1)
                    k = int(np.argmin(dd))
                    v = (float(dd[k]), a, ids[j + 1 + k])
                    if best is None or v < best:
                        best = v
                _, a, b = best
                self.fallbacks += 1
            c = self.next_id
            self.next_id += 1
            adj = (self.neighbors[a] | self.neighbors[b]) - {a, b}
            self.members[c] = self.members.pop(a) + self.members.pop(b)
            self.count[c] = self.count.pop(a) + self.count.pop(b)
            self.sums[c] = self.sums.pop(a) + self.sums.pop(b)
            self.previous[c] = self.previous.pop(a) + self.previous.pop(b)
            self.inc[c] = dict(Counter(self.inc.pop(a)) + Counter(self.inc.pop(b)))
            self.neighbors.pop(a)
            self.neighbors.pop(b)
            self.neighbors[c] = adj
            for d in sorted(adj):
                self.neighbors[d].discard(a)
                self.neighbors[d].discard(b)
                self.neighbors[d].add(c)
                self.push(c, d)
            self.merges += 1
        return sorted([sorted(c) for c in self.members.values()], key=lambda c: c[0])


def build(snap, X, cfg, variant, previous=None, knn=None):
    start = time.perf_counter()
    nodes = snap["nodes"]
    ids = [n["id"] for n in nodes]
    index = {v: i for i, v in enumerate(ids)}
    n = len(ids)
    clusters = [[v] for v in ids]
    links = knn if knn is not None else semantic_links(X, cfg["knn_k"])
    links = set(links)
    if variant != "semantic":
        for e in snap["hyperedges"]:
            links.update(
                (min(index[a], index[b]), max(index[a], index[b]))
                for a, b in combinations(e["members"], 2)
            )
    edges = (
        project(snap["hyperedges"])[0] if variant == "pairwise" else snap["hyperedges"]
    )
    if variant == "semantic":
        edges = []
    levels = {3: clusters}
    stage_logs = []
    coarse = {}
    for level, budget in [
        (2, cfg["budgets"][2]),
        (1, cfg["budgets"][1]),
        (0, cfg["budgets"][0]),
    ]:
        membership = {v: str(i) for i, c in enumerate(clusters) for v in c}
        # Actual collapsed edge distributions feed the next stage's objective.
        working = collapse(edges, membership, snap["snapshot"])
        prev = (
            {v: i for i, c in enumerate(previous["partitions"][str(level)]) for v in c}
            if previous
            else {}
        )
        a = 1.0 if variant == "semantic" else cfg["alpha"]
        lam = cfg["lambda"] if variant == "temporal" and previous else 0.0
        engine = Merger(clusters, X, index, working, links, prev, a, lam, n)
        clusters = engine.run(min(budget, n))
        levels[level] = clusters
        stage_logs.append(
            dict(
                level=level,
                merges=engine.merges,
                fallbacks=engine.fallbacks,
                candidate_pairs=engine.initial_pairs,
                lambda_used=lam,
            )
        )
        # Remap the updated union-neighbor graph exactly; no fresh proposal search.
        active = sorted(engine.members, key=lambda i: min(engine.members[i]))
        remap = {old: new for new, old in enumerate(active)}
        links = {
            (min(remap[a], remap[b]), max(remap[a], remap[b]))
            for a in active
            for b in engine.neighbors[a]
            if a != b
        }
        coarse[str(level)] = collapse(
            snap["hyperedges"],
            {
                v: f"{snap['snapshot']}_L{level}_{i}"
                for i, c in enumerate(clusters)
                for v in c
            },
            snap["snapshot"],
        )
    result = dict(
        snapshot=snap["snapshot"],
        variant=variant,
        partitions={str(l): levels[l] for l in range(4)},
        coarsened_hyperedges=coarse,
        stages=stage_logs,
        seconds=time.perf_counter() - start,
    )
    validate(result, set(ids), cfg["budgets"])
    return result


def validate(h, ids, budgets):
    for l in range(4):
        cs = h["partitions"][str(l)]
        flat = [v for c in cs for v in c]
        assert (
            len(flat) == len(ids) and set(flat) == ids and len(set(flat)) == len(flat)
        )
        if l < 3:
            assert len(cs) <= budgets[l]
            parent = {v: i for i, c in enumerate(cs) for v in c}
            for child in h["partitions"][str(l + 1)]:
                assert len({parent[v] for v in child}) == 1
            for i, c in enumerate(cs):
                assert set(c) == {
                    v
                    for child in h["partitions"][str(l + 1)]
                    if parent[child[0]] == i
                    for v in child
                }
