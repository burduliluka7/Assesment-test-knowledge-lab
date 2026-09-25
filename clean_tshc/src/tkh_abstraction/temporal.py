from collections import Counter
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score


def vi(a, b):
    if len(a) < 2:
        return 0.0
    n = len(a)
    ca = Counter(a)
    cb = Counter(b)
    joint = Counter(zip(a, b))
    entropy = lambda c: -sum(m / n * math.log(m / n) for m in c.values())
    return max(0.0, 2 * entropy(joint) - entropy(ca) - entropy(cb))


def track(h, previous, threshold=0.1):
    records = []
    events = []
    year = h["snapshot"]
    oldrecords = previous["supernodes"] if previous else []
    overlaps = {}
    for level in range(4):
        current = [set(c) for c in h["partitions"][str(level)]]
        old = [c for c in oldrecords if c["level"] == level]
        oldsets = [set(c["member_ids"]) for c in old]
        mapping = {}
        rows = []
        if level == 3:
            mapping = {j: "leaf_" + next(iter(c)) for j, c in enumerate(current)}
        elif old:
            shared = set().union(*current) & set().union(*oldsets)
            inter = np.array([[len(a & b) for b in current] for a in oldsets])
            union = np.array([[len(a | b) for b in current] for a in oldsets])
            jac = inter / np.maximum(union, 1)
            aa, bb = linear_sum_assignment(-jac)
            mapping = {
                int(j): old[int(i)]["persistent_id"]
                for i, j in zip(aa, bb)
                if inter[i, j] > 0
            }
            for i, a in enumerate(oldsets):
                for j, b in enumerate(current):
                    count = int(inter[i, j])
                    if count:
                        rows.append(
                            dict(
                                old_id=old[i]["persistent_id"],
                                new_index=j,
                                intersection=count,
                                jaccard=float(jac[i, j]),
                                old_fraction=count / max(1, len(a & shared)),
                                new_fraction=count / max(1, len(b & shared)),
                            )
                        )
        for j, c in enumerate(current):
            pid = mapping.get(j, f"L{level}_{year}_{j:04d}")
            records.append(
                dict(
                    id=f"{year}_L{level}_{j}",
                    snapshot=year,
                    level=level,
                    parent_id=None,
                    child_ids=[],
                    member_ids=sorted(c),
                    size=len(c),
                    label=None,
                    gloss=None,
                    persistent_id=pid,
                )
            )
        if level == 3:
            continue
        new = [c for c in records if c["level"] == level]

        def emit(kind, os, ns, rr):
            events.append(
                dict(
                    from_snapshot=previous["snapshot"] if previous else None,
                    to_snapshot=year,
                    level=level,
                    event_type=kind,
                    old_ids=os,
                    new_ids=ns,
                    overlaps=rr,
                )
            )

        for r in rows:
            r["new_id"] = new[r.pop("new_index")]["persistent_id"]
        overlaps[str(level)] = rows
        for c in new:
            rr = [r for r in rows if r["new_id"] == c["persistent_id"]]
            if not rr:
                emit("birth", [], [c["persistent_id"]], [])
            else:
                strong = [
                    r
                    for r in rr
                    if r["intersection"] >= 2 and r["new_fraction"] >= threshold
                ]
                if len(strong) > 1:
                    emit(
                        "merge",
                        [r["old_id"] for r in strong],
                        [c["persistent_id"]],
                        strong,
                    )
                prior = next(
                    (o for o in old if o["persistent_id"] == c["persistent_id"]), None
                )
                if prior:
                    emit(
                        (
                            "growth"
                            if len(set(c["member_ids"]) - set(prior["member_ids"]))
                            else "continuation"
                        ),
                        [prior["persistent_id"]],
                        [c["persistent_id"]],
                        rr,
                    )
        for c in old:
            rr = [r for r in rows if r["old_id"] == c["persistent_id"]]
            if not rr:
                emit("death", [c["persistent_id"]], [], [])
            strong = [
                r
                for r in rr
                if r["intersection"] >= 2 and r["old_fraction"] >= threshold
            ]
            if len(strong) > 1:
                emit(
                    "split", [c["persistent_id"]], [r["new_id"] for r in strong], strong
                )
    for level in range(1, 4):
        parent = {
            v: c for c in records if c["level"] == level - 1 for v in c["member_ids"]
        }
        for c in [r for r in records if r["level"] == level]:
            p = parent[c["member_ids"][0]]
            c["parent_id"] = p["id"]
            p["child_ids"].append(c["id"])
    h["supernodes"] = records
    h["temporal_overlap"] = overlaps
    for l in range(4):
        ps = [c["persistent_id"] for c in records if c["level"] == l]
        assert len(ps) == len(set(ps))
    return events


def transition(old, new):
    result = {}
    for level in range(3):
        aa = {
            v: c["persistent_id"]
            for c in old["supernodes"]
            if c["level"] == level
            for v in c["member_ids"]
        }
        bb = {
            v: c["persistent_id"]
            for c in new["supernodes"]
            if c["level"] == level
            for v in c["member_ids"]
        }
        ids = sorted(aa.keys() & bb.keys())
        a = [aa[v] for v in ids]
        b = [bb[v] for v in ids]
        result[str(level)] = dict(
            shared_nodes=len(ids),
            ari=float(adjusted_rand_score(a, b)),
            vi=vi(a, b),
            normalized_vi=vi(a, b) / (2 * math.log(len(ids))) if len(ids) > 1 else 0.0,
            lineage_retention=float(np.mean(np.array(a) == np.array(b))),
        )
    return result
