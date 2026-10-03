"""A bounded direct context document and identical cosine scorer for both searches."""

from collections import defaultdict
import numpy as np


def documents(snap, cfg):
    nodes = {n["id"]: n for n in snap["nodes"]}
    neighbors = defaultdict(set)
    edge_ids = defaultdict(list)
    visible_articles = {
        a
        for n in snap["nodes"]
        if n["type"] == "article"
        for a in n["provenance"].get("articles", [])
    }
    for e in snap["hyperedges"]:
        if e["relation_type"] not in cfg["content_relations"]:
            continue
        for v in e["members"]:
            neighbors[v].update(
                w for w in e["members"] if w != v and nodes[w]["type"] != "author"
            )
            edge_ids[v].append(e["id"])
    result = []
    for v, n in sorted(nodes.items()):
        if n["type"] == "author":
            continue
        choices = sorted(neighbors[v], key=lambda w: (nodes[w]["type"] != "claim", w))
        used = set()
        picked = []
        for w in choices:
            text = nodes[w]["surface_form"]
            if text in used or text == n["surface_form"]:
                continue
            picked.append(w)
            used.add(text)
            if len(picked) >= cfg["context_cap"]:
                break
        article_ids = set()
        for w in [v] + picked:
            article_ids.update(nodes[w]["provenance"].get("articles", []))
        relevant_edges = [
            e
            for e in snap["hyperedges"]
            if e["id"] in edge_ids[v] and set(e["members"]) & set(picked)
        ]
        article_ids.update(e["provenance"].get("article_id") for e in relevant_edges)
        article_ids &= visible_articles
        result.append(
            dict(
                node_id=v,
                node_type=n["type"],
                text=f"[{n['type']}] {n['surface_form']}"
                + "\n"
                + "\n".join(
                    f"[{nodes[w]['type']}] {nodes[w]['surface_form']}" for w in picked
                ),
                context_ids=picked,
                hyperedge_ids=[e["id"] for e in relevant_edges],
                article_ids=sorted(article_ids),
            )
        )
    return result


def search(h, docs, X, q, beam=None):
    index = {d["node_id"]: i for i, d in enumerate(docs)}
    trace = []
    count = 0
    if beam is None:
        eligible = list(range(len(docs)))
    else:
        parents = None
        for level in [0, 1, 2]:
            candidates = [
                c
                for c in h["supernodes"]
                if c["level"] == level
                and (parents is None or c["parent_id"] in parents)
            ]
            scored = []
            for c in candidates:
                ii = [index[v] for v in c["member_ids"] if v in index]
                if not ii:
                    continue
                center = X[ii].mean(axis=0)
                center /= max(np.linalg.norm(center), 1e-15)
                score = float(center @ q)
                count += 1
                trace.append(
                    dict(
                        kind="cluster",
                        id=c["id"],
                        level=level,
                        comparison=count,
                        score=score,
                    )
                )
                scored.append((score, c["id"]))
            parents = {
                cid for _, cid in sorted(scored, key=lambda x: (-x[0], x[1]))[:beam]
            }
        leaves = {
            v for c in h["supernodes"] if c["id"] in parents for v in c["member_ids"]
        }
        eligible = sorted(index[v] for v in leaves if v in index)
    scored = []
    for i in eligible:
        count += 1
        score = float(X[i] @ q)
        v = docs[i]["node_id"]
        trace.append(dict(kind="leaf", id=v, comparison=count, score=score))
        scored.append((score, v))
    ranked = [
        dict(node_id=v, score=s) for s, v in sorted(scored, key=lambda x: (-x[0], x[1]))
    ]
    return dict(
        beam=beam,
        ranked=ranked,
        comparisons=count,
        leaf_comparisons=len(eligible),
        cluster_comparisons=count - len(eligible),
        trace=trace,
    )
