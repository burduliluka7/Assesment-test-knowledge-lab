"""Deterministic generation-only labels; held-out evaluation is a separate step."""

import hashlib
import re
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from .data import write


def split_members(members, nodes, seed, fraction):
    generation = []
    held = []
    for typ in sorted({nodes[v]["type"] for v in members}):
        vv = sorted(
            [v for v in members if nodes[v]["type"] == typ],
            key=lambda v: hashlib.sha256(f"{seed}:{v}".encode()).hexdigest(),
        )
        take = max(1, min(len(vv) - 1, round(len(vv) * fraction))) if len(vv) > 1 else 1
        generation += vv[:take]
        held += vv[take:]
    assert not set(generation) & set(held) and set(generation) | set(held) == set(
        members
    )
    return sorted(generation), sorted(held)


def generate(h, snap, X, cfg, out):
    nodes = {n["id"]: n for n in snap["nodes"]}
    index = {n["id"]: i for i, n in enumerate(snap["nodes"])}
    packets = []
    assert all(n["first_seen_year"] <= snap["snapshot"] for n in snap["nodes"])
    assert all(
        e["year"] <= snap["snapshot"]
        and (e["provenance"].get("article_year") or e["year"]) <= snap["snapshot"]
        and set(e["members"]) <= nodes.keys()
        for e in snap["hyperedges"]
    )
    visible_articles = {
        a
        for n in snap["nodes"]
        if n["type"] == "article"
        for a in n["provenance"].get("articles", [])
    }
    for level in [0, 1]:
        clusters = [c for c in h["supernodes"] if c["level"] == level]
        splits = {
            c["id"]: split_members(
                c["member_ids"], nodes, cfg["seed"], cfg["label_generation_fraction"]
            )
            for c in clusters
        }
        vocab_ids = sorted(
            {
                v
                for gen, _ in splits.values()
                for v in gen
                if nodes[v]["type"] != "author"
            }
        )
        texts = [nodes[v]["surface_form"] for v in vocab_ids]
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 3), stop_words="english", lowercase=True
        )
        try:
            matrix = vectorizer.fit_transform(texts)
            features = vectorizer.get_feature_names_out()
            ix = {v: i for i, v in enumerate(vocab_ids)}
        except ValueError:
            matrix = None
            features = []
            ix = {}
        for c in clusters:
            gen, held = splits[c["id"]]
            gen_set = set(gen)
            held_set = set(held)
            edges = [
                e
                for e in snap["hyperedges"]
                if set(e["members"]) <= gen_set
                and e["relation_type"] not in ["authored_by", "cites"]
            ]
            he = [
                e
                for e in snap["hyperedges"]
                if set(e["members"]) <= held_set
                and e["relation_type"] not in ["authored_by", "cites"]
            ]
            candidates = [v for v in gen if nodes[v]["type"] != "author"]
            # Generation-only centroid, so held-out members cannot choose the label.
            center = (
                X[[index[v] for v in candidates]].mean(axis=0)
                if candidates
                else np.zeros(X.shape[1])
            )
            representatives = sorted(
                candidates, key=lambda v: (-float(X[index[v]] @ center), v)
            )
            label = "Scientific evidence"
            if matrix is not None and candidates:
                scores = np.asarray(
                    matrix[[ix[v] for v in candidates]].mean(axis=0)
                ).ravel()
                order = sorted(
                    np.flatnonzero(scores),
                    key=lambda j: (
                        -scores[j],
                        next(
                            (
                                i
                                for i, v in enumerate(representatives)
                                if features[j] in nodes[v]["surface_form"].lower()
                            ),
                            len(representatives),
                        ),
                        features[j],
                    ),
                )
                suitable = [
                    features[j]
                    for j in order
                    if any(ch.isalpha() for ch in features[j])
                    and not all(t in ENGLISH_STOP_WORDS for t in features[j].split())
                ]
                if suitable:
                    label = suitable[0]
            label = " ".join(label.split()[: cfg["label_max_words"]])
            names = [nodes[v]["surface_form"] for v in representatives[:2]]
            # Node names may themselves be long claims; the gloss stays bounded.
            short = [n for n in names if len(n) <= 100]
            gloss = (
                f"This cluster groups evidence concerning {label}"
                + (", including " + "; ".join(short) if short else "")
                + "."
            )
            c["label"] = label
            c["gloss"] = gloss
            assert all(nodes[v]["first_seen_year"] <= snap["snapshot"] for v in gen)
            assert all(
                e["year"] <= snap["snapshot"]
                and (e["provenance"].get("article_year") or e["year"])
                <= snap["snapshot"]
                for e in edges
            )
            generation_input = dict(
                cutoff=snap["snapshot"],
                generation_members=[
                    dict(
                        id=v,
                        type=nodes[v]["type"],
                        surface_form=nodes[v]["surface_form"],
                    )
                    for v in gen
                ],
                generation_type_counts=dict(Counter(nodes[v]["type"] for v in gen)),
                internal_hyperedges=edges,
                tfidf_scope="Generation members only, within current snapshot and level",
                phrase=label,
                representative_ids=representatives[:2],
            )
            held_text = "\n".join(
                f"{nodes[v]['type']}: {nodes[v]['surface_form']}"
                for v in held
                if nodes[v]["type"] != "author"
            )
            packet = dict(
                id=c["id"],
                level=level,
                snapshot=snap["snapshot"],
                variant=h["variant"],
                generation_input=generation_input,
                held_out_ids=held,
                held_out_hyperedges=he,
                held_out_text=held_text[: cfg["label_evidence_max_chars"]],
                character_truncated=len(held_text) > cfg["label_evidence_max_chars"],
                label=label,
                gloss=gloss,
                evaluable=len([v for v in held if nodes[v]["type"] != "author"]) >= 2,
            )
            write(
                out / f"labels/inputs/{h['variant']}/{c['id']}.json", generation_input
            )
            packets.append(packet)
    return packets


def evaluate_labels(packets, nli):
    scored = nli.score(
        [(p["held_out_text"], p["gloss"]) for p in packets if p["evaluable"]]
    )
    it = iter(scored)
    result = []
    for p in packets:
        row = dict(p)
        tokens = set(re.findall(r"[a-z]{3,}", p["label"].lower())) - ENGLISH_STOP_WORDS
        held = set(re.findall(r"[a-z]{3,}", p["held_out_text"].lower()))
        row["lexical_support"] = len(tokens & held) / len(tokens) if tokens else None
        row["nli"] = next(it) if p["evaluable"] else None
        row["overclaim_proxy"] = (
            int(row["nli"]["label"] != "entailment") if row["nli"] else None
        )
        result.append(row)
    return result
