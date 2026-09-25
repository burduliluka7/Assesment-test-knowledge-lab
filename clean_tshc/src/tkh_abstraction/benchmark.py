"""Evaluation-only gold annotations. Imported only after predictions are frozen."""

import csv
import hashlib
import re
import unicodedata
from collections import Counter
import numpy as np
from .data import read, write


def normalize(s):
    s = unicodedata.normalize("NFKC", s).casefold()
    return " ".join(s.translate(str.maketrans({c: "-" for c in "‐‑‒–−"})).split())


def support_audit(snap, truth, annotations):
    nodes = {n["id"]: n for n in snap["nodes"]}
    rows = []
    literal = []
    method = []
    basic = lambda s: " ".join(unicodedata.normalize("NFKC", s).casefold().split())
    for qid, q in truth.items():
        for target in q.get("expected_methods", []):
            exact = [
                v
                for v, n in nodes.items()
                if normalize(n["surface_form"]) == normalize(target)
            ]
            literal.append(
                any(basic(n["surface_form"]) == basic(target) for n in nodes.values())
            )
            method.append(
                any(
                    basic(n["surface_form"]) == basic(target) and n["type"] == "method"
                    for n in nodes.values()
                )
            )
            ann = annotations.get(target, {})
            if exact:
                status = "EXACT"
                components = [exact]
                rationale = "Normalized full surface-form identity, including typographic hyphens; heterogeneous node types accepted."
            else:
                components = [
                    [v for v in c if v in nodes] for c in ann.get("components", [])
                ]
                status = ann.get("status", "ABSENT")
                rationale = ann.get(
                    "rationale", "No defensible identity or component mapping found."
                )
                if not any(components):
                    status = "ABSENT"
            matched = {v for c in components for v in c}
            edges = [e for e in snap["hyperedges"] if matched & set(e["members"])]
            rows.append(
                dict(
                    question_id=qid,
                    target=target,
                    status=status,
                    components=components,
                    node_ids=sorted(matched),
                    hyperedge_ids=[e["id"] for e in edges],
                    source_articles=sorted(
                        {
                            a
                            for v in matched
                            for a in nodes[v]["provenance"].get("articles", [])
                        }
                    ),
                    evidence=[
                        dict(
                            node_id=v,
                            surface_form=nodes[v]["surface_form"],
                            type=nodes[v]["type"],
                            provenance=nodes[v]["provenance"],
                        )
                        for v in sorted(matched)
                    ],
                    rationale=rationale,
                    verification_method="Fresh normalized matching; assistant inspection of explicit node text, relation and provenance for nonexact cases. No human expert adjudication.",
                )
            )
    return dict(
        records=rows,
        counts=dict(Counter(r["status"] for r in rows)),
        target_occurrences=len(rows),
        unique_targets=len({r["target"] for r in rows}),
        basic_normalized_exact_occurrences=sum(literal),
        basic_method_only_exact_occurrences=sum(method),
        basic_no_literal_occurrences=len(rows) - sum(literal),
        limitations="PARTIAL targets remain in conditional denominator but receive no full raw recovery credit; their component coverage is reported separately. ABSENT means no defensible mapping found by this audit, not a proof of semantic absence.",
    )


def expected_claims(truth):
    rows = []
    for qid, q in truth.items():
        if q["type"] == "A":
            for target, claims in q.get("method_claims", {}).items():
                sources = [
                    e.get("source")
                    for e in q.get("method_evidence", {}).get(target, [])
                    if e.get("source")
                ]
                for i, text in enumerate(claims):
                    rows.append(
                        dict(
                            id=f"{qid}:{target}:{i}",
                            question_id=qid,
                            type="A",
                            target=target,
                            text=text,
                            sources=sources,
                        )
                    )
        else:
            for c in q.get("required_claims", []):
                sources = [q.get("sources", {}).get(s, s) for s in c.get("sources", [])]
                rows.append(
                    dict(
                        id=c["id"],
                        question_id=qid,
                        type="B",
                        target=None,
                        text=c["text"],
                        sources=sources,
                    )
                )
    return rows


def resolve_source(source, snap):
    source = source if isinstance(source, dict) else {"key": source}
    nodes = {n["id"]: n for n in snap["nodes"]}
    articles = [n for n in snap["nodes"] if n["type"] == "article"]
    found = set()
    method = "unmapped"
    for n in articles:
        p = n["provenance"]
        match = False
        if source.get("doi") and normalize(source["doi"]) == normalize(
            p.get("doi") or ""
        ):
            match = True
            method = "doi"
        if source.get("title") and normalize(source["title"]) == normalize(
            n["surface_form"]
        ):
            match = True
            method = "full_title"
        if match:
            found.update(p.get("articles", []))
    if not found:
        key = source.get("key", "")
        year = re.search(r"\b(19|20)\d{2}\b", key)
        surname = re.match(r"([\w-]+)", key)
        if year and surname:
            author_ids = {
                v
                for v, n in nodes.items()
                if n["type"] == "author"
                and re.search(
                    r"(?<!\w)" + re.escape(surname[1]) + r"(?!\w)",
                    n["surface_form"],
                    re.I,
                )
            }
            for e in snap["hyperedges"]:
                if (
                    e["relation_type"] == "authored_by"
                    and author_ids & set(e["members"])
                    and e["provenance"].get("article_year") == int(year[0])
                ):
                    found.add(e["provenance"]["article_id"])
            if found:
                method = "unique_author_year"
    return dict(
        source=source,
        article_ids=sorted(found) if len(found) == 1 else [],
        status="MATCHED" if len(found) == 1 else "AMBIGUOUS" if found else "UNMAPPED",
        rule=method,
    )


def align_claims(truth, snap, encoder, annotations=None):
    annotations = annotations or {}
    claims = expected_claims(truth)
    nodes = [n for n in snap["nodes"] if n["type"] == "claim"]
    x = encoder.encode([n["surface_form"] for n in nodes])
    q = encoder.encode([c["text"] for c in claims])
    rows = []
    for c, vector in zip(claims, q):
        refs = [resolve_source(s, snap) for s in c["sources"]]
        sources = {a for r in refs for a in r["article_ids"]}
        eligible = [
            i
            for i, n in enumerate(nodes)
            if sources & set(n["provenance"].get("articles", []))
        ]
        candidates = eligible if eligible else list(range(len(nodes)))
        scores = x[candidates] @ vector
        best = sorted(
            zip(scores, candidates), key=lambda t: (-float(t[0]), nodes[t[1]]["id"])
        )[:3]
        exact = [
            n["id"]
            for n in nodes
            if normalize(n["surface_form"]).rstrip(".")
            == normalize(c["text"]).rstrip(".")
            and sources & set(n["provenance"].get("articles", []))
        ]
        reviewed = annotations.get(c["id"])
        if reviewed:
            assert reviewed["claim_text"] == c["text"]
            index = {n["id"]: n for n in nodes}
            assert reviewed["node_ids"] and all(
                v in index for v in reviewed["node_ids"]
            )
            assert all(
                sources & set(index[v]["provenance"].get("articles", []))
                for v in reviewed["node_ids"]
            )
            assert set(reviewed["source_article_ids"]) <= sources
            exact = reviewed["node_ids"]
        rows.append(
            dict(
                c,
                source_resolution=refs,
                source_article_ids=sorted(sources),
                aligned_node_ids=exact,
                status=(
                    "REVIEWED_TEXT_AND_SOURCE"
                    if reviewed
                    else "EXACT_TEXT_AND_SOURCE" if exact else "NOT_EVALUABLE_ALIGNMENT"
                ),
                reviewed_annotation=reviewed,
                proposals=[
                    dict(
                        node_id=nodes[i]["id"],
                        text=nodes[i]["surface_form"],
                        cosine=float(s),
                        provenance=nodes[i]["provenance"],
                    )
                    for s, i in best
                ],
                decision=(
                    reviewed["rationale"]
                    if reviewed
                    else "Only exact claim text plus resolved source is automatically accepted. Semantic proposals require adjudication; similarity alone is not entailment."
                ),
            )
        )
    return rows


def evaluate_predictions(predictions, docs, support, claims, topks):
    lookup = {d["node_id"]: d for d in docs}
    results = []
    for p in predictions:
        qid = p["question_id"]
        targets = [r for r in support["records"] if r["question_id"] == qid]
        expected = [r for r in claims if r["question_id"] == qid]
        row = {
            k: p[k]
            for k in [
                "question_id",
                "question_type",
                "system",
                "beam",
                "comparisons",
                "leaf_comparisons",
                "cluster_comparisons",
            ]
        }
        row["at_k"] = {}
        for k in topks:
            selected = [r["node_id"] for r in p["ranked"][:k]]
            direct = set(selected)
            evidence = direct | {v for d in selected for v in lookup[d]["context_ids"]}
            sources = {a for d in selected for a in lookup[d]["article_ids"]}
            found = []
            partial = []
            coverage = []
            direct_identity = []
            composite_bundles = []
            for t in targets:
                components = t["components"]
                accessible = (
                    evidence if t["status"] in ["COMPOSITE", "PARTIAL"] else direct
                )
                cc = [bool(set(c) & accessible) for c in components]
                fraction = sum(cc) / len(cc) if cc else None
                complete = bool(cc) and all(cc)
                if complete and t["status"] in ["EXACT", "ALIAS", "COMPOSITE"]:
                    found.append(t["target"])
                    if t["status"] == "COMPOSITE":
                        composite_bundles.append(t["target"])
                    else:
                        direct_identity.append(t["target"])
                if complete and t["status"] == "PARTIAL":
                    partial.append(t["target"])
                if t["status"] in ["COMPOSITE", "PARTIAL"]:
                    coverage.append(
                        dict(
                            target=t["target"],
                            status=t["status"],
                            component_coverage=fraction,
                        )
                    )
            supported = sum(t["status"] != "ABSENT" for t in targets)
            alignable = [c for c in expected if c["aligned_node_ids"]]
            hit = sum(bool(set(c["aligned_node_ids"]) & evidence) for c in alignable)
            expected_sources = {a for c in expected for a in c["source_article_ids"]}
            row["at_k"][str(k)] = dict(
                found_targets=found,
                raw_recall=len(found) / len(targets) if targets else None,
                supported_recall=len(found) / supported if supported else None,
                raw_denominator=len(targets),
                supported_denominator=supported,
                direct_identity_targets=direct_identity,
                composite_bundle_targets=composite_bundles,
                direct_identity_raw_recall=(
                    len(direct_identity) / len(targets) if targets else None
                ),
                partial_evidence_targets=partial,
                component_coverage=coverage,
                expected_claims=len(expected),
                alignable_claims=len(alignable),
                retrieved_claims=hit,
                strict_claim_recall=hit / len(alignable) if alignable else None,
                direct_claim_recall=(
                    sum(bool(set(c["aligned_node_ids"]) & direct) for c in alignable)
                    / len(alignable)
                    if alignable
                    else None
                ),
                recovery_unit="Top-k documents: direct node identity for EXACT/ALIAS; union of supplied document context for composite bundles and aligned claim evidence. Direct-only counts exported separately.",
                claim_status="EVALUABLE" if alignable else "NOT_EVALUABLE_ALIGNMENT",
                source_recovery=(
                    len(sources & expected_sources) / len(expected_sources)
                    if expected_sources
                    else None
                ),
                source_overlap=sorted(sources & expected_sources),
                selected_node_ids=selected,
                selected_evidence_ids=sorted(evidence),
            )
        direct_target = {
            v
            for t in targets
            if t["status"] in ["EXACT", "ALIAS"]
            for v in t["node_ids"]
        }
        first = [
            t["comparison"]
            for t in p["trace"]
            if t["kind"] == "leaf" and t["id"] in direct_target
        ]
        row["comparisons_to_first_direct_target"] = min(first) if first else None
        row["first_target_definition"] = (
            "First EXACT/ALIAS node encountered in fixed scoring order; not final rank and not a composite stopping guarantee."
        )
        results.append(row)
    return results
