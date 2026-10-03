"""Evaluation-only exact V1 normalization and target mapping, copied AFTER freeze."""
import unicodedata
from collections import Counter

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
