"""Fresh independent metrics, fixed controls, and evaluation-only benchmark scoring."""

import csv
import hashlib
import json
import math
from collections import Counter
from copy import deepcopy
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t
from scipy.sparse import csr_matrix
from sklearn.metrics import adjusted_rand_score
from .data import read, write, sha
from .embeddings import Encoder, NLI
from .hierarchy import build
from .hypergraph import project
from .labels import generate, evaluate_labels
from .temporal import track, transition
from .retrieval import documents, search


def stats(values):
    a = np.array(values, dtype=float)
    n = len(a)
    if not n:
        return dict(n=0, mean=None, std=None, ci95=None, values=[])
    mean = float(a.mean())
    sd = float(a.std(ddof=1)) if n > 1 else 0.0
    half = float(student_t.ppf(0.975, n - 1) * sd / math.sqrt(n)) if n > 1 else 0.0
    return dict(
        n=n,
        mean=mean,
        std=sd,
        ci95=[mean - half, mean + half],
        values=list(map(float, a)),
    )


def coherence_value(labels, Y):
    k = int(max(labels)) + 1
    slots = csr_matrix(
        (np.ones(len(labels)), (labels, np.arange(len(labels)))), shape=(k, len(labels))
    )
    sums = slots @ Y
    sizes = np.bincount(labels, minlength=k)
    norm = np.linalg.norm(sums, axis=1)
    # Include every nonempty cluster, including singletons, in both observation and null.
    valid = sizes > 0
    scores = norm[valid] / sizes[valid]
    return dict(
        macro=float(scores.mean()) if len(scores) else None,
        size_weighted=(
            float(norm[valid].sum() / sizes[valid].sum()) if valid.any() else None
        ),
        clusters=int(valid.sum()),
        singletons=int((sizes == 1).sum()),
    )


def coherence(h, snap, Y, cfg):
    index = {n["id"]: i for i, n in enumerate(snap["nodes"])}
    types = np.array([n["type"] for n in snap["nodes"]])
    result = {}
    for level in range(3):
        labels = np.empty(len(Y), dtype=int)
        for j, c in enumerate(h["partitions"][str(level)]):
            labels[[index[v] for v in c]] = j
        observed = coherence_value(labels, Y)
        null = []
        null_macro = []
        rng = np.random.default_rng(cfg["seed"])
        for _ in range(cfg["null_replicates"]):
            perm = labels.copy()
            for typ in sorted(set(types)):
                ii = np.flatnonzero(types == typ)
                perm[ii] = rng.permutation(labels[ii])
                assert Counter(perm[ii]) == Counter(labels[ii])
            nc = coherence_value(perm, Y)
            null.append(nc["size_weighted"])
            null_macro.append(nc["macro"])
        ns = stats(null)
        effect = observed["size_weighted"] - ns["mean"]
        nm = stats(null_macro)
        me = observed["macro"] - nm["mean"]
        result[str(level)] = dict(
            observed=observed,
            null=ns,
            null_macro=nm,
            effect=effect,
            standardized_effect=effect / ns["std"] if ns["std"] else None,
            p_upper=(1 + sum(v >= observed["size_weighted"] for v in null))
            / (len(null) + 1),
            macro_effect=me,
            macro_standardized_effect=me / nm["std"] if nm["std"] else None,
            macro_p_upper=(1 + sum(v >= observed["macro"] for v in null_macro))
            / (len(null_macro) + 1),
            seed=cfg["seed"],
            null_policy="Exact cluster-size and per-cluster type-count slots; shuffle labels within node types; singleton treatment matches observation.",
        )
    return result


def partition_ari(a, b):
    result = {}
    for level in range(3):
        aa = {v: i for i, c in enumerate(a["partitions"][str(level)]) for v in c}
        bb = {v: i for i, c in enumerate(b["partitions"][str(level)]) for v in c}
        ids = sorted(aa.keys() & bb.keys())
        result[str(level)] = float(
            adjusted_rand_score([aa[v] for v in ids], [bb[v] for v in ids])
        )
    return result


def execute_evaluation(
    graph, snaps, snapshot_stats, all_h, transitions, vectors, base, cfg, out
):
    assert cfg["construction_model"] != cfg["evaluation_model"]
    write(
        out / "audit/gate_H.json",
        dict(
            status="PASS",
            construction=cfg["construction_model"],
            evaluation=cfg["evaluation_model"],
        ),
    )
    enc = Encoder(
        cfg["evaluation_model"],
        cfg["evaluation_revision"],
        Path(cfg["cache_dir"]) / "evaluation",
        cfg["batch_size"],
    )
    Yall = enc.encode([f"{n['type']}: {n['surface_form']}" for n in graph["nodes"]])
    coh = {}
    packets = []
    for variant, hs in all_h.items():
        coh[variant] = {}
        for snap, h in zip(snaps, hs):
            ii = [base[n["id"]] for n in snap["nodes"]]
            print("COHERENCE/LABELS", variant, snap["snapshot"], flush=True)
            coh[variant][str(snap["snapshot"])] = coherence(h, snap, Yall[ii], cfg)
            packets += generate(h, snap, vectors[ii], cfg, out)
            write(out / f"hierarchies/{variant}/hierarchy_{h['snapshot']}.json", h)
            if variant == "temporal":
                write(out / f"hierarchies/hierarchy_{h['snapshot']}.json", h)
    write(
        out / "audit/gate_G.json",
        dict(
            status="PASS",
            samples=[
                dict(
                    id=p["id"],
                    label=p["label"],
                    gloss=p["gloss"],
                    input=f"labels/inputs/temporal/{p['id']}.json",
                )
                for year in cfg["snapshots"]
                for level in [0, 1]
                for p in [
                    x
                    for x in packets
                    if x["variant"] == "temporal"
                    and x["snapshot"] == year
                    and x["level"] == level
                ][:5]
            ],
            limitation="Assistant/programmatic inspection of generation support; no human expert rating.",
        ),
    )
    print("LABEL NLI", len(packets), flush=True)
    nli = NLI(cfg)
    label_rows = evaluate_labels(packets, nli)
    del nli
    write(out / "labels/faithfulness.json", label_rows)
    label_summary = {}
    for variant in cfg["variants"]:
        label_summary[variant] = {}
        for level in [0, 1]:
            rr = [
                r for r in label_rows if r["variant"] == variant and r["level"] == level
            ]
            valid = [r for r in rr if r["nli"]]
            label_summary[variant][str(level)] = dict(
                samples=len(rr),
                evaluable=len(valid),
                not_evaluable=len(rr) - len(valid),
                overclaim_proxy=stats([r["overclaim_proxy"] for r in valid]),
                labels=dict(Counter(r["nli"]["label"] for r in valid)),
                lexical_support=stats(
                    [
                        r["lexical_support"]
                        for r in rr
                        if r["lexical_support"] is not None
                    ]
                ),
                truncated=sum(
                    r["character_truncated"] or bool(r["nli"] and r["nli"]["truncated"])
                    for r in rr
                ),
            )
    snap = next(s for s in snaps if s["snapshot"] == cfg["perturbation_snapshot"])
    si = snaps.index(snap)
    ii = [base[n["id"]] for n in snap["nodes"]]
    x = vectors[ii]
    Y = Yall[ii]
    perturb = {}
    for variant in cfg["variants"]:
        rows = []
        ref = all_h[variant][si]
        prev = all_h[variant][si - 1] if si else None
        for seed in cfg["perturbation_seeds"]:
            rng = np.random.default_rng(seed)
            remove = set(
                map(
                    int,
                    rng.choice(
                        len(snap["hyperedges"]),
                        round(cfg["perturbation_fraction"] * len(snap["hyperedges"])),
                        replace=False,
                    ),
                )
            )
            damaged = dict(
                snap,
                hyperedges=[
                    e for i, e in enumerate(snap["hyperedges"]) if i not in remove
                ],
            )
            print("PERTURB", variant, seed, flush=True)
            p = build(damaged, x, cfg, variant, prev)
            write(out / f"perturbations/{variant}_{seed}.json", p)
            rows.append(
                dict(
                    seed=seed,
                    removed=len(remove),
                    ari=partition_ari(ref, p),
                    seconds=p["seconds"],
                )
            )
        perturb[variant] = dict(
            snapshot=snap["snapshot"],
            fraction=cfg["perturbation_fraction"],
            seeds=cfg["perturbation_seeds"],
            rows=rows,
            levels={str(l): stats([r["ari"][str(l)] for r in rows]) for l in range(3)},
            interpretation="Conditional final-snapshot perturbation with fixed previous hierarchy; intervals across five deletion seeds, not across corpora.",
        )
    sensitivity = []
    ref = all_h["temporal"][si]
    prev = all_h["temporal"][si - 1] if si else None
    for key, values in [
        ("alpha", cfg["alpha_sensitivity"]),
        ("lambda", cfg["lambda_sensitivity"]),
        ("knn_k", cfg["knn_sensitivity"]),
    ]:
        for value in values:
            if value == cfg[key]:
                p = ref
            else:
                cc = dict(cfg)
                cc[key] = value
                print("SENSITIVITY", key, value, flush=True)
                p = build(snap, x, cc, "temporal", prev)
                track(p, prev, cfg["event_contribution"])
                write(out / f"sensitivity/{key}_{value}.json", p)
            pc = {}
            for l in range(3):
                membership = {
                    v: j for j, c in enumerate(p["partitions"][str(l)]) for v in c
                }
                pc[str(l)] = coherence_value(
                    np.array([membership[n["id"]] for n in snap["nodes"]]), Y
                )
            sensitivity.append(
                dict(
                    parameter=key,
                    value=value,
                    snapshot=snap["snapshot"],
                    ari_vs_primary=partition_ari(ref, p),
                    temporal_stability=transition(prev, p) if prev else None,
                    coherence=pc,
                )
            )
    # Freeze all prediction documents and rankings before importing gold evaluation.
    snap = snaps[-1]
    docs = documents(snap, cfg)
    write(out / "retrieval/documents.json", docs)
    DX = enc.encode([d["text"] for d in docs])
    with (Path(cfg["data_dir"]) / "questions.csv").open(
        encoding="utf-8-sig", newline=""
    ) as f:
        questions = list(csv.DictReader(f, delimiter=";"))
    Q = enc.encode([q["question"] for q in questions])
    predictions = []
    for q, vector in zip(questions, Q):
        for beam in [None] + cfg["retrieval_beams"]:
            p = search(all_h["temporal"][-1], docs, DX, vector, beam)
            p.update(
                question_id=q["question_id"],
                question_type=q["type"],
                system="flat" if beam is None else f"beam{beam}",
            )
            predictions.append(p)
    write(out / "retrieval/predictions.json", predictions)
    prediction_sha = sha(out / "retrieval/predictions.json")
    write(
        out / "audit/gate_J.json",
        dict(
            status="PREDICTIONS_FROZEN",
            sha256=prediction_sha,
            document_sha256=sha(out / "retrieval/documents.json"),
            scorer="Identical normalized BGE document cosine; all nonauthor leaves; no reranking",
            beams=cfg["retrieval_beams"],
        ),
    )
    from .benchmark import support_audit, align_claims, evaluate_predictions

    truth = read(Path(cfg["data_dir"]) / "ground_truth.json")
    support = support_audit(snap, truth, read("notes/target_annotations.json"))
    write(out / "audit/benchmark_support_audit.json", support)
    write(out / "benchmark_support_audit.json", support)
    claims = align_claims(truth, snap, enc, read("notes/claim_annotations.json"))
    write(out / "audit/claim_alignment.json", claims)
    retrieved = evaluate_predictions(
        predictions, docs, support, claims, cfg["retrieval_top_ks"]
    )
    write(out / "retrieval/per_question.json", retrieved)
    assert prediction_sha == sha(out / "retrieval/predictions.json")
    audit = read(out / "audit/data_audit.json")
    audit["benchmark"] = dict(
        questions=len(questions),
        question_types=dict(Counter(q["type"] for q in questions)),
        target_occurrences=support["target_occurrences"],
        unique_targets=support["unique_targets"],
        literal=support["basic_normalized_exact_occurrences"],
        method_literal=support["basic_method_only_exact_occurrences"],
        no_literal=support["basic_no_literal_occurrences"],
    )
    write(out / "audit/data_audit.json", audit)
    grouped = {}
    for system in sorted({r["system"] for r in retrieved}):
        rows = [r for r in retrieved if r["system"] == system]
        aa = [r for r in rows if r["question_type"] == "A"]
        bb = [r for r in rows if r["question_type"] == "B"]
        kk = {}
        for k in cfg["retrieval_top_ks"]:
            rr = [r["at_k"][str(k)] for r in aa]
            den = sum(r["raw_denominator"] for r in rr)
            sup = sum(r["supported_denominator"] for r in rr)
            hits = sum(len(r["found_targets"]) for r in rr)
            kk[str(k)] = dict(
                raw_micro_recall=hits / den,
                raw_macro_recall=float(np.mean([r["raw_recall"] for r in rr])),
                supported_micro_recall=hits / sup if sup else None,
                found=hits,
                raw_targets=den,
                supported_targets=sup,
                direct_identity_found=sum(
                    len(r["direct_identity_targets"]) for r in rr
                ),
                composite_bundle_found=sum(
                    len(r["composite_bundle_targets"]) for r in rr
                ),
                typeB_claims=sum(r["at_k"][str(k)]["expected_claims"] for r in bb),
                typeB_alignable=sum(r["at_k"][str(k)]["alignable_claims"] for r in bb),
                typeB_source_recovery=stats(
                    [
                        r["at_k"][str(k)]["source_recovery"]
                        for r in bb
                        if r["at_k"][str(k)]["source_recovery"] is not None
                    ]
                ),
            )
        grouped[system] = dict(
            at_k=kk,
            comparisons=stats([r["comparisons"] for r in rows]),
            fine_comparisons=stats([r["leaf_comparisons"] for r in rows]),
            first_target_comparisons_median=(
                float(
                    np.median(
                        [
                            r["comparisons_to_first_direct_target"]
                            for r in aa
                            if r["comparisons_to_first_direct_target"] is not None
                        ]
                    )
                )
                if any(r["comparisons_to_first_direct_target"] is not None for r in aa)
                else None
            ),
            first_target_not_found=sum(
                r["comparisons_to_first_direct_target"] is None for r in aa
            ),
        )
    independence = dict(
        construction=cfg["construction_model"],
        coherence_and_retrieval=cfg["evaluation_model"],
        models_distinct=cfg["construction_model"] != cfg["evaluation_model"],
        gold_loaded_after_prediction_freeze=read(out / "audit/gate_J.json")["sha256"]
        == prediction_sha
        == sha(out / "retrieval/predictions.json"),
        prediction_sha256=prediction_sha,
        construction_revision=cfg["construction_revision"],
        evaluation_revision=cfg["evaluation_revision"],
        labels="Generation-only member/type/relation packets; generation-only TF-IDF and centroid; disjoint held-out members.",
        null="Independent fixed-seed within-type slot permutations",
        annotations="Evaluation-only inspected support statuses; never imported by retrieval or hierarchy.",
    )
    write(out / "audit/evaluation_independence.json", independence)
    write(
        out / "audit/gates_EFI.json",
        dict(
            E=dict(
                static_vs_temporal={
                    str(s["snapshot"]): partition_ari(
                        all_h["static"][i], all_h["temporal"][i]
                    )
                    for i, s in enumerate(snaps)
                }
            ),
            F=dict(
                event_examples={
                    kind: next(
                        (
                            e
                            for e in read(out / "temporal_events.json")
                            if e["variant"] == "temporal" and e["event_type"] == kind
                        ),
                        None,
                    )
                    for kind in [
                        "birth",
                        "continuation",
                        "growth",
                        "merge",
                        "split",
                        "death",
                    ]
                }
            ),
            I=dict(
                status="AUDITED",
                support_counts=support["counts"],
                claim_statuses=dict(Counter(c["status"] for c in claims)),
            ),
        ),
    )
    return dict(
        snapshot_statistics=snapshot_stats,
        coherence=coh,
        perturbation_stability=perturb,
        cross_snapshot_stability=transitions,
        label_faithfulness=label_summary,
        projection_loss=project(graph["hyperedges"])[1],
        parameter_sensitivity=sensitivity,
        benchmark_support=support,
        extrinsic=grouped,
        claim_alignment=dict(
            total=len(claims),
            alignable=sum(bool(c["aligned_node_ids"]) for c in claims),
            by_type={
                typ: dict(
                    expected=sum(c["type"] == typ for c in claims),
                    alignable=sum(
                        c["type"] == typ and bool(c["aligned_node_ids"]) for c in claims
                    ),
                )
                for typ in ["A", "B"]
            },
        ),
        independence=independence,
        config=cfg,
        construction_runtimes={
            v: [h["seconds"] for h in hs] for v, hs in all_h.items()
        },
    )
