"""Small static figures, complete tables, and a report generated from this run."""

import csv
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .data import read


def table(headers, rows):
    def cell(x):
        return f"{x:.4f}" if isinstance(x, (float, np.floating)) else str(x)

    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        + ["| " + " | ".join(cell(x) for x in row) + " |" for row in rows]
    )


def csv_table(path, headers, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)


def deliver(m, all_h, cfg, out):
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    td = out / "tables"
    td.mkdir(exist_ok=True)

    def save(name):
        plt.tight_layout()
        plt.savefig(figdir / f"{name}.png", dpi=200)
        plt.savefig(figdir / f"{name}.pdf")
        plt.close()

    ss = m["snapshot_statistics"]
    years = [s["snapshot"] for s in ss]
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6))
    for ax, key in zip(axes, ["nodes", "hyperedges"]):
        ax.plot(years, [s[key] for s in ss], marker="o")
        ax.set(xlabel="Annual snapshot", ylabel=key.capitalize(), xticks=years)
    save("snapshot_growth")
    fig, ax = plt.subplots(figsize=(6, 2.6))
    for s in ss:
        hist = sorted((int(k), v) for k, v in s["arity"]["histogram"].items())
        ax.plot(*zip(*hist), marker=".", label=str(s["snapshot"]))
    ax.set(xlabel="Original hyperedge arity", ylabel="Hyperedges", yscale="log")
    ax.legend(ncol=4, fontsize=8)
    save("arity_distribution")
    rows = [
        [
            s["snapshot"],
            s["nodes"],
            s["hyperedges"],
            s["new_nodes"],
            s["new_edges"],
            s["node_growth_percent"],
            s["edge_growth_percent"],
            s["arity"]["min"],
            s["arity"]["max"],
            s["arity"]["mean"],
            s["arity"]["median"],
            s["arity"]["fraction_gt2"],
            s["deferred_anomalous_edges"],
        ]
        for s in ss
    ]
    csv_table(
        td / "snapshot_summary.csv",
        [
            "snapshot",
            "nodes",
            "edges",
            "new_nodes",
            "new_edges",
            "node_growth_pct",
            "edge_growth_pct",
            "arity_min",
            "arity_max",
            "arity_mean",
            "arity_median",
            "fraction_gt2",
            "deferred_anomalous_edges",
        ],
        rows,
    )
    variants = cfg["variants"]
    last = str(years[-1])
    cohrows = []
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.7), sharey=True)
    for l, ax in enumerate(axes):
        vals = [m["coherence"][v][last][str(l)] for v in variants]
        xx = np.arange(4)
        ax.plot(
            xx, [r["observed"]["size_weighted"] for r in vals], "o", label="Observed"
        )
        ax.errorbar(
            xx,
            [r["null"]["mean"] for r in vals],
            yerr=[r["null"]["std"] for r in vals],
            fmt="x",
            label="Null mean ± SD",
        )
        ax.set(xticks=xx, xticklabels=variants, title=f"2026 level {l}")
        ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("Independent BGE coherence")
    axes[-1].legend(fontsize=7)
    save("coherence_vs_null")
    for v in variants:
        for year, levels in m["coherence"][v].items():
            for l, r in levels.items():
                cohrows.append(
                    [
                        v,
                        year,
                        l,
                        r["observed"]["macro"],
                        r["observed"]["size_weighted"],
                        r["null"]["mean"],
                        r["null"]["std"],
                        *r["null"]["ci95"],
                        r["effect"],
                        r["standardized_effect"],
                        r["p_upper"],
                        r["null_macro"]["mean"],
                        r["macro_effect"],
                        r["macro_p_upper"],
                    ]
                )
    csv_table(
        td / "coherence.csv",
        [
            "variant",
            "year",
            "level",
            "macro",
            "size_weighted",
            "null_mean",
            "null_sd",
            "null_ci_low",
            "null_ci_high",
            "absolute_effect",
            "standardized_effect",
            "p_upper",
            "null_macro_mean",
            "macro_effect",
            "macro_p",
        ],
        cohrows,
    )
    temporalrows = []
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.7), sharey=True)
    for l, ax in enumerate(axes):
        for v in ["static", "temporal"]:
            ts = m["cross_snapshot_stability"][v]
            ax.plot(
                range(len(ts)),
                [x[str(l)]["ari"] for x in ts.values()],
                marker="o",
                label=v,
            )
        ax.set(
            xticks=range(3), xticklabels=["20→22", "22→24", "24→26"], title=f"Level {l}"
        )
    axes[0].set_ylabel("ARI on shared nodes")
    axes[-1].legend()
    save("temporal_stability")
    for v, ts in m["cross_snapshot_stability"].items():
        for tr, levels in ts.items():
            for l, r in levels.items():
                temporalrows.append(
                    [
                        v,
                        tr,
                        l,
                        r["shared_nodes"],
                        r["ari"],
                        r["vi"],
                        r["normalized_vi"],
                        r["lineage_retention"],
                    ]
                )
    csv_table(
        td / "temporal_stability.csv",
        [
            "variant",
            "transition",
            "level",
            "shared_nodes",
            "ari",
            "vi",
            "normalized_vi",
            "lineage_retention",
        ],
        temporalrows,
    )
    from collections import Counter

    events = read(out / "temporal_events.json")
    ec = Counter(
        (
            e["variant"],
            e["from_snapshot"],
            e["to_snapshot"],
            e["level"],
            e["event_type"],
        )
        for e in events
    )
    csv_table(
        td / "temporal_events.csv",
        ["variant", "from", "to", "level", "event", "count"],
        [
            list(key) + [count]
            for key, count in sorted(ec.items(), key=lambda kv: str(kv[0]))
        ],
    )
    fig, ax = plt.subplots(figsize=(6, 3))
    perturbrows = []
    for l in range(3):
        rr = [m["perturbation_stability"][v]["levels"][str(l)] for v in variants]
        xx = np.arange(4) + (l - 1) * 0.14
        ax.errorbar(
            xx,
            [r["mean"] for r in rr],
            yerr=[r["mean"] - r["ci95"][0] for r in rr],
            fmt="o",
            label=f"Level {l}",
        )
        for v, r in zip(variants, rr):
            perturbrows.append([v, l, r["n"], r["mean"], r["std"], *r["ci95"]])
    ax.set(
        xticks=range(4),
        xticklabels=variants,
        ylabel="ARI, mean and 95% Student-t CI",
        title="2026: five 10% edge deletions",
    )
    ax.legend()
    save("perturbation_stability")
    csv_table(
        td / "perturbation.csv",
        ["variant", "level", "n", "mean_ari", "sd", "ci_low", "ci_high"],
        perturbrows,
    )
    systems = ["flat"] + [f"beam{b}" for b in cfg["retrieval_beams"]]
    retrievalrows = []
    fig, ax = plt.subplots(figsize=(6, 3))
    for k in cfg["retrieval_top_ks"]:
        for system in systems:
            r = m["extrinsic"][system]
            s = r["at_k"][str(k)]
            ax.scatter(
                r["comparisons"]["mean"],
                s["raw_micro_recall"],
                marker="o" if k == 10 else "x",
            )
            ax.annotate(
                f"{system} @{k}",
                (r["comparisons"]["mean"], s["raw_micro_recall"]),
                fontsize=7,
                xytext=(3, 4 if k == 10 else -10),
                textcoords="offset points",
            )
            retrievalrows.append(
                [
                    system,
                    k,
                    s["found"],
                    s["raw_targets"],
                    s["supported_targets"],
                    s["raw_micro_recall"],
                    s["supported_micro_recall"],
                    r["comparisons"]["mean"],
                    r["first_target_comparisons_median"],
                    r["first_target_not_found"],
                ]
            )
    ax.set(
        xlabel="Mean cluster + leaf cosine comparisons",
        ylabel="Raw Type-A target recall",
        title="Identical BGE documents and leaf cosine scorer",
    )
    save("retrieval_work_tradeoff")
    csv_table(
        td / "retrieval_summary.csv",
        [
            "system",
            "k",
            "found",
            "raw_targets",
            "supported_targets",
            "raw_recall",
            "supported_recall",
            "mean_comparisons",
            "median_first_direct_hit",
            "questions_without_direct_hit",
        ],
        retrievalrows,
    )
    per = read(out / "retrieval/per_question.json")
    pr = []
    for p in per:
        for k, r in p["at_k"].items():
            pr.append(
                [
                    p["question_id"],
                    p["question_type"],
                    p["system"],
                    k,
                    r["raw_recall"],
                    r["supported_recall"],
                    r["strict_claim_recall"],
                    r["alignable_claims"],
                    r["expected_claims"],
                    r["source_recovery"],
                    p["comparisons"],
                    p["comparisons_to_first_direct_target"],
                ]
            )
    csv_table(
        td / "retrieval_per_question.csv",
        [
            "question",
            "type",
            "system",
            "k",
            "raw_recall",
            "supported_recall",
            "strict_claim_recall",
            "alignable_claims",
            "expected_claims",
            "source_recovery",
            "comparisons",
            "first_direct_target_comparisons",
        ],
        pr,
    )
    sr = []
    for r in m["parameter_sensitivity"]:
        for l in range(3):
            sr.append(
                [
                    r["parameter"],
                    r["value"],
                    l,
                    r["ari_vs_primary"][str(l)],
                    r["temporal_stability"][str(l)]["ari"],
                    r["coherence"][str(l)]["size_weighted"],
                ]
            )
    csv_table(
        td / "sensitivity.csv",
        ["parameter", "value", "level", "ari_vs_primary", "temporal_ari", "coherence"],
        sr,
    )
    lr = []
    for v, levels in m["label_faithfulness"].items():
        for l, r in levels.items():
            lr.append(
                [
                    v,
                    l,
                    r["samples"],
                    r["evaluable"],
                    r["not_evaluable"],
                    r["overclaim_proxy"]["mean"],
                    r["labels"].get("entailment", 0),
                    r["labels"].get("neutral", 0),
                    r["labels"].get("contradiction", 0),
                    r["truncated"],
                ]
            )
    csv_table(
        td / "label_faithfulness.csv",
        [
            "variant",
            "level",
            "labels",
            "evaluable",
            "not_evaluable",
            "overclaim_proxy",
            "entailment",
            "neutral",
            "contradiction",
            "truncated",
        ],
        lr,
    )
    report(m, cfg, out, cohrows, perturbrows, retrievalrows, lr)


def report(m, cfg, out, cohrows, perturbrows, retrievalrows, lr):
    audit = read(out / "audit/data_audit.json")
    p = m["projection_loss"]
    b = m["benchmark_support"]
    ss = m["snapshot_statistics"]
    summary = []
    for v in cfg["variants"]:
        c = m["coherence"][v]["2026"]
        t = m["cross_snapshot_stability"][v]
        summary.append(
            [
                v,
                *[c[str(l)]["observed"]["size_weighted"] for l in range(3)],
                float(
                    np.mean([r[str(l)]["ari"] for r in t.values() for l in range(3)])
                ),
                *[
                    m["perturbation_stability"][v]["levels"][str(l)]["mean"]
                    for l in range(3)
                ],
            ]
        )
    sens = [
        [r["parameter"], r["value"], *[r["ari_vs_primary"][str(l)] for l in range(3)]]
        for r in m["parameter_sensitivity"]
    ]
    primary = m["extrinsic"]["beam3"]
    flat = m["extrinsic"]["flat"]
    a = m["claim_alignment"]
    labels = [r for r in lr if r[0] == "temporal"]
    an = audit["anomalies"]
    text = r"""# Temporal Semantic Hypergraph Coarsening

## 1. Problem and formal statement

We construct a laminar hierarchy by greedily balancing semantic compactness against native hyperedge fragmentation, with variation-of-information regularization preserving continuity across annual snapshots. We independently evaluate coherence, stability, labels and downstream navigation. This is a small research implementation; no learned clustering, benchmark tuning or historical result reuse is involved.

For the visible hypergraph $H_t=(V_t,E_t)$, partitions $P_3\preceq P_2\preceq P_1\preceq P_0$ have singleton leaves and at most 120/40/12 clusters. These **design choices are display budgets**, not estimated natural community counts. Unit MiniLM vectors encode only `type: surface_form`. With $\mu_C=|C|^{-1}\sum_{v\in C}x_v$, $p_{e,C}=|e\cap C|/|e|$, and shared nodes $U_t$, our objective is

$$J_t(P)=\alpha\underbrace{\frac{\sum_C\sum_{v\in C}\|x_v-\mu_C\|^2}{4|V_t|}}_{S(P)}+(1-\alpha)\underbrace{\frac{1}{|E_t|}\sum_e\frac{-\sum_Cp_{e,C}\log p_{e,C}}{\log|e|}}_{F(P)}+\lambda\underbrace{\frac{VI(P|_{U_t},P_{t-1}|_{U_t})}{2\log|U_t|}}_{T(P,P_{t-1})}.$$

Set $T=0$ for fewer than two shared nodes. Fixed $\alpha=.5$ gives the normalized terms equal nominal status; $\lambda=.2$ treats history as a regularizer, with zero history weight in 2020. Normalization does not equalize their realized influence: unit-vector $S$ is at most .25. New nodes do not contribute to VI. Every stage uses the previous snapshot's partition at its own target level.

The algorithm performs **restricted-candidate greedy minimization** of this objective. Start at singletons, propose the union of semantic 10-nearest-neighbor and hyperedge co-membership pairs, and repeatedly merge the pair with minimum exact Ward/fragmentation/VI delta in a lazy heap. New neighbors are the union of merged neighbors. Continue through 120→40→12 without reclustering. If the proposal graph disconnects before the budget, join the nearest current centroids deterministically. Co-membership pairs propose merges; the native objective uses original edge endpoint distributions. No global optimum is claimed.

| Property | Status | Basis |
| --- | --- | --- |
| P1 Laminarity | Guaranteed by construction; tested | Only merges; unique parent and union-of-children invariants |
| P2 Budgets | Guaranteed by construction; tested | Each stage stops at its display budget |
| P3 Coherence | Empirical | Independent BGE evaluation against type/size-preserving nulls |
| P4 Fidelity | Structural preservation; empirical partition quality | Original edge identity, arity, multiplicity and provenance survive collapse |
| P5 Stability | Encouraged during construction; empirical degree | Shared-node VI regularization; ARI/VI measured afterward |
| P6 Faithfulness | Conservative generation; empirical proxy | Disjoint held-out evidence and NLI/lexical checks |

Brute-force semantic proposals cost $O(n^2d)$ plus sorting, and embedding cost depends on token lengths and encoder inference. For $M$ initial proposals and $Q$ subsequent pair updates, heap work is $O((M+Q)\log(M+Q))$; each delta uses $O(d)$ semantic arithmetic, shared incident-edge counts, and overlap counts with at most 120 previous clusters. Updating unions touches incident edges and neighboring candidates. Dense proposals and total updates can approach quadratic behavior; the implementation makes no scaling claim beyond this corpus. Hungarian matching costs $O(K^3)$ per level with $K\le120$.

## 2. Data, native representation and temporal identities

**Verified input audit:** @AUDIT@. Input SHA256 hashes and schemas are saved in `outputs/audit/input_manifest.json`. Original inputs are untouched.

@SNAPSHOTS@

Visibility requires `first_seen_year(v) <= t`, `edge.year <= t`, asserting article year <= t when known, and every endpoint visible. Origin year describes an entity's origin, not corpus visibility. **Verified:** @ANOMALIES@. Whole inconsistent edges are delayed, explaining why strict counts can be below the task guide's illustrative slicing. Ingestion timestamps are not publication dates. **Limitation:** the 2026 snapshot has annual temporal resolution; therefore the implementation cannot guarantee exclusion of material appearing after February 2026 within that year.

For every actual hierarchy level, collapse each original edge to `(supernode, multiplicity)` records and retain internal edges. For example, five endpoints mapping to A,A,A,B,C become `{A:3,B:1,C:1}`. All internal, partial and ≥3-supernode cases conserve original arity. Coarse endpoint identities are unavailable from that edge record alone; member lists and original input permit reconstruction. Relation, provenance, identity and endpoint multiplicities remain explicit. This preserves higher-order grouping, without asserting irreducible superiority over every pairwise statistic.

The pairwise baseline adds $1/\binom{r}{2}$ to each endpoint pair and minimizes normalized weighted cut with the same semantic term. **Verified projection loss:** @PROJECTION@. Aggregation conflates overlapping source edges and cannot uniquely reconstruct their identity, arity or grouping. Clique projection is the credible simpler alternative, but is not our default representation. A neural clustering model would add training assumptions and tuning without a written requirement here.

After construction, Hungarian maximum-Jaccard matching assigns persistent IDs. This is bookkeeping, not stabilization. The many-to-many overlap log records continuation, growth, merge, split, birth and death when observed; merge/split contributions require ≥2 nodes and ≥10% of the relevant shared historical membership. Events describe membership, not scientific discoveries. Cumulative visibility need not produce deaths.

The design borrows multilevel aggregation, native hyperedge grouping and evolutionary regularization ideas, rather than reproducing a published optimizer. [Schaub et al. (2023)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.107.054305) motivate distinguishing a displayed hierarchy from evidence for natural hierarchical communities. [Chen et al. (2022)](https://link.springer.com/article/10.1007/s40324-021-00282-x) survey coarsening; [Zhou et al. (NIPS 2006)](https://papers.nips.cc/paper/2006/file/dff8e9c2ac33381546d96deea9922999-Paper.pdf) motivate hyperedge grouping; [Chi et al. (2007)](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/01/evospe.pdf) motivate a current/history tradeoff. Our entropy objective and level-specific VI differ from their spectral formulations. Full citations and borrowing boundaries are in `references.md`.

## 3. Evaluation validity

MiniLM constructs the hierarchy; BGE-small independently evaluates coherence and encodes retrieval documents. Exact model revisions, seeds and all scientific choices are in `configs/default.yaml`. Ground truth is loaded for scoring only after documents and predictions have been saved and hashed. The construction modules cannot read benchmark files; target annotations are evaluation-only. A separate schema audit may count benchmark records, but passes no benchmark content to construction.

Coherence is mean member cosine to each independent evaluation centroid, reported macro and size-weighted at every level and snapshot, including singletons. Each of 100 null partitions preserves every cluster's exact size and type counts by shuffling labels within node types (seed42). We export absolute/standardized effects, null SD/mean intervals and empirical upper-tail p-values $(1+\#\{null\ge observed\})/101$. These are conditional null tests, not evidence of performance across new corpora; multiple tests are descriptive and uncorrected.

At 2026, each method is rebuilt after deleting 10% of edges with five fixed seeds (11,23,37,53,71). Previous 2024 hierarchies remain fixed: this measures sensitivity to current evidence, not propagated historical damage. ARI means, SDs and Student-t 95% intervals quantify deletion-seed variation. Semantic-only ignores both edges and edge proposals; ARI=1 denotes edge blindness, not superior structural robustness. Real temporal ARI, VI and lineage retention use shared nodes for all three adjacent transitions. Three transitions give limited statistical power; their descriptive average is not nine independent replications.

L0/L1 labels use generation-only TF-IDF 1–3-grams, generation-centroid representatives and a conservative gloss template. Authors are excluded from phrase candidates. Hash/type-stratified splitting approximates 70/30; singleton type groups go to generation. Exact inputs are retained. Held-out member evidence is capped at 1,800 characters, with premise-only NLI token truncation recorded. The overclaim proxy is one minus the held-out entailment fraction; small unevaluable clusters are null. No NLI output enters construction or retrieval. Domain mismatch, omitted evidence and easy-to-entail vague glosses limit this proxy. There was no human scientific audit. Unversioned surface forms and pretrained model knowledge prevent perfect historical information isolation despite strict graph-evidence cutoffs.

Retrieval encodes each nonauthor leaf as its type/name plus at most 12 deduplicated direct content-hyperedge neighbors. No recursive traversal, alias expansion, diffusion or reranking is used. `authored_by` and broad `cites` are excluded from scientific context. Flat scores all candidate documents; hierarchy scores descendant-document centroids through P0→P1→P2 and then the surviving leaves using exactly the same cosine. Global beams 1/3/5/10 are fixed, with 3 primary. Work counts every cluster/leaf comparison; it is not a measured latency speedup. First-target work means the first EXACT/ALIAS leaf encountered in deterministic scoring order, not first final-rank answer or a composite stopping guarantee.

The supplied target list is heterogeneous. **Verified support audit:** @SUPPORT@. Full normalized identity accepts any graph type; safe aliases require explicit evidence. Composite labels receive component-bundle evaluation, not fabricated graph nodes. Partial evidence never earns complete-target recovery; those targets remain in the conditional denominator. Raw recall keeps all 63 supplied occurrences. Non-gold returns are unjudged. Source overlap is reported separately from strict claim-node recovery. **Claim alignment:** @CLAIMS@. Semantic candidates are suggestions; only defensible accepted alignments are scored, with the remainder `NOT_EVALUABLE_ALIGNMENT`. A paper hit cannot establish its scientific claim.

## 4. Results

All numbers below come from this clean execution. The four variants are semantic-only, semantic+pairwise, static native and temporal native.

@VARIANTS@

Coherence columns are 2026 size-weighted BGE; temporal ARI averages three transitions and three levels descriptively; perturbation columns are five-seed means. Full null effects, p-values, macro values, all snapshots and CIs are in `outputs/tables/` and `outputs/metrics.json`. Do not interpret equal nominal weights as a guarantee that the structural and temporal terms have comparable effects.

@TEMPORAL_DECISION@

@LABELS@

The table gives primary temporal labels pooled across four snapshots; the overclaim rate is an NLI evaluation proxy, not a human scientific error rate. Generation-member names quoted in glosses can be absent from disjoint held-out evidence; sorted-ID truncation can omit later evidence types. Consequently this proxy mixes unsupported generalization, evidence omission and classifier behavior. All variant probabilities, neutral/contradiction counts and truncation flags remain available.

@RETRIEVAL@

**Observed navigation result:** @NAVIGATION@. A reduced candidate pool can promote an item by removing competitors, so beam recall is not mathematically bounded by flat top-k recall. No scorer or beam was changed to rescue this result. Full Q1–Q18 recall, composite component coverage, source recovery, null strict-claim scores and first-hit work are exported separately. Raw target recovery combines direct identity with explicitly mapped composite document bundles; it is not a count of gold nodes ranked in the first k slots. Direct identity and composite contributions are separate fields. The first-hit median excludes misses and depends on arbitrary ID scan order; it is an encounter diagnostic, not an efficiency ranking. Total work versus recall is the efficiency comparison.

@SENSITIVITY@

Sensitivity varies one parameter at a time at 2026 with the fixed primary 2024 history, not a Cartesian sweep or downstream selection. ARI versus primary quantifies proposal/weight dependence, not scientific correctness. Final preference must consider temporal continuity, independent coherence, labels and perturbation together; native representational fidelity alone does not establish retrieval superiority.

## 5. Limitations and four more weeks

The optimization is greedy and proposal-dependent; the corpus is small and dates are annual. Labels may be generic and the NLI proxy is imperfect. Claim alignment accepts exact text/source matches or explicitly inspected evaluation-only annotations; other valid claims may remain unevaluable. One assistant-reviewed Q18.C16 alignment links long-range transferability to clai_00167 and the same source article. This is no substitute for independent expert review. Partial/composite annotations also require review. The benchmark has been repeatedly examined and supplies incomplete target lists; this is development evaluation, with no held-out external question set. No production scalability or causal scientific validity is claimed.

Actual inspection found large dominant clusters (primary final L0/L1/L2 maxima: @CLUSTER_MAX@ nodes), generic phrases and occasional author-like cited-work representatives. Fixed budgets do not imply balanced or useful communities. Top retrieval documents can discuss gold methods while direct gold nodes rank below 25; an evidence hit is not automatically a target-identity hit. See `notes/output_inspection.md` for traces. The reported temporal average also hides a qualification: @LAST_TRANSITION@. These limitations are retained rather than repaired by outcome-based tuning.

With four more weeks, prioritize blinded expert label/target audits and claim alignment, precise publication-date versioning, genuinely held-out downstream questions, then scalable candidate generation if profiling justifies it. A stronger objective would follow diagnosed failures rather than benchmark-driven tuning. Reproduction instructions, dependency pins and tests are in `README.md`; `AI_USAGE.md` records the master prompt, Codex implementation, Ultralight attempts and verification. Gate K records a separate complete fresh-output run and comparison, when available, rather than treating cached historical reports as evidence.

## Figures

![Strict annual graph growth](outputs/figures/snapshot_growth.png)
![Native arity distribution](outputs/figures/arity_distribution.png)
![Independent coherence and type-preserving null](outputs/figures/coherence_vs_null.png)
![Real temporal stability](outputs/figures/temporal_stability.png)
![Perturbation confidence intervals](outputs/figures/perturbation_stability.png)
![Downstream comparison work and recall](outputs/figures/retrieval_work_tradeoff.png)
"""
    replacements = {
        "@CLUSTER_MAX@": "/".join(
            str(
                max(
                    map(
                        len,
                        read(
                            out / f"hierarchies/hierarchy_{cfg['snapshots'][-1]}.json"
                        )["partitions"][str(l)],
                    )
                )
            )
            for l in range(3)
        ),
        "@LAST_TRANSITION@": f"temporal ARI is below static at {sum(m['cross_snapshot_stability']['temporal'][list(m['cross_snapshot_stability']['temporal'])[-1]][str(l)]['ari']<m['cross_snapshot_stability']['static'][list(m['cross_snapshot_stability']['static'])[-1]][str(l)]['ari'] for l in range(3))}/3 levels in the last transition",
        "@AUDIT@": f"{audit['article_nodes']} article nodes, {ss[-1]['nodes']} nodes, {ss[-1]['hyperedges']} hyperedges, {len(ss[-1]['node_types'])} node types and {len(ss[-1]['relation_types'])} relation types; {audit['benchmark']['questions']} questions rather than the README's 17",
        "@SNAPSHOTS@": table(
            ["Year", "Nodes", "Edges", "Max arity", "Arity >2"],
            [
                [
                    s["snapshot"],
                    s["nodes"],
                    s["hyperedges"],
                    s["arity"]["max"],
                    s["arity"]["fraction_gt2"],
                ]
                for s in ss
            ],
        ),
        "@ANOMALIES@": f"{an['before_member_visibility']} edge years predate an endpoint's visibility and {an['before_asserting_article']} predate the asserting article",
        "@PROJECTION@": f"{p['original_hyperedges']} original edges produce {p['pair_instances']} pair instances and {p['unique_pairs']} unique pairs; {p['collision_pairs']} pairs ({p['collision_fraction']:.2%}) have multiple contributing edges",
        "@SUPPORT@": f"{b['target_occurrences']} target occurrences / {b['unique_targets']} unique labels; basic normalized exact matching finds {b['basic_normalized_exact_occurrences']} anywhere and {b['basic_method_only_exact_occurrences']} as method nodes, leaving {b['basic_no_literal_occurrences']} without literal identity. Refined status counts (typographic hyphens normalized) are {b['counts']}",
        "@CLAIMS@": f"{a['alignable']}/{a['total']} accepted overall, including {a['by_type']['B']['alignable']}/{a['by_type']['B']['expected']} Type-B claims",
        "@VARIANTS@": table(
            [
                "Variant",
                "Coh L0",
                "Coh L1",
                "Coh L2",
                "Temporal ARI",
                "Perturb L0",
                "Perturb L1",
                "Perturb L2",
            ],
            summary,
        ),
        "@TEMPORAL_DECISION@": f"**Computed tradeoff:** temporal versus static native changes descriptive mean cross-snapshot ARI from {summary[2][4]:.4f} to {summary[3][4]:.4f}, and mean 2026 coherence across levels from {np.mean(summary[2][1:4]):.4f} to {np.mean(summary[3][1:4]):.4f}. This {'supports a continuity benefit' if summary[3][4]>summary[2][4] else 'does not support a continuity benefit'} on this corpus. Perturbation and label quality remain separate criteria; the result does not establish universal superiority or justify downstream tuning.",
        "@LABELS@": table(
            [
                "Variant",
                "Level",
                "Labels",
                "Evaluable",
                "Unavailable",
                "Overclaim",
                "Entail",
                "Neutral",
                "Contradict",
                "Truncated",
            ],
            labels,
        ),
        "@RETRIEVAL@": table(
            [
                "System",
                "K",
                "Recovered",
                "Raw N",
                "Supported N",
                "Raw recall",
                "Conditional recall",
                "Mean comparisons",
                "First-hit median",
                "No first hit",
            ],
            retrievalrows,
        ),
        "@NAVIGATION@": f"primary beam3 recovers {primary['at_k']['10']['found']}/63 at ten using {primary['comparisons']['mean']:.1f} mean comparisons; flat recovers {flat['at_k']['10']['found']}/63 using {flat['comparisons']['mean']:.1f}",
        "@SENSITIVITY@": table(
            ["Parameter", "Value", "ARI vs primary L0", "L1", "L2"], sens
        ),
    }
    for key, value in replacements.items():
        text = text.replace(key, value)
    # Root report is submission-facing; each output directory also retains its own report.
    Path("report.md").write_text(text, encoding="utf8")
    (out / "report.md").write_text(
        text.replace("(outputs/figures/", "(figures/"), encoding="utf8"
    )
