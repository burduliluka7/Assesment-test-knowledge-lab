# Evidence-aware candidate reranking

The fixed evidence scorer improves conservative 2025 top-ten recovery from H2’s 3/47 to 6/47 canonical target instances (R3: 4/47). It brings three H2 rank 11–100 targets into the top ten and preserves all three existing H2 hits. This is a modest gain: 41/47 still fail at ten, and median gold rank worsens from 66 to 89 at K100 and 104 at K200. Enlarging the candidate pool provides no additional top-ten hits. The evidence hypothesis is only partially supported; final answer quality does not yet justify hierarchy approximation as the next priority.

These are development diagnostics on a repeatedly used benchmark, not held-out estimates. No answer-trained weights, parameter sweep, new aliases, remote relevance API, hierarchy change, or change to Type B retrieval was made. The previous [diffusion report](hypergraph_diffusion.md) and every historical metric section remain unchanged.

## Frozen protocol and evidence

`FrozenH2Candidates` composes the existing `ContextIndex`, original extractive query view, normalized BGE embeddings, native `ThetaOperator`, unchanged `MentionIndex`, entity grouping and RRF. H2 remains alpha=.85, M=100, uniform native relation weights, original convergence settings and RRF60. Existing questions replay archived full node vectors and reconstruct the complete ranking; all 28 top-200 dictionaries and per-question metric dictionaries match historical H2 exactly before scoring. New questions execute the same H2 functions. Only each frozen top-K prefix is reordered; the entire tail retains its original order. Candidate recall always uses the original H2 prefix. H3 remains an archived comparison.

`CandidateEvidenceIndex.build_candidate_evidence` uses graph-only pools, then ranks evidence by tier, specificity and original-question cosine, with stable ID tie breaks. Caps are direct claims 3, safe explicit-name claims 3, task/problem 3, technical 2, dataset/metric 2, titles 2 and paths 2. Identical normalized text cannot occupy multiple selected fields. Direct relations are limited to claims/addresses/solves/uses_technique/uses_component/evaluated_on/presents with arity ≤16. Direct items are labelled low-arity (≤3) or co-membership only; undirected incidence is not a subject–predicate assertion. Explicit names use the frozen safe-boundary MentionIndex. Only titles from presenting or provenance publications are included; whole-article claim expansion is excluded.

Genericness combines the positive-degree 90th percentile, short labels (≤3 tokens), broad node type or ≥3 source articles, and an identifier exemption. It filters context nodes, never removes candidate entities. Up to two approximate high-product paths from the twenty highest positive semantic seeds are kept within three hops. The traversal reuses native Theta transition factors, bounds retained paths per vertex, and is not exact diffusion attribution. Generic bridges remain in diagnostic paths; only non-generic claim/task/problem source text can enter semantic fields. Final path destinations are labelled target entities. Named but semantically broad nodes can escape this heuristic. `specific_evidence_count` means selected direct/name-linked non-title evidence is present, not that it entails an answer.

RR1 reuses BGE-small-en-v1.5 revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` and H2’s cached atomic question/text cosines. Negative cosines are clamped to zero; each field uses its maximum. Fixed weights are name .10, direct claims .20, explicit mentions .20, task/problem .20, technical .10, dataset .10, article .05, path source .05. These are new predeclared reranker defaults, not inherited R3 weights or fitted values.

`score = (sum_available(weight × field_cosine) / sum_available(weight)) × (0.5 + 0.5 × sqrt(sum_available(weight) / sum_all(weight)))`. Missing fields are excluded from the semantic denominator, while coverage mildly discounts sparse evidence. The discount can still disadvantage sparse graph extractions; it does not repair them. Metadata H2/R3 rank and score never enter semantic text. RR3 fuses H2 with RR1 using the existing RRF60 and average ranks for exact-score ties. “Best available” means RR1 because no eligible RR2 is local, not retrospective score-based selection.

RR2 is **unavailable**. The local cache contains BGE, MiniLM and a three-way NLI DeBERTa classifier; none is a relevance cross-encoder. `CandidateReranker` and `LocalCrossEncoderReranker` supply the optional interface, explicit local-only loaders and whole-block token packing at a 480-token pair budget, with included/omitted blocks and token counts. No RR2 metric is fabricated and no model is downloaded. The CE path is covered by loader/packing safeguards, not a real-model relevance experiment.

All 14 Type A questions appear in every stage and cutoff; Q5 (Bead-mapping, GNN+GPP) and Q11 (CE→MC→NN→PF, hybrid frameworks) have no canonical named-entity evaluation units, so canonical recall is explicitly null while their supplied labels remain in the raw /63 denominator. Canonical counts are 47 instances/34 unique labels at 2025 and 50/37 at 2026. The unchanged resolver records individual composite, evidence-backed, ambiguous or unresolved reasons; noncanonical targets are not silently dropped.

## Quality and cost: 2025

| Variant | K | Hit1 | Hit5 | Hit10 | Hit20 | MacroR5 | MacroR10 | MacroR20 | MicroR5 | MicroR10 | MicroR20 | UniqueR10 | Raw /63 | MRR | R-Prec | R@R | Median rank | Mean rank | RR calls | RR seconds | Total seconds* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R3 | — | 0.0833 | 0.1667 | 0.3333 | 0.3333 | 0.0917 | 0.2028 | 0.2028 | 0.0426 | 0.0851 | 0.0851 | 0.1176 | 4/63 | 0.1589 | 0.0917 | 0.0917 | 160.0000 | 378.8723 | 0 | 0 | 0.4295 |
| H3 | — | 0.0833 | 0.1667 | 0.2500 | 0.3333 | 0.1667 | 0.1750 | 0.2028 | 0.0426 | 0.0638 | 0.0851 | 0.0882 | 3/63 | 0.1587 | 0.0917 | 0.0917 | 59.0000 | 102.5106 | 0 | 0 | 0.4798 |
| RR0 | — | 0.0833 | 0.1667 | 0.2500 | 0.3333 | 0.1667 | 0.1750 | 0.2028 | 0.0426 | 0.0638 | 0.0851 | 0.0882 | 3/63 | 0.1602 | 0.0917 | 0.0917 | 66.0000 | 103.4043 | 0.0000 | 0.0000 | 0.5225 |
| RR1-100 | 100 | 0.2500 | 0.3333 | 0.5000 | 0.5833 | 0.2028 | 0.2208 | 0.3041 | 0.0851 | 0.1277 | 0.1489 | 0.1765 | 6/63 | 0.3278 | 0.2208 | 0.2208 | 89.0000 | 107.7872 | 100.0000 | 0.0019 | 1.4108 |
| RR1-200 | 200 | 0.2500 | 0.3333 | 0.5000 | 0.5833 | 0.2028 | 0.2208 | 0.3041 | 0.0851 | 0.1277 | 0.1489 | 0.1765 | 6/63 | 0.3281 | 0.2208 | 0.2208 | 104.0000 | 123.2979 | 200.0000 | 0.0033 | 1.5000 |
| RR3-100 | 100 | 0.0833 | 0.3333 | 0.4167 | 0.5833 | 0.2028 | 0.2104 | 0.3041 | 0.0851 | 0.1064 | 0.1489 | 0.1471 | 5/63 | 0.2051 | 0.0992 | 0.0992 | 71.0000 | 106.1702 | 100.0000 | 0.0024 | 1.4114 |
| RR3-200 | 200 | 0.0833 | 0.3333 | 0.4167 | 0.6667 | 0.2028 | 0.2104 | 0.3160 | 0.0851 | 0.1064 | 0.1702 | 0.1471 | 5/63 | 0.2063 | 0.0992 | 0.0992 | 78.0000 | 111.2766 | 200.0000 | 0.0042 | 1.5009 |

RR2-100 and RR2-200: unavailable; no quality/cost values. Hit and macro recall average the 12 evaluable questions; costs average all 14. Unique recall is the union of canonical target names recovered across associated questions. R-Precision and Recall@R coincide for these distinct grouped answer units.

| Variant | Frozen C100 (macro) | Frozen C200 (macro) | Candidate preservation | Evidence selection seconds | Unique cached scores consulted | New evidence dots |
| --- | --- | --- | --- | --- | --- | --- |
| RR0 | 0.7146 | 0.9674 | 1.0000 | 0.0000 | 0.0000 | 0 |
| RR1-100 | 0.7146 | 0.9674 | 1.0000 | 0.8865 | 1585.5714 | 0 |
| RR1-200 | 0.7146 | 0.9674 | 1.0000 | 0.9742 | 1959.4286 | 0 |
| RR3-100 | 0.7146 | 0.9674 | 1.0000 | 0.8865 | 1585.5714 | 0 |
| RR3-200 | 0.7146 | 0.9674 | 1.0000 | 0.9742 | 1959.4286 | 0 |

## Quality and cost: 2026

| Variant | K | Hit1 | Hit5 | Hit10 | Hit20 | MacroR5 | MacroR10 | MacroR20 | MicroR5 | MicroR10 | MicroR20 | UniqueR10 | Raw /63 | MRR | R-Prec | R@R | Median rank | Mean rank | RR calls | RR seconds | Total seconds* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R3 | — | 0.0833 | 0.1667 | 0.2500 | 0.3333 | 0.0909 | 0.1742 | 0.2020 | 0.0400 | 0.0600 | 0.0800 | 0.0811 | 3/63 | 0.1546 | 0.0909 | 0.0909 | 181.5000 | 333.0400 | 0 | 0 | 0.5050 |
| H3 | — | 0.0833 | 0.2500 | 0.2500 | 0.3333 | 0.1742 | 0.1742 | 0.2020 | 0.0600 | 0.0600 | 0.0800 | 0.0811 | 3/63 | 0.1634 | 0.0909 | 0.0909 | 71.5000 | 101.0400 | 0 | 0 | 0.5626 |
| RR0 | — | 0.0833 | 0.2500 | 0.2500 | 0.3333 | 0.1742 | 0.1742 | 0.2020 | 0.0600 | 0.0600 | 0.0800 | 0.0811 | 3/63 | 0.1694 | 0.0909 | 0.0909 | 73.0000 | 98.9600 | 0.0000 | 0.0000 | 0.6156 |
| RR1-100 | 100 | 0.1667 | 0.4167 | 0.4167 | 0.6667 | 0.2124 | 0.2124 | 0.3141 | 0.1000 | 0.1000 | 0.1600 | 0.1351 | 5/63 | 0.2613 | 0.2188 | 0.2188 | 90.5000 | 101.8200 | 100.0000 | 0.0014 | 1.4854 |
| RR1-200 | 200 | 0.1667 | 0.4167 | 0.4167 | 0.6667 | 0.2124 | 0.2124 | 0.3141 | 0.1000 | 0.1000 | 0.1600 | 0.1351 | 5/63 | 0.2616 | 0.2188 | 0.2188 | 90.0000 | 116.5200 | 200.0000 | 0.0025 | 1.5590 |
| RR3-100 | 100 | 0.0833 | 0.2500 | 0.4167 | 0.6667 | 0.1742 | 0.2124 | 0.3141 | 0.0600 | 0.1000 | 0.1600 | 0.1351 | 5/63 | 0.2132 | 0.1077 | 0.1077 | 77.0000 | 101.3400 | 100.0000 | 0.0017 | 1.4856 |
| RR3-200 | 200 | 0.0833 | 0.2500 | 0.4167 | 0.6667 | 0.1742 | 0.2124 | 0.3141 | 0.0600 | 0.1000 | 0.1600 | 0.1351 | 5/63 | 0.2145 | 0.1077 | 0.1077 | 82.0000 | 106.3800 | 200.0000 | 0.0030 | 1.5595 |

RR2-100 and RR2-200: unavailable; no quality/cost values. Hit and macro recall average the 12 evaluable questions; costs average all 14. Unique recall is the union of canonical target names recovered across associated questions. R-Precision and Recall@R coincide for these distinct grouped answer units.

| Variant | Frozen C100 (macro) | Frozen C200 (macro) | Candidate preservation | Evidence selection seconds | Unique cached scores consulted | New evidence dots |
| --- | --- | --- | --- | --- | --- | --- |
| RR0 | 0.7040 | 0.8859 | 1.0000 | 0.0000 | 0.0000 | 0 |
| RR1-100 | 0.7040 | 0.8859 | 1.0000 | 0.8684 | 1635.1429 | 0 |
| RR1-200 | 0.7040 | 0.8859 | 1.0000 | 0.9409 | 2125.9286 | 0 |
| RR3-100 | 0.7040 | 0.8859 | 1.0000 | 0.8684 | 1635.1429 | 0 |
| RR3-200 | 0.7040 | 0.8859 | 1.0000 | 0.9409 | 2125.9286 | 0 |

*Total seconds is a component-sum estimate: measured historical H2 search time + current path/evidence construction + current reranking, not a fresh end-to-end H2 execution. Cold model loading, snapshot/index building, embedding encoding and evaluation/export are excluded. Path traversal is shared per query; K100 is charged shared traversal plus its first 100 bundle builds, K200 all 200. Scalar reranker calls double, but cached dot products do not. Existing H2 search comparisons remain in RR0 cost; new reranker comparisons are separate. Query encoding/replay wall time is not substituted for historical search timing. RR3 cost includes scoring and fusion. Timings are indicative single-run measurements; part of the final run overlapped test verification, so these are not controlled latency benchmarks.

## Candidate-pool oracle

The evaluation-only oracle maximizes distinct canonical coverage in ten candidate slots using exact finite subset coverage. Alias labels merge before optimization; raw label-instance recovery is recorded separately. It never feeds expected labels or gold membership to evidence/scoring. For distinct answer groups and >10 golds, ten slots cannot recover all targets. Missing candidates and top-ten capacity both limit this diagnostic.

| Cutoff | K | Oracle found | Micro oracle R10 | Macro oracle R10 | RR1 R10 | Gap (micro) | RR3 R10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2025 | 100 | 31/47 | 0.6596 | 0.7146 | 0.1277 | 0.5319 | 0.1064 |
| 2025 | 200 | 43/47 | 0.9149 | 0.9674 | 0.1277 | 0.7872 | 0.1064 |
| 2026 | 100 | 32/50 | 0.6400 | 0.7040 | 0.1000 | 0.5400 | 0.1000 |
| 2026 | 200 | 44/50 | 0.8800 | 0.8794 | 0.1000 | 0.7800 | 0.1000 |

| Cutoff | Question | Canonical targets | Present100 | Oracle100 | Present200 | Oracle200 |
| --- | --- | --- | --- | --- | --- | --- |
| 2025 | Q1 | 7 | 5 | 0.7143 | 7 | 1.0000 |
| 2025 | Q2 | 3 | 3 | 1.0000 | 3 | 1.0000 |
| 2025 | Q3 | 1 | 1 | 1.0000 | 1 | 1.0000 |
| 2025 | Q4 | 1 | 1 | 1.0000 | 1 | 1.0000 |
| 2025 | Q5 | 0 | 0 | — | 0 | — |
| 2025 | Q6 | 1 | 1 | 1.0000 | 1 | 1.0000 |
| 2025 | Q7 | 1 | 0 | 0.0000 | 1 | 1.0000 |
| 2025 | Q8 | 1 | 1 | 1.0000 | 1 | 1.0000 |
| 2025 | Q9 | 2 | 0 | 0.0000 | 2 | 1.0000 |
| 2025 | Q10 | 1 | 1 | 1.0000 | 1 | 1.0000 |
| 2025 | Q11 | 0 | 0 | — | 0 | — |
| 2025 | Q12 | 11 | 7 | 0.6364 | 10 | 0.9091 |
| 2025 | Q13 | 8 | 5 | 0.6250 | 8 | 1.0000 |
| 2025 | Q14 | 10 | 6 | 0.6000 | 7 | 0.7000 |
| 2026 | Q1 | 7 | 4 | 0.5714 | 7 | 1.0000 |
| 2026 | Q2 | 3 | 3 | 1.0000 | 3 | 1.0000 |
| 2026 | Q3 | 1 | 1 | 1.0000 | 1 | 1.0000 |
| 2026 | Q4 | 1 | 1 | 1.0000 | 1 | 1.0000 |
| 2026 | Q5 | 0 | 0 | — | 0 | — |
| 2026 | Q6 | 1 | 1 | 1.0000 | 1 | 1.0000 |
| 2026 | Q7 | 1 | 0 | 0.0000 | 0 | 0.0000 |
| 2026 | Q8 | 1 | 1 | 1.0000 | 1 | 1.0000 |
| 2026 | Q9 | 2 | 0 | 0.0000 | 2 | 1.0000 |
| 2026 | Q10 | 1 | 1 | 1.0000 | 1 | 1.0000 |
| 2026 | Q11 | 0 | 0 | — | 0 | — |
| 2026 | Q12 | 13 | 8 | 0.6154 | 11 | 0.7692 |
| 2026 | Q13 | 8 | 5 | 0.6250 | 7 | 0.8750 |
| 2026 | Q14 | 11 | 7 | 0.6364 | 10 | 0.9091 |

## Evidence availability and failure categories

| Cutoff | Group | Candidate/question pairs | Specific evidence present | Fraction |
| --- | --- | --- | --- | --- |
| 2025 | gold | 43 | 41 | 0.9535 |
| 2025 | not in answer key | 2757 | 2350 | 0.8524 |
| 2026 | gold | 45 | 43 | 0.9556 |
| 2026 | not in answer key | 2755 | 2398 | 0.8704 |

Counts above are candidate-group/question pairs, not raw labels or globally unique entities. At 2025, 41/43 in-pool canonical target instances have selected specific evidence; at 2026, 43/45 do. Presence is a structural proxy. Non-gold means absent from the benchmark answer list, not scientifically incorrect.

| Cutoff | Group | Field | Mean selected | Mean available before cap | Presence fraction | Mean best cosine if available |
| --- | --- | --- | --- | --- | --- | --- |
| 2025 | gold | direct_claims | 1.2558 | 4.6744 | 0.4186 | 0.7156 |
| 2025 | gold | explicit_mentions | 2.3488 | 8.9767 | 0.8372 | 0.7266 |
| 2025 | gold | task_problem | 0.4186 | 1.7209 | 0.1395 | 0.7682 |
| 2025 | gold | technical | 0.5116 | 3.7674 | 0.2558 | 0.6362 |
| 2025 | gold | dataset | 0.5116 | 3.6744 | 0.2558 | 0.6302 |
| 2025 | gold | article | 1.6744 | 4.4884 | 1.0000 | 0.7625 |
| 2025 | gold | path_evidence | 0.6279 | 0.6279 | 0.6047 | 0.8194 |
| 2025 | not gold | direct_claims | 1.1284 | 5.1440 | 0.3761 | 0.7144 |
| 2025 | not gold | explicit_mentions | 0.6768 | 1.8796 | 0.3250 | 0.6843 |
| 2025 | not gold | task_problem | 0.5136 | 2.0116 | 0.1712 | 0.7272 |
| 2025 | not gold | technical | 0.5209 | 3.7048 | 0.2604 | 0.6679 |
| 2025 | not gold | dataset | 0.5245 | 2.2818 | 0.2713 | 0.6326 |
| 2025 | not gold | article | 0.9768 | 1.5477 | 0.7668 | 0.7182 |
| 2025 | not gold | path_evidence | 1.1143 | 1.1143 | 0.7515 | 0.8028 |
| 2026 | gold | direct_claims | 1.2000 | 4.4667 | 0.4000 | 0.7156 |
| 2026 | gold | explicit_mentions | 2.3333 | 9.3778 | 0.8222 | 0.7343 |
| 2026 | gold | task_problem | 0.4000 | 1.6444 | 0.1333 | 0.7682 |
| 2026 | gold | technical | 0.4889 | 3.6000 | 0.2444 | 0.6362 |
| 2026 | gold | dataset | 0.6222 | 3.9333 | 0.3111 | 0.6435 |
| 2026 | gold | article | 1.7111 | 5.4889 | 1.0000 | 0.7714 |
| 2026 | gold | path_evidence | 0.6667 | 0.6667 | 0.6222 | 0.8248 |
| 2026 | not gold | direct_claims | 1.2490 | 5.7016 | 0.4163 | 0.7185 |
| 2026 | not gold | explicit_mentions | 0.6958 | 2.0359 | 0.3176 | 0.6867 |
| 2026 | not gold | task_problem | 0.5423 | 2.1303 | 0.1808 | 0.7294 |
| 2026 | not gold | technical | 0.4980 | 3.4301 | 0.2490 | 0.6698 |
| 2026 | not gold | dataset | 0.5721 | 2.7597 | 0.2933 | 0.6308 |
| 2026 | not gold | article | 0.9673 | 1.7020 | 0.7466 | 0.7203 |
| 2026 | not gold | path_evidence | 1.0817 | 1.0817 | 0.7670 | 0.8032 |

Full min/quartile/median/max count distributions and per-candidate similarities are in `reranker_failure_analysis.json`. Counts after selection reflect genericness filtering and cross-field deduplication; they do not prove the original graph lacks useful evidence.

| Cutoff | Group | Low-arity selected | Co-membership only selected | Explicit-name selected | Indirect source via generic bridge | Diagnostic paths via generic bridge |
| --- | --- | --- | --- | --- | --- | --- |
| 2025 | gold | 84 | 42 | 101 | 2 | 4 |
| 2025 | not gold | 5008 | 3032 | 1866 | 1761 | 2410 |
| 2026 | gold | 88 | 44 | 105 | 2 | 6 |
| 2026 | not gold | 5538 | 3045 | 1917 | 1667 | 2387 |

Paths through generic bridges are retained and flagged, not recast as direct linkage. Their non-generic source text can still contribute the fixed .05 path-field weight. This is weaker attribution than direct or explicit-name evidence, and high source similarity is partly induced by semantic seed selection. Co-membership-only text is also explicitly marked in the compact document. Neither diagnostic is used to tune weights after seeing outcomes.

| Cutoff | Stage | NOT_IN_H2_100 | NOT_IN_H2_200 | IN_CANDIDATES_NO_SPECIFIC_EVIDENCE | IN_CANDIDATES_EVIDENCE_PRESENT_RERANK_FAILED | RERANK_IMPROVED_BUT_BELOW_10 | RETRIEVED_TOP10 | RERANK_DEGRADED_EXISTING_GOOD_RESULT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025 | RR1-100 | 16 | 0 | 0 | 16 | 9 | 6 | 0 |
| 2025 | RR1-200 | 0 | 4 | 2 | 22 | 13 | 6 | 0 |
| 2025 | RR3-100 | 16 | 0 | 0 | 16 | 10 | 5 | 0 |
| 2025 | RR3-200 | 0 | 4 | 2 | 21 | 15 | 5 | 0 |
| 2026 | RR1-100 | 18 | 0 | 0 | 16 | 11 | 5 | 0 |
| 2026 | RR1-200 | 0 | 5 | 2 | 22 | 16 | 5 | 0 |
| 2026 | RR3-100 | 18 | 0 | 0 | 16 | 11 | 5 | 0 |
| 2026 | RR3-200 | 0 | 5 | 2 | 22 | 16 | 5 | 0 |

Categories are mutually exclusive per stage/pool. Absence from that H2 pool takes precedence; top-ten recovery then takes precedence over degradation. A good R3/H2 result is a top-ten target. Remaining in-pool targets are classified by selected specific evidence and rank direction. “Evidence present/rerank failed” is operational, not a judgment that the evidence entails the answer.

## Gains and degradation

| Cutoff | Stage | Original bin | Targets | Improved | Unchanged | Worsened | Now top10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2025 | RR1-100 | R3 top10 | 4 | 3 | 1 | 0 | 4 |
| 2025 | RR1-100 | H2 top10 | 3 | 2 | 1 | 0 | 3 |
| 2025 | RR1-100 | H2 11-50 | 16 | 6 | 0 | 10 | 3 |
| 2025 | RR1-100 | H2 51-100 | 12 | 6 | 0 | 6 | 0 |
| 2025 | RR1-100 | H2 101-200 | 12 | 0 | 12 | 0 | 0 |
| 2025 | RR1-200 | R3 top10 | 4 | 3 | 1 | 0 | 4 |
| 2025 | RR1-200 | H2 top10 | 3 | 2 | 1 | 0 | 3 |
| 2025 | RR1-200 | H2 11-50 | 16 | 6 | 0 | 10 | 3 |
| 2025 | RR1-200 | H2 51-100 | 12 | 5 | 0 | 7 | 0 |
| 2025 | RR1-200 | H2 101-200 | 12 | 5 | 0 | 7 | 0 |
| 2025 | RR3-100 | R3 top10 | 4 | 2 | 1 | 1 | 4 |
| 2025 | RR3-100 | H2 top10 | 3 | 1 | 2 | 0 | 3 |
| 2025 | RR3-100 | H2 11-50 | 16 | 7 | 0 | 9 | 2 |
| 2025 | RR3-100 | H2 51-100 | 12 | 5 | 1 | 6 | 0 |
| 2025 | RR3-100 | H2 101-200 | 12 | 0 | 12 | 0 | 0 |
| 2025 | RR3-200 | R3 top10 | 4 | 2 | 1 | 1 | 4 |
| 2025 | RR3-200 | H2 top10 | 3 | 1 | 2 | 0 | 3 |
| 2025 | RR3-200 | H2 11-50 | 16 | 7 | 0 | 9 | 2 |
| 2025 | RR3-200 | H2 51-100 | 12 | 5 | 0 | 7 | 0 |
| 2025 | RR3-200 | H2 101-200 | 12 | 5 | 0 | 7 | 0 |
| 2026 | RR1-100 | R3 top10 | 3 | 1 | 1 | 1 | 3 |
| 2026 | RR1-100 | H2 top10 | 3 | 2 | 1 | 0 | 3 |
| 2026 | RR1-100 | H2 11-50 | 15 | 6 | 0 | 9 | 2 |
| 2026 | RR1-100 | H2 51-100 | 14 | 7 | 0 | 7 | 0 |
| 2026 | RR1-100 | H2 101-200 | 13 | 0 | 13 | 0 | 0 |
| 2026 | RR1-200 | R3 top10 | 3 | 1 | 1 | 1 | 3 |
| 2026 | RR1-200 | H2 top10 | 3 | 2 | 1 | 0 | 3 |
| 2026 | RR1-200 | H2 11-50 | 15 | 6 | 0 | 9 | 2 |
| 2026 | RR1-200 | H2 51-100 | 14 | 6 | 0 | 8 | 0 |
| 2026 | RR1-200 | H2 101-200 | 13 | 6 | 0 | 7 | 0 |
| 2026 | RR3-100 | R3 top10 | 3 | 1 | 2 | 0 | 3 |
| 2026 | RR3-100 | H2 top10 | 3 | 1 | 2 | 0 | 3 |
| 2026 | RR3-100 | H2 11-50 | 15 | 7 | 0 | 8 | 2 |
| 2026 | RR3-100 | H2 51-100 | 14 | 6 | 0 | 8 | 0 |
| 2026 | RR3-100 | H2 101-200 | 13 | 0 | 13 | 0 | 0 |
| 2026 | RR3-200 | R3 top10 | 3 | 1 | 2 | 0 | 3 |
| 2026 | RR3-200 | H2 top10 | 3 | 1 | 2 | 0 | 3 |
| 2026 | RR3-200 | H2 11-50 | 15 | 7 | 0 | 8 | 2 |
| 2026 | RR3-200 | H2 51-100 | 14 | 6 | 0 | 8 | 0 |
| 2026 | RR3-200 | H2 101-200 | 13 | 5 | 0 | 8 | 0 |

R3-bin movement is relative to R3; H2 bins are relative to H2. Bins overlap across reference systems. In 2025 all four original R3 top-ten targets and all three H2 top-ten targets are recovered by both RR1 and RR3; substantial degradation nevertheless occurs among H2’s rank 11–100 targets. RR3 moderates overall displacement but also keeps SchNet at rank 11 instead of RR1’s 8.

## Actual evidence traces

### Improvement: Q2, D4FT

Question: Which methods (by Feb 2026) are best suited for accelerating electronic structure calculations when self-consistent field iterations become prohibitively expensive for large systems (>10³ atoms)?

R3 → H2 → RR1-100/200 → RR3-100/200: 10 → 12 → 2/2 → 4/4. RR1 score 0.6926; available semantic mean 0.7015; coverage 0.95.

| Field | Node | Support | Cosine | Contribution | Selected maximum-scoring text |
| --- | --- | --- | --- | --- | --- |
| direct_claims | clai_00427 | direct:claims | 0.7487 | 0.1556 | Incorporating a neural local scaling transformation as a learnable basis (preserving the overlap matrix) enhances wave‑function expressiveness and yields lower LSDA ground‑state energies for atoms (He, Li, Be, C, N, O) compared with the raw STO‑3g basis. |
| explicit_mentions | clai_00430 | safe_explicit_name | 0.7081 | 0.1472 | D4FT successfully converges on unstable fullerene fragments (C₁₂₀, C₁₄₀) where PySCF fails to converge within the default iteration limit. |
| task_problem | prob_00156 | direct:solves | 0.7840 | 0.1630 | Inefficiency of CPU‑bound quantum chemistry packages |
| technical | comp_00280 | direct:uses_component | 0.6508 | 0.0676 | GPU‑accelerated tensor backend |
| dataset | metr_00032 | direct:evaluated_on | 0.6523 | 0.0678 | ground state energy (Ha) |
| article | arti_00019 | direct:presents | 0.6816 | 0.0354 | D4FT: A DEEP LEARNING APPROACH TO KOHN-SHAM DENSITY FUNCTIONAL THEORY |

These are actual score inputs and additive contributions, not causal proof from an evidence-removal ablation. Full selected texts, source years, provenance, discarded items and approximate paths are in `candidate_evidence.json`.

### Improvement: Q12, M3GNet

Question: What are the leading (by Feb 2026) ML-based interatomic potential methods for large-scale atomistic simulation?

R3 → H2 → RR1-100/200 → RR3-100/200: 97 → 29 → 8/8 → 10/10. RR1 score 0.6817; available semantic mean 0.7423; coverage 0.70.

| Field | Node | Support | Cosine | Contribution | Selected maximum-scoring text |
| --- | --- | --- | --- | --- | --- |
| direct_claims | clai_00646 | direct:claims | 0.7534 | 0.1977 | M3GNet molecular‑dynamics simulations of Li₃YCl₆ reproduce ionic conductivity (14.3 mS cm⁻¹) and activation energy (0.22 eV) consistent with ab‑initio MD, and short‑time MD correctly identifies known lithium superionic conductors by their high mean‑squared displacements at elevated temperatures. |
| explicit_mentions | clai_00343 | safe_explicit_name | 0.8238 | 0.2162 | The SevenNet machine-learned interatomic potential outperforms the earlier M3GNet model on the Matbench Discovery benchmark. |
| technical | comp_00483 | direct:uses_component | 0.6009 | 0.0788 | MLIP-NEB relaxation framework |
| article | arti_00050 | provenance_title_fallback | 0.8258 | 0.0542 | Machine Learning Interatomic Potentials: A Review and Framework for Understanding MLIPs |
| path_evidence | clai_00350 | short_path_source | 0.8372 | 0.0549 | Using an ML‑trained interatomic potential (ML‑IAP) enables validation of many more candidate structures than conventional AIMD, as demonstrated by MD simulations on 30 random and 100 top‑Ξ structures. |

These are actual score inputs and additive contributions, not causal proof from an evidence-removal ablation. Full selected texts, source years, provenance, discarded items and approximate paths are in `candidate_evidence.json`.

### Improvement: Q13, SchNet

Question: What are the leading (by Feb 2026) GNN architectures for predicting properties of solid-state crystalline materials?

R3 → H2 → RR1-100/200 → RR3-100/200: 34 → 36 → 8/8 → 11/11. RR1 score 0.6853; available semantic mean 0.7033; coverage 0.90.

| Field | Node | Support | Cosine | Contribution | Selected maximum-scoring text |
| --- | --- | --- | --- | --- | --- |
| direct_claims | clai_00083 | direct:claims | 0.7063 | 0.1529 | Including force information in the training loss causally improves SchNet’s generalization to previously unseen chemical structures in ISO17. |
| explicit_mentions | clai_00490 | safe_explicit_name | 0.7603 | 0.1646 | MEGNet models trained on approximately 60,000 crystals outperform prior models (SchNet and CGCNN) for formation energy, band gap, bulk modulus, and shear modulus, achieving errors comparable to or better than density‑functional‑theory uncertainties. |
| task_problem | task_00025 | direct:addresses | 0.6871 | 0.1488 | Interatomic force prediction |
| technical | comp_00022 | direct:uses_component | 0.7384 | 0.0799 | Atom-wise dense layers |
| article | arti_00002 | direct:presents | 0.7183 | 0.0389 | SchNet: A continuous-filter convolutional neural network for modeling quantum interactions |
| path_evidence | clai_00146 | short_path_source | 0.8254 | 0.0447 | The E(3)‑equivariant GNN surrogate provides superior accuracy compared with third‑ and fifth‑order Gaussian quadrature for thermal expansion of copper, martensitic phase transition of iron, and grain‑boundary energy calculations, matching molecular dynamics reference data. |

These are actual score inputs and additive contributions, not causal proof from an evidence-removal ablation. Full selected texts, source years, provenance, discarded items and approximate paths are in `candidate_evidence.json`.

D4FT benefits from task/claim evidence about self-consistency, scaling and electronic structure. M3GNet and SchNet gain from relevant claims across selected fields. However, their highest explicit-mention text sometimes states that a *different* model outperforms them. Max-cosine pooling cannot identify the favored subject or comparison polarity; a recovered benchmark answer is therefore not proof of constraint-level reasoning.

### Remaining failure/degradation: Q2, HamGNN

Question: Which methods (by Feb 2026) are best suited for accelerating electronic structure calculations when self-consistent field iterations become prohibitively expensive for large systems (>10³ atoms)?

R3 → H2 → RR1-100/200 → RR3-100/200: 90 → 24 → 91/136 → 49/59. RR1 score 0.5442; available semantic mean 0.6838; coverage 0.35.

| Field | Node | Support | Cosine | Contribution | Selected maximum-scoring text |
| --- | --- | --- | --- | --- | --- |
| direct_claims | clai_00199 | direct:claims | 0.7425 | 0.3376 | Incorporating orbital‑energy loss in a second fine‑tuning round markedly improves band‑structure predictions, yielding energy bands and Fermi surfaces that closely match DFT calculations for the test set. |
| article | arti_00029 | direct:presents | 0.6679 | 0.0759 | Universal Machine Learning Kohn-Sham Hamiltonian for Materials |

These are actual score inputs and additive contributions, not causal proof from an evidence-removal ablation. Full selected texts, source years, provenance, discarded items and approximate paths are in `candidate_evidence.json`.

### Remaining failure/degradation: Q1, MACE

Question: Which methods (by Feb 2026) are best suited for predicting interatomic potentials when balancing accuracy close to DFT with computational efficiency for systems of millions of atoms?

R3 → H2 → RR1-100/200 → RR3-100/200: 333 → 114 → 114/145 → 114/133. RR1 score 0.5431; available semantic mean 0.6654; coverage 0.40.

| Field | Node | Support | Cosine | Contribution | Selected maximum-scoring text |
| --- | --- | --- | --- | --- | --- |
| explicit_mentions | clai_00604 | safe_explicit_name | 0.7050 | 0.2877 | On the 3BPA molecule energy profile, MACE predictions are closest to the DFT ground truth across three dihedral slices, outperforming BOTNet and NequIP. |
| article | arti_00050 | provenance_title_fallback | 0.8009 | 0.0817 | Machine Learning Interatomic Potentials: A Review and Framework for Understanding MLIPs |
| path_evidence | clai_00343 | short_path_source | 0.8184 | 0.0835 | The SevenNet machine-learned interatomic potential outperforms the earlier M3GNet model on the Matbench Discovery benchmark. |

These are actual score inputs and additive contributions, not causal proof from an evidence-removal ablation. Full selected texts, source years, provenance, discarded items and approximate paths are in `candidate_evidence.json`.

HamGNN is an example of an available gold candidate demoted by scoring, while MACE has selected explicit mentions but sparse direct/task fields. The availability denominator avoids zero-filled penalties, yet the fixed coverage confidence still discounts sparse bundles. High-similarity path-source text may describe a neighbor’s comparative result. These observations identify extraction/attribution ambiguity and cosine weakness; the experiment cannot causally apportion them.

### Suppressed unjudged candidate: Q1, MLIP_framework_analysis

H2 rank 10 → RR1-200 rank 105; score 0.5664, coverage 0.35. This entry is not a canonical gold answer for this question. It remains **unjudged**, not a proven false positive. The question requests scientific methods/models; article/framework nodes can be topically related without identifying the requested method. Its complete selected semantic input is:

```text
Candidate: MLIP_framework_analysis
Types: method
direct_claims [low_arity_direct]: ColabFit Exchange provides open-access datasets for data-driven interatomic potentials.
direct_claims [low_arity_direct]: U-MLIPs systematically underpredict energies and forces due to softening of the potential energy surface, attributed to biased sampling in training datasets from DFT ionic relaxations near local PES minima.
direct_claims [low_arity_direct]: Elemental-SDNNFF produces accurate forces for Heusler alloys constituting 55 different elements and accurate predictions of phonon properties.
article [low_arity_direct]: Machine Learning Interatomic Potentials: A Review and Framework for Understanding MLIPs
Graph support: presents -> candidate
Graph support: extends -> extends -> candidate
```

### Suppressed unjudged candidate: Q12, MLIP_framework_analysis

H2 rank 10 → RR1-200 rank 81; score 0.5878, coverage 0.35. This entry is not a canonical gold answer for this question. It remains **unjudged**, not a proven false positive. The question requests scientific methods/models; article/framework nodes can be topically related without identifying the requested method. Its complete selected semantic input is:

```text
Candidate: MLIP_framework_analysis
Types: method
direct_claims [low_arity_direct]: ColabFit Exchange provides open-access datasets for data-driven interatomic potentials.
direct_claims [low_arity_direct]: U-MLIPs systematically underpredict energies and forces due to softening of the potential energy surface, attributed to biased sampling in training datasets from DFT ionic relaxations near local PES minima.
direct_claims [low_arity_direct]: Elemental-SDNNFF produces accurate forces for Heusler alloys constituting 55 different elements and accurate predictions of phonon properties.
article [low_arity_direct]: Machine Learning Interatomic Potentials: A Review and Framework for Understanding MLIPs
Graph support: presents -> candidate
Graph support: extends -> extends -> candidate
```

Sparse or less query-specific selected fields explain these demotions numerically; no external scientific adjudication or counterfactual evidence test was conducted.

## Every canonical target

| Cutoff | Q | Target | R3 | H2 | RR1-100 | RR1-200 | RR3-100 | RR3-200 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025 | Q1 | MACE | 333 | 114 | 114 | 145 | 114 | 133 |
| 2025 | Q1 | M3GNet | 101 | 26 | 22 | 22 | 21 | 19 |
| 2025 | Q1 | CHGNet | 119 | 88 | 100 | 187 | 99 | 130 |
| 2025 | Q1 | GAP | 206 | 47 | 93 | 140 | 76 | 81 |
| 2025 | Q1 | NequIP | 188 | 91 | 37 | 40 | 68 | 62 |
| 2025 | Q1 | ACE | 269 | 154 | 154 | 171 | 154 | 175 |
| 2025 | Q1 | EquiformerV2 | 23 | 42 | 75 | 99 | 64 | 69 |
| 2025 | Q2 | DeepH | 260 | 42 | 74 | 104 | 64 | 71 |
| 2025 | Q2 | D4FT | 10 | 12 | 2 | 2 | 4 | 4 |
| 2025 | Q2 | HamGNN | 90 | 24 | 91 | 136 | 49 | 59 |
| 2025 | Q3 | DeepH-E3 | 1 | 1 | 1 | 1 | 1 | 1 |
| 2025 | Q4 | VGNN | 2292 | 65 | 39 | 46 | 53 | 51 |
| 2025 | Q6 | LiFlow | 8 | 2 | 1 | 1 | 2 | 2 |
| 2025 | Q7 | symbolic regression | 159 | 186 | 186 | 150 | 186 | 178 |
| 2025 | Q8 | ConvLSTM | 138 | 25 | 28 | 28 | 23 | 23 |
| 2025 | Q9 | E(3)-equivariant GNNs | 2266 | 119 | 119 | 133 | 119 | 133 |
| 2025 | Q9 | NequIP | 196 | 111 | 111 | 53 | 111 | 78 |
| 2025 | Q10 | SchNet | 28 | 23 | 16 | 17 | 16 | 15 |
| 2025 | Q12 | MACE | 334 | 104 | 104 | 133 | 104 | 122 |
| 2025 | Q12 | NequIP | 216 | 87 | 45 | 50 | 67 | 67 |
| 2025 | Q12 | CHGNet | 107 | 73 | 97 | 173 | 92 | 106 |
| 2025 | Q12 | M3GNet | 97 | 29 | 8 | 8 | 10 | 10 |
| 2025 | Q12 | SevenNet | 175 | 41 | 51 | 57 | 50 | 48 |
| 2025 | Q12 | GAP | 215 | 50 | 89 | 140 | 74 | 86 |
| 2025 | Q12 | DeePMD | 229 | 111 | 111 | 110 | 111 | 110 |
| 2025 | Q12 | MatterSim | 130 | 35 | 68 | 79 | 56 | 58 |
| 2025 | Q12 | Orb | 398 | 130 | 130 | 188 | 130 | 170 |
| 2025 | Q12 | eSEN | 2320 | 283 | 283 | 283 | 283 | 283 |
| 2025 | Q12 | EquiformerV2 | 24 | 32 | 82 | 118 | 59 | 63 |
| 2025 | Q13 | CGCNN | 160 | 107 | 107 | 65 | 107 | 82 |
| 2025 | Q13 | SchNet | 34 | 36 | 8 | 8 | 11 | 11 |
| 2025 | Q13 | MEGNet | 185 | 185 | 185 | 188 | 185 | 199 |
| 2025 | Q13 | M3GNet | 111 | 55 | 23 | 24 | 40 | 39 |
| 2025 | Q13 | NequIP | 186 | 72 | 53 | 56 | 65 | 67 |
| 2025 | Q13 | MACE | 253 | 106 | 106 | 155 | 106 | 131 |
| 2025 | Q13 | CHGNet | 136 | 100 | 99 | 189 | 100 | 139 |
| 2025 | Q13 | EquiformerV2 | 24 | 57 | 79 | 99 | 71 | 76 |
| 2025 | Q14 | Matbench Discovery | 2 | 8 | 1 | 1 | 3 | 3 |
| 2025 | Q14 | Matbench | 97 | 53 | 88 | 124 | 71 | 78 |
| 2025 | Q14 | JARVIS-DFT | 547 | 541 | 541 | 541 | 541 | 541 |
| 2025 | Q14 | Materials Project | 106 | 35 | 51 | 56 | 48 | 48 |
| 2025 | Q14 | OQMD | 118 | 101 | 101 | 88 | 101 | 95 |
| 2025 | Q14 | AFLOW | 260 | 66 | 94 | 143 | 81 | 97 |
| 2025 | Q14 | Alexandria | 3381 | 353 | 353 | 353 | 353 | 353 |
| 2025 | Q14 | GNoME | 126 | 41 | 30 | 33 | 38 | 39 |
| 2025 | Q14 | WBM | 674 | 726 | 726 | 726 | 726 | 726 |
| 2025 | Q14 | OMat24 | 475 | 71 | 90 | 132 | 83 | 99 |
| 2026 | Q1 | MACE | 439 | 128 | 128 | 147 | 128 | 141 |
| 2026 | Q1 | M3GNet | 114 | 29 | 19 | 19 | 16 | 16 |
| 2026 | Q1 | CHGNet | 135 | 96 | 99 | 184 | 100 | 132 |
| 2026 | Q1 | GAP | 241 | 57 | 55 | 63 | 63 | 64 |
| 2026 | Q1 | NequIP | 219 | 103 | 103 | 52 | 103 | 79 |
| 2026 | Q1 | ACE | 320 | 170 | 170 | 154 | 170 | 172 |
| 2026 | Q1 | EquiformerV2 | 27 | 32 | 90 | 121 | 60 | 66 |
| 2026 | Q2 | DeepH | 309 | 73 | 96 | 157 | 88 | 109 |
| 2026 | Q2 | D4FT | 12 | 14 | 3 | 3 | 6 | 6 |
| 2026 | Q2 | HamGNN | 102 | 64 | 91 | 151 | 82 | 97 |
| 2026 | Q3 | DeepH-E3 | 1 | 1 | 1 | 1 | 1 | 1 |
| 2026 | Q4 | VGNN | 2567 | 59 | 44 | 52 | 55 | 53 |
| 2026 | Q6 | LiFlow | 9 | 2 | 1 | 1 | 2 | 2 |
| 2026 | Q7 | symbolic regression | 184 | 225 | 225 | 225 | 225 | 225 |
| 2026 | Q8 | ConvLSTM | 158 | 29 | 30 | 30 | 25 | 24 |
| 2026 | Q9 | E(3)-equivariant GNNs | 2391 | 139 | 139 | 91 | 139 | 122 |
| 2026 | Q9 | NequIP | 232 | 111 | 111 | 61 | 111 | 88 |
| 2026 | Q10 | SchNet | 32 | 25 | 18 | 18 | 18 | 15 |
| 2026 | Q12 | MACE | 448 | 110 | 110 | 138 | 110 | 123 |
| 2026 | Q12 | NequIP | 251 | 92 | 50 | 57 | 76 | 77 |
| 2026 | Q12 | CHGNet | 121 | 73 | 98 | 174 | 92 | 110 |
| 2026 | Q12 | M3GNet | 109 | 31 | 11 | 11 | 13 | 13 |
| 2026 | Q12 | SevenNet | 200 | 42 | 55 | 63 | 53 | 55 |
| 2026 | Q12 | GAP | 250 | 50 | 68 | 82 | 63 | 66 |
| 2026 | Q12 | DeePMD | 265 | 56 | 34 | 36 | 49 | 48 |
| 2026 | Q12 | MatterSim | 148 | 37 | 74 | 89 | 58 | 61 |
| 2026 | Q12 | Orb | 446 | 112 | 112 | 185 | 112 | 151 |
| 2026 | Q12 | eSEN | 266 | 116 | 116 | 179 | 116 | 152 |
| 2026 | Q12 | PET | 404 | 233 | 233 | 233 | 233 | 233 |
| 2026 | Q12 | UMA | 2564 | 402 | 402 | 402 | 402 | 402 |
| 2026 | Q12 | EquiformerV2 | 28 | 32 | 93 | 152 | 61 | 71 |
| 2026 | Q13 | CGCNN | 179 | 130 | 130 | 67 | 130 | 93 |
| 2026 | Q13 | SchNet | 38 | 25 | 5 | 5 | 8 | 8 |
| 2026 | Q13 | MEGNet | 206 | 205 | 205 | 205 | 205 | 205 |
| 2026 | Q13 | M3GNet | 124 | 87 | 26 | 27 | 55 | 52 |
| 2026 | Q13 | NequIP | 207 | 79 | 54 | 59 | 70 | 71 |
| 2026 | Q13 | MACE | 386 | 105 | 105 | 160 | 105 | 139 |
| 2026 | Q13 | CHGNet | 150 | 86 | 99 | 188 | 98 | 134 |
| 2026 | Q13 | EquiformerV2 | 25 | 49 | 80 | 107 | 68 | 74 |
| 2026 | Q14 | Matbench Discovery | 2 | 4 | 3 | 3 | 2 | 2 |
| 2026 | Q14 | Matbench | 110 | 58 | 90 | 132 | 76 | 85 |
| 2026 | Q14 | JARVIS-DFT | 611 | 620 | 620 | 620 | 620 | 620 |
| 2026 | Q14 | Materials Project | 117 | 34 | 55 | 61 | 52 | 52 |
| 2026 | Q14 | MPtrj | 154 | 157 | 157 | 177 | 157 | 174 |
| 2026 | Q14 | OQMD | 131 | 96 | 73 | 89 | 88 | 92 |
| 2026 | Q14 | AFLOW | 301 | 56 | 95 | 146 | 78 | 89 |
| 2026 | Q14 | Alexandria | 467 | 147 | 147 | 200 | 147 | 180 |
| 2026 | Q14 | GNoME | 143 | 43 | 32 | 34 | 40 | 42 |
| 2026 | Q14 | WBM | 185 | 183 | 183 | 156 | 183 | 178 |
| 2026 | Q14 | OMat24 | 124 | 41 | 53 | 59 | 55 | 55 |

## Decisions after this pass

1. At 2025, RR1 improves micro Recall@10 by 3/47 = .0638 over H2 and 2/47 = .0426 over R3, reaching .1277. RR3 reaches .1064. At 2026, RR1/RR3 reach 5/50 = .10 versus H2’s 3/50 = .06.
2. K200 does not improve RR1/RR3 top-ten recovery over K100 at either cutoff. It doubles scoring calls and adds evidence construction; RR3-200 gains one 2025 hit at twenty. That does not establish a sufficient quality benefit for K200, though it remains essential as a diagnostic of discrimination.
3. Oracle micro Recall@10 is 31/47 = .6596 and 43/47 = .9149 at 2025 for K100/K200; 32/50 = .64 and 44/50 = .88 at 2026. Macro oracle values are separately tabulated and must not be confused with these micro denominators.
4. RR1’s oracle gaps are 25/47 = .5319 and 37/47 = .7872 at 2025; .54 and .78 at 2026. High candidate coverage does not translate to final discrimination.
5. Selected specific evidence is present for 41/43 primary in-pool gold instances and 43/45 at 2026. Only two in-pool instances at each cutoff have none. This does not establish that their evidence is discriminative or sufficient.
6. Failures predominantly have selected evidence present, rather than total evidence absence. Comparative-subject ambiguity, sparse extraction, and max-cosine’s inability to evaluate conjunctions, direction, scale and scientific constraints are plausible mechanisms supported by traces. Their relative causal contributions remain unmeasured.
7. Deterministic evidence scoring yields three new 2025 top-ten hits but worsens median rank. It helps a few answers and does not solve the ranking bottleneck.
8. A relevance cross-encoder comparison is unavailable; the local NLI classifier is not substituted. No conclusion about cross-encoder superiority is supported.
9. RRF preserves all H2 primary top-ten strengths and moderates rank degradation, but sacrifices SchNet’s top-ten gain. It does not beat RR1 at ten.
10. Hierarchy approximation is not justified as the next priority. The quality prerequisite remains unmet. A future pass should test stronger relevance/constraint scoring and better evidence attribution on unseen questions before optimizing the same weak ranking for speed. No hierarchy implementation was added.

Semantic retrieval locates query-relevant text; diffusion transfers that relevance to structurally related entities; reranking must test which entities satisfy the question. Neither graph proximity nor semantic cosine is entailment, and this pass does not fix graph extraction.

## Reproduction and verification

```powershell
$env:HF_HUB_OFFLINE='1'
.\.venv\Scripts\python.exe -m tkh_abstraction.cli evaluate-reranking --config configs/reranking.yaml
.\.venv\Scripts\python.exe -m pytest -q --junitxml=artifacts/evaluation/reranker_tests.xml
.\.venv\Scripts\python.exe scripts/report_reranking.py
.\.venv\Scripts\python.exe scripts/package_submission.py
```

For unseen questions on the same graph, pass `--questions questions_new.csv --ground-truth ground_truth_new.json`; use separate configured output directories to preserve assessment exports. New data additionally requires `reranker_verify_history: false` and explicit data/output paths. `--cutoff` selects a partial experiment; the assessment report audit expects both cutoffs. Current question IDs and target names occur in evaluation/report fixtures, not production evidence/scoring.

The complete suite passed 155 tests with no skips/errors/failures. New tests cover deterministic capped evidence, explicit mentions/dedup/temporal exclusion, generic bridges, normalized missing fields, coverage and pooling, K100/K200 preservation, unchanged groups, RRF determinism, multi-answer/alias oracle coverage, frozen candidate metrics, no oracle/gold production inputs, offline/NLI rejection, all-null unseen evaluation, Type B/history preservation, ground-truth perturbation invariance, and exact RR0 historical metrics. The report audit also checks source hashes, every frozen prefix, all 14 question statuses, old artifact/metric hashes, native hierarchy manifest, temporal evidence containment and score decomposition.

Implementation: `candidate_evidence.py`, `reranking.py`, `reranker_pipeline.py`, `evaluation/reranker_oracle.py`, CLI extension, `configs/reranking.yaml`, `tests/test_reranking.py`, and this reporting script. Required exports are `reranker_results.json`, `reranker_per_target.json`, `candidate_evidence.json`, `reranker_failure_analysis.json`, `reranker_oracle.json`, and `reranker_tests.xml`; detailed protocols/stage exports/audit are under `artifacts/evaluation/reranking`. Pre-pass metrics and diffusion report are preserved under `artifacts/baseline_diffusion`.

Ultralight used serial read-only Claude Opus planning and Claude Sonnet review through the official local subscription helper. Codex implemented, integrated and executed the work. Prompts, results and decisions are under `notes/reranking`; no human scientific validation is claimed.
