# Requirement-aware candidate reranking

At the conservative 2025 cutoff, CR0 (unchanged RR1-100) recovers 6/47 canonical target instances at ten. Requirement cosine CR1 recovers 11/47; controlled NLI CR2 recovers 6/47; H2 fusion CR3/CR4 recover 10/47 and 5/47. The frozen H2-100 oracle remains 31/47. These are fixed-default development diagnostics on the repeatedly used benchmark; no held-out generalization, human scientific validation or calibrated entailment probability is claimed.

The experiment reuses `FrozenH2Candidates`, the exact prior `CandidateEvidenceIndex` bundles, `CandidateReranker`, `ThetaOperator`, `MentionIndex`, grouped entities, resolver, BGE cache, local `NLI`, temporal snapshots, reranking metrics and the unchanged oracle. H2 alpha=.85/M100/weights/eligibility/RRF and every previous source module except the CLI remain unchanged. Every historical H2 prefix, per-question metric, CR0 ranking/metric, previous report and metric section is audited. Primary K is 100; CR1/CR2 K200 are diagnostics. CR5 was not added. No hierarchy or new candidate-discovery mechanism was implemented.

## Generic decomposition and fixed scoring

`decompose_question(question, declared_type)` uses generic interrogative framing, task/condition connectors, conjunctions, negative markers and dates. Fields are original contiguous substrings with exact start/end offsets; requirements are free-form strings, with no materials-science taxonomy or expected-answer logic. No suitable local generative model exists; no model was used for decomposition, no model downloaded and no remote decomposition API called. Original question text is always retained. Temporal phrases are diagnostic and excluded from support vectors; the existing selected snapshot remains authoritative, including the original 2025 conservative / 2026 annual sensitivity interpretation. Dates such as “after” do not introduce a new lower-bound filter.

The parser is deliberately shallow: it can fragment conjunctions, mistake descriptive answer-type modifiers for framing, or leave a compound task intact. It does not infer missing properties. The report exports every decomposition rather than claiming correctness from exact source spans. A source span verifies extraction fidelity, not that the phrase is a logically complete condition.

A concrete limitation is Q12: the extracted task is “large-scale atomistic simulation”, while the answer-head qualifiers “ML-based interatomic potential” are not retained as separate scored requirements. Q5 splits “crystalline symmetries” away from the later shared predicate “must be preserved”. These are qualitative decomposition concerns, not automatically inferred from retrieval failure. They limit a causal claim that the experiment understands every condition. No answer-specific paraphrase was inserted to repair them.

| Q | Answer type | Core task | Requirements | Exclusions | Temporal | Warnings |
| --- | --- | --- | --- | --- | --- | --- |
| Q1 | method | predicting interatomic potentials | accuracy close to DFT; computational efficiency; systems of millions of atoms |  | by Feb 2026 |  |
| Q2 | method | accelerating electronic structure calculations | self-consistent field iterations become prohibitively expensive; large systems (>10³ atoms) |  | by Feb 2026 |  |
| Q3 | method | learning DFT Hamiltonians | dealing with twisted van der Waals heterostructures containing thousands of magnetic atoms |  | by Feb 2026 |  |
| Q4 | method | phonon property prediction | universal MLIPs show inaccuracies despite good energy/force performance |  | by Feb 2026 |  |
| Q5 | method | coarse-graining molecular dynamics | crystalline symmetries; long-range elastic interactions must be preserved |  | by Feb 2026 |  |
| Q6 | method | predicting ionic conductivity | full ab initio MD is too expensive; accuracy must be maintained |  | by Feb 2026 |  |
| Q7 | method | predicting phase transitions | screening ~50,000 inorganic compounds at high throughput |  | by Feb 2026 |  |
| Q8 | method | predicting fracture patterns | generalizing from atomistic MD data to unseen bicrystalline structures |  | by Feb 2026 |  |
| Q9 | method | ensuring physical consistency | transferring information between scales | violating thermodynamic principles | by Feb 2026 |  |
| Q10 | method | representing atomic environments | the system exhibits significant compositional diversity, defects; disorder |  | by Feb 2026 |  |
| Q11 | method | integrating features from three or more distinct scales simultaneously within a unified framework |  |  | by Feb 2026 | Alternative clause retained intact; no Boolean parsing |
| Q12 | method | large-scale atomistic simulation |  |  | by Feb 2026 |  |
| Q13 | architecture | predicting properties of solid-state crystalline materials |  |  | by Feb 2026 |  |
| Q14 | — | evaluate MLIPs and property prediction models for materials discovery |  |  | by Feb 2026 |  |
| Q15 | — | What are the most important open scientific challenges ( | in AI-based multiscale modeling of solid-state materials |  | by Feb 2026 |  |
| Q16 | — | How does E(3)-equivariance improve GNN-based interatomic potentials in accuracy and data efficiency in comparison | vanilla GNN |  | — |  |
| Q17 | — | Give evidences | ML-based coarse-grained molecular dynamics is effective; studying ionic transport; phase transitions in solid-state systems ( |  | by Feb 2026 |  |
| Q18 | system | sizes and conditions beyond the training set ( |  |  | by Feb 2026 |  |

Type A mean explicit requirements/question: 1.1429; mean positive components including the core: 2.1429. Type B decompositions are diagnostics only; Type B retrieval and all its prior outputs remain unchanged.

CR1 compares each task/requirement string with each previously selected evidence text using normalized BGE-small-en-v1.5 vectors. Evidence embeddings are reused from `ContextIndex.X`; only component strings need encoding. Each component takes the maximum of nonnegative cosine × attribution weight. Different requirements can select different items.

Attribution weights, fixed before outcomes: direct subject 1, safe explicit-name support .8, comparison winner 1, comparison baseline 0, mention-only .15, uncertain .25, indirect graph neighbor 0. Safe names use unchanged boundaries and graph-derived aliases. Generic active/passive comparison direction is checked before subject/list heuristics. An unnegated comparison baseline cannot unconditionally boost the named candidate. A direct low-arity edge without an explicit subject remains UNCERTAIN; co-membership is not a predicate. Unnamed path/provenance text is diagnostic-only; attributable path text may qualify by its textual role. Original graph labels and paths are preserved. These heuristics are not a complete relation extractor.

For positive component supports `s`, the ranking score is `core_support × geometric_mean(s) × product(1 − exclusion_violation) × soft_type_prior`. A missing core gates the score to zero; one strong component cannot compensate freely for weak support elsewhere. Answer type uses only a .95 mismatch factor for recognized generic entity families and never filters candidates. “Requirement coverage” is separately defined as the fraction of positive components ≥.5; this fixed descriptive threshold is not used in ranking. Geometric support is a smooth strength/coverage proxy, not a probability of satisfying every condition.

CR2 uses `cross-encoder/nli-deberta-v3-small`, pinned revision `fa2804872c3b4bd748f38c0185cc85775361e735`, strictly local. It is a three-way NLI classifier, **not a relevance cross-encoder**. For each candidate/component, the single highest attributed-cosine eligible item is preselected. The hypothesis explicitly includes the candidate: task “<candidate> is suitable for <task>”; condition “<candidate> supports <task> with <condition>” or “is suitable … when …”. This literal templating avoids invented paraphrases but sometimes produces awkward or stronger-than-premise statements.

Positive NLI support is `weight × max(0, entailment − contradiction)`. Exclusions use a positive violation hypothesis, “<candidate> involves <excluded phrase> for <task>”; entailment reduces the final score. Neutral neither supplies positive task support nor proves exclusion compliance. CR1 uses similarity to the excluded phrase as a coarse violation proxy and cannot distinguish its negation. CR2 preserves all three probabilities, predicted labels, input token counts and premise-only truncation flags. Any NLI execution failure/unavailability makes the entire question use explicit CR1 fallback, avoiding mixed cosine/NLI scales; failed probabilities remain null. Fallback is never silently presented as NLI success. Exact premise/hypothesis cache reuse and forward work are separately counted.

CR3 and CR4 fuse H2 with CR1 and CR2 respectively using unchanged RRF60 and deterministic average ranks for exact score ties. No mixing weights, support thresholds, parser rules, attribution weights or NLI parameters were selected by benchmark score. An attribution or decomposition change motivated by an actual generic bug is logged separately from modeling limitations.

## Metrics: 2025

| Stage | K | Hit1 | Hit5 | Hit10 | Hit20 | MacroR5 | MacroR10 | MacroR20 | MicroR5 | MicroR10 | MicroR20 | UniqueR10 | Raw /63 | MRR | R-Prec | R@R | Median | Mean rank | Preserved | Support comparisons | NLI requests | NLI new pairs | NLI sec | New stage sec | Total sec* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CR0-100 | 100 | 0.2500 | 0.3333 | 0.5000 | 0.5833 | 0.2028 | 0.2208 | 0.3041 | 0.0851 | 0.1277 | 0.1489 | 0.1765 | 6/63 | 0.3278 | 0.2208 | 0.2208 | 89.0000 | 107.7872 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0012 | 1.4102 |
| CR1-100 | 100 | 0.4167 | 0.6667 | 0.6667 | 0.7500 | 0.4026 | 0.4102 | 0.4924 | 0.2128 | 0.2340 | 0.3617 | 0.2941 | 11/63 | 0.5283 | 0.3268 | 0.3268 | 29.0000 | 90.8723 | 1.0000 | 1443.8571 | 0.0000 | 0.0000 | 0.0000 | 0.0865 | 1.4955 |
| CR2-100 | 100 | 0.2500 | 0.4167 | 0.4167 | 0.4167 | 0.2006 | 0.2006 | 0.2248 | 0.1277 | 0.1277 | 0.1915 | 0.1765 | 6/63 | 0.3454 | 0.2006 | 0.2006 | 73.0000 | 103.0213 | 1.0000 | 1443.8571 | 208.6429 | 2.5000 | 0.1825 | 0.2783 | 1.6872 |
| CR3-100 | 100 | 0.2500 | 0.5000 | 0.5833 | 0.8333 | 0.2958 | 0.3470 | 0.5693 | 0.1277 | 0.2128 | 0.3830 | 0.2647 | 10/63 | 0.3366 | 0.2915 | 0.2915 | 53.0000 | 93.2979 | 1.0000 | 1443.8571 | 0.0000 | 0.0000 | 0.0000 | 0.0869 | 1.4959 |
| CR4-100 | 100 | 0.1667 | 0.3333 | 0.3333 | 0.5833 | 0.2576 | 0.2652 | 0.3276 | 0.0851 | 0.1064 | 0.2128 | 0.1471 | 5/63 | 0.2571 | 0.1818 | 0.1818 | 65.0000 | 100.7021 | 1.0000 | 1443.8571 | 208.6429 | 2.5000 | 0.1825 | 0.2785 | 1.6875 |
| CR1-200 | 200 | 0.4167 | 0.5833 | 0.6667 | 0.7500 | 0.3867 | 0.4102 | 0.4797 | 0.1702 | 0.2340 | 0.3404 | 0.2941 | 11/63 | 0.5261 | 0.3268 | 0.3268 | 34.0000 | 86.5319 | 1.0000 | 2446.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1708 | 1.6711 |
| CR2-200 | 200 | 0.2500 | 0.5000 | 0.5000 | 0.5833 | 0.2763 | 0.2839 | 0.3498 | 0.1277 | 0.1489 | 0.2340 | 0.2059 | 7/63 | 0.3642 | 0.2006 | 0.2006 | 125.0000 | 127.7660 | 1.0000 | 2446.0000 | 379.5000 | 6.7143 | 0.4170 | 0.5973 | 2.0975 |

| Reference | MicroR10 | Median gold rank | Frozen C100 macro | Frozen C200 macro |
| --- | --- | --- | --- | --- |
| R3 | 0.0851 | 160.0000 | 0.3701 | 0.7140 |
| H2 | 0.0638 | 66.0000 | 0.7146 | 0.9674 |
| H3 | 0.0638 | 59.0000 | 0.7230 | 0.8841 |

## Metrics: 2026

| Stage | K | Hit1 | Hit5 | Hit10 | Hit20 | MacroR5 | MacroR10 | MacroR20 | MicroR5 | MicroR10 | MicroR20 | UniqueR10 | Raw /63 | MRR | R-Prec | R@R | Median | Mean rank | Preserved | Support comparisons | NLI requests | NLI new pairs | NLI sec | New stage sec | Total sec* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CR0-100 | 100 | 0.1667 | 0.4167 | 0.4167 | 0.6667 | 0.2124 | 0.2124 | 0.3141 | 0.1000 | 0.1000 | 0.1600 | 0.1351 | 5/63 | 0.2613 | 0.2188 | 0.2188 | 90.5000 | 101.8200 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0013 | 1.4852 |
| CR1-100 | 100 | 0.4167 | 0.5833 | 0.6667 | 0.7500 | 0.3779 | 0.4047 | 0.4883 | 0.1400 | 0.2200 | 0.3600 | 0.2703 | 11/63 | 0.5218 | 0.3214 | 0.3214 | 28.5000 | 84.8400 | 1.0000 | 1482.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0896 | 1.5736 |
| CR2-100 | 100 | 0.2500 | 0.4167 | 0.4167 | 0.4167 | 0.1975 | 0.2051 | 0.2190 | 0.1200 | 0.1400 | 0.1800 | 0.1892 | 7/63 | 0.3425 | 0.2051 | 0.2051 | 72.0000 | 96.7600 | 1.0000 | 1482.0000 | 210.0000 | 0.2143 | 0.0276 | 0.1641 | 1.6480 |
| CR3-100 | 100 | 0.3333 | 0.5000 | 0.5000 | 0.7500 | 0.2918 | 0.2982 | 0.4715 | 0.1200 | 0.1400 | 0.3200 | 0.1892 | 7/63 | 0.3977 | 0.2768 | 0.2768 | 50.5000 | 87.9600 | 1.0000 | 1482.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0900 | 1.5739 |
| CR4-100 | 100 | 0.1667 | 0.2500 | 0.3333 | 0.5833 | 0.2500 | 0.2628 | 0.3237 | 0.0600 | 0.1000 | 0.2000 | 0.1351 | 5/63 | 0.2474 | 0.1795 | 0.1795 | 67.0000 | 95.8800 | 1.0000 | 1482.0000 | 210.0000 | 0.2143 | 0.0276 | 0.1644 | 1.6483 |
| CR1-200 | 200 | 0.4167 | 0.5833 | 0.6667 | 0.7500 | 0.3779 | 0.3983 | 0.4732 | 0.1400 | 0.2000 | 0.3200 | 0.2703 | 10/63 | 0.5195 | 0.3214 | 0.3214 | 37.0000 | 80.1000 | 1.0000 | 2514.0714 | 0.0000 | 0.0000 | 0.0000 | 0.1486 | 1.7089 |
| CR2-200 | 200 | 0.2500 | 0.4167 | 0.4167 | 0.4167 | 0.1911 | 0.2115 | 0.2266 | 0.1000 | 0.1600 | 0.2000 | 0.2162 | 8/63 | 0.3432 | 0.2115 | 0.2115 | 117.5000 | 114.6400 | 1.0000 | 2514.0714 | 385.9286 | 0.2143 | 0.0299 | 0.2255 | 1.7858 |

| Reference | MicroR10 | Median gold rank | Frozen C100 macro | Frozen C200 macro |
| --- | --- | --- | --- | --- |
| R3 | 0.0600 | 181.5000 | 0.3245 | 0.6527 |
| H2 | 0.0600 | 73.0000 | 0.7040 | 0.8859 |
| H3 | 0.0600 | 71.5000 | 0.7159 | 0.8859 |

All 14 Type A questions remain in every stage. Q5/Q11 have no canonical entity units and explicit null recall; their labels remain in the raw /63 denominator. Macro metrics average 12 evaluable questions; micro denominators are 47 canonical instances at 2025 and 50 at 2026. Unique recall retains the prior union-of-target-names definition. Costs average all 14. Candidate preservation is exact at both pools; candidates outside the reranked prefix retain their H2 order.

*Total seconds is historical H2 search + prior evidence construction + measured new preparation/scoring/fusion, not a cold end-to-end latency benchmark. Shared top200 component/evidence-vector preparation and NLI aggregation are conservatively charged to primary runs; this can overcount K100. CR4 reuses CR2’s batch results but is reported with standalone component cost. Cached NLI calls do not imply new neural forward work. Model loading, evaluation/export and most existing graph indexing are outside the per-query component sum. Inspect exact cache hits/forward pairs and run logs before comparing machines or reruns.

## Oracle gap

| Year | Stage | Unchanged oracle | Oracle microR10 | Actual microR10 | Gap |
| --- | --- | --- | --- | --- | --- |
| 2025 | CR0-100 | 31/47 | 0.6596 | 0.1277 | 0.5319 |
| 2025 | CR1-100 | 31/47 | 0.6596 | 0.2340 | 0.4255 |
| 2025 | CR2-100 | 31/47 | 0.6596 | 0.1277 | 0.5319 |
| 2025 | CR3-100 | 31/47 | 0.6596 | 0.2128 | 0.4468 |
| 2025 | CR4-100 | 31/47 | 0.6596 | 0.1064 | 0.5532 |
| 2025 | CR1-200 | 43/47 | 0.9149 | 0.2340 | 0.6809 |
| 2025 | CR2-200 | 43/47 | 0.9149 | 0.1489 | 0.7660 |
| 2026 | CR0-100 | 32/50 | 0.6400 | 0.1000 | 0.5400 |
| 2026 | CR1-100 | 32/50 | 0.6400 | 0.2200 | 0.4200 |
| 2026 | CR2-100 | 32/50 | 0.6400 | 0.1400 | 0.5000 |
| 2026 | CR3-100 | 32/50 | 0.6400 | 0.1400 | 0.5000 |
| 2026 | CR4-100 | 32/50 | 0.6400 | 0.1000 | 0.5400 |
| 2026 | CR1-200 | 44/50 | 0.8800 | 0.2000 | 0.6800 |
| 2026 | CR2-200 | 44/50 | 0.8800 | 0.1600 | 0.7200 |

The oracle remains evaluation-only and byte-equivalent per question to the preceding experiment. It maximizes distinct canonical coverage in ten slots; no oracle membership enters decomposition, attribution, cosine, NLI, or fusion.

## Requirement and attribution diagnostics

| Year | K | Group | Scorer | Candidates | Mean coverage | Soft coverage | All supported | Partial | None | Contradiction | No direct attributable evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025 | 100 | gold | CR1 | 31 | 0.7312 | 0.5637 | 22 | 1 | 8 | 0 | 7 |
| 2025 | 100 | gold | CR2 | 31 | 0.1183 | 0.0615 | 1 | 6 | 24 | 9 | 7 |
| 2025 | 100 | not gold | CR1 | 1369 | 0.2740 | 0.3061 | 321 | 102 | 946 | 0 | 915 |
| 2025 | 100 | not gold | CR2 | 1369 | 0.0247 | 0.0101 | 8 | 61 | 1300 | 312 | 915 |
| 2025 | 200 | gold | CR1 | 43 | 0.7190 | 0.5315 | 29 | 3 | 11 | 0 | 10 |
| 2025 | 200 | gold | CR2 | 43 | 0.1027 | 0.0451 | 1 | 8 | 34 | 11 | 10 |
| 2025 | 200 | not gold | CR1 | 2757 | 0.2244 | 0.2657 | 527 | 174 | 2056 | 0 | 1954 |
| 2025 | 200 | not gold | CR2 | 2757 | 0.0183 | 0.0082 | 15 | 86 | 2656 | 588 | 1954 |
| 2026 | 100 | gold | CR1 | 32 | 0.8021 | 0.5791 | 25 | 1 | 6 | 0 | 6 |
| 2026 | 100 | gold | CR2 | 32 | 0.1146 | 0.0597 | 1 | 6 | 25 | 8 | 6 |
| 2026 | 100 | not gold | CR1 | 1368 | 0.2795 | 0.3131 | 330 | 99 | 939 | 0 | 903 |
| 2026 | 100 | not gold | CR2 | 1368 | 0.0255 | 0.0109 | 9 | 63 | 1296 | 316 | 903 |
| 2026 | 200 | gold | CR1 | 45 | 0.7537 | 0.5332 | 32 | 3 | 10 | 0 | 10 |
| 2026 | 200 | gold | CR2 | 45 | 0.0981 | 0.0439 | 1 | 8 | 36 | 10 | 10 |
| 2026 | 200 | not gold | CR1 | 2755 | 0.2172 | 0.2702 | 507 | 176 | 2072 | 0 | 1949 |
| 2026 | 200 | not gold | CR2 | 2755 | 0.0193 | 0.0091 | 16 | 89 | 2650 | 604 | 1949 |

Counts are candidate/question pairs, with fraction = count/candidates. “Supported” means model score ≥.5 under the fixed attribution policy, not human-verified entailment. No direct attributable evidence means no direct-subject, explicit-name, or comparison-winner item among the capped prior bundle; uncertain graph-linked items can still contribute .25. Neither this count nor a zero NLI score proves the underlying graph lacks sufficient information. Gold membership is applied only in evaluation.

| Year | K | Group | DIRECT_SUBJECT_SUPPORT | EXPLICIT_NAME_SUPPORT | COMPARISON_WINNER | COMPARISON_BASELINE | MENTION_ONLY | INDIRECT_GRAPH_NEIGHBOR | UNCERTAIN | Co-membership flag | Path-only flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025 | 100 | gold | 43 | 30 | 4 | 4 | 20 | 58 | 95 | 42 | 19 |
| 2025 | 100 | not gold | 464 | 542 | 14 | 32 | 251 | 2175 | 5355 | 1932 | 1524 |
| 2025 | 200 | gold | 49 | 40 | 5 | 4 | 27 | 87 | 104 | 42 | 27 |
| 2025 | 200 | not gold | 611 | 951 | 28 | 49 | 390 | 5079 | 7932 | 3032 | 3069 |
| 2026 | 100 | gold | 40 | 32 | 4 | 4 | 23 | 63 | 97 | 43 | 22 |
| 2026 | 100 | not gold | 450 | 529 | 37 | 31 | 280 | 2136 | 5584 | 1959 | 1498 |
| 2026 | 200 | gold | 48 | 41 | 5 | 4 | 30 | 96 | 110 | 44 | 30 |
| 2026 | 200 | not gold | 568 | 974 | 56 | 52 | 424 | 4884 | 8487 | 3045 | 2977 |

Attribution categories are mutually exclusive per item; graph co-membership/path flags overlap them. Comparison winner/baseline labels refer to the detected text relation, not whether a method is globally better or meets the question.

## Rank movement and failures

| Year | Stage | Original bin | N | Improved | Same | Worsened | Entered10 | Left10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025 | CR1-100 | H2 1-10 | 3 | 0 | 2 | 1 | 0 | 1 |
| 2025 | CR1-100 | H2 11-50 | 16 | 13 | 0 | 3 | 5 | 0 |
| 2025 | CR1-100 | H2 51-100 | 12 | 8 | 0 | 4 | 4 | 0 |
| 2025 | CR1-100 | H2 >100 | 16 | 0 | 16 | 0 | 0 | 0 |
| 2025 | CR1-100 | H2 prior top10 | 3 | 0 | 2 | 1 | 0 | 1 |
| 2025 | CR1-100 | CR0 prior top10 | 6 | 1 | 1 | 4 | 0 | 3 |
| 2025 | CR2-100 | H2 1-10 | 3 | 0 | 1 | 2 | 0 | 2 |
| 2025 | CR2-100 | H2 11-50 | 16 | 5 | 0 | 11 | 3 | 0 |
| 2025 | CR2-100 | H2 51-100 | 12 | 9 | 0 | 3 | 2 | 0 |
| 2025 | CR2-100 | H2 >100 | 16 | 0 | 16 | 0 | 0 | 0 |
| 2025 | CR2-100 | H2 prior top10 | 3 | 0 | 1 | 2 | 0 | 2 |
| 2025 | CR2-100 | CR0 prior top10 | 6 | 0 | 1 | 5 | 0 | 5 |
| 2025 | CR3-100 | H2 1-10 | 3 | 1 | 2 | 0 | 0 | 0 |
| 2025 | CR3-100 | H2 11-50 | 16 | 14 | 0 | 2 | 6 | 0 |
| 2025 | CR3-100 | H2 51-100 | 12 | 7 | 1 | 4 | 1 | 0 |
| 2025 | CR3-100 | H2 >100 | 16 | 0 | 16 | 0 | 0 | 0 |
| 2025 | CR3-100 | H2 prior top10 | 3 | 1 | 2 | 0 | 0 | 0 |
| 2025 | CR3-100 | CR0 prior top10 | 6 | 1 | 2 | 3 | 0 | 1 |
| 2025 | CR4-100 | H2 1-10 | 3 | 0 | 2 | 1 | 0 | 1 |
| 2025 | CR4-100 | H2 11-50 | 16 | 5 | 3 | 8 | 3 | 0 |
| 2025 | CR4-100 | H2 51-100 | 12 | 3 | 4 | 5 | 0 | 0 |
| 2025 | CR4-100 | H2 >100 | 16 | 0 | 16 | 0 | 0 | 0 |
| 2025 | CR4-100 | H2 prior top10 | 3 | 0 | 2 | 1 | 0 | 1 |
| 2025 | CR4-100 | CR0 prior top10 | 6 | 0 | 1 | 5 | 0 | 4 |
| 2025 | CR1-200 | H2 1-10 | 3 | 0 | 2 | 1 | 0 | 1 |
| 2025 | CR1-200 | H2 11-50 | 16 | 10 | 0 | 6 | 5 | 0 |
| 2025 | CR1-200 | H2 51-100 | 12 | 7 | 0 | 5 | 4 | 0 |
| 2025 | CR1-200 | H2 >100 | 16 | 10 | 4 | 2 | 0 | 0 |
| 2025 | CR1-200 | H2 prior top10 | 3 | 0 | 2 | 1 | 0 | 1 |
| 2025 | CR1-200 | CR0 prior top10 | 6 | 0 | 1 | 5 | 0 | 3 |
| 2025 | CR2-200 | H2 1-10 | 3 | 0 | 1 | 2 | 0 | 2 |
| 2025 | CR2-200 | H2 11-50 | 16 | 5 | 0 | 11 | 3 | 0 |
| 2025 | CR2-200 | H2 51-100 | 12 | 6 | 0 | 6 | 2 | 0 |
| 2025 | CR2-200 | H2 >100 | 16 | 6 | 4 | 6 | 1 | 0 |
| 2025 | CR2-200 | H2 prior top10 | 3 | 0 | 1 | 2 | 0 | 2 |
| 2025 | CR2-200 | CR0 prior top10 | 6 | 0 | 1 | 5 | 0 | 5 |
| 2026 | CR1-100 | H2 1-10 | 3 | 0 | 2 | 1 | 0 | 1 |
| 2026 | CR1-100 | H2 11-50 | 15 | 13 | 0 | 2 | 4 | 0 |
| 2026 | CR1-100 | H2 51-100 | 14 | 10 | 0 | 4 | 5 | 0 |
| 2026 | CR1-100 | H2 >100 | 18 | 0 | 18 | 0 | 0 | 0 |
| 2026 | CR1-100 | H2 prior top10 | 3 | 0 | 2 | 1 | 0 | 1 |
| 2026 | CR1-100 | CR0 prior top10 | 5 | 0 | 1 | 4 | 0 | 3 |
| 2026 | CR2-100 | H2 1-10 | 3 | 0 | 1 | 2 | 0 | 2 |
| 2026 | CR2-100 | H2 11-50 | 15 | 6 | 0 | 9 | 4 | 0 |
| 2026 | CR2-100 | H2 51-100 | 14 | 10 | 0 | 4 | 2 | 0 |
| 2026 | CR2-100 | H2 >100 | 18 | 0 | 18 | 0 | 0 | 0 |
| 2026 | CR2-100 | H2 prior top10 | 3 | 0 | 1 | 2 | 0 | 2 |
| 2026 | CR2-100 | CR0 prior top10 | 5 | 0 | 1 | 4 | 0 | 4 |
| 2026 | CR3-100 | H2 1-10 | 3 | 2 | 1 | 0 | 0 | 0 |
| 2026 | CR3-100 | H2 11-50 | 15 | 14 | 0 | 1 | 4 | 0 |
| 2026 | CR3-100 | H2 51-100 | 14 | 9 | 0 | 5 | 0 | 0 |
| 2026 | CR3-100 | H2 >100 | 18 | 0 | 18 | 0 | 0 | 0 |
| 2026 | CR3-100 | H2 prior top10 | 3 | 2 | 1 | 0 | 0 | 0 |
| 2026 | CR3-100 | CR0 prior top10 | 5 | 1 | 2 | 2 | 0 | 1 |
| 2026 | CR4-100 | H2 1-10 | 3 | 0 | 2 | 1 | 0 | 1 |
| 2026 | CR4-100 | H2 11-50 | 15 | 5 | 1 | 9 | 3 | 0 |
| 2026 | CR4-100 | H2 51-100 | 14 | 4 | 4 | 6 | 0 | 0 |
| 2026 | CR4-100 | H2 >100 | 18 | 0 | 18 | 0 | 0 | 0 |
| 2026 | CR4-100 | H2 prior top10 | 3 | 0 | 2 | 1 | 0 | 1 |
| 2026 | CR4-100 | CR0 prior top10 | 5 | 0 | 1 | 4 | 0 | 3 |
| 2026 | CR1-200 | H2 1-10 | 3 | 0 | 2 | 1 | 0 | 1 |
| 2026 | CR1-200 | H2 11-50 | 15 | 10 | 1 | 4 | 3 | 0 |
| 2026 | CR1-200 | H2 51-100 | 14 | 9 | 0 | 5 | 5 | 0 |
| 2026 | CR1-200 | H2 >100 | 18 | 10 | 5 | 3 | 0 | 0 |
| 2026 | CR1-200 | H2 prior top10 | 3 | 0 | 2 | 1 | 0 | 1 |
| 2026 | CR1-200 | CR0 prior top10 | 5 | 0 | 1 | 4 | 0 | 3 |
| 2026 | CR2-200 | H2 1-10 | 3 | 0 | 1 | 2 | 0 | 2 |
| 2026 | CR2-200 | H2 11-50 | 15 | 5 | 0 | 10 | 4 | 0 |
| 2026 | CR2-200 | H2 51-100 | 14 | 6 | 0 | 8 | 2 | 0 |
| 2026 | CR2-200 | H2 >100 | 18 | 7 | 5 | 6 | 1 | 0 |
| 2026 | CR2-200 | H2 prior top10 | 3 | 0 | 1 | 2 | 0 | 2 |
| 2026 | CR2-200 | CR0 prior top10 | 5 | 0 | 1 | 4 | 0 | 4 |

| Year | Stage | CONTRADICTORY_EVIDENCE | NLI_FAILURE | NOT_IN_H2_100 | NOT_IN_H2_200 | NO_ATTRIBUTABLE_EVIDENCE | PARTIAL_REQUIREMENT_SUPPORT | RETRIEVED_TOP10 | SEMANTIC_SCORER_FAILURE | SUPPORTED_BUT_RANKED_BELOW_10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025 | CR1-100 | 0 | 0 | 16 | 0 | 7 | 1 | 11 | 1 | 11 |
| 2025 | CR2-100 | 7 | 8 | 16 | 0 | 7 | 3 | 6 | 0 | 0 |
| 2025 | CR3-100 | 0 | 0 | 16 | 0 | 7 | 1 | 10 | 0 | 13 |
| 2025 | CR4-100 | 6 | 10 | 16 | 0 | 7 | 3 | 5 | 0 | 0 |
| 2025 | CR1-200 | 0 | 0 | 0 | 4 | 10 | 3 | 11 | 1 | 18 |
| 2025 | CR2-200 | 9 | 12 | 0 | 4 | 10 | 5 | 7 | 0 | 0 |
| 2026 | CR1-100 | 0 | 0 | 18 | 0 | 6 | 1 | 11 | 0 | 14 |
| 2026 | CR2-100 | 6 | 10 | 18 | 0 | 6 | 3 | 7 | 0 | 0 |
| 2026 | CR3-100 | 0 | 0 | 18 | 0 | 6 | 1 | 7 | 0 | 18 |
| 2026 | CR4-100 | 5 | 13 | 18 | 0 | 6 | 3 | 5 | 0 | 0 |
| 2026 | CR1-200 | 0 | 0 | 0 | 5 | 10 | 3 | 10 | 0 | 22 |
| 2026 | CR2-200 | 8 | 14 | 0 | 5 | 10 | 5 | 8 | 0 | 0 |

Primary categories are mutually exclusive: pool absence, top-ten recovery, no direct attributable item, predicted contradiction, zero/partial diagnostic coverage, then supported-but-below-ten. A zero-coverage NLI failure is an operational scoring failure relative to a gold target, not proof the classifier misunderstood the premise. Decomposition errors and comparison-attribution errors require qualitative adjudication; they are not fabricated automatically from answer membership. Additional flags are exported.

## Actual traces and known failure patterns

### Required diagnostic: Q12, M3GNet (2025)

Question: What are the leading (by Feb 2026) ML-based interatomic potential methods for large-scale atomistic simulation?

R3 97 → H2 29; CR0-100 8, CR1-100 7, CR2-100 74, CR3-100 7, CR4-100 48, CR1-200 9, CR2-200 145.

CR1 coverage 1.000, score 0.587456; CR2 coverage 0.000, score 0.000000.

| Kind | Extracted condition | Evidence node | Attribution | Evidence text | Cosine support | NLI support | Entail | Neutral | Contradict | Candidate-specific hypothesis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| core_task | large-scale atomistic simulation | clai_00646 | DIRECT_SUBJECT_SUPPORT | M3GNet molecular‑dynamics simulations of Li₃YCl₆ reproduce ionic conductivity (14.3 mS cm⁻¹) and activation energy (0.22 eV) consistent with ab‑initio MD, and short‑time MD correctly identifies known lithium superionic conductors by their high mean‑squared displacements at elevated temperatures. | 0.7665 | 0.0000 | 0.0006 | 0.0041 | 0.9953 | M3GNet is suitable for large-scale atomistic simulation. |

### Required diagnostic: Q13, SchNet (2025)

Question: What are the leading (by Feb 2026) GNN architectures for predicting properties of solid-state crystalline materials?

R3 34 → H2 36; CR0-100 8, CR1-100 67, CR2-100 41, CR3-100 63, CR4-100 50, CR1-200 98, CR2-200 69.

CR1 coverage 0.000, score 0.035454; CR2 coverage 0.000, score 0.000000.

| Kind | Extracted condition | Evidence node | Attribution | Evidence text | Cosine support | NLI support | Entail | Neutral | Contradict | Candidate-specific hypothesis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| core_task | predicting properties of solid-state crystalline materials | task_00025 | UNCERTAIN | Interatomic force prediction | 0.1883 | 0.0000 | 0.0017 | 0.9952 | 0.0030 | SchNet is suitable for predicting properties of solid-state crystalline materials. |

### Required diagnostic: Q2, HamGNN (2025)

Question: Which methods (by Feb 2026) are best suited for accelerating electronic structure calculations when self-consistent field iterations become prohibitively expensive for large systems (>10³ atoms)?

R3 90 → H2 24; CR0-100 91, CR1-100 55, CR2-100 80, CR3-100 44, CR4-100 24, CR1-200 91, CR2-200 159.

CR1 coverage 0.000, score 0.034376; CR2 coverage 0.000, score 0.000000.

| Kind | Extracted condition | Evidence node | Attribution | Evidence text | Cosine support | NLI support | Entail | Neutral | Contradict | Candidate-specific hypothesis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| core_task | accelerating electronic structure calculations | clai_00199 | UNCERTAIN | Incorporating orbital‑energy loss in a second fine‑tuning round markedly improves band‑structure predictions, yielding energy bands and Fermi surfaces that closely match DFT calculations for the test set. | 0.1935 | 0.0000 | 0.0035 | 0.9672 | 0.0293 | Universal HamGNN Hamiltonian model is suitable for accelerating electronic structure calculations. |
| requirement | self-consistent field iterations become prohibitively expensive | arti_00029 | UNCERTAIN | Universal Machine Learning Kohn-Sham Hamiltonian for Materials | 0.1604 | 0.0000 | 0.0004 | 0.9948 | 0.0048 | Universal HamGNN Hamiltonian model is suitable for accelerating electronic structure calculations when self-consistent field iterations become prohibitively expensive. |
| requirement | large systems (>10³ atoms) | clai_00193 | UNCERTAIN | The universal HamGNN model accurately predicts the energy band structures and charge densities of complex multi‑element crystals (e.g., Hf2Zr9Ta6Ti5Nb5B54, HfTaTiB4MoC4, K3Ba3Li2Al4B6O20F), matching DFT results. | 0.1807 | 0.0000 | 0.0010 | 0.9980 | 0.0010 | Universal HamGNN Hamiltonian model is suitable for accelerating electronic structure calculations when large systems (>10³ atoms). |

### Required diagnostic: Q1, MACE (2025)

Question: Which methods (by Feb 2026) are best suited for predicting interatomic potentials when balancing accuracy close to DFT with computational efficiency for systems of millions of atoms?

R3 333 → H2 114; CR0-100 114, CR1-100 114, CR2-100 114, CR3-100 114, CR4-100 114, CR1-200 51, CR2-200 156.

CR1 coverage 0.500, score 0.297104; CR2 coverage 0.250, score 0.000000.

| Kind | Extracted condition | Evidence node | Attribution | Evidence text | Cosine support | NLI support | Entail | Neutral | Contradict | Candidate-specific hypothesis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| core_task | predicting interatomic potentials | clai_00604 | EXPLICIT_NAME_SUPPORT | On the 3BPA molecule energy profile, MACE predictions are closest to the DFT ground truth across three dihedral slices, outperforming BOTNet and NequIP. | 0.5659 | 0.1386 | 0.3568 | 0.4598 | 0.1835 | MACE is suitable for predicting interatomic potentials. |
| requirement | accuracy close to DFT | clai_00604 | EXPLICIT_NAME_SUPPORT | On the 3BPA molecule energy profile, MACE predictions are closest to the DFT ground truth across three dihedral slices, outperforming BOTNet and NequIP. | 0.5499 | 0.7787 | 0.9749 | 0.0236 | 0.0015 | MACE is suitable for predicting interatomic potentials when accuracy close to DFT. |
| requirement | computational efficiency | clai_00604 | EXPLICIT_NAME_SUPPORT | On the 3BPA molecule energy profile, MACE predictions are closest to the DFT ground truth across three dihedral slices, outperforming BOTNet and NequIP. | 0.4946 | 0.0006 | 0.0016 | 0.9976 | 0.0008 | MACE supports predicting interatomic potentials with computational efficiency. |
| requirement | systems of millions of atoms | clai_00604 | EXPLICIT_NAME_SUPPORT | On the 3BPA molecule energy profile, MACE predictions are closest to the DFT ground truth across three dihedral slices, outperforming BOTNet and NequIP. | 0.4936 | 0.0000 | 0.0001 | 0.9978 | 0.0021 | MACE supports predicting interatomic potentials with systems of millions of atoms. |

M3GNet Q12 is a clear technical NLI failure in this trace: the premise reports successful molecular-dynamics simulations and compatible conductivity/activation energies, yet the model assigns about .995 contradiction to suitability for large-scale atomistic simulation. The premise does not establish the requested scale, so neutrality would be defensible; it does not assert that large-scale use is impossible. Contradiction minus entailment clips support to zero and the zero-score tie policy further affects rank. This is a qualitative logical assessment of the actual trace, not an external scientific performance certification.

A remaining attribution error affects SchNet: “Including force information in the training loss causally improves SchNet’s generalization …” is classified as MENTION_ONLY because the generic “including” heuristic mistakes a participial subject for an illustrative list. Its graph provenance is retained, but the .15 weight can make a less specific task label win evidence selection. This is reported as an attribution limitation; the rule was not patched specifically for this benchmark sentence.

| Q | Candidate | Node | Detected role | Comparative premise | New positive support weight |
| --- | --- | --- | --- | --- | --- |
| Q1 | M3GNet | clai_00649 | COMPARISON_WINNER | For a MgO test set, M3GNet outperforms Moment Tensor Potentials in extrapolating equation‑of‑state behavior beyond the training regime and exhibits long‑range interaction characteristics similar to a Buckingham + Coulomb model. | 1.0000 |
| Q1 | M3GNet | clai_00343 | COMPARISON_BASELINE | The SevenNet machine-learned interatomic potential outperforms the earlier M3GNet model on the Matbench Discovery benchmark. | 0.0000 |
| Q1 | SchNet | clai_00490 | UNCERTAIN | MEGNet models trained on approximately 60,000 crystals outperform prior models (SchNet and CGCNN) for formation energy, band gap, bulk modulus, and shear modulus, achieving errors comparable to or better than density‑functional‑theory uncertainties. | 0.2500 |
| Q2 | SchNet | clai_00490 | UNCERTAIN | MEGNet models trained on approximately 60,000 crystals outperform prior models (SchNet and CGCNN) for formation energy, band gap, bulk modulus, and shear modulus, achieving errors comparable to or better than density‑functional‑theory uncertainties. | 0.2500 |
| Q2 | M3GNet | clai_00343 | COMPARISON_BASELINE | The SevenNet machine-learned interatomic potential outperforms the earlier M3GNet model on the Matbench Discovery benchmark. | 0.0000 |
| Q2 | M3GNet | clai_00649 | COMPARISON_WINNER | For a MgO test set, M3GNet outperforms Moment Tensor Potentials in extrapolating equation‑of‑state behavior beyond the training regime and exhibits long‑range interaction characteristics similar to a Buckingham + Coulomb model. | 1.0000 |
| Q3 | M3GNet | clai_00649 | COMPARISON_WINNER | For a MgO test set, M3GNet outperforms Moment Tensor Potentials in extrapolating equation‑of‑state behavior beyond the training regime and exhibits long‑range interaction characteristics similar to a Buckingham + Coulomb model. | 1.0000 |
| Q3 | M3GNet | clai_00343 | COMPARISON_BASELINE | The SevenNet machine-learned interatomic potential outperforms the earlier M3GNet model on the Matbench Discovery benchmark. | 0.0000 |

Detected baselines now have zero positive support. The multi-comparison MEGNet/SchNet statement is UNCERTAIN and receives .25 rather than unconditional full support. The former SevenNet/M3GNet baseline statement receives zero. This tests attribution behavior; it does not establish that the resulting rank improves. A comparison can still be relevant background even when it is not positive evidence for the requested condition.

### All scored requirements supported and rank improved: Q12, NequIP (2025)

Question: What are the leading (by Feb 2026) ML-based interatomic potential methods for large-scale atomistic simulation?

R3 216 → H2 87; CR0-100 45, CR1-100 3, CR2-100 41, CR3-100 18, CR4-100 89, CR1-200 4, CR2-200 72.

CR1 coverage 1.000, score 0.627507; CR2 coverage 0.000, score 0.000000.

| Kind | Extracted condition | Evidence node | Attribution | Evidence text | Cosine support | NLI support | Entail | Neutral | Contradict | Candidate-specific hypothesis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| core_task | large-scale atomistic simulation | clai_00190 | DIRECT_SUBJECT_SUPPORT | NequIP reproduces structural and kinetic properties from ab‑initio molecular dynamics simulations with high fidelity. | 0.7922 | 0.0000 | 0.0009 | 0.0041 | 0.9949 | NequIP is suitable for large-scale atomistic simulation. |

### Partial support and demotion: Q10, SchNet (2025)

Question: Which methods (by Feb 2026) are best suited for representing atomic environments when the system exhibits significant compositional diversity, defects, and disorder?

R3 28 → H2 23; CR0-100 16, CR1-100 29, CR2-100 30, CR3-100 20, CR4-100 30, CR1-200 45, CR2-200 58.

CR1 coverage 0.667, score 0.234129; CR2 coverage 0.000, score 0.000000.

| Kind | Extracted condition | Evidence node | Attribution | Evidence text | Cosine support | NLI support | Entail | Neutral | Contradict | Candidate-specific hypothesis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| core_task | representing atomic environments | clai_00488 | EXPLICIT_NAME_SUPPORT | MEGNet models achieve lower mean absolute errors than SchNet on 11 of the 13 QM9 molecular properties evaluated. | 0.5035 | 0.0000 | 0.0013 | 0.9949 | 0.0038 | SchNet is suitable for representing atomic environments. |
| requirement | the system exhibits significant compositional diversity, defects | clai_00488 | EXPLICIT_NAME_SUPPORT | MEGNet models achieve lower mean absolute errors than SchNet on 11 of the 13 QM9 molecular properties evaluated. | 0.5213 | 0.0000 | 0.0011 | 0.9960 | 0.0029 | SchNet is suitable for representing atomic environments when the system exhibits significant compositional diversity, defects. |
| requirement | disorder | clai_00488 | EXPLICIT_NAME_SUPPORT | MEGNet models achieve lower mean absolute errors than SchNet on 11 of the 13 QM9 molecular properties evaluated. | 0.3831 | 0.0000 | 0.0019 | 0.9925 | 0.0056 | SchNet is suitable for representing atomic environments when disorder. |

### NLI improves a gold rank relative to cosine: Q14, Matbench (2025)

Question: What datasets and benchmarks are commonly used (by Feb 2026) to evaluate MLIPs and property prediction models for materials discovery?

R3 97 → H2 53; CR0-100 88, CR1-100 67, CR2-100 18, CR3-100 73, CR4-100 31, CR1-200 100, CR2-200 20.

CR1 coverage 0.000, score 0.035811; CR2 coverage 0.000, score 0.000020.

| Kind | Extracted condition | Evidence node | Attribution | Evidence text | Cosine support | NLI support | Entail | Neutral | Contradict | Candidate-specific hypothesis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| core_task | evaluate MLIPs and property prediction models for materials discovery | data_00117 | UNCERTAIN | Matbench experimental metallicity classification | 0.1892 | 0.0045 | 0.0198 | 0.9783 | 0.0019 | Matbench is suitable for evaluate MLIPs and property prediction models for materials discovery. |

### NLI worsens a gold rank relative to cosine: Q1, GAP (2025)

Question: Which methods (by Feb 2026) are best suited for predicting interatomic potentials when balancing accuracy close to DFT with computational efficiency for systems of millions of atoms?

R3 206 → H2 47; CR0-100 93, CR1-100 11, CR2-100 93, CR3-100 15, CR4-100 48, CR1-200 14, CR2-200 182.

CR1 coverage 1.000, score 0.507742; CR2 coverage 0.500, score 0.000000.

| Kind | Extracted condition | Evidence node | Attribution | Evidence text | Cosine support | NLI support | Entail | Neutral | Contradict | Candidate-specific hypothesis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| core_task | predicting interatomic potentials | clai_00683 | DIRECT_SUBJECT_SUPPORT | GAP sp3 concentration predictions are closer to DFT reference than empirical Brenner and Tersoff potentials, particularly at lower densities; empirical potentials underestimate sp3 concentration in tetrahedral amorphous carbon. | 0.7482 | 0.8546 | 0.8751 | 0.1043 | 0.0206 | GAP is suitable for predicting interatomic potentials. |
| requirement | accuracy close to DFT | clai_00683 | DIRECT_SUBJECT_SUPPORT | GAP sp3 concentration predictions are closer to DFT reference than empirical Brenner and Tersoff potentials, particularly at lower densities; empirical potentials underestimate sp3 concentration in tetrahedral amorphous carbon. | 0.7357 | 0.9828 | 0.9837 | 0.0153 | 0.0010 | GAP is suitable for predicting interatomic potentials when accuracy close to DFT. |
| requirement | computational efficiency | clai_00683 | DIRECT_SUBJECT_SUPPORT | GAP sp3 concentration predictions are closer to DFT reference than empirical Brenner and Tersoff potentials, particularly at lower densities; empirical potentials underestimate sp3 concentration in tetrahedral amorphous carbon. | 0.5960 | 0.0019 | 0.0020 | 0.9979 | 0.0001 | GAP supports predicting interatomic potentials with computational efficiency. |
| requirement | systems of millions of atoms | clai_00683 | DIRECT_SUBJECT_SUPPORT | GAP sp3 concentration predictions are closer to DFT reference than empirical Brenner and Tersoff potentials, particularly at lower densities; empirical potentials underestimate sp3 concentration in tetrahedral amorphous carbon. | 0.6466 | 0.0000 | 0.0002 | 0.9988 | 0.0010 | GAP supports predicting interatomic potentials with systems of millions of atoms. |

### Good H2 rank but no direct attributable evidence: Q2, HamGNN (2025)

Question: Which methods (by Feb 2026) are best suited for accelerating electronic structure calculations when self-consistent field iterations become prohibitively expensive for large systems (>10³ atoms)?

R3 90 → H2 24; CR0-100 91, CR1-100 55, CR2-100 80, CR3-100 44, CR4-100 24, CR1-200 91, CR2-200 159.

CR1 coverage 0.000, score 0.034376; CR2 coverage 0.000, score 0.000000.

| Kind | Extracted condition | Evidence node | Attribution | Evidence text | Cosine support | NLI support | Entail | Neutral | Contradict | Candidate-specific hypothesis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| core_task | accelerating electronic structure calculations | clai_00199 | UNCERTAIN | Incorporating orbital‑energy loss in a second fine‑tuning round markedly improves band‑structure predictions, yielding energy bands and Fermi surfaces that closely match DFT calculations for the test set. | 0.1935 | 0.0000 | 0.0035 | 0.9672 | 0.0293 | Universal HamGNN Hamiltonian model is suitable for accelerating electronic structure calculations. |
| requirement | self-consistent field iterations become prohibitively expensive | arti_00029 | UNCERTAIN | Universal Machine Learning Kohn-Sham Hamiltonian for Materials | 0.1604 | 0.0000 | 0.0004 | 0.9948 | 0.0048 | Universal HamGNN Hamiltonian model is suitable for accelerating electronic structure calculations when self-consistent field iterations become prohibitively expensive. |
| requirement | large systems (>10³ atoms) | clai_00193 | UNCERTAIN | The universal HamGNN model accurately predicts the energy band structures and charge densities of complex multi‑element crystals (e.g., Hf2Zr9Ta6Ti5Nb5B54, HfTaTiB4MoC4, K3Ba3Li2Al4B6O20F), matching DFT results. | 0.1807 | 0.0000 | 0.0010 | 0.9980 | 0.0010 | Universal HamGNN Hamiltonian model is suitable for accelerating electronic structure calculations when large systems (>10³ atoms). |

## Local NLI validity diagnostics

Standalone exact-score ties retain the existing entity-ID tie break; fusion uses average tied ranks before combining with H2. A zero component makes the geometric score zero, so large tied groups can lose H2 ordering. No epsilon or new tie policy was fitted after observing results. The core is deliberately counted both as a gate and inside the geometric mean: its exponent is 1 + 1/n for n positive components.

| Year | Q | K | Scorer | Exact zeros | Distinct scores | Largest tie | Score std | Fallback candidates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025 | Q1 | 100 | CR1 | 3 | 96 | 3 | 0.1981 | 0 |
| 2025 | Q1 | 100 | CR2 | 97 | 4 | 97 | 0.0214 | 0 |
| 2025 | Q1 | 200 | CR1 | 21 | 168 | 21 | 0.1821 | 0 |
| 2025 | Q1 | 200 | CR2 | 196 | 5 | 196 | 0.0151 | 0 |
| 2025 | Q2 | 100 | CR1 | 5 | 93 | 5 | 0.1926 | 0 |
| 2025 | Q2 | 100 | CR2 | 100 | 1 | 100 | 0.0000 | 0 |
| 2025 | Q2 | 200 | CR1 | 19 | 171 | 19 | 0.1765 | 0 |
| 2025 | Q2 | 200 | CR2 | 200 | 1 | 200 | 0.0000 | 0 |
| 2025 | Q3 | 100 | CR1 | 8 | 88 | 8 | 0.1792 | 0 |
| 2025 | Q3 | 100 | CR2 | 98 | 3 | 98 | 0.0186 | 0 |
| 2025 | Q3 | 200 | CR1 | 28 | 163 | 28 | 0.1615 | 0 |
| 2025 | Q3 | 200 | CR2 | 198 | 3 | 198 | 0.0132 | 0 |
| 2025 | Q4 | 100 | CR1 | 8 | 87 | 8 | 0.1649 | 0 |
| 2025 | Q4 | 100 | CR2 | 99 | 2 | 99 | 0.0028 | 0 |
| 2025 | Q4 | 200 | CR1 | 25 | 151 | 25 | 0.1444 | 0 |
| 2025 | Q4 | 200 | CR2 | 197 | 4 | 197 | 0.0020 | 0 |
| 2025 | Q5 | 100 | CR1 | 10 | 91 | 10 | 0.1841 | 0 |
| 2025 | Q5 | 100 | CR2 | 99 | 2 | 99 | 0.0424 | 0 |
| 2025 | Q5 | 200 | CR1 | 39 | 154 | 39 | 0.1641 | 0 |
| 2025 | Q5 | 200 | CR2 | 199 | 2 | 199 | 0.0300 | 0 |
| 2025 | Q6 | 100 | CR1 | 2 | 97 | 2 | 0.1584 | 0 |
| 2025 | Q6 | 100 | CR2 | 100 | 1 | 100 | 0.0000 | 0 |
| 2025 | Q6 | 200 | CR1 | 26 | 172 | 26 | 0.1428 | 0 |
| 2025 | Q6 | 200 | CR2 | 199 | 2 | 199 | 0.0002 | 0 |
| 2025 | Q7 | 100 | CR1 | 1 | 97 | 3 | 0.1877 | 0 |
| 2025 | Q7 | 100 | CR2 | 98 | 3 | 98 | 0.0000 | 0 |
| 2025 | Q7 | 200 | CR1 | 22 | 170 | 22 | 0.1729 | 0 |
| 2025 | Q7 | 200 | CR2 | 198 | 3 | 198 | 0.0000 | 0 |
| 2025 | Q8 | 100 | CR1 | 4 | 96 | 4 | 0.1687 | 0 |
| 2025 | Q8 | 100 | CR2 | 97 | 4 | 97 | 0.0031 | 0 |
| 2025 | Q8 | 200 | CR1 | 24 | 157 | 24 | 0.1530 | 0 |
| 2025 | Q8 | 200 | CR2 | 191 | 10 | 191 | 0.0022 | 0 |
| 2025 | Q9 | 100 | CR1 | 4 | 97 | 4 | 0.0475 | 0 |
| 2025 | Q9 | 100 | CR2 | 61 | 40 | 61 | 0.1085 | 0 |
| 2025 | Q9 | 200 | CR1 | 25 | 161 | 25 | 0.0463 | 0 |
| 2025 | Q9 | 200 | CR2 | 134 | 67 | 134 | 0.0807 | 0 |
| 2025 | Q10 | 100 | CR1 | 10 | 90 | 10 | 0.1564 | 0 |
| 2025 | Q10 | 100 | CR2 | 90 | 11 | 90 | 0.0666 | 0 |
| 2025 | Q10 | 200 | CR1 | 49 | 149 | 49 | 0.1411 | 0 |
| 2025 | Q10 | 200 | CR2 | 187 | 14 | 187 | 0.0490 | 0 |
| 2025 | Q11 | 100 | CR1 | 17 | 80 | 17 | 0.1747 | 0 |
| 2025 | Q11 | 100 | CR2 | 37 | 64 | 37 | 0.1009 | 0 |
| 2025 | Q11 | 200 | CR1 | 44 | 150 | 44 | 0.1505 | 0 |
| 2025 | Q11 | 200 | CR2 | 89 | 112 | 89 | 0.1093 | 0 |
| 2025 | Q12 | 100 | CR1 | 8 | 92 | 8 | 0.2148 | 0 |
| 2025 | Q12 | 100 | CR2 | 73 | 28 | 73 | 0.1058 | 0 |
| 2025 | Q12 | 200 | CR1 | 27 | 165 | 27 | 0.1984 | 0 |
| 2025 | Q12 | 200 | CR2 | 155 | 46 | 155 | 0.1013 | 0 |
| 2025 | Q13 | 100 | CR1 | 5 | 95 | 5 | 0.1951 | 0 |
| 2025 | Q13 | 100 | CR2 | 74 | 27 | 74 | 0.0296 | 0 |
| 2025 | Q13 | 200 | CR1 | 36 | 154 | 36 | 0.1832 | 0 |
| 2025 | Q13 | 200 | CR2 | 169 | 32 | 169 | 0.0210 | 0 |
| 2025 | Q14 | 100 | CR1 | 2 | 95 | 2 | 0.1699 | 0 |
| 2025 | Q14 | 100 | CR2 | 64 | 37 | 64 | 0.0319 | 0 |
| 2025 | Q14 | 200 | CR1 | 24 | 155 | 24 | 0.1642 | 0 |
| 2025 | Q14 | 200 | CR2 | 141 | 60 | 141 | 0.0227 | 0 |
| 2026 | Q1 | 100 | CR1 | 1 | 98 | 2 | 0.2072 | 0 |
| 2026 | Q1 | 100 | CR2 | 96 | 5 | 96 | 0.0214 | 0 |
| 2026 | Q1 | 200 | CR1 | 21 | 170 | 21 | 0.1892 | 0 |
| 2026 | Q1 | 200 | CR2 | 195 | 6 | 195 | 0.0151 | 0 |
| 2026 | Q2 | 100 | CR1 | 4 | 94 | 4 | 0.1971 | 0 |
| 2026 | Q2 | 100 | CR2 | 100 | 1 | 100 | 0.0000 | 0 |
| 2026 | Q2 | 200 | CR1 | 18 | 171 | 18 | 0.1769 | 0 |
| 2026 | Q2 | 200 | CR2 | 200 | 1 | 200 | 0.0000 | 0 |
| 2026 | Q3 | 100 | CR1 | 6 | 90 | 6 | 0.1851 | 0 |
| 2026 | Q3 | 100 | CR2 | 98 | 3 | 98 | 0.0186 | 0 |
| 2026 | Q3 | 200 | CR1 | 23 | 164 | 23 | 0.1664 | 0 |
| 2026 | Q3 | 200 | CR2 | 198 | 3 | 198 | 0.0132 | 0 |
| 2026 | Q4 | 100 | CR1 | 9 | 87 | 9 | 0.1809 | 0 |
| 2026 | Q4 | 100 | CR2 | 100 | 1 | 100 | 0.0000 | 0 |
| 2026 | Q4 | 200 | CR1 | 21 | 168 | 21 | 0.1536 | 0 |
| 2026 | Q4 | 200 | CR2 | 197 | 4 | 197 | 0.0000 | 0 |
| 2026 | Q5 | 100 | CR1 | 10 | 91 | 10 | 0.1700 | 0 |
| 2026 | Q5 | 100 | CR2 | 99 | 2 | 99 | 0.0424 | 0 |
| 2026 | Q5 | 200 | CR1 | 36 | 153 | 36 | 0.1569 | 0 |
| 2026 | Q5 | 200 | CR2 | 199 | 2 | 199 | 0.0300 | 0 |
| 2026 | Q6 | 100 | CR1 | 1 | 98 | 2 | 0.1633 | 0 |
| 2026 | Q6 | 100 | CR2 | 100 | 1 | 100 | 0.0000 | 0 |
| 2026 | Q6 | 200 | CR1 | 25 | 173 | 25 | 0.1453 | 0 |
| 2026 | Q6 | 200 | CR2 | 199 | 2 | 199 | 0.0002 | 0 |
| 2026 | Q7 | 100 | CR1 | 0 | 98 | 3 | 0.1919 | 0 |
| 2026 | Q7 | 100 | CR2 | 98 | 3 | 98 | 0.0000 | 0 |
| 2026 | Q7 | 200 | CR1 | 13 | 181 | 13 | 0.1700 | 0 |
| 2026 | Q7 | 200 | CR2 | 198 | 3 | 198 | 0.0000 | 0 |
| 2026 | Q8 | 100 | CR1 | 5 | 95 | 5 | 0.1712 | 0 |
| 2026 | Q8 | 100 | CR2 | 98 | 3 | 98 | 0.0031 | 0 |
| 2026 | Q8 | 200 | CR1 | 15 | 169 | 15 | 0.1496 | 0 |
| 2026 | Q8 | 200 | CR2 | 193 | 8 | 193 | 0.0022 | 0 |
| 2026 | Q9 | 100 | CR1 | 4 | 96 | 4 | 0.0490 | 0 |
| 2026 | Q9 | 100 | CR2 | 62 | 39 | 62 | 0.1057 | 0 |
| 2026 | Q9 | 200 | CR1 | 24 | 171 | 24 | 0.0475 | 0 |
| 2026 | Q9 | 200 | CR2 | 131 | 70 | 131 | 0.0817 | 0 |
| 2026 | Q10 | 100 | CR1 | 12 | 89 | 12 | 0.1556 | 0 |
| 2026 | Q10 | 100 | CR2 | 91 | 10 | 91 | 0.0666 | 0 |
| 2026 | Q10 | 200 | CR1 | 50 | 151 | 50 | 0.1377 | 0 |
| 2026 | Q10 | 200 | CR2 | 188 | 13 | 188 | 0.0490 | 0 |
| 2026 | Q11 | 100 | CR1 | 13 | 83 | 13 | 0.1601 | 0 |
| 2026 | Q11 | 100 | CR2 | 31 | 70 | 31 | 0.1240 | 0 |
| 2026 | Q11 | 200 | CR1 | 42 | 152 | 42 | 0.1424 | 0 |
| 2026 | Q11 | 200 | CR2 | 77 | 124 | 77 | 0.1317 | 0 |
| 2026 | Q12 | 100 | CR1 | 4 | 95 | 4 | 0.2277 | 0 |
| 2026 | Q12 | 100 | CR2 | 65 | 36 | 65 | 0.0898 | 0 |
| 2026 | Q12 | 200 | CR1 | 21 | 171 | 21 | 0.1991 | 0 |
| 2026 | Q12 | 200 | CR2 | 148 | 53 | 148 | 0.0704 | 0 |
| 2026 | Q13 | 100 | CR1 | 3 | 97 | 3 | 0.1969 | 0 |
| 2026 | Q13 | 100 | CR2 | 74 | 27 | 74 | 0.0296 | 0 |
| 2026 | Q13 | 200 | CR1 | 27 | 163 | 27 | 0.1797 | 0 |
| 2026 | Q13 | 200 | CR2 | 170 | 31 | 170 | 0.0210 | 0 |
| 2026 | Q14 | 100 | CR1 | 3 | 93 | 3 | 0.1807 | 0 |
| 2026 | Q14 | 100 | CR2 | 66 | 35 | 66 | 0.0319 | 0 |
| 2026 | Q14 | 200 | CR1 | 20 | 163 | 20 | 0.1673 | 0 |
| 2026 | Q14 | 200 | CR2 | 143 | 58 | 143 | 0.0227 | 0 |

Selected evidence/hypothesis pairs: 10716 (includes repeated uses across candidates/cutoffs, unlike unique forward-pair cost). Predicted labels: {"neutral": 8247, "contradiction": 2035, "entailment": 434}; truncated premises: 0. Mean entailment 0.0470, mean neutral 0.7501.

### Highest entailment

| Year/Q | Candidate | Evidence | Hypothesis | Entail | Neutral | Contradict |
| --- | --- | --- | --- | --- | --- | --- |
| 2025/Q8 | ConvLSTM | The ConvLSTM model accurately predicts fracture patterns, crack‑length trends, and both mode‑I (tensile) and mode‑II (shear) loading responses across crystal orientations, closely matching atomistic molecular dynamics simulations, with only a minor discrepancy for the x100 orientation under mode‑II shear. | ConvLSTM is suitable for predicting fracture patterns. | 0.9967 | 0.0031 | 0.0002 |
| 2026/Q8 | ConvLSTM | The ConvLSTM model accurately predicts fracture patterns, crack‑length trends, and both mode‑I (tensile) and mode‑II (shear) loading responses across crystal orientations, closely matching atomistic molecular dynamics simulations, with only a minor discrepancy for the x100 orientation under mode‑II shear. | ConvLSTM is suitable for predicting fracture patterns. | 0.9967 | 0.0031 | 0.0002 |
| 2025/Q1 | MatterSim | MatterSim, a zero‑shot machine‑learned interatomic potential trained on >17 million DFT‑labeled structures, supports universal simulation across all elements, temperatures and pressures, achieving energy prediction errors below 50 meV/atom even in high‑temperature, non‑equilibrium regimes. | MatterSim is suitable for predicting interatomic potentials. | 0.9952 | 0.0043 | 0.0005 |

### Highest contradiction

| Year/Q | Candidate | Evidence | Hypothesis | Entail | Neutral | Contradict |
| --- | --- | --- | --- | --- | --- | --- |
| 2026/Q5 | Comprehensive review of AI-driven multiscale modeling approaches for solid‑state physics and chemistry | computational scalability of quantum mechanical methods | Comprehensive review of AI-driven multiscale modeling approaches for solid‑state physics and chemistry is suitable for coarse-graining molecular dynamics. | 0.0000 | 0.0001 | 0.9999 |
| 2026/Q5 | Comprehensive review of AI-driven multiscale modeling approaches for solid‑state physics and chemistry | computational scalability of quantum mechanical methods | Comprehensive review of AI-driven multiscale modeling approaches for solid‑state physics and chemistry is suitable for coarse-graining molecular dynamics when long-range elastic interactions must be preserved. | 0.0000 | 0.0002 | 0.9998 |
| 2025/Q5 | Orb-v3: atomistic simulation at scale | Orb-v3 models achieve competitive thermal conductivity prediction (κSRME), demonstrating direct models can yield smooth second- and third-order derivatives of the potential energy surface. | Orb-v3: atomistic simulation at scale is suitable for coarse-graining molecular dynamics when long-range elastic interactions must be preserved. | 0.0000 | 0.0003 | 0.9996 |

### Highest neutral

| Year/Q | Candidate | Evidence | Hypothesis | Entail | Neutral | Contradict |
| --- | --- | --- | --- | --- | --- | --- |
| 2025/Q7 | AFLOW high-throughput materials discovery framework | AFLOW: an automatic framework for high-throughput materials discovery | AFLOW high-throughput materials discovery framework is suitable for predicting phase transitions when screening ~50,000 inorganic compounds at high throughput. | 0.0001 | 0.9997 | 0.0002 |
| 2026/Q7 | AFLOW high-throughput materials discovery framework | AFLOW: an automatic framework for high-throughput materials discovery | AFLOW high-throughput materials discovery framework is suitable for predicting phase transitions when screening ~50,000 inorganic compounds at high throughput. | 0.0001 | 0.9997 | 0.0002 |
| 2026/Q1 | LEARNING INTER-ATOMIC POTENTIALS WITHOUT EXPLICIT EQUIVARIANCE | A generic unconstrained Transformer can serve as an effective backbone for machine-learned interatomic potentials without explicit equivariance constraints. | LEARNING INTER-ATOMIC POTENTIALS WITHOUT EXPLICIT EQUIVARIANCE supports predicting interatomic potentials with systems of millions of atoms. | 0.0002 | 0.9996 | 0.0002 |

### Numerical/scaling wording

| Year/Q | Candidate | Evidence | Hypothesis | Entail | Neutral | Contradict |
| --- | --- | --- | --- | --- | --- | --- |
| 2025/Q1 | Orb-v3: atomistic simulation at scale | Orb-v3 can simulate the carbonic anhydrase II enzyme system with over 20,000 atoms under fully solvated conditions for over 700 ps without unphysical behavior. | Orb-v3: atomistic simulation at scale supports predicting interatomic potentials with systems of millions of atoms. | 0.9692 | 0.0257 | 0.0051 |
| 2026/Q1 | Orb-v3: atomistic simulation at scale | Orb-v3 can simulate the carbonic anhydrase II enzyme system with over 20,000 atoms under fully solvated conditions for over 700 ps without unphysical behavior. | Orb-v3: atomistic simulation at scale supports predicting interatomic potentials with systems of millions of atoms. | 0.9692 | 0.0257 | 0.0051 |
| 2025/Q1 | Orb-v3 | Orb-v3 can simulate the carbonic anhydrase II enzyme system with over 20,000 atoms under fully solvated conditions for over 700 ps without unphysical behavior. | Orb-v3 supports predicting interatomic potentials with systems of millions of atoms. | 0.9646 | 0.0250 | 0.0104 |

### Negation/exclusion

| Year/Q | Candidate | Evidence | Hypothesis | Entail | Neutral | Contradict |
| --- | --- | --- | --- | --- | --- | --- |
| 2025/Q4 | VGNN | VGNN achieves direct prediction of Γ-phonon spectra and full phonon dispersion using only atomic coordinates as input without prior knowledge of interatomic forces. | VGNN is suitable for phonon property prediction. | 0.9940 | 0.0056 | 0.0003 |
| 2026/Q4 | VGNN | VGNN achieves direct prediction of Γ-phonon spectra and full phonon dispersion using only atomic coordinates as input without prior knowledge of interatomic forces. | VGNN is suitable for phonon property prediction. | 0.9940 | 0.0056 | 0.0003 |
| 2026/Q10 | LEARNING INTER-ATOMIC POTENTIALS WITHOUT EXPLICIT EQUIVARIANCE | A generic unconstrained Transformer can serve as an effective backbone for machine-learned interatomic potentials without explicit equivariance constraints. | LEARNING INTER-ATOMIC POTENTIALS WITHOUT EXPLICIT EQUIVARIANCE is suitable for representing atomic environments. | 0.9899 | 0.0099 | 0.0002 |

### Conjunction

| Year/Q | Candidate | Evidence | Hypothesis | Entail | Neutral | Contradict |
| --- | --- | --- | --- | --- | --- | --- |
| 2025/Q10 | Atomic cluster expansion of scalar, vectorial and tensorial properties and including magnetism and charge transfer | The atomic cluster expansion (ACE) is generalized to model scalar, vectorial and tensorial atomic properties and to include additional atomic degrees of freedom—species, magnetic moments, and charges—on equal footing with atomic positions. | Atomic cluster expansion of scalar, vectorial and tensorial properties and including magnetism and charge transfer is suitable for representing atomic environments. | 0.9845 | 0.0148 | 0.0007 |
| 2026/Q10 | Atomic cluster expansion of scalar, vectorial and tensorial properties and including magnetism and charge transfer | The atomic cluster expansion (ACE) is generalized to model scalar, vectorial and tensorial atomic properties and to include additional atomic degrees of freedom—species, magnetic moments, and charges—on equal footing with atomic positions. | Atomic cluster expansion of scalar, vectorial and tensorial properties and including magnetism and charge transfer is suitable for representing atomic environments. | 0.9845 | 0.0148 | 0.0007 |
| 2025/Q11 | MatterSim | MatterSim, a zero‑shot machine‑learned interatomic potential trained on >17 million DFT‑labeled structures, supports universal simulation across all elements, temperatures and pressures, achieving energy prediction errors below 50 meV/atom even in high‑temperature, non‑equilibrium regimes. | MatterSim is suitable for integrating features from three or more distinct scales simultaneously within a unified framework. | 0.9781 | 0.0210 | 0.0009 |

### Controlled comparison and technical probes

| Kind | Premise | Hypothesis | Entail | Neutral | Contradict |
| --- | --- | --- | --- | --- | --- |
| actual_comparison_baseline | The SevenNet machine-learned interatomic potential outperforms the earlier M3GNet model on the Matbench Discovery benchmark. | M3GNet outperforms the other model. | 0.0018 | 0.0008 | 0.9973 |
| paired_actual_comparison_subject | The SevenNet machine-learned interatomic potential outperforms the earlier M3GNet model on the Matbench Discovery benchmark. | SevenNet outperforms the other model. | 0.9935 | 0.0057 | 0.0008 |
| paired_actual_comparison_subject | The SevenNet machine-learned interatomic potential outperforms the earlier M3GNet model on the Matbench Discovery benchmark. | M3GNet outperforms the other model. | 0.0018 | 0.0008 | 0.9973 |
| technical_scale_control | D4FT successfully converges on unstable fullerene fragments (C₁₂₀, C₁₄₀) where PySCF fails to converge within the default iteration limit. | D4FT converges on carbon fragments containing 120 and 140 atoms. | 0.9807 | 0.0098 | 0.0095 |
| technical_scale_control | D4FT successfully converges on unstable fullerene fragments (C₁₂₀, C₁₄₀) where PySCF fails to converge within the default iteration limit. | D4FT converges on carbon fragments containing more than one million atoms. | 0.0001 | 0.9972 | 0.0027 |
| technical_identity_control | The SevenNet machine-learned interatomic potential outperforms the earlier M3GNet model on the Matbench Discovery benchmark. | The SevenNet machine-learned interatomic potential outperforms the earlier M3GNet model on the Matbench Discovery benchmark. | 0.9838 | 0.0147 | 0.0015 |
| technical_negation_control | The SevenNet machine-learned interatomic potential outperforms the earlier M3GNet model on the Matbench Discovery benchmark. | It is not true that the SevenNet machine-learned interatomic potential outperforms the earlier M3GNet model on the Matbench Discovery benchmark. | 0.0003 | 0.0016 | 0.9981 |
| technical_identity_control | D4FT successfully converges on unstable fullerene fragments (C₁₂₀, C₁₄₀) where PySCF fails to converge within the default iteration limit. | D4FT successfully converges on unstable fullerene fragments (C₁₂₀, C₁₄₀) where PySCF fails to converge within the default iteration limit. | 0.9826 | 0.0148 | 0.0027 |
| technical_negation_control | D4FT successfully converges on unstable fullerene fragments (C₁₂₀, C₁₄₀) where PySCF fails to converge within the default iteration limit. | It is not true that d4FT successfully converges on unstable fullerene fragments (C₁₂₀, C₁₄₀) where PySCF fails to converge within the default iteration limit. | 0.0003 | 0.0022 | 0.9976 |
| technical_identity_control | M3GNet molecular‑dynamics simulations of Li₃YCl₆ reproduce ionic conductivity (14.3 mS cm⁻¹) and activation energy (0.22 eV) consistent with ab‑initio MD, and short‑time MD correctly identifies known lithium superionic conductors by their high mean‑squared displacements at elevated temperatures. | M3GNet molecular‑dynamics simulations of Li₃YCl₆ reproduce ionic conductivity (14.3 mS cm⁻¹) and activation energy (0.22 eV) consistent with ab‑initio MD, and short‑time MD correctly identifies known lithium superionic conductors by their high mean‑squared displacements at elevated temperatures. | 0.9604 | 0.0315 | 0.0081 |
| technical_negation_control | M3GNet molecular‑dynamics simulations of Li₃YCl₆ reproduce ionic conductivity (14.3 mS cm⁻¹) and activation energy (0.22 eV) consistent with ab‑initio MD, and short‑time MD correctly identifies known lithium superionic conductors by their high mean‑squared displacements at elevated temperatures. | It is not true that m3GNet molecular‑dynamics simulations of Li₃YCl₆ reproduce ionic conductivity (14.3 mS cm⁻¹) and activation energy (0.22 eV) consistent with ab‑initio MD, and short‑time MD correctly identifies known lithium superionic conductors by their high mean‑squared displacements at elevated temperatures. | 0.0029 | 0.0159 | 0.9812 |
| actual_high_entailment | MatterSim, a zero‑shot machine‑learned interatomic potential trained on >17 million DFT‑labeled structures, supports universal simulation across all elements, temperatures and pressures, achieving energy prediction errors below 50 meV/atom even in high‑temperature, non‑equilibrium regimes. | MatterSim is suitable for predicting interatomic potentials. | 0.9952 | 0.0043 | 0.0005 |
| mismatched_premise_control | The computational cost of LiFlow scales linearly with the number of atoms (O(n)), in contrast to the cubic scaling (O(n³)) of ab initio molecular dynamics. | MatterSim is suitable for predicting interatomic potentials. | 0.0294 | 0.9097 | 0.0609 |
| actual_high_entailment | DeepH-E3 enables accurate electronic‑structure calculations for supercells larger than 10⁴ atoms, reducing computational cost by several orders of magnitude and allowing calculations in minutes on a single GPU versus months on thousands of CPU cores. | DeepH-E3 is suitable for accelerating electronic structure calculations. | 0.9946 | 0.0049 | 0.0004 |
| mismatched_premise_control | The SevenNet machine-learned interatomic potential outperforms the earlier M3GNet model on the Matbench Discovery benchmark. | DeepH-E3 is suitable for accelerating electronic structure calculations. | 0.0142 | 0.9488 | 0.0370 |
| actual_high_entailment | DeepH-E3 provides a universal E(3)-equivariant deep-learning framework that exactly preserves Euclidean symmetry of the DFT Hamiltonian, including spin-orbit coupling. | DeepH-E3 is suitable for learning DFT Hamiltonians. | 0.9906 | 0.0089 | 0.0005 |
| mismatched_premise_control | The SevenNet machine-learned interatomic potential outperforms the earlier M3GNet model on the Matbench Discovery benchmark. | DeepH-E3 is suitable for learning DFT Hamiltonians. | 0.0237 | 0.6028 | 0.3735 |
| actual_high_entailment | VGNN achieves direct prediction of Γ-phonon spectra and full phonon dispersion using only atomic coordinates as input without prior knowledge of interatomic forces. | VGNN is suitable for phonon property prediction. | 0.9940 | 0.0056 | 0.0003 |
| mismatched_premise_control | The SevenNet machine-learned interatomic potential outperforms the earlier M3GNet model on the Matbench Discovery benchmark. | VGNN is suitable for phonon property prediction. | 0.0107 | 0.7786 | 0.2107 |
| actual_high_entailment | The ConvLSTM model accurately predicts fracture patterns, crack‑length trends, and both mode‑I (tensile) and mode‑II (shear) loading responses across crystal orientations, closely matching atomistic molecular dynamics simulations, with only a minor discrepancy for the x100 orientation under mode‑II shear. | ConvLSTM is suitable for coarse-graining molecular dynamics. | 0.9401 | 0.0459 | 0.0141 |
| mismatched_premise_control | The SevenNet machine-learned interatomic potential outperforms the earlier M3GNet model on the Matbench Discovery benchmark. | ConvLSTM is suitable for coarse-graining molecular dynamics. | 0.0024 | 0.0103 | 0.9874 |
| actual_high_entailment | CHGNet molecular dynamics simulations of Li superionic conductors reproduce room‑temperature ionic conductivities and activation energies within the error bars of AIMD, correctly distinguishing faster from slower conductors. | Molecular dynamics is suitable for predicting ionic conductivity. | 0.9892 | 0.0101 | 0.0007 |
| mismatched_premise_control | The SevenNet machine-learned interatomic potential outperforms the earlier M3GNet model on the Matbench Discovery benchmark. | Molecular dynamics is suitable for predicting ionic conductivity. | 0.0160 | 0.8084 | 0.1757 |
| actual_high_entailment | Validation against experimental reports and independent first‑principles calculations confirms the robustness and predictive power of the ML‑guided framework, with reasonable agreement for transition temperatures of KNO₃, KNO₂ and CaCO₃, DFT‑calculated free‑energy differences matching ML predictions for a ten‑material subset, and thermal‑conductivity ratios for Li₄TiS₄ and NaNO₃ closely reproduced by DFT. | first-principles calculations is suitable for predicting phase transitions. | 0.9678 | 0.0298 | 0.0024 |
| mismatched_premise_control | The SevenNet machine-learned interatomic potential outperforms the earlier M3GNet model on the Matbench Discovery benchmark. | first-principles calculations is suitable for predicting phase transitions. | 0.0197 | 0.2805 | 0.6999 |
| actual_high_entailment | The ConvLSTM model accurately predicts fracture patterns, crack‑length trends, and both mode‑I (tensile) and mode‑II (shear) loading responses across crystal orientations, closely matching atomistic molecular dynamics simulations, with only a minor discrepancy for the x100 orientation under mode‑II shear. | ConvLSTM is suitable for predicting fracture patterns. | 0.9967 | 0.0031 | 0.0002 |
| mismatched_premise_control | The SevenNet machine-learned interatomic potential outperforms the earlier M3GNet model on the Matbench Discovery benchmark. | ConvLSTM is suitable for predicting fracture patterns. | 0.0165 | 0.9623 | 0.0211 |
| actual_high_entailment | The M3GNet architecture preserves continuity of energies, forces, and stresses with respect to changes in bond count, satisfying a necessary physical constraint for interatomic potentials. | M3GNet is suitable for ensuring physical consistency. | 0.9940 | 0.0056 | 0.0003 |
| mismatched_premise_control | The SevenNet machine-learned interatomic potential outperforms the earlier M3GNet model on the Matbench Discovery benchmark. | M3GNet is suitable for ensuring physical consistency. | 0.1296 | 0.7920 | 0.0785 |
| actual_high_entailment | MatterSim, a zero‑shot machine‑learned interatomic potential trained on >17 million DFT‑labeled structures, supports universal simulation across all elements, temperatures and pressures, achieving energy prediction errors below 50 meV/atom even in high‑temperature, non‑equilibrium regimes. | MatterSim is suitable for representing atomic environments. | 0.9870 | 0.0099 | 0.0031 |
| mismatched_premise_control | The SevenNet machine-learned interatomic potential outperforms the earlier M3GNet model on the Matbench Discovery benchmark. | MatterSim is suitable for representing atomic environments. | 0.0702 | 0.8702 | 0.0595 |
| actual_high_entailment | Combined two-body, three-body, and SOAP many-body descriptors are necessary for GAP to achieve the accuracy limit imposed by force locality and make physically meaningful predictions; models using only subsets of descriptors produce unphysical structures during molecular dynamics simulations. | GAP is suitable for integrating features from three or more distinct scales simultaneously within a unified framework. | 0.9616 | 0.0336 | 0.0048 |
| mismatched_premise_control | The SevenNet machine-learned interatomic potential outperforms the earlier M3GNet model on the Matbench Discovery benchmark. | GAP is suitable for integrating features from three or more distinct scales simultaneously within a unified framework. | 0.0326 | 0.9485 | 0.0189 |
| actual_high_entailment | MatterSim, a zero‑shot machine‑learned interatomic potential trained on >17 million DFT‑labeled structures, supports universal simulation across all elements, temperatures and pressures, achieving energy prediction errors below 50 meV/atom even in high‑temperature, non‑equilibrium regimes. | MatterSim is suitable for large-scale atomistic simulation. | 0.9551 | 0.0393 | 0.0056 |
| mismatched_premise_control | The SevenNet machine-learned interatomic potential outperforms the earlier M3GNet model on the Matbench Discovery benchmark. | MatterSim is suitable for large-scale atomistic simulation. | 0.0270 | 0.9591 | 0.0140 |
| actual_high_entailment | In polycrystalline aluminum, the GNN predicts atomic potential energy with a mean normalized relative error of 11.0% (maximum ≈13%) and achieves R² = 0.99 for mean potential energy versus grain number. | normalized relative error is suitable for predicting properties of solid-state crystalline materials. | 0.7229 | 0.2531 | 0.0240 |
| mismatched_premise_control | The SevenNet machine-learned interatomic potential outperforms the earlier M3GNet model on the Matbench Discovery benchmark. | normalized relative error is suitable for predicting properties of solid-state crystalline materials. | 0.0019 | 0.9930 | 0.0051 |
| actual_high_entailment | Predicting ionic conductivity in solids from the machine-learned potential energy landscape | Frozen-framework potential energy surface (PES) descriptor screening is suitable for evaluate MLIPs and property prediction models for materials discovery. | 0.8724 | 0.0923 | 0.0352 |
| mismatched_premise_control | The SevenNet machine-learned interatomic potential outperforms the earlier M3GNet model on the Matbench Discovery benchmark. | Frozen-framework potential energy surface (PES) descriptor screening is suitable for evaluate MLIPs and property prediction models for materials discovery. | 0.0444 | 0.3949 | 0.5607 |

These post-hoc controls do not enter any ranking. Exact identity/negation and paired comparison subjects test logical consistency; numerical extrapolation is not entailed merely by a smaller-system result. Mismatched graph premises are an imperfect negative control, not independently annotated false statements. The probabilities and any counterexamples are retained even when they undermine the NLI model.

The actual rank disagreements above separate cosine/NLI behavior, but not their causal scientific correctness. High neutral probabilities can be appropriate when a premise states a property but the template asserts general suitability; a high-entailment numerical extrapolation or wrong comparison subject is stronger evidence of failure. The supplied answer key does not label these premise/hypothesis pairs, so classifier errors are not equated automatically with gold rank losses.

## Every canonical target

| Year | Q | Target | R3 | H2 | CR0-100 | CR1-100 | CR2-100 | CR3-100 | CR4-100 | CR1-200 | CR2-200 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025 | Q1 | ACE | 269 | 154 | 154 | 154 | 154 | 154 | 154 | 57 | 177 |
| 2025 | Q1 | CHGNet | 119 | 88 | 100 | 97 | 69 | 95 | 88 | 170 | 137 |
| 2025 | Q1 | EquiformerV2 | 23 | 42 | 75 | 23 | 59 | 24 | 45 | 34 | 115 |
| 2025 | Q1 | GAP | 206 | 47 | 93 | 11 | 93 | 15 | 48 | 14 | 182 |
| 2025 | Q1 | M3GNet | 101 | 26 | 22 | 14 | 63 | 11 | 29 | 17 | 127 |
| 2025 | Q1 | MACE | 333 | 114 | 114 | 114 | 114 | 114 | 114 | 51 | 156 |
| 2025 | Q1 | NequIP | 188 | 91 | 37 | 17 | 21 | 53 | 91 | 22 | 38 |
| 2025 | Q2 | D4FT | 10 | 12 | 2 | 11 | 90 | 4 | 12 | 13 | 181 |
| 2025 | Q2 | DeepH | 260 | 42 | 74 | 1 | 73 | 7 | 42 | 1 | 143 |
| 2025 | Q2 | HamGNN | 90 | 24 | 91 | 55 | 80 | 44 | 24 | 91 | 159 |
| 2025 | Q3 | DeepH-E3 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 2025 | Q4 | VGNN | 2292 | 65 | 39 | 1 | 51 | 17 | 65 | 1 | 111 |
| 2025 | Q6 | LiFlow | 8 | 2 | 1 | 2 | 60 | 1 | 2 | 2 | 121 |
| 2025 | Q7 | symbolic regression | 159 | 186 | 186 | 186 | 186 | 186 | 186 | 26 | 4 |
| 2025 | Q8 | ConvLSTM | 138 | 25 | 28 | 1 | 1 | 1 | 1 | 1 | 1 |
| 2025 | Q9 | E(3)-equivariant GNNs | 2266 | 119 | 119 | 119 | 119 | 119 | 119 | 141 | 19 |
| 2025 | Q9 | NequIP | 196 | 111 | 111 | 111 | 111 | 111 | 111 | 26 | 91 |
| 2025 | Q10 | SchNet | 28 | 23 | 16 | 29 | 30 | 20 | 30 | 45 | 58 |
| 2025 | Q12 | CHGNet | 107 | 73 | 97 | 92 | 77 | 89 | 77 | 161 | 155 |
| 2025 | Q12 | DeePMD | 229 | 111 | 111 | 111 | 111 | 111 | 111 | 17 | 161 |
| 2025 | Q12 | EquiformerV2 | 24 | 32 | 82 | 22 | 71 | 16 | 51 | 33 | 138 |
| 2025 | Q12 | GAP | 215 | 50 | 89 | 23 | 12 | 33 | 20 | 34 | 20 |
| 2025 | Q12 | M3GNet | 97 | 29 | 8 | 7 | 74 | 7 | 48 | 9 | 145 |
| 2025 | Q12 | MACE | 334 | 104 | 104 | 104 | 104 | 104 | 104 | 54 | 167 |
| 2025 | Q12 | MatterSim | 130 | 35 | 68 | 2 | 1 | 4 | 4 | 2 | 1 |
| 2025 | Q12 | NequIP | 216 | 87 | 45 | 3 | 41 | 18 | 89 | 4 | 72 |
| 2025 | Q12 | Orb | 398 | 130 | 130 | 130 | 130 | 130 | 130 | 187 | 125 |
| 2025 | Q12 | SevenNet | 175 | 41 | 51 | 5 | 5 | 9 | 7 | 6 | 8 |
| 2025 | Q12 | eSEN | 2320 | 283 | 283 | 283 | 283 | 283 | 283 | 283 | 283 |
| 2025 | Q13 | CGCNN | 160 | 107 | 107 | 107 | 107 | 107 | 107 | 29 | 194 |
| 2025 | Q13 | CHGNet | 136 | 100 | 99 | 94 | 76 | 100 | 100 | 152 | 144 |
| 2025 | Q13 | EquiformerV2 | 24 | 57 | 79 | 29 | 67 | 53 | 65 | 45 | 128 |
| 2025 | Q13 | M3GNet | 111 | 55 | 23 | 1 | 2 | 5 | 12 | 1 | 2 |
| 2025 | Q13 | MACE | 253 | 106 | 106 | 106 | 106 | 106 | 106 | 49 | 162 |
| 2025 | Q13 | MEGNet | 185 | 185 | 185 | 185 | 185 | 185 | 185 | 180 | 119 |
| 2025 | Q13 | NequIP | 186 | 72 | 53 | 11 | 37 | 40 | 75 | 17 | 57 |
| 2025 | Q13 | SchNet | 34 | 36 | 8 | 67 | 41 | 63 | 50 | 98 | 69 |
| 2025 | Q14 | AFLOW | 260 | 66 | 94 | 5 | 2 | 17 | 19 | 7 | 2 |
| 2025 | Q14 | Alexandria | 3381 | 353 | 353 | 353 | 353 | 353 | 353 | 353 | 353 |
| 2025 | Q14 | GNoME | 126 | 41 | 30 | 24 | 66 | 28 | 58 | 42 | 133 |
| 2025 | Q14 | JARVIS-DFT | 547 | 541 | 541 | 541 | 541 | 541 | 541 | 541 | 541 |
| 2025 | Q14 | Matbench | 97 | 53 | 88 | 67 | 18 | 73 | 31 | 100 | 20 |
| 2025 | Q14 | Matbench Discovery | 2 | 8 | 1 | 28 | 41 | 8 | 25 | 47 | 66 |
| 2025 | Q14 | Materials Project | 106 | 35 | 51 | 15 | 11 | 15 | 14 | 26 | 11 |
| 2025 | Q14 | OMat24 | 475 | 71 | 90 | 82 | 78 | 83 | 79 | 121 | 156 |
| 2025 | Q14 | OQMD | 118 | 101 | 101 | 101 | 101 | 101 | 101 | 30 | 26 |
| 2025 | Q14 | WBM | 674 | 726 | 726 | 726 | 726 | 726 | 726 | 726 | 726 |
| 2026 | Q1 | ACE | 320 | 170 | 170 | 170 | 170 | 170 | 170 | 54 | 179 |
| 2026 | Q1 | CHGNet | 135 | 96 | 99 | 99 | 68 | 99 | 96 | 174 | 135 |
| 2026 | Q1 | EquiformerV2 | 27 | 32 | 90 | 25 | 58 | 19 | 36 | 36 | 113 |
| 2026 | Q1 | GAP | 241 | 57 | 55 | 15 | 93 | 25 | 58 | 17 | 183 |
| 2026 | Q1 | M3GNet | 114 | 29 | 19 | 17 | 61 | 13 | 33 | 20 | 125 |
| 2026 | Q1 | MACE | 439 | 128 | 128 | 128 | 128 | 128 | 128 | 48 | 156 |
| 2026 | Q1 | NequIP | 219 | 103 | 103 | 103 | 103 | 103 | 103 | 24 | 39 |
| 2026 | Q2 | D4FT | 12 | 14 | 3 | 11 | 90 | 4 | 14 | 14 | 180 |
| 2026 | Q2 | DeepH | 309 | 73 | 96 | 1 | 71 | 18 | 73 | 1 | 144 |
| 2026 | Q2 | HamGNN | 102 | 64 | 91 | 58 | 79 | 71 | 64 | 94 | 160 |
| 2026 | Q3 | DeepH-E3 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 2026 | Q4 | VGNN | 2567 | 59 | 44 | 1 | 48 | 16 | 59 | 1 | 110 |
| 2026 | Q6 | LiFlow | 9 | 2 | 1 | 2 | 59 | 1 | 2 | 2 | 124 |
| 2026 | Q7 | symbolic regression | 184 | 225 | 225 | 225 | 225 | 225 | 225 | 225 | 225 |
| 2026 | Q8 | ConvLSTM | 158 | 29 | 30 | 1 | 1 | 1 | 1 | 1 | 1 |
| 2026 | Q9 | E(3)-equivariant GNNs | 2391 | 139 | 139 | 139 | 139 | 139 | 139 | 141 | 21 |
| 2026 | Q9 | NequIP | 232 | 111 | 111 | 111 | 111 | 111 | 111 | 29 | 91 |
| 2026 | Q10 | SchNet | 32 | 25 | 18 | 26 | 36 | 21 | 32 | 39 | 59 |
| 2026 | Q12 | CHGNet | 121 | 73 | 98 | 95 | 76 | 90 | 82 | 170 | 149 |
| 2026 | Q12 | DeePMD | 265 | 56 | 34 | 17 | 81 | 28 | 70 | 20 | 156 |
| 2026 | Q12 | EquiformerV2 | 28 | 32 | 93 | 24 | 70 | 20 | 56 | 31 | 133 |
| 2026 | Q12 | GAP | 250 | 50 | 68 | 25 | 16 | 41 | 25 | 32 | 23 |
| 2026 | Q12 | M3GNet | 109 | 31 | 11 | 10 | 73 | 8 | 55 | 11 | 142 |
| 2026 | Q12 | MACE | 448 | 110 | 110 | 110 | 110 | 110 | 110 | 50 | 162 |
| 2026 | Q12 | MatterSim | 148 | 37 | 74 | 2 | 1 | 4 | 6 | 2 | 1 |
| 2026 | Q12 | NequIP | 251 | 92 | 50 | 6 | 49 | 25 | 93 | 6 | 76 |
| 2026 | Q12 | Orb | 446 | 112 | 112 | 112 | 112 | 112 | 112 | 191 | 122 |
| 2026 | Q12 | PET | 404 | 233 | 233 | 233 | 233 | 233 | 233 | 233 | 233 |
| 2026 | Q12 | SevenNet | 200 | 42 | 55 | 8 | 4 | 12 | 8 | 8 | 6 |
| 2026 | Q12 | UMA | 2564 | 402 | 402 | 402 | 402 | 402 | 402 | 402 | 402 |
| 2026 | Q12 | eSEN | 266 | 116 | 116 | 116 | 116 | 116 | 116 | 41 | 10 |
| 2026 | Q13 | CGCNN | 179 | 130 | 130 | 130 | 130 | 130 | 130 | 27 | 194 |
| 2026 | Q13 | CHGNet | 150 | 86 | 99 | 96 | 74 | 96 | 89 | 163 | 145 |
| 2026 | Q13 | EquiformerV2 | 25 | 49 | 80 | 29 | 65 | 48 | 60 | 43 | 127 |
| 2026 | Q13 | M3GNet | 124 | 87 | 26 | 1 | 2 | 14 | 18 | 1 | 2 |
| 2026 | Q13 | MACE | 386 | 105 | 105 | 105 | 105 | 105 | 105 | 45 | 163 |
| 2026 | Q13 | MEGNet | 206 | 205 | 205 | 205 | 205 | 205 | 205 | 205 | 205 |
| 2026 | Q13 | NequIP | 207 | 79 | 54 | 12 | 37 | 42 | 82 | 17 | 57 |
| 2026 | Q13 | SchNet | 38 | 25 | 5 | 69 | 41 | 53 | 39 | 101 | 69 |
| 2026 | Q14 | AFLOW | 301 | 56 | 95 | 7 | 2 | 14 | 13 | 9 | 2 |
| 2026 | Q14 | Alexandria | 467 | 147 | 147 | 147 | 147 | 147 | 147 | 186 | 105 |
| 2026 | Q14 | GNoME | 143 | 43 | 32 | 28 | 61 | 34 | 61 | 45 | 133 |
| 2026 | Q14 | JARVIS-DFT | 611 | 620 | 620 | 620 | 620 | 620 | 620 | 620 | 620 |
| 2026 | Q14 | MPtrj | 154 | 157 | 157 | 157 | 157 | 157 | 157 | 153 | 48 |
| 2026 | Q14 | Matbench | 110 | 58 | 90 | 74 | 17 | 82 | 31 | 109 | 20 |
| 2026 | Q14 | Matbench Discovery | 2 | 4 | 3 | 25 | 39 | 1 | 20 | 39 | 66 |
| 2026 | Q14 | Materials Project | 117 | 34 | 55 | 17 | 10 | 13 | 12 | 26 | 10 |
| 2026 | Q14 | OMat24 | 124 | 41 | 53 | 24 | 38 | 27 | 59 | 38 | 63 |
| 2026 | Q14 | OQMD | 131 | 96 | 73 | 20 | 21 | 61 | 50 | 31 | 26 |
| 2026 | Q14 | WBM | 185 | 183 | 183 | 183 | 183 | 183 | 183 | 29 | 16 |

## Decision questions

1. CR1 versus CR0 at 2025: 11/47 versus 6/47; micro Recall@10 change +0.1064.
2. CR2 versus CR1: 6/47 versus 11/47; change -0.1064.
3. Candidate-specific NLI is evaluated as a controlled entailment component only. Its technical suitability is assessed by the quality table, rank losses, neutral/contradiction distributions and actual probes; existence of a local classifier is not evidence it should become default.
   Here it does not improve the cosine result: CR2 retrieves 6/47 against CR1’s 11/47, with the technical contradiction failure illustrated above.
4. Oracle gap: CR0 0.5319, CR1 0.4255, CR2 0.5319. The combined decomposition/attribution/aggregation ablation cannot isolate the causal effect of decomposition alone.
5. New top-ten entries from H2 ranks 11–100: CR1-100: 9, CR2-100: 5, CR3-100: 7, CR4-100: 3.
6. Existing top-ten losses are tabulated separately against H2 and CR0; the two references overlap and are not added together.
   Primary losses (H2 / CR0): CR1-100: 1 / 3, CR2-100: 2 / 5, CR3-100: 0 / 1, CR4-100: 1 / 4.
7. The SevenNet/M3GNet statement receives zero positive support. The multi-comparison SchNet statement is UNCERTAIN (.25). Unconditional full support is removed, but actual ranks need not improve.
8. Primary failures and diagnostic flags distinguish absent candidates, weak textual attribution, zero/partial support, model contradiction and supported-but-below-ten. The experiment cannot reliably quantify decomposition correctness or scientific attribution accuracy without independent annotations.
9. CR3/CR4 retention and entry counts are in the movement table; RRF can retain H2 relevance but also suppress a standalone scorer’s gains. Fixed fusion is not an automatic improvement.
   At 2025, CR3 preserves all three H2 hits and five of six CR0 hits, while CR1 preserves two and three respectively. CR4 preserves two H2 hits and two CR0 hits. CR3 is the better retention tradeoff in this run.
10. The next-step decision is based on absolute recovery, protection of prior hits, oracle gap and trace validity, not the best single benchmark score. No hierarchical approximation, extra model, CR5 or parameter sweep was added in response to poor results.

**Decision:** retain generic requirement decomposition and attributed cosine as a promising experimental direction. Reject this pinned local NLI model/template/top-one configuration as the default for this retrieval use case: it underperforms CR1, adds inference work and loses prior good answers. This is not a rejection of every possible NLI model. CR3 offers a retention tradeoff, not a reason to replace H2 discovery. With roughly three quarters of canonical targets still absent from the top ten, imperfect decomposition/attribution, and no unseen-question validation, hierarchy approximation remains premature. The next quality study should validate these mechanisms on unseen questions before optimizing fine-level retrieval for speed.

## Reproduction, tests and provenance

```powershell
$env:HF_HUB_OFFLINE='1'
.\.venv\Scripts\python.exe -m tkh_abstraction.cli evaluate-constraints --config configs/constraints.yaml
.\.venv\Scripts\python.exe -m pytest -q --junitxml=artifacts/evaluation/constraint_reranking_tests.xml
.\.venv\Scripts\python.exe scripts/constraint_nli_probes.py
.\.venv\Scripts\python.exe scripts/report_constraints.py
.\.venv\Scripts\python.exe scripts/package_submission.py
```

Unseen questions use the existing `--questions` and `--ground-truth` CLI arguments and separate configured output paths. For another graph, or an isolated output directory without copied frozen baseline artifacts, set `constraint_verify_history: false`; no new question IDs or answer names need code changes. Changing the answer key changes evaluation only. The assessment audit expects the supplied full two-cutoff run. Do not run old report writers to overwrite historical reports.

Full suite: 193 passed, no errors/failures/skips. Added tests cover frozen old sources/oracle, CR0 parity, exact extractive spans, unrelated question wording, independent requirements, exclusions, temporal isolation, identity-bearing hypotheses, different evidence per condition, comparison direction/negation, mention/path/co-membership discount, core gating, NLI probabilities and fallback, offline loaders, no benchmark-rule strings, candidate preservation, ground-truth perturbation invariance and Type B/history preservation. Existing H2 recomputation, RRF and hierarchy regression tests remain in the suite.

Files: `question_decomposition.py` (`QuestionDecomposition`, `decompose_question`, `candidate_hypothesis`), `evidence_attribution.py` (`classify_attribution`), `constraint_scoring.py` (`ControlledLocalNLI`, `RequirementCosineReranker`, `RequirementNLIReranker`, `PreparedReranker`, `aggregate_support`), `constraint_pipeline.py` (`run_constraints`, diagnostic classification), CLI extension, `configs/constraints.yaml`, tests and two report/probe scripts. All seven requested machine-readable/test artifacts and this report are under `artifacts/evaluation`; checkpoints/protocol/audit/probes are under `constraints/`. Previous metrics/report are also archived under `artifacts/baseline_reranking`.

Ultralight local adaptation provided serial read-only Claude Opus planning and Claude Sonnet review via the official first-party subscription helper. Codex implemented, integrated and executed. Advice was checked rather than adopted blindly: graph co-membership is not subject support, and unavailable NLI probabilities are null rather than invented neutral results. Prompts/results, generic bug fixes and decisions are retained in `notes/constraints`. No human scientific adjudication is claimed.
