# Multi-resolution semantic abstraction over an evolving knowledge hypergraph

## Problem and formalization (Deliverable 0)

We ask whether native higher-order structure, text semantics and temporal continuity can yield a useful, budget-constrained scientific browsing hierarchy. At publication cutoff t, H_t=(V_t,E_t) contains nodes first seen by t and edges whose fact date, asserting-paper date and complete membership are visible by t. This is a retrospective corpus-publication simulation, not a reconstruction of the database's ingestion history. Model pretraining is also retrospective: frozen encoders can contain later background knowledge.

P_0(t),…,P_K(t) are exclusive, covering partitions of V_t. Every child in P_(k+1) has exactly one parent in P_k. Our chosen display budgets are (12,40,120), followed by |V_t| singleton leaves; these intermediate budgets are design choices. A current unit stores its underlying members, count, type distribution and base-embedding mean. A coarse edge retains its identity, provenance, original endpoints and multiplicities m_eC. At each level we greedily minimize merge increments of

**J(P_t) = λ_h J_hyper + λ_s J_spec + λ_m J_sem + λ_t J_temp**, subject to partition coverage, exclusivity, laminarity and |P_k|≤N_k.

J_hyper is the edge-weighted mean of (1−Σ_C(m_eC/n_e)²)/(1−1/n_e), including internal evidence with zero fragmentation. J_sem and J_spec are within-cluster sums of squares divided by their respective total input variance at that resolution. Semantic coordinates are normalized MiniLM text embeddings; supernode centroids are member-count-weighted means. Spectral coordinates are low-frequency eigenvectors of the current hypergraph operator. J_temp is VI against the predecessor at the same level, restricted to common base nodes and normalized by log(number of common nodes); it is zero for ≤1 common node. New nodes carry no direct VI penalty.

Precisely, for current-unit coordinates x_u and masses n_u, J_x=Σ_C Σ_(u∈C) n_u‖x_u−μ_C‖² / Σ_u n_u‖x_u−x̄‖², with a 10⁻¹² denominator floor. Its Ward merge increase before normalization is n_A n_B‖μ_A−μ_B‖²/(n_A+n_B). Within-unit dispersion is a fixed offset at that resolution. VI(P,Q)=H(P|Q)+H(Q|P), so both splitting and merging previous groups are penalized; the predecessor Q is fixed during construction.

Default weights (0.30,0.20,0.35,0.15) and unit relation weights are fixed design choices. We renormalize active weights, disabling the temporal term at the first snapshot. We optimize exact local deltas over restricted candidates, without claiming a global optimum. P1 and P2 hold by merging whole units and stopping at the budget. Native P4 is enforced procedurally. P3 semantic usefulness, P5 stability and P6 faithfulness are empirical properties, not guarantees.

## Literature positioning

[Zhou, Huang and Schölkopf (NIPS 2006)](https://papers.nips.cc/paper/2006/file/dff8e9c2ac33381546d96deea9922999-Paper.pdf) motivates the normalized incidence-based hypergraph operator. We retain that geometry but add direct multiplicity-aware fragmentation and semantics. [Chen, Saad and Zhang (2022)](https://doi.org/10.1007/s40324-021-00282-x) surveys recursive coarsening; we borrow the multilevel principle and recompute structure on each actual coarse hypergraph. [Schaub, Li and Peel (2023)](https://doi.org/10.1103/PhysRevE.107.054305) distinguishes hierarchy detection from merely producing a dendrogram: our imposed display levels are not evidence of a naturally occurring hierarchy. [Chi et al. (KDD 2007)](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/01/evospe.pdf) motivates balancing current fit against historical smoothness. We use partition VI, not their evolutionary spectral objective.

## Method and alternatives

For binary incidence B on non-internal current edges, D_e is support arity, W is edge weight and D_v=diag(Bw). We apply Θ=D_v^(-1/2)BW D_e^(-1)BᵀD_v^(-1/2), L=I−Θ through sparse incidence products. Isolated vertices have zero inverse degree and L_ii=1. Seeded Lanczos computes up to eight low-frequency coordinates. Ward distances are invariant to eigenvector sign; truncated degenerate eigenspaces can still be unstable. Coordinates stay fixed during individual merges and are recomputed after each completed resolution. Native fragmentation supplies local multiplicity information while the spectrum supplies global geometry.

Candidates combine native support cycles, semantic/spectral nearest neighbours, previous-cluster links and a deterministic connectivity chain. A heap invalidates consumed units lazily. VI deltas use sparse contingency rows; hyperedge deltas use incident-edge multiplicities. At each resolution T4 collapse adds endpoint multiplicities: {a,b,c,d} can become {S1:3,S2:1}; three or more supports remain one hyperedge. Fully internal edges remain evidence records but leave the spectral incidence matrix. The output of collapse is the next level's input. This loses endpoint distinctions in the working coarse geometry, although original members remain traceable.

Semantic-only, native structural, static hybrid, temporal hybrid and pairwise hybrid variants share budgets and deterministic construction. The pairwise baseline assigns w/choose(n,2) to each pair, conserving total edge weight; it is a credible alternative, not the main representation. Across the final native graph it produces 72,701 pair instances and 70,022 unique pairs (50.88× edges); 1,510 pairs combine multiple source edges. A triple of weight one and three pair edges of weight one-third have identical projected adjacency. Pure semantic clustering omits relations, pure structure omits scientific meaning, and independent snapshots omit construction-time continuity. These are useful controls, not universally inferior approaches.

Hungarian overlap matching assigns primary persistent identities after construction, while many-to-many overlap records split/merge interpretations. Thresholds are 0.20 Jaccard for continuation and 0.10 member share for events. Birth/death denote identity creation/retirement, which can coexist with split/merge; ambiguous overlaps are retained. Matching alone is not the temporal mechanism.

## Evaluation and validity

Inspection of preliminary failures found exact scientific names stored as `cited_work`; both method/data routes now include that type. Citation mapping requires exact names. Containment is restricted to method/technique/component/dataset names, so a future-work sentence merely mentioning a model is not a safe alias. These generic schema repairs occurred during development, with no score-based choice of thresholds or hierarchy settings. V2 node text uses a type prefix consistently for all retrieval systems; historical v1 used surface-only vectors.

The core hierarchy and its 25 validated exports are unchanged. Source, construction-configuration, input and artifact hashes prove reuse; the original metrics/report/archive remain under `artifacts/baseline_v1`. The new evaluation repairs a demonstrated schema mismatch: `expected_methods` contains heterogeneous targets. Unicode/case/space normalization preserves hyphens, plus signs, E(3) and versions. Exact names precede whole-identifier containment; distinct containing forms remain ambiguous. Identical normalized mentions are treated as one named target, a lexical assumption rather than expert entity resolution. Routing uses only question text and declared type; methods, datasets and conceptual questions get appropriate broad types, with an all-type fallback.

Four deterministic farthest-point member prototypes per eligible supernode augment its centroid: score = 0.5 centroid cosine + 0.5 mean of the best two prototype cosines. A best-first queue replaces the fixed global beam of three. Every actual query-vector comparison costs one: a four-prototype cluster costs five, a leaf one. Offline embedding/index work is excluded and logged separately. Scored roots/children follow deterministic ID order when the remaining budget cannot cover all candidates. Budgets 25 and 50 can fail to score the root layer: these are insufficient-budget failure diagnostics, not evidence of useful low-cost abstraction. Partial expansions can produce non-monotone curves; more search can also replace relevant outputs in the top ten.

All systems share normalized BGE node/query vectors, route filters and output K=10. Typed fixed-beam isolates search changes; the original method-only result remains a separate historical baseline. Exhaustive flat is plotted at its actual cost; a seeded query-independent flat scan is only a secondary equal-work reference. Budget grid 25,50,100,200,500, prototype count and weights were fixed before scoring, with no answer-key tuning. This development benchmark has already informed evaluator repairs and is not an untouched final test set.

Expected recall counts mapped named targets; strict precision counts returned nodes matching those names. Extended precision additionally accepts `valid_but_unlisted`, without adding recall credit. Wrong-domain and outdated-main rates are separate; other outputs remain unjudged. Empty-output precision is zero; questions without mapped targets have null recall. Type A and B results are stratified, with evaluable counts and mapping coverage in every curve row.

Claim retrieval follows returned entities through native relations and article provenance, then scores candidate claims against the question. Direct neighbors precede provenance candidates, with ID ties and a separate 100-comparison budget; reused exact claim-leaf scores are logged. This same path serves flat and hierarchical entities. An additional exhaustive claim-corpus reference is recorded separately. Evaluation-only alignment first resolves article titles/DOIs or author/year candidates, then checks lexical overlap, independent MiniLM similarity (at least .78), NLI entailment (at least .80), numbers and ambiguity. Retrieval never sees these alignments. Source overlap measures evidence provenance, not claim truth. All unaligned claims remain explicit. End-2025 is conservative primary evidence; annual 2026 is a sensitivity that can contain material after February.

Coherence uses surface-only BGE, separate from MiniLM construction, and excludes singleton clusters. Exact-size and within-type null permutations accompany observed means; p=(1+#null≥observed)/51 is descriptive and uncorrected across levels. Five seeds (11,23,37,53,71) remove round(10% of edges), rebuild each hierarchy with a fixed predecessor, and report ARI/VI at every level. Approximate t intervals quantify those removals; real cross-snapshot stability is measured separately.

New labels use extractive noun phrases and a one-sentence, non-comparative gloss. A deterministic type-stratified split reserves approximately 30% of members when feasible; generation sees only the other members and fully contained relations, with generation-local IDF. Inputs have exact keys `cutoff`, `generation_members` (id, type, surface_form, first_seen_year), `generation_type_counts`, and `generation_only_relation_counts`. A real member record is `{"id":"arti_00024","type":"article","surface_form":"The Joint Automated Repository for Various Integrated Simulations (JARVIS) for data-driven materials design","first_seen_year":2020}`. The complete corresponding packet and output are reproduced verbatim in the appendix; every full packet and label overlay is exported. Original hierarchy labels remain historical; v2 overlays are the revised labels.

Gloss NLI retains DeBERTa and the .50 threshold. Paired old/new glosses use identical, label-independent held-out chunks (at most six members, three chunks); truncation is recorded. Maximum entailment tests the new gloss's weaker existential claim, and per-chunk rates are also exported. This protocol cannot be compared directly with the old shared-evidence rate. Legacy generation already saw the later-held-out members; neither protocol withholds members from clustering or model pretraining. Lexical label support is separate from gloss NLI. Generation and other-cluster controls test proxy discrimination. Shorter, weaker glosses can score better while conveying less information. Blind human audit ratings remain blank.

## Measured results

The corpus has 5,798 nodes, 1,429 edges, 12 node types and 11 relations; historical origins span 1901–2026 and first appearances 1997–2026. There are 170 edges predating a member and 134 predating the asserting article (overlapping counts). Three nodes have origin after first appearance. Raw guide counts test edge year alone; our visible counts additionally require the asserting paper and every endpoint. The difference is deliberate temporal filtering, not dropped data.

| Cutoff | Nodes | Raw dated edges | Consistently visible edges | New nodes / edges |
|---|---:|---:|---:|---|
| 2020 | 1505 | 405 | 374 | 1505 / 374 |
| 2022 | 2164 | 589 | 526 | 659 / 152 |
| 2024 | 4164 | 1063 | 983 | 2000 / 457 |
| 2026 | 5798 | 1429 | 1429 | 1634 / 446 |

Type and arity distributions, temporal spans and growth are exported in `descriptive.json`. Event counts by transition/level, including birth, growth, continuation, merge, split and death, appear in the appendix.

| Variant | BGE coherence / null | Perturbation ARI [95% CI] | Real temporal ARI | Typed legacy / prototype recall |
|---|---|---|---|---|
| semantic | 0.783 / 0.760 | 1.000 [1.000, 1.000] | 0.397 | 0.000 / 0.000 |
| structural | 0.763 / 0.760 | 0.512 [0.392, 0.633] | 0.703 | 0.000 / 0.000 |
| static | 0.777 / 0.760 | 0.464 [0.439, 0.490] | 0.351 | 0.000 / 0.000 |
| temporal | 0.775 / 0.760 | 0.550 [0.497, 0.603] | 0.636 | 0.000 / 0.000 |
| pairwise | 0.776 / 0.760 | 0.644 [0.589, 0.700] | 0.696 | 0.000 / 0.000 |

The final L0 comparison above retains original intrinsic results; retrieval uses conservative 2025 and budget 500. Temporal coherence exceeds its size null by 0.015 (p=.020). Native structure need not win every metric: pairwise has higher perturbation ARI. Semantic-only ARI=1 reflects ignoring edges, not structural robustness.

Mapping changes from 21/50 unique names in the old method-only evaluator to 33/50; 0 are ambiguous and 17 unmapped. The new mapped target-instance count is 46/63 over the full Type A benchmark. The CSV contains 14 Type A questions, all retained in metrics.json; 12 have at least one mapped target and enter mapped-target macro recall, while Q5 (Bead-mapping, GNN+GPP); Q11 (CE→MC→NN→PF, hybrid frameworks) have no exact or accepted safe-alias node matches at the conservative 2025 cutoff and therefore have explicit not_evaluable status and null recall, rather than being omitted from the benchmark. Dataset routing now includes dataset/task/metric/article nodes. Claim alignment covers 0/188 Type A claims and 0/36 Type B claims. These are automated proxies, not verified ground truth; low coverage sharply limits interpretation.

No claim clears every fixed alignment gate, so claim recall is unavailable. Separately, 2/188 supplemental Type A evidence excerpts align and receive their own recall field; they are not substituted for the 224 claims. Unmapped names include composites/descriptions (GNN+GPP, hybrid frameworks), absent or variant-specific names, and ambiguous acronyms. Their exclusion from conditional recall is accompanied by full denominator counts and an all-expected lower bound; no aliases were added to chase scores.

| Search, temporal variant | Type A recall | Entity comparisons | Type B claim recall | Type B source hit | Total comparisons, Type B |
|---|---:|---:|---:|---:|---:|
| legacy_typed_fixed_beam | 0.000 | 84.286 | N/A | 0.482 | 164.750 |
| prototype_best_first | 0.000 | 500.000 | N/A | 0.214 | 582.500 |
| flat_exhaustive | 0.000 | 2393.643 | N/A | 0.518 | 2396.750 |
| flat_query_independent_scan | 0.000 | 500.000 | N/A | 0.429 | 594.750 |

The historical method-only recall was .020. Under the new typed protocol, temporal fixed-beam/prototype/flat recall is 0.000/0.000/0.000. The prototype-minus-fixed-beam difference is 0.000; the Type B source-hit difference is -0.268. Poor exhaustive-flat results limit what branch search alone can repair. Lower cost alone is not a retrieval-efficiency success.
Type A prototype strict/extended precision is 0.000/0.000; wrong-domain/outdated-main rates are 0.000/0.000. Type B source coverage is 0.847; claim-recall evaluable questions number 0/4. Annual-2026 prototype Type A recall is 0.000, compared with 0.000 in 2025. Lower comparison cost alone does not establish useful retrieval. Full curves, ranks, failed mappings, evidence and branch diagnostics remain available.

![Quality versus charged comparisons](figures/retrieval_v2.png)

| Level, across four snapshots | Evaluable / sampled | Legacy unsupported | New unsupported | New entail / neutral / contradiction | New lexical token coverage |
|---|---|---:|---:|---|---:|
| L0 | 48 / 48 | 0.958 | 0.542 | 0.458 / 0.479 / 0.062 | 0.823 |
| L1 | 48 / 48 | 0.875 | 0.479 | 0.521 / 0.354 / 0.125 | 0.705 |

The historical 2026 temporal L0/L1 non-entailment rates were .917/1.000 under shared evidence. The table instead gives the paired held-out comparison across all four snapshots. New single-chunk control entailment rates are {"generation": 0.2916666666666667, "other_cluster": 0.052083333333333336, "own_holdout": 0.23958333333333334}; weak own-versus-other separation would undermine interpretation as a discriminative faithfulness measure. NLI/BGE are fallible proxies and do not establish scientific correctness.

## Limitations, verified versus assumed, and next steps

Retain temporal hybrid as the assessment's evolving native-hypergraph browser: its final-transition ARI .636 exceeds static .351, while coherence decreases from .777 to .775. This is a measured stability–coherence tradeoff, not a scalar winner. Retrieval and incomplete alignment do not establish any variant as a validated question-answering system. Unmatched claims, sparse label holdouts and unfilled expert judgments remain explicit limitations; no numerical improvement is required for this repair to be valid.

Exact higher-order edge identity survives collapse, but the fragmentation statistic has a quadratic endpoint-pair interpretation and binary spectral incidence omits multiplicity. Greedy candidates and display budgets can dominate weights. Frozen language models may contain later knowledge; first-seen filtering cannot prove that unversioned 2026-export surface text never absorbed later wording. Source/year ambiguity and incomplete relevance labels remain. Four original intrinsic weight sensitivities are preserved, not selected against questions.

Verified facts are executed invariant/delta tests, hash-validated artifacts, measured corpus/metric values and explicit output traces. Assumptions include date semantics, lexical co-reference, automated claim alignments, design budgets/weights and NLI as support. No human verification is claimed. With four more weeks, obtain versioned text and source papers, adjudicate alignments, complete a blind expert label audit, reserve new questions for final testing, and only then investigate split-aware refinement and candidate sensitivity.

## Complexity

For n units, m edges, I incidences, semantic dimension d and spectral dimension q, storage is O(n+I); each sparse operator product costs O(I+n), plus Lanczos orthogonalization. No dense Laplacian is formed. Brute-force neighbors cost O(n²(d+q)) arithmetic in bounded memory; initial candidates are O(I+nk), but unions/heap updates can approach quadratic. Each level performs n−N merges with sparse incidence/contingency updates and O(d+q) centroid updates. Coarse Hungarian matching costs O(N³). Prototype construction is offline O(pnd) per level/type family, with p≤4. Query comparison counts omit embedding inference, graph enumeration and queue overhead; wall times are descriptive shared-machine observations. No large-scale complexity or speedup claim is made.
