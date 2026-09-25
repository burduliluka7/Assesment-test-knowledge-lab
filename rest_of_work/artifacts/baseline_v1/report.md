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

## Evaluation validity and circularity controls

| Component | Optimization or generation input | Evaluation input | Control and limitation |
|---|---|---|---|
| Coherence | MiniLM with optional node-type prefix | BGE-small-en-v1.5, surface form only | Different family; exact-size random-label null and within-type null; shared training sources may remain |
| Temporal construction | Previous partition via VI | Edge-removal ARI and real-snapshot ARI/VI | Five perturbations at the final cutoff; predecessor held fixed; real temporal agreement is optimization-adjacent |
| Labels | Snapshot-visible TF-IDF keyphrases, deterministic gloss template | Independent DeBERTa NLI plus blind audit CSV | NLI shares source evidence and is only a support proxy; no independent paper text or human ratings |
| Retrieval | Hierarchy, with no question-based tuning | Supplied questions and method ground truth | Same frozen BGE scorer, output K=10; all scored objects charged, including unopened supernodes |

Coherence excludes singleton clusters and reports both node- and cluster-weighted means. Null permutations preserve exact cluster sizes; the additional within-type permutation preserves type composition. Upper-tail p=(1+#null≥observed)/(R+1) is descriptive and uncorrected across multiple levels. Perturbations remove round(10% of visible edges) at seeds 11,23,37,53,71, retaining nodes and the original predecessor. Approximate t intervals describe these five removals, not population uncertainty. Cross-snapshot ARI/VI is reported separately with graph growth and event counts.

The benchmark actually contains 18 questions; 14 have expected methods and 10 have mappable targets at the primary cutoff. Only 21/50 unique expected names map. The expected_methods field also names datasets; our primary view is explicitly method-node-only. Claim-only questions are excluded from method scores. Mapping uses exact normalized names, then whole-token containment, never fuzzy semantic matches. A mapping audit corrected variant conflation by preserving hyphens/plus signs (DeepH is not DeepH-E3); retrieval was recomputed without changing construction or tuning. Ambiguous mappings remain visible. We report mapped recall and an explicitly labeled all-expected lower bound; strict precision can penalize valid but unlisted methods. Claim paraphrases lack verified node alignment, so supporting-claim recall is unavailable. Questions refer to February 2026 but dates are annual: 2025 is primary, 2026 a potentially post-cutoff sensitivity.

Hierarchical retrieval scores complete candidate batches and follows a fixed global beam of three; a batch that cannot fit the remaining budget is not scored. Flat retrieval exhaustively ranks method nodes. Both use the same method filter and BGE cosine scorer; centroids use only eligible method members. We compare quality at fixed output K and disclose different computation costs. Paired question bootstrap intervals accompany recall differences. Returned rank and scoring inspections are distinct; failures are censored. Both systems can additionally score up to 50 article-linked claims after method retrieval, with separate charged inspections; these are contextual evidence, not adjudicated claim support.

Labels summarize three corpus-derived keyphrases. Every generator input is exported. NLI flags entailment probability below 0.50 as a conservative non-entailment/overclaim proxy, including neutral predictions. Premises use up to eight keyword-relevant members and token truncation. This can reject vague but harmless labels and can miss factual errors inherited from the corpus. The blind human audit is unfilled; no manual faithfulness result is claimed.

## Results

The supplied data contains 5,798 nodes, 1,429 hyperedges, 12 node types and 11 relations. Historical origins span [1901, 2026]; corpus first appearances span [1997, 2026]. Validation records 170 edges before member visibility and 134 before their asserting paper. Deferring whole inconsistent edges avoids inventing truncated relations.

| Cutoff | Nodes | Visible edges | New nodes | New edges | Deferred dated edges |
|---|---:|---:|---:|---:|---:|
| 2020 | 1505 | 374 | 1505 | 374 | 31 |
| 2022 | 2164 | 526 | 659 | 152 | 63 |
| 2024 | 4164 | 983 | 2000 | 457 | 80 |
| 2026 | 5798 | 1429 | 1634 | 446 | 0 |

![Corpus growth and final arities](figures/growth_arity.png)

Final-cutoff L0 results follow. Coherence and null use BGE; temporal ARI is the last real transition. NLI is the proxy above. Retrieval is conservative-2025 recall@10 with a maximum 500 scored-object budget.

| Variant | Coherence | Null | Perturbation ARI [95% CI] | Temporal ARI | NLI flag rate | Retrieval recall |
|---|---:|---:|---|---:|---:|---:|
| semantic | 0.783 | 0.760 | 1.000 [1.000, 1.000] | 0.397 | 0.917 | 0.037 |
| structural | 0.763 | 0.760 | 0.512 [0.392, 0.633] | 0.703 | 1.000 | 0.020 |
| static | 0.777 | 0.760 | 0.464 [0.439, 0.490] | 0.351 | 0.917 | 0.040 |
| temporal | 0.775 | 0.760 | 0.550 [0.497, 0.603] | 0.636 | 0.917 | 0.020 |
| pairwise | 0.776 | 0.760 | 0.644 [0.589, 0.700] | 0.696 | 0.917 | 0.000 |

![Independent coherence and perturbation stability](figures/coherence_stability.png)

For the temporal variant, L0 coherence minus its size-preserving null is 0.015 (permutation p=0.020). Its 2025 mapped recall@10 is 0.020, versus flat 0.020, at mean 52.200 versus 408.000 score inspections. These are observed trade-offs on a small benchmark, not evidence of general superiority. Per-question scores, mapping coverage, both benchmark cutoffs, L1/L2 metrics, event counts and sensitivity runs are in metrics.json.

| Temporal variant NLI sample | n | Non-entailment flag rate |
|---|---:|---:|
| 2020:L0 | 12 | 0.750 |
| 2020:L1 | 12 | 0.833 |
| 2022:L0 | 12 | 0.750 |
| 2022:L1 | 12 | 0.917 |
| 2024:L0 | 12 | 1.000 |
| 2024:L1 | 12 | 1.000 |
| 2026:L0 | 12 | 0.917 |
| 2026:L1 | 12 | 1.000 |

![Search cost and recall](figures/retrieval.png)

## Limitations, shipping decision and next steps

Semantic-only has the strongest final L0 coherence (0.783) and takes 4.0s to construct, versus 23.0s for temporal. Its perturbation ARI of 1 is expected because removing edges cannot affect an edge-agnostic method; it is not evidence of robust use of structure. Static and temporal coherence are close (0.777 versus 0.775), but their last-transition ARIs differ (0.351 versus 0.636). The pairwise variant has higher mean perturbation ARI (0.644) than temporal (0.550), so native structure does not win every comparison. Retain temporal as the assessment's evolving native-hypergraph browser: continuity materially improves over static while coherence declines modestly. Do not ship any variant as a validated question-answering system: absolute retrieval recall is very low and alignment coverage is limited. The high NLI flag rates also mean label faithfulness has not been established; expert audit is required before relying on these abstractions.

Arbitrary display budgets, sparse candidates and a greedy trajectory can dominate weight settings. The four sensitivity settings vary semantic balance or temporal strength without tuning on questions, conditional on the default predecessor; they do not establish an optimal configuration.

The method's native fragmentation is a quadratic endpoint-pair agreement statistic: preserving edge objects prevents representational loss, but does not make this objective irreducibly higher-order. Binary spectral incidence also omits multiplicity intentionally. Labels are modest lexical abstractions, not expert-written macro-concepts. Evidence is unversioned surface text from a 2026 export; first-seen filtering cannot prove that a merged surface form never incorporated later wording. Frozen models may encode later knowledge. Neither NLI nor BGE constitutes an expert scientific audit. The supplied benchmark is small, includes mapping ambiguity and has incomplete relevance labels.

With four more weeks, obtain versioned text and source papers, independently adjudicate method/claim alignments, complete a blind expert label audit, and reserve a separate question set for final testing. Investigate split-aware local refinement, candidate sensitivity and eigenspace degeneracy only after these validity improvements.

## Complexity, verified facts and design choices

Let n,m,I,d,q be current units, edges, support incidences, semantic dimension and spectral dimension. Loading/storage cost O(n+I); each sparse operator-vector product costs O(I+n), plus Lanczos orthogonalization. We never form a dense Laplacian. Brute-force nearest-neighbour search costs O(n²(d+q)) arithmetic in bounded memory; the sparse candidate set starts at O(I+nk), but neighbour unions and heap pushes can grow toward quadratic. Each level takes n−N_k merges. Hyperedge and VI updates use sparse incident/contingency maps; centroid updates cost O(d+q), and heap updates add log(heap size). Coarse matching uses an O(N_k³) Hungarian solve; singleton matching is direct. Encoder inference depends on token length and model architecture. Recorded wall times are descriptive shared-machine observations, not a controlled runtime benchmark. No large-graph scalability claim is made.

Verified facts are the measured corpus statistics, executed invariant/delta tests, actual hierarchy validation and exported metric values. All snapshot hierarchies are validated before export. Budgets, weights, thresholds, model selection, corpus-date interpretation and NLI as a proxy are assumptions/design choices. Human audit ratings and verified supporting-claim recall remain unavailable. Exact model revisions, input hashes, configuration, seeds, dependency versions and AI assistance are disclosed in the repository.
