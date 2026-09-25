# Query-seeded native hypergraph diffusion

This is a Type A candidate-generation experiment. Graph construction, hierarchy memberships, resolver, grouping rules, historical R0–R5 artifacts and Type B evidence outputs are preserved. Defaults were declared before diffusion outcomes: alpha=0.85, top-M=100, tolerance=1e-8, maximum 100 iterations, RRF k=60. Sensitivities do not select a new default.

## Mathematical mechanism and attribution

[Zhou, Huang and Schölkopf (NIPS 2006), sections 5 and 7](https://proceedings.neurips.cc/paper_files/paper/2006/file/dff8e9c2ac33381546d96deea9922999-Paper.pdf) define the symmetric normalized hypergraph operator and transductive inference from initial **labels**. They did not evaluate semantic-query retrieval. Our adaptation supplies atomic query–node cosine relevance as the initial signal. The existing spectral code returns L=I−Theta, so `ThetaOperator.apply_theta(x)` reuses `x−L@x`. Dense allocation and inversion occur only in tiny correctness tests.

```text
Theta = Dv^(-1/2) H W De^(-1) H.T Dv^(-1/2)
y = L1_normalize(positive top-M atomic semantic scores)
f_next = (1-alpha)*y + alpha*Theta*f
f_limit = (1-alpha)*(I-alpha*Theta)^(-1)*y
```

The scalar (1−alpha) does not change rankings relative to the unscaled linear solve. The reused incidence path keeps edge-cardinality normalization and weighted vertex degrees. Original export edges have no numeric weight field, so native weight=1; optional explicit future weights are supported by the retrieval adapter. No confidence field is silently promoted to a weight. Singleton/internal edges follow the existing spectral exclusion policy; the supplied fine graph has valid arity≥2. Isolated seeded vertices retain only their restart term.

The symmetric operator does not conserve L1 probability mass: exported per-type “mass” is a sum of relevance scores. We do not renormalize between iterations. Stopping uses the specified L1 change; exported L2 error bounds divide the last change by (1−alpha). A maximum-iteration stop is explicitly marked unconverged. Test solves verify scaled equality at alpha=.50/.70/.85/.95 with stricter tolerance and sufficient test iterations.

The pairwise control uses the existing weight-conserving clique projection (w/choose(arity,2) per pair), followed by the same symmetric arity-2 operator. On nonisolated vertices this is 0.5*(I+normalized adjacency), including self retention. It is a lazy symmetric graph diffusion, not ordinary non-lazy PPR. Native Theta is itself a linear operator with an equivalent weighted pair expansion; an empirical difference here tests this particular projection/normalization, not irreducible higher-order expressivity.

## Frozen retrieval protocol

Atomic BGE vectors and original-question vectors reuse the existing exact-text/model cache and are explicitly normalized. Authors get no initial semantic mass, but remain structural intermediates in H0. Top-M ties break by node ID; negatives are zeroed. If all selected scores are nonpositive, the deterministic fallback is uniform over the selected eligible nodes and is flagged. A genuinely empty eligible pool retains zero signal.

Final answer groups admit method, technique, component, dataset, cited_work, article and metric nodes. Claims, tasks, problems, future topics and authors remain in the propagation graph. A group qualifies when any member has an admitted type; score aggregation is maximum, never duplicate-counting sum. H0 uses no added numeric type prior; fused variants inherit R3 priors through ranks. R3_answer_pool is the required matched-pool control; every canonical target remains eligible at both cutoffs. Historical R3 alone is not a fair causal contrast for filtering gains.

H1 combines matched-pool R3 and H0 by equal-weight RRF with fixed k=60 and average ranks for exact score ties; an arbitrary zero-score ID ordering contributes no distinct fusion ranks. A flagged nonpositive/empty-seed fallback makes fused variants retain R3 rather than treating an uninformative diffusion order as semantic evidence. H2 builds a graph-only explicit-mention index using bounded whole-name matching: distinctive identifiers or sufficiently informative multiword names, no numeric-only names, stopword acronyms, fuzzy aliases or generic short strings. It raises a group’s maximum own seed to its maximum mentioning-node seed, writing only one deterministic member to avoid replication across duplicate nodes, then globally normalizes. The evidence source keeps its seed, so total pre-normalization mass can grow and every unmodified seed is rescaled downward. All updates read original y, so mention links cannot cascade. H2 consequently can increase the number of nonzero seed nodes; it is not a fixed-support M control. H2 then diffuses and fuses with R3. Atomic top-M itself is node-level and can include repeated text at different nodes; max answer aggregation does not undo that seed-level multiplicity.

H3 keeps H2 and uses predeclared relation multipliers: claims/presents/solves/addresses=1; evaluated_on/uses_technique/uses_component=.75; extends/cites=.25; proposes_future_work=.10; authored_by=0. Weighted vertex degrees are recomputed. These are design priors, not fitted weights; changing a relation also changes normalization, so H3 is not a single-relation causal ablation. H3 was executed only after the uniform/mention checkpoints.

The shuffled control independently randomizes each edge’s distinct memberships, retaining node count, arity, relation labels and edge weights with seed42. It does not preserve node degrees, type mixes or connectivity. One shuffle gives a limited structural sanity check, not a significance test or clean semantic-only intervention. Pairwise and shuffled controls use exactly H0’s seeds. Type B is not scored or replaced by diffusion.

## 2025 conservative primary

| Variant | Candidate R@20 / 50 / 100 / 200 (macro) | R@10 micro | Raw recovered /63 | Median / mean gold rank | MRR | R-Prec = R@R |
| --- | --- | --- | --- | --- | --- | --- |
| R0 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | 0.0000 | 0/63 | 4231.0000 / 3843.9574 | 0.0004 | 0.0000 |
| R1 | 0.2028 / 0.2327 / 0.3542 / 0.5174 | 0.0851 | 4/63 | 233.0000 / 573.3191 | 0.1296 | 0.0917 |
| R3 | 0.2028 / 0.3264 / 0.3701 / 0.7140 | 0.0851 | 4/63 | 160.0000 / 378.8723 | 0.1589 | 0.0917 |
| H0 | 0.1944 / 0.3716 / 0.4994 / 0.5751 | 0.0426 | 2/63 | 184.0000 / 275.7447 | 0.0465 | 0.0000 |
| H0_shuffled | 0.0083 / 0.0917 / 0.0917 / 0.1833 | 0.0000 | 0/63 | 829.0000 / 996.7021 | 0.0105 | 0.0000 |
| H0_pairwise | 0.1944 / 0.3716 / 0.4994 / 0.5452 | 0.0213 | 1/63 | 261.0000 / 330.7872 | 0.0388 | 0.0000 |
| H1 | 0.2028 / 0.4549 / 0.5473 / 0.7764 | 0.0638 | 3/63 | 109.0000 / 243.5106 | 0.0908 | 0.0083 |
| H2 | 0.2028 / 0.5257 / 0.7146 / 0.9674 | 0.0638 | 3/63 | 66.0000 / 103.4043 | 0.1602 | 0.0917 |
| H3 | 0.2028 / 0.5166 / 0.7230 / 0.8841 | 0.0638 | 3/63 | 59.0000 / 102.5106 | 0.1587 | 0.0917 |

| Variant | Hit@1 / 5 / 10 / 20 | Recall@5 / 10 / 20 / 50 / 100 (macro) | Precision@5 / 10 | Unique-target R@10 | Ranked / canonical |
| --- | --- | --- | --- | --- | --- |
| R0 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 / 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 | 47/47 |
| R1 | 0.0833 / 0.1667 / 0.3333 / 0.3333 | 0.0917 / 0.2028 / 0.2028 / 0.2327 / 0.3542 | 0.0286 / 0.0286 | 0.1176 | 47/47 |
| R3 | 0.0833 / 0.1667 / 0.3333 / 0.3333 | 0.0917 / 0.2028 / 0.2028 / 0.3264 / 0.3701 | 0.0286 / 0.0286 | 0.1176 | 47/47 |
| H0 | 0.0000 / 0.0000 / 0.1667 / 0.2500 | 0.0000 / 0.1111 / 0.1944 / 0.3716 / 0.4994 | 0.0000 / 0.0143 | 0.0588 | 47/47 |
| H0_shuffled | 0.0000 / 0.0000 / 0.0000 / 0.0833 | 0.0000 / 0.0000 / 0.0083 / 0.0917 / 0.0917 | 0.0000 / 0.0000 | 0.0000 | 47/47 |
| H0_pairwise | 0.0000 / 0.0000 / 0.0833 / 0.2500 | 0.0000 / 0.0833 / 0.1944 / 0.3716 / 0.4994 | 0.0000 / 0.0071 | 0.0294 | 47/47 |
| H1 | 0.0000 / 0.0833 / 0.2500 / 0.3333 | 0.0833 / 0.1750 / 0.2028 / 0.4549 / 0.5473 | 0.0143 / 0.0214 | 0.0882 | 47/47 |
| H2 | 0.0833 / 0.1667 / 0.2500 / 0.3333 | 0.1667 / 0.1750 / 0.2028 / 0.5257 / 0.7146 | 0.0286 / 0.0214 | 0.0882 | 47/47 |
| H3 | 0.0833 / 0.1667 / 0.2500 / 0.3333 | 0.1667 / 0.1750 / 0.2028 / 0.5166 / 0.7230 | 0.0286 / 0.0214 | 0.0882 | 47/47 |

| Pool-only diagnostic | Candidate R@50 / 100 | R@10 micro | Median rank |
| --- | --- | --- | --- |
| R3_answer_pool | 0.3264 / 0.3979 | 0.0851 | 144.0000 |
| atomic_answer_pool | 0.0000 / 0.0000 | 0.0000 | 2198.0000 |

| Variant | Seed / R3 dot products | Theta applications (=iterations) | Incidence entry visits upper bound | Component runtime seconds | Unconverged queries |
| --- | --- | --- | --- | --- | --- |
| R0 | 0.0000 / 4792.0000 | 0.0000 | 0.0000 | 0.1431 | 0 |
| R1 | 0.0000 / 4792.0000 | 0.0000 | 0.0000 | 0.4593 | 0 |
| R3 | 0.0000 / 4792.0000 | 0.0000 | 0.0000 | 0.4295 | 0 |
| H0 | 4506.0000 / 0.0000 | 78.2857 | 1187750.8571 | 0.0384 | 0 |
| H0_shuffled | 4506.0000 / 0.0000 | 56.5714 | 858301.7143 | 0.0348 | 0 |
| H0_pairwise | 4506.0000 / 0.0000 | 79.1429 | 18124980.5714 | 0.0551 | 0 |
| H1 | 4506.0000 / 4792.0000 | 78.2857 | 1187750.8571 | 0.4757 | 0 |
| H2 | 4506.0000 / 4792.0000 | 77.7143 | 1179081.1429 | 0.5225 | 0 |
| H3 | 4506.0000 / 4792.0000 | 78.1429 | 1185583.4286 | 0.4798 | 0 |

## 2026 annual sensitivity; may include evidence after February

| Variant | Candidate R@20 / 50 / 100 / 200 (macro) | R@10 micro | Raw recovered /63 | Median / mean gold rank | MRR | R-Prec = R@R |
| --- | --- | --- | --- | --- | --- | --- |
| R0 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | 0.0000 | 0/63 | 4934.5000 / 4507.3000 | 0.0004 | 0.0000 |
| R1 | 0.2020 / 0.2188 / 0.3245 / 0.4875 | 0.0600 | 3/63 | 261.5000 / 517.0400 | 0.1222 | 0.0909 |
| R3 | 0.2020 / 0.3245 / 0.3245 / 0.6527 | 0.0600 | 3/63 | 181.5000 / 333.0400 | 0.1546 | 0.0909 |
| H0 | 0.1742 / 0.3296 / 0.4124 / 0.5485 | 0.0200 | 1/63 | 199.5000 / 259.4200 | 0.0330 | 0.0000 |
| H0_shuffled | 0.0000 / 0.0000 / 0.0000 / 0.0259 | 0.0000 | 0/63 | 911.5000 / 1169.5400 | 0.0025 | 0.0000 |
| H0_pairwise | 0.1667 / 0.3296 / 0.3941 / 0.4443 | 0.0200 | 1/63 | 264.0000 / 343.5000 | 0.0328 | 0.0000 |
| H1 | 0.2020 / 0.4407 / 0.5154 / 0.6045 | 0.0600 | 3/63 | 119.5000 / 217.5400 | 0.0888 | 0.0076 |
| H2 | 0.2020 / 0.4681 / 0.7040 / 0.8859 | 0.0600 | 3/63 | 73.0000 / 98.9600 | 0.1694 | 0.0909 |
| H3 | 0.2020 / 0.4617 / 0.7159 / 0.8859 | 0.0600 | 3/63 | 71.5000 / 101.0400 | 0.1634 | 0.0909 |

| Variant | Hit@1 / 5 / 10 / 20 | Recall@5 / 10 / 20 / 50 / 100 (macro) | Precision@5 / 10 | Unique-target R@10 | Ranked / canonical |
| --- | --- | --- | --- | --- | --- |
| R0 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 / 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 | 50/50 |
| R1 | 0.0833 / 0.0833 / 0.2500 / 0.3333 | 0.0833 / 0.1742 / 0.2020 / 0.2188 / 0.3245 | 0.0143 / 0.0214 | 0.0811 | 50/50 |
| R3 | 0.0833 / 0.1667 / 0.2500 / 0.3333 | 0.0909 / 0.1742 / 0.2020 / 0.3245 / 0.3245 | 0.0286 / 0.0214 | 0.0811 | 50/50 |
| H0 | 0.0000 / 0.0000 / 0.0833 / 0.2500 | 0.0000 / 0.0833 / 0.1742 / 0.3296 / 0.4124 | 0.0000 / 0.0071 | 0.0270 | 50/50 |
| H0_shuffled | 0.0000 / 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 / 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 | 50/50 |
| H0_pairwise | 0.0000 / 0.0000 / 0.0833 / 0.1667 | 0.0000 / 0.0833 / 0.1667 / 0.3296 / 0.3941 | 0.0000 / 0.0071 | 0.0270 | 50/50 |
| H1 | 0.0000 / 0.1667 / 0.2500 / 0.3333 | 0.0909 / 0.1742 / 0.2020 / 0.4407 / 0.5154 | 0.0286 / 0.0214 | 0.0811 | 50/50 |
| H2 | 0.0833 / 0.2500 / 0.2500 / 0.3333 | 0.1742 / 0.1742 / 0.2020 / 0.4681 / 0.7040 | 0.0429 / 0.0214 | 0.0811 | 50/50 |
| H3 | 0.0833 / 0.2500 / 0.2500 / 0.3333 | 0.1742 / 0.1742 / 0.2020 / 0.4617 / 0.7159 | 0.0429 / 0.0214 | 0.0811 | 50/50 |

| Pool-only diagnostic | Candidate R@50 / 100 | R@10 micro | Median rank |
| --- | --- | --- | --- |
| R3_answer_pool | 0.3245 / 0.3245 | 0.0600 | 165.5000 |
| atomic_answer_pool | 0.0000 / 0.0000 | 0.0000 | 2542.0000 |

| Variant | Seed / R3 dot products | Theta applications (=iterations) | Incidence entry visits upper bound | Component runtime seconds | Unconverged queries |
| --- | --- | --- | --- | --- | --- |
| R0 | 0.0000 / 5555.0000 | 0.0000 | 0.0000 | 0.1484 | 0 |
| R1 | 0.0000 / 5555.0000 | 0.0000 | 0.0000 | 0.5356 | 0 |
| R3 | 0.0000 / 5555.0000 | 0.0000 | 0.0000 | 0.5050 | 0 |
| H0 | 5237.0000 / 0.0000 | 78.7857 | 1406955.2857 | 0.0443 | 0 |
| H0_shuffled | 5237.0000 / 0.0000 | 56.7143 | 1012803.7143 | 0.0405 | 0 |
| H0_pairwise | 5237.0000 / 0.0000 | 79.4286 | 22246989.7143 | 0.0644 | 0 |
| H1 | 5237.0000 / 5555.0000 | 78.7857 | 1406955.2857 | 0.5577 | 0 |
| H2 | 5237.0000 / 5555.0000 | 77.8571 | 1390372.8571 | 0.6156 | 0 |
| H3 | 5237.0000 / 5555.0000 | 78.4286 | 1400577.4286 | 0.5626 | 0 |

All 14 Type A records remain explicit. Canonical recall uses 47 target instances at 2025 and 50 at 2026; Q5 and Q11 remain non-evaluable for canonical recall, with their expected labels retained in the raw /63 denominator. Macro candidate recall averages 12 evaluable questions; micro recall counts canonical instances; unique recall counts targets recovered in any associated question. Missing ranks would be censored at pool-size+1, with coverage shown; none are omitted here. Candidate prefixes and output ranks are identical because this pass has no reranker. Precision is fixed-K judged-target precision under incomplete labels. All per-K macro/micro/unique/raw values are in JSON.

Cost columns are per-query means. A sparse Theta application performs two sparse incidence multiplies; incidence entry visits are an operation-count proxy, not FLOPs or seconds. R3 dots and atomic-seed dots are separate. Fusion rows charge the sum of standalone component work even when the experiment reuses shared results; actual-new application counts are also exported. Runtime is a sum of measured query components including seed preparation, mention assistance, diffusion, answer aggregation and fusion; it excludes offline model/index/operator construction and diagnostic path searches. H3 weighting and pairwise projection are built once per snapshot; operator build time, nnz and edge counts are exported. This is local timing, not a controlled serving benchmark.

## Mechanistic evidence and limitations

| Seed type | Selected node occurrences across 14 queries | Mean initial score mass |
| --- | --- | --- |
| task | 334 | 0.2394 |
| problem | 317 | 0.2269 |
| claim | 300 | 0.2134 |
| future_topic | 118 | 0.0845 |
| article | 101 | 0.0719 |
| technique | 68 | 0.0484 |
| method | 61 | 0.0437 |
| component | 58 | 0.0412 |
| cited_work | 27 | 0.0192 |
| dataset | 15 | 0.0106 |
| metric | 1 | 0.0007 |

| Selected H2 top-100 path relation | Occurrences |
| --- | --- |
| cites | 65 |
| extends | 54 |
| addresses | 45 |
| presents | 37 |
| solves | 35 |
| uses_technique | 13 |
| proposes_future_work | 7 |
| claims | 3 |
| evaluated_on | 1 |

| Diagnostic category (overlapping) | Instances | H3 better / same / worse than matched R3 | Median rank gain |
| --- | --- | --- | --- |
| R3_11_100 | 8 | 3 / 1 / 4 | -1.5000 |
| R3_above100_or_missing | 35 | 31 / 0 / 4 | 95.0000 |
| R3_top10 | 4 | 1 / 1 / 2 | -1.0000 |
| positive_seed_path_within_four_hops | 47 | 35 / 2 / 10 | 67.0000 |
| sparse_cited_work | 4 | 4 / 0 / 0 | 1667.5000 |
| top20_evidence_seed_within_two_hops | 14 | 9 / 1 / 4 | 28.0000 |

All 47 primary canonical instances have at least one selected positive-seed structural path within four hops, so the no-path category has zero instances. This does not imply that every path carries relevant scientific evidence: several failures connect only through generic materials-science or machine-learning nodes. Across the 35 targets originally below rank 100 in full R3, H3 improves 31 relative to matched-pool R3; among the four original top-ten targets, two worsen.

“Strong evidence” here means a top-20 semantic claim/article/task/problem seed with a selected path of one or two hyperedges; it is not a human relevance judgment. Zero-hop direct seeds and lexical mention transfers are reported separately from structural propagation. The no-path category means no positive top-M structural seed path found within four hops, not proof that the full graph is disconnected. Paths are undirected incidence paths, ordered toward large seeds, short connections and weighted small edges. They include actual node texts, relation IDs, arity and local Theta transition factors, but do not uniquely decompose the full diffusion score. Structural paths do not prove that a claim entails an answer.

| Variant | Mean final type fractions (largest four) | Top-degree-decile score share | Arity>16 edge-output share | Degree-score Spearman |
| --- | --- | --- | --- | --- |
| H0 | task: 0.2056, problem: 0.1970, claim: 0.1638, method: 0.0785 | 0.1787 | 0.4149 | 0.1912 |
| H0_shuffled | task: 0.1511, claim: 0.1422, technique: 0.1347, problem: 0.1308 | 0.3046 | 0.3977 | 0.7642 |
| H0_pairwise | problem: 0.2142, task: 0.2124, claim: 0.1927, method: 0.0722 | 0.1961 | 0.0000 | 0.2060 |
| H2 | task: 0.1675, problem: 0.1590, claim: 0.1395, cited_work: 0.1295 | 0.2074 | 0.4594 | 0.2213 |
| H3 | task: 0.1768, problem: 0.1681, claim: 0.1583, cited_work: 0.1232 | 0.6474 | 0.4622 | 0.3180 |

These measures expose generic-node/hub drift without treating correlation as causation. Each query exports full type-score history, top nodes/edges, relation contributions and compact NPZ vectors of semantic scores, seeds, propagated scores and weighted degrees. Edge mass decomposes one final Theta application, not every walk contributing to the inverse. High-degree share uses each graph’s own 90th-percentile threshold; degree ties may include more than 10% of vertices.

### Examples of gains and failures

| Question / target | Matched R3 → H0 → H2 → H3 rank | Approximate support |
| --- | --- | --- |
| Q8 / ConvLSTM | 123 → 27 → 25 → 29 | problem prob_00168 (Fracture pattern prediction from MD simulations) → solves[h_00159] → method meth_00230 (ConvLSTM-based fracture prediction model) → presents[h_00292] → article arti_00030 (Using Deep Learning to Predict Fracture Patterns in Crystalline Solids) → cites[h_01313] → cited_work cite_00001 (ConvLSTM)<br>claim clai_00023 (A model trained exclusively on single‑crystal data generalizes to bicrystal and gradient‑orientation crystals, reproducing MD frac) → claims[h_00629] → article arti_00030 (Using Deep Learning to Predict Fracture Patterns in Crystalline Solids) → cites[h_01313] → cited_work cite_00001 (ConvLSTM)<br>clai_00021 → explicit mention → cite_00001 |
| Q12 / MatterSim | 117 → 157 → 35 → 40 | task task_00164 (Large‑scale atomistic simulations) → addresses[h_00152] → task task_00003 (materials science) → addresses[h_00191] → method meth_00414 (Scalable EquiVariance-Enabled Neural NETwork) → extends[h_00205] → method meth_00091 (MatterSim)<br>problem prob_00538 (large-scale atomistic simulation) → solves[h_00206] → method meth_00313 (Orb-v3) → extends[h_00205] → method meth_00091 (MatterSim)<br>clai_00459 → explicit mention → cite_00060 |
| Q12 / SevenNet | 159 → 285 → 41 → 39 | task task_00164 (Large‑scale atomistic simulations) → addresses[h_00152] → task task_00003 (materials science) → addresses[h_00038] → method meth_00134 (Frozen-framework potential energy surface (PES) descriptor screening) → presents[h_00313] → article arti_00008 (Predicting ionic conductivity in solids from the machine-learned potential ) → cites[h_01302] → cited_work cite_00054 (SevenNet)<br>problem prob_00538 (large-scale atomistic simulation) → solves[h_00206] → method meth_00313 (Orb-v3) → presents[h_00321] → article arti_00038 (Orb-v3: atomistic simulation at scale) → cites[h_01320] → cited_work cite_00054 (SevenNet)<br>clai_00343 → explicit mention → cite_00054 |

| Question / target | Matched R3 → H3 rank | Failure evidence |
| --- | --- | --- |
| Q14 / WBM | 431 → 707 | problem prob_00271 (Materials property prediction benchmarking) → solves[h_00230] → method meth_00307 (Automatminer) → addresses[h_00227] → task task_00003 (materials science) → addresses[h_00203] → method meth_00313 (Orb-v3) → evaluated_on[h_00481] → dataset data_00175 (WBM)<br>technique tech_00725 (Property prediction benchmarking) → uses_technique[h_00277] → method meth_00388 (MLIP_framework_analysis) → addresses[h_00273] → task task_00003 (materials science) → addresses[h_00203] → method meth_00313 (Orb-v3) → evaluated_on[h_00481] → dataset data_00175 (WBM) |
| Q14 / JARVIS-DFT | 341 → 525 | problem prob_00271 (Materials property prediction benchmarking) → solves[h_00230] → method meth_00307 (Automatminer) → addresses[h_00227] → task task_00001 (machine learning) → addresses[h_00129] → method meth_00247 (JARVIS integrated materials design infrastructure) → uses_component[h_00132] → component comp_00335 (JARVIS‑DFT)<br>technique tech_00725 (Property prediction benchmarking) → uses_technique[h_00277] → method meth_00388 (MLIP_framework_analysis) → addresses[h_00273] → task task_00003 (materials science) → addresses[h_00129] → method meth_00247 (JARVIS integrated materials design infrastructure) → uses_component[h_00132] → component comp_00335 (JARVIS‑DFT) |
| Q7 / symbolic regression | 143 → 210 | claim clai_00510 (Screening of approximately 50 000 inorganic crystals with the machine‑learning model identified over 2 000 temperature‑induced sol) → claims[h_01116] → method meth_00140 (Uncertainty-aware graph convolutional neural network for vibrational free‑e) → addresses[h_00044] → task task_00003 (materials science) → addresses[h_00278] → method meth_00437 (two-step supervised-unsupervised melting temperature prediction with symbol) → extends[h_00280] → method meth_00068 (symbolic regression)<br>task task_00083 (High‑throughput prediction of temperature‑induced solid‑solid phase transitions) → addresses[h_00044] → task task_00003 (materials science) → addresses[h_00278] → method meth_00437 (two-step supervised-unsupervised melting temperature prediction with symbol) → extends[h_00280] → method meth_00068 (symbolic regression) |

H2 gains cannot be attributed to copying all article context: only safe literal name mentions transfer seed support. Nevertheless a lexical mention may describe a comparison, limitation or negation, and the symmetric graph has no entailment semantics. Fusion can elevate many moderately supported entities while lowering an already-correct top result. H3 removes author-edge transmission and changes weighted normalization, so it can also suppress a useful incidental connection. Inspect the before/after rows and trace text rather than interpreting every short path as scientifically valid.

## Alpha and seed-count sensitivity

The default remains alpha=.85, M=100. These are one-at-a-time checks, not a Cartesian search or model selection exercise.

| Cutoff / variant | alpha | M | Macro candidate R@100 | Micro R@10 | Median gold rank | Mean iterations | Unconverged |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2025 / H0 | 0.5 | 100 | 0.3958 | 0.0000 | 224.0000 | 21.2143 | 0 |
| 2025 / H0 | 0.7 | 100 | 0.4979 | 0.0000 | 190.0000 | 38.9286 | 0 |
| 2025 / H0 | 0.85 | 50 | 0.4236 | 0.0638 | 218.0000 | 79.5714 | 0 |
| 2025 / H0 | 0.85 | 100 | 0.4994 | 0.0426 | 184.0000 | 78.2857 | 0 |
| 2025 / H0 | 0.85 | 200 | 0.4729 | 0.0213 | 173.0000 | 77.0000 | 0 |
| 2025 / H0 | 0.95 | 100 | 0.5070 | 0.0426 | 128.0000 | 100.0000 | 14 |
| 2025 / H3 | 0.5 | 100 | 0.7667 | 0.0851 | 58.0000 | 21.0000 | 0 |
| 2025 / H3 | 0.7 | 100 | 0.7667 | 0.0851 | 61.0000 | 38.5000 | 0 |
| 2025 / H3 | 0.85 | 50 | 0.7444 | 0.0638 | 55.0000 | 79.3571 | 0 |
| 2025 / H3 | 0.85 | 100 | 0.7230 | 0.0638 | 59.0000 | 78.1429 | 0 |
| 2025 / H3 | 0.85 | 200 | 0.6277 | 0.0426 | 80.0000 | 76.6429 | 0 |
| 2025 / H3 | 0.95 | 100 | 0.6368 | 0.0851 | 75.0000 | 100.0000 | 14 |
| 2026 / H0 | 0.5 | 100 | 0.3929 | 0.0000 | 222.5000 | 21.2857 | 0 |
| 2026 / H0 | 0.7 | 100 | 0.4005 | 0.0000 | 213.5000 | 39.1429 | 0 |
| 2026 / H0 | 0.85 | 50 | 0.4150 | 0.0400 | 201.0000 | 79.7143 | 0 |
| 2026 / H0 | 0.85 | 100 | 0.4124 | 0.0200 | 199.5000 | 78.7857 | 0 |
| 2026 / H0 | 0.85 | 200 | 0.3861 | 0.0000 | 202.5000 | 77.3571 | 0 |
| 2026 / H0 | 0.95 | 100 | 0.4817 | 0.0400 | 180.5000 | 100.0000 | 14 |
| 2026 / H3 | 0.5 | 100 | 0.7083 | 0.0600 | 73.5000 | 21.0000 | 0 |
| 2026 / H3 | 0.7 | 100 | 0.7040 | 0.0600 | 74.0000 | 38.6429 | 0 |
| 2026 / H3 | 0.85 | 50 | 0.7013 | 0.0600 | 53.5000 | 79.3571 | 0 |
| 2026 / H3 | 0.85 | 100 | 0.7159 | 0.0600 | 71.5000 | 78.4286 | 0 |
| 2026 / H3 | 0.85 | 200 | 0.6039 | 0.0600 | 92.5000 | 76.9286 | 0 |
| 2026 / H3 | 0.95 | 100 | 0.6118 | 0.0800 | 83.0000 | 100.0000 | 14 |

If any 100-iteration run is unconverged, its row is a bounded-iteration sensitivity result, not a certified fixed point; inspect the exported residual/error bound. No alpha or seed count was changed to maximize these results.

In the executed runs all default rows converged and no seed fallback was used. Every alpha=.95 sensitivity reached the 100-iteration cap. At 2025, H3 C@100 is .7667 at alpha=.50/.70, .7230 at .85 and .6368 in the truncated .95 run; M=50/100/200 gives .7444/.7230/.6277 at alpha=.85. These variations show sensitivity to propagation depth and seed breadth; they do not license selecting the best development-benchmark setting. The seed/claim/graph text remains the same across these controls.

## Thirteen explicit answers

1. **Does pure native diffusion improve candidates?** H0 macro C@50/C@100 = 0.3716/0.4994, versus R3 0.3264/0.3701 and matched-pool R3 0.3264/0.3979. This separates diffusion from output filtering.
2. **Does pure diffusion improve median rank?** H0 median 184.0 versus full R3 160.0 and matched R3 144.0. Candidate recall and median rank need not move together. H2/H3 fusion medians are 66.0/59.0.
3. **Which seed types are useful?** The mass and path tables identify which types actually supplied seeds and connected to recovered targets. These are observed support proxies; there was no type-removal causal ablation, so they cannot establish a unique best seed type.
4. **Does evidence transfer reach correct entities?** Yes where the exported positive-weight claim/task/article paths or explicit-mention transfers terminate at mapped target nodes. Selected examples above demonstrate the mechanism; they do not certify claim entailment.
5. **Real versus shuffled?** H0 C@100 0.4994, shuffled 0.0917; medians 184.0/829.0. This favors the real graph over this null, but degree/connectivity changes and one shuffle prevent a strong semantic-only causal claim.
6. **Native versus pairwise?** C@100 is 0.4994 native and 0.4994 pairwise; medians 184.0/261.0. The full K tables expose differences; no general native-higher-order advantage follows from one projection comparison.
7. **Does mention support help?** H1→H2 macro C@100 0.5473→0.7146, median 109.0→66.0; sparse cited-work rows and mention-source IDs are exported. This is an association with the prescribed mention mechanism, including its changed seed support and normalization.
8. **Does relation weighting help?** H2→H3 C@100 0.7146→0.7230, C@200 0.9674→0.8841, median 66.0→59.0. Treat mixed changes as sensitivity to a design prior, not learned scientific relation quality.
9. **Which relations carry useful relevance?** The selected recovered-target path counts and edge-mass diagnostics show their frequencies. Counts include shared paths and generic hubs; useful scientific support requires checking the actual statements.
10. **What causes failures?** No seed path within four hops is a connectivity/seed-support warning; a path through high-degree generic nodes indicates possible drift; direct semantic/mention support can still be weak or irrelevant. These are diagnostic proxies, not exhaustive causal labels. Candidate gains alongside weaker top-ten retrieval show that remaining answer ranking is unsolved.
11. **Does top-ten recovery exceed 4/47?** H0/H1/H2/H3 recover 2/3/3/3 instances, respectively, versus R3’s 4/47. Raw denominator remains 63.
12. **Does candidate recall improve beyond ~.37?** H2/H3 macro C@100 reaches 0.7146/0.7230, against full R3 0.3701 and matched-pool R3 0.3979. The exact per-question and micro values are exported; twelve evaluable questions are too few to claim broad generalization.
13. **Investigate hierarchy next?** The candidate improvements justify a later controlled approximation study of the fine-graph relevance target, especially H2, with candidate-quality loss and actual sparse work measured. They do not justify deploying hierarchical retrieval or claiming solved QA. No hierarchy diffusion, cross-encoder, LLM reranking or tuned relation-weight search was implemented in this pass. Independent questions, stronger structural nulls and source/claim adjudication remain necessary.

## Every primary canonical target: ranks

| Q / target | R3 | R3 matched | H0 | H1 | H2 | H3 | H3 gain vs matched R3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Q1 / MACE | 333 | 302 | 232 | 243 | 114 | 120 | 182 |
| Q1 / M3GNet | 101 | 96 | 22 | 42 | 26 | 29 | 67 |
| Q1 / CHGNet | 119 | 108 | 168 | 98 | 88 | 90 | 18 |
| Q1 / GAP | 206 | 190 | 219 | 152 | 47 | 54 | 136 |
| Q1 / NequIP | 188 | 172 | 29 | 62 | 91 | 59 | 113 |
| Q1 / ACE | 269 | 251 | 420 | 294 | 154 | 158 | 93 |
| Q1 / EquiformerV2 | 23 | 23 | 73 | 35 | 42 | 41 | -18 |
| Q2 / DeepH | 260 | 242 | 184 | 178 | 42 | 41 | 201 |
| Q2 / D4FT | 10 | 10 | 33 | 11 | 12 | 12 | -2 |
| Q2 / HamGNN | 90 | 90 | 7 | 22 | 24 | 39 | 51 |
| Q3 / DeepH-E3 | 1 | 1 | 7 | 2 | 1 | 1 | 0 |
| Q4 / VGNN | 2292 | 1806 | 230 | 460 | 65 | 68 | 1738 |
| Q6 / LiFlow | 8 | 8 | 17 | 6 | 2 | 2 | 6 |
| Q7 / symbolic regression | 159 | 143 | 344 | 188 | 186 | 210 | -67 |
| Q8 / ConvLSTM | 138 | 123 | 27 | 39 | 25 | 29 | 94 |
| Q9 / E(3)-equivariant GNNs | 2266 | 1825 | 1314 | 1762 | 119 | 117 | 1708 |
| Q9 / NequIP | 196 | 180 | 285 | 187 | 111 | 113 | 67 |
| Q10 / SchNet | 28 | 28 | 84 | 22 | 23 | 28 | 0 |
| Q12 / MACE | 334 | 300 | 201 | 199 | 104 | 111 | 189 |
| Q12 / NequIP | 216 | 200 | 34 | 58 | 87 | 56 | 144 |
| Q12 / CHGNet | 107 | 100 | 160 | 90 | 73 | 78 | 22 |
| Q12 / M3GNet | 97 | 94 | 25 | 36 | 29 | 28 | 66 |
| Q12 / SevenNet | 175 | 159 | 285 | 165 | 41 | 39 | 120 |
| Q12 / GAP | 215 | 199 | 248 | 176 | 50 | 53 | 146 |
| Q12 / DeePMD | 229 | 212 | 295 | 211 | 111 | 115 | 97 |
| Q12 / MatterSim | 130 | 117 | 157 | 99 | 35 | 40 | 77 |
| Q12 / Orb | 398 | 334 | 305 | 315 | 130 | 135 | 199 |
| Q12 / eSEN | 2320 | 1818 | 556 | 1052 | 283 | 292 | 1526 |
| Q12 / EquiformerV2 | 24 | 24 | 74 | 32 | 32 | 35 | -11 |
| Q13 / CGCNN | 160 | 144 | 203 | 129 | 107 | 115 | 29 |
| Q13 / SchNet | 34 | 34 | 181 | 53 | 36 | 37 | -3 |
| Q13 / MEGNet | 185 | 169 | 414 | 227 | 185 | 195 | -26 |
| Q13 / M3GNet | 111 | 102 | 37 | 47 | 55 | 43 | 59 |
| Q13 / NequIP | 186 | 170 | 121 | 109 | 72 | 75 | 95 |
| Q13 / MACE | 253 | 234 | 201 | 186 | 106 | 98 | 136 |
| Q13 / CHGNet | 136 | 122 | 347 | 172 | 100 | 101 | 21 |
| Q13 / EquiformerV2 | 24 | 24 | 470 | 58 | 57 | 56 | -32 |
| Q14 / Matbench Discovery | 2 | 2 | 24 | 6 | 8 | 8 | -6 |
| Q14 / Matbench | 97 | 93 | 74 | 66 | 53 | 55 | 38 |
| Q14 / JARVIS-DFT | 547 | 341 | 696 | 499 | 541 | 525 | -184 |
| Q14 / Materials Project | 106 | 96 | 95 | 75 | 35 | 37 | 59 |
| Q14 / OQMD | 118 | 106 | 78 | 72 | 101 | 81 | 25 |
| Q14 / AFLOW | 260 | 226 | 326 | 247 | 66 | 63 | 163 |
| Q14 / Alexandria | 3381 | 1951 | 1323 | 1989 | 353 | 324 | 1627 |
| Q14 / GNoME | 126 | 112 | 25 | 40 | 41 | 36 | 76 |
| Q14 / WBM | 674 | 431 | 1247 | 696 | 726 | 707 | -276 |
| Q14 / OMat24 | 475 | 315 | 1063 | 538 | 71 | 69 | 246 |

Positive gain means a lower/better rank. Both cutoffs, score values, groups, mention sources, selected paths, and categories are in `diffusion_per_target.json` and `propagation_traces.json`.

## Implementation, verification and reproduction

The full suite passed **126 tests**, including new dense/sparse and dense-solve equivalence, De/Dv normalization, zero/weighted/isolated edges, deterministic seeds and fallback, temporal separation, group max, evidence intermediates, conservative mentions, RRF, reproducible shuffle, projection weights, no dense sparse conversion, and a complete synthetic pipeline preserving historical files and Type B data. Automated artifact audit verifies all 14 questions per variant/cutoff, raw denominators, canonical rank coverage, score decompositions, current source hashes, frozen historical artifacts/metrics, original hierarchy hashes and exact R0/R1/R3 metric parity.

New implementation: `src/tkh_abstraction/diffusion.py` (`ThetaOperator`, `diffuse`, `semantic_seeds`, `MentionIndex`, `rank_entities`, `rrf_fuse`, diagnostics/paths) and `diffusion_pipeline.py` (`run_diffusion`, metric adapter). They reuse spectral incidence/operator, native graph classes, projection, snapshots, normalized cached BGE vectors, entity groups, question analysis, unchanged resolver, metrics and artifact writers. CLI adds `evaluate-diffusion`; defaults live in `configs/diffusion.yaml`; tests in `tests/test_diffusion.py`; reporting/audit in `scripts/report_diffusion.py`. Documentation and archive packaging were updated. Construction and historical evaluators were not rewritten.

Executed primary commands (the initial checkpoint stopped before H3; full run includes both cutoffs and sensitivities):

```powershell
$env:HF_HUB_OFFLINE='1'
.venv\Scripts\python.exe -m tkh_abstraction.cli evaluate-diffusion --config configs/diffusion.yaml --cutoff 2025 --diffusion-stage R0 --diffusion-stage R1 --diffusion-stage R3 --diffusion-stage H0 --diffusion-stage H0_shuffled --diffusion-stage H0_pairwise --diffusion-stage H1 --diffusion-stage H2 --no-sensitivity
.venv\Scripts\python.exe -m pytest -q --junitxml=artifacts/evaluation/diffusion/tests.xml
.venv\Scripts\python.exe -m tkh_abstraction.cli evaluate-diffusion --config configs/diffusion.yaml
.venv\Scripts\python.exe scripts/report_diffusion.py
.venv\Scripts\python.exe scripts/package_submission.py
```

`--diffusion-stage` and `--cutoff` select subsets; `--no-sensitivity` disables only the sensitivity pass. Partial invocations replace only the diffusion artifact set, never R0–R5 historical outputs. Run the complete command before the full report audit. For another dataset, configure input/output paths and disable assessment-specific `diffusion_verify_reuse`; no current Q ID or expected answer is used by production retrieval. The report’s benchmark assertions intentionally validate this supplied assessment.

Ultralight Opus planning and Sonnet review were read-only consultations through the official local subscription helper; Codex implemented and executed all experiments. Advice was checked against the code/math: suggestions to double the pairwise operator or assume a stochastic L1 error bound were rejected. The first benchmark attempt exposed a duplicate diagnostics-key error, which was fixed with regression coverage before successful checkpoints. Full prompts, decisions and run logs are preserved in `notes/diffusion`. No expert scientific validation is claimed.
