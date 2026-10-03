# Clean TSHC Interview Bank: 100 Questions and Answers

Scope: **clean_tshc only**. These questions are based on the supplied Intern Candidate Assessment Task and the current `clean_tshc/report/full_report.tex`. They deliberately exclude `rest_of_work`.

Use this as an interview study appendix. The answers distinguish what the implementation guarantees, what it measures empirically, and what it does **not** establish.

---

## A. Task, data, and problem formulation

### 1. What problem is TSHC trying to solve?
TSHC builds a multi-resolution, human-navigable abstraction of an evolving scientific knowledge **hypergraph**. Instead of showing thousands of fine-grained nodes at once, it produces nested super-nodes at a few display scales, while trying to preserve semantic meaning, native hyperedge structure, and continuity over time. The core challenge is not merely compression; it is compression that remains scientifically interpretable and temporally stable.

### 2. What are P1-P6?
P1 is laminar refinement, P2 is hard size budgets, P3 is semantic coherence, P4 is hyperedge fidelity, P5 is temporal stability with trackable identities, and P6 is faithful temporally honest labels. P1 and P2 are structural constraints. P3-P6 are the substantive quality requirements: meaning, higher-order structure, evolution, and interpretation.

### 3. Why is this not just ordinary clustering?
Ordinary clustering normally gives one partition. The task requires a **nested hierarchy** with several resolutions, hard cluster-count budgets, persistence across snapshots, labels, and preservation of higher-order hyperedges. It also requires evaluation controls against circularity and an extrinsic retrieval task, so the problem is broader than simply maximizing one clustering score.

### 4. Why are hypergraphs necessary here?
A TKH relation can involve more than two entities simultaneously. For example, one comparison relation may connect several methods, a dataset, and a metric. A normal graph edge can connect only two endpoints, so a plain graph cannot directly encode the fact that all those entities participated in one higher-order relation.

### 5. Why is sampling or degree-thresholding not enough?
Sampling and degree thresholding make the graph smaller, but they do not create a meaningful abstraction. They can remove important low-degree concepts and do not guarantee that each remaining item represents a coherent scientific idea. TSHC instead aggregates nodes into super-nodes so that every original node is still represented somewhere in the hierarchy.

### 6. Why are the display budgets 12, 40, and 120?
They are human-navigation design budgets, not estimates of a “true” number of scientific communities. The coarsest budget of 12 lies within the task’s requested range of roughly 10-15 macro concepts, and 40/120 provide progressively finer intermediate views. The report explicitly avoids claiming that these are natural community counts.

### 7. What snapshots are used?
The implementation uses four cumulative snapshots: 2020, 2022, 2024, and 2026. Each snapshot contains only nodes and hyperedges that satisfy the strict visibility rules by that year.

### 8. What is the difference between `origin_year` and `first_seen_year`?
`origin_year` means when the scientific object was introduced in the world. `first_seen_year` means when the supplied corpus first contains evidence of it. For temporal honesty, TSHC uses `first_seen_year`, because the hierarchy should not claim knowledge that the corpus had not yet observed.

### 9. What is the strict visibility rule?
A node is visible at year t if its `first_seen_year <= t`. A hyperedge is visible only if its own year is <= t, its asserting article is not newer than t when that year is known, and all of its endpoints are visible. This prevents dangling or temporally inconsistent relations from appearing too early.

### 10. What data-quality problems did the audit find?
The audit found 170 hyperedges dated earlier than at least one endpoint’s first-seen year and 134 hyperedges dated earlier than the asserting article. Rather than silently accepting them, TSHC defers those edges until all visibility conditions are satisfied. The report also notes that annual dates cannot enforce the benchmark’s “by February 2026” cutoff exactly.

---

## B. Hypergraphs and projection

### 11. What information does clique projection lose?
Clique projection replaces one r-ary hyperedge with all pairwise edges among its endpoints. After projection, one true multi-node relation can be indistinguishable from several unrelated pairwise relations. The original grouping, arity, and edge identity can be lost once pair weights are aggregated.

### 12. How is the pairwise baseline weighted?
An arity-r hyperedge is expanded into `choose(r,2)` pairs, with each pair receiving weight `1 / choose(r,2)`. This conserves total edge weight per original hyperedge. It is used as a credible baseline, not as the primary representation.

### 13. What is hyperedge arity?
The arity of a hyperedge is the number of endpoints it contains. In the supplied graph, arity ranges from 2 to 65, and most hyperedges have arity greater than 2. That is why higher-order structure is not a minor corner case.

### 14. How does TSHC measure hyperedge fragmentation?
For each hyperedge, it looks at the fraction of its endpoints that lie in each current cluster. It computes the Shannon entropy of that endpoint distribution and divides by `log(r)` for an edge of arity r. Fragmentation is 0 when all endpoints are together and 1 when every endpoint is separated.

### 15. Why divide the hyperedge entropy by `log(r)`?
The maximum entropy of an r-endpoint hyperedge occurs when all endpoints are in different clusters, and that maximum is `log(r)`. Dividing by `log(r)` normalizes fragmentation to [0,1], so edges of different arities are comparable.

### 16. What happens to fragmentation as clusters merge?
At the singleton partition, every hyperedge is maximally fragmented, so F starts at 1. As clusters merge, endpoints become grouped together, so F can only decrease or remain unchanged. This creates a genuine tension with semantic dispersion, which increases as unrelated semantic clusters are merged.

### 17. What are the three required T4 collapse cases?
If all n endpoints fall in one super-node, the edge becomes internal and is kept with multiplicity n. If only m of n endpoints fall together, the edge becomes a coarse relation such as `{A:m, B:n-m}`. If endpoints occupy three or more super-nodes, the edge remains a genuine higher-order coarse hyperedge rather than being split into pairs.

### 18. Why keep fully internal hyperedges?
An internal edge is evidence explaining why the members of that super-node belong together. Deleting it would throw away provenance and structural evidence. Keeping it also makes the collapse representation faithful to the original relation inventory.

### 19. What exactly is lost during collapse?
A coarse record tells us how many original endpoints landed in each super-node, but not which fine nodes inside a super-node participated. That fine identity can still be recovered from the saved membership lists and original graph. TSHC therefore preserves edge identity, arity, multiplicity, relation type, and provenance, but coarse geometry cannot by itself reconstruct every fine endpoint.

### 20. Does using native hyperedges prove irreducible higher-order expressivity?
No. The report is careful not to claim that no pairwise representation could ever encode equivalent information. The implementation guarantee is narrower: original edge identity, grouping, arity, and multiplicity are explicitly retained and used by the objective, whereas the chosen clique projection loses those distinctions.

---

## C. The formal objective

### 21. What objective does TSHC optimize?
At each level and snapshot, TSHC minimizes
`J = alpha*S + (1-alpha)*F + lambda*T`.
S is semantic dispersion, F is native hyperedge fragmentation, and T is temporal drift from the previous snapshot’s partition at the same level. The primary parameters are `alpha=0.5` and `lambda=0.2`.

### 22. What is the semantic term S?
S is the within-cluster sum of squared distances between MiniLM node embeddings and their cluster centroids, normalized by `4|V|`. It is a Ward-style compactness objective: semantically similar nodes produce smaller dispersion.

### 23. Why divide S by `4|V|`?
The node embeddings are unit-normalized. Squared distance between two unit vectors is at most 4, so dividing the total by `4|V|` gives a bounded scale. In practice the report derives a tighter bound of 0.25 for this centroid-based dispersion.

### 24. What is the structural term F?
F is the average normalized entropy of each native hyperedge’s endpoint distribution across current clusters. It rewards partitions that keep endpoints participating in the same scientific relation together. Importantly, the unit of structure is the original hyperedge, not projected pairs.

### 25. What is the temporal term T?
T is the Variation of Information between the current partition and the previous snapshot’s partition at the same level, restricted to shared nodes and normalized by `2 log |U|`. New nodes do not contribute because they have no previous cluster membership to preserve.

### 26. Why choose `alpha = 0.5`?
The goal was to give semantic and structural terms equal nominal weight after normalization, without fitting the value to the benchmark. The report explicitly notes that equal coefficients do **not** mean equal realized influence because S and F have different empirical ranges.

### 27. Why choose `lambda = 0.2`?
`lambda=0.2` makes temporal history a regularizer rather than the dominant objective. It was fixed before observing downstream results. Sensitivity experiments show that other lambda values can change memberships substantially, so it should be treated as a design choice rather than a universally optimal constant.

### 28. Why is the structure/semantics trade-off real?
As clusters merge, semantic dispersion S can only increase, because clusters become less homogeneous. Hyperedge fragmentation F can only decrease, because more endpoints become grouped. Therefore the two objectives pull in opposing directions, and alpha genuinely controls a compromise instead of combining two perfectly aligned signals.

### 29. What is the semantic merge cost?
For merging clusters A and B, the increase is the Ward term:
`Delta S = (1/(4|V|)) * (n_A n_B/(n_A+n_B)) * ||mu_A-mu_B||^2`.
Only the cluster sizes and vector sums are needed.

### 30. What is the fragmentation merge cost?
Only hyperedges touching both A and B can change. The exact delta uses the multiplicities of the edge endpoints in A and B and the function `phi(x)=x log x`. This makes the structural update local and avoids recomputing fragmentation for all edges after every merge.

### 31. Why is `Delta F <= 0`?
Merging two endpoint groups makes the endpoint distribution of an affected hyperedge less fragmented. Entropy cannot increase when two categories are combined in this way. Therefore the fragmentation term either decreases or stays the same.

### 32. How is the temporal merge delta computed?
The implementation maintains counts of how many shared nodes in each current cluster came from each previous cluster. When A and B merge, only their rows in the current/previous contingency table change. That allows exact changes in current entropy and joint entropy, and therefore in VI, to be computed without scanning all members.

### 33. Does the method find the global minimum of J?
No. It uses deterministic greedy agglomeration over a restricted candidate set. Each selected merge is the exact lowest-cost **current candidate merge**, but the whole merge sequence is not guaranteed to be globally optimal.

### 34. Were alpha, lambda, or k tuned on retrieval performance?
No. The primary settings were fixed before benchmark results were inspected. The later one-factor sensitivity study is explicitly a dependence diagnostic, not parameter selection.

### 35. Which P1-P6 properties are guaranteed versus empirical?
P1 laminarity and P2 budgets are guaranteed and tested. P4’s structural preservation is guaranteed, although whether it improves clustering quality is empirical. P3 coherence, the degree of P5 stability, and P6 faithfulness are evaluated empirically; temporal regularization encourages P5 but does not mathematically guarantee a specific ARI or VI.

---

## D. Construction algorithm and implementation

### 36. Why use bottom-up agglomerative clustering?
Every operation merges two existing clusters. That automatically makes later partitions coarser than earlier ones, so laminarity comes for free. It also makes hard cluster-count budgets easy: stop and record the partition exactly when the number of live clusters reaches 120, 40, and 12.

### 37. What is the actual hierarchy cascade?
The finest level P3 is all singleton nodes. The algorithm then continuously merges until 120 clusters and records P2, continues to 40 and records P1, and finally continues to 12 and records P0. It is one continuous merge cascade, not three independent clusterings.

### 38. What pairs are allowed as merge candidates?
The candidate graph is the union of semantic proposals and structural proposals. Semantic proposals are each node’s 10 nearest MiniLM neighbors, symmetrized. Structural proposals connect nodes that co-occur in a native hyperedge.

### 39. Are the structural proposal pairs a pairwise approximation of the objective?
No. Co-membership pairs only determine which merges are considered. The **cost** of every candidate merge is still computed using the native hyperedge multiplicities and the full objective. Candidate generation and objective evaluation are separate.

### 40. Why use k=10 nearest semantic neighbors?
It keeps the candidate graph much smaller than all `n(n-1)/2` possible pairs while still proposing semantically plausible merges. Sensitivity tests at k=5 and k=20 produce hierarchies very close to k=10, with ARI at least about 0.93 versus the primary hierarchy.

### 41. What is the lazy heap?
All candidate merge costs are stored in a min-heap. When two clusters merge, old heap entries mentioning either dead cluster become stale and are skipped when popped. New entries involving the new cluster are inserted with freshly computed costs.

### 42. Why can the lazy heap still return the exact current best candidate?
A merge cost depends only on the sufficient statistics of its two live clusters plus global constants. If neither cluster has changed, its cost has not changed. If one cluster changed, the old cluster id is dead and that heap entry is rejected, so all live entries are current.

### 43. What happens if the candidate graph becomes disconnected?
The algorithm has a deterministic fallback: merge the two clusters with nearest semantic centroids. However, the report states that this fallback was never triggered in the executed main, perturbation, or sensitivity runs.

### 44. What is the main computational bottleneck?
The brute-force semantic neighbor search is `O(n^2 d)`, with `d=384`, followed by heap operations for many candidate updates. Co-membership proposals add about 72,701 pair instances in the full graph. The largest temporal 2026 build took about 133 seconds on CPU.

### 45. Why is there no scalability claim?
The corpus has only 5,798 nodes, and brute-force similarity is acceptable at that size. The task emphasizes research judgment over production infrastructure. The report explicitly says that approximate nearest-neighbor indexing should be considered only if profiling on a larger corpus justifies it.

### 46. How reproducible is the implementation?
Model revisions and package versions are pinned, random seeds are fixed, tie-breaking is deterministic, label splits are hash-based, and predictions are frozen before gold evaluation. Two fresh output runs reproduced scientific metrics, memberships, label probabilities, and retrieval predictions exactly.

### 47. What do the tests actually protect against?
Tests cover objective-delta correctness, heap correctness, laminarity, budget enforcement, snapshot visibility, multiplicity conservation, temporal bookkeeping, label split separation, null-model reproducibility, retrieval parity, evaluator positive controls, and benchmark isolation. The suite is aimed at scientific invariants rather than just code coverage.

### 48. Why was spectral hypergraph clustering rejected as the main method?
It is strong for hypergraph partitioning, but it does not naturally produce the exact nested hard-budget hierarchy required here, and adding the same temporal coupling would require additional design. TSHC instead uses a merge-based objective with exact local deltas and guaranteed nesting.

### 49. Why reject a two-stage “semantic first, structural repair later” approach?
A two-stage scheme can hide the trade-off across separate heuristics. TSHC keeps semantics, structure, and time in one explicit objective, so the exchange among them is visible and auditable.

### 50. Why reject a learned/neural clustering method?
There is no supervised target defining the “correct” hierarchy, and learning would add extra training assumptions and tuning opportunities. For an assessment emphasizing interpretability and evaluation validity, a small explicit objective is easier to defend and verify.

---

## E. Variants and temporal treatment

### 51. What are the four compared variants?
`semantic` uses only Ward semantic cost and kNN proposals. `pairwise` uses the clique projection with no temporal term. `static` uses native hyperedges with no temporal regularization. `temporal` uses native hyperedges plus the VI regularizer and is the primary variant.

### 52. Why are static and temporal identical in 2020?
There is no previous snapshot in 2020. Therefore the temporal term T is set to zero, so the static and temporal objectives are the same for the first snapshot.

### 53. What is the difference between temporal regularization and identity matching?
Regularization changes the clustering by penalizing departures from the previous partition. Hungarian identity matching happens **after** construction and only decides which new super-node inherits which persistent id. Matching is bookkeeping; it cannot make an unstable clustering stable.

### 54. How does Hungarian matching work here?
At each level, the implementation computes Jaccard similarity between every old cluster and every new cluster. It then solves the one-to-one assignment that maximizes total similarity. A matched new cluster with nonzero overlap inherits the old persistent id.

### 55. Why use Jaccard similarity for cluster identity?
Jaccard compares shared membership to the union of memberships, so it rewards large overlap while accounting for size differences. It is simple, interpretable, and directly tied to cluster membership rather than embedding similarity.

### 56. What are birth, continuation, growth, merge, split, and death?
Birth means no overlap with an old cluster. Continuation means a matched cluster gained no new members. Growth means a matched cluster acquired at least one member. Merge and split are many-to-one or one-to-many strong-overlap patterns, and death means an old cluster has no overlap with any new cluster.

### 57. What counts as a “strong” overlap for merge/split events?
The report uses at least 2 shared nodes and at least 10% of the relevant cluster’s shared membership. This deliberately ignores one-node contamination so tiny accidental overlaps do not create merge or split events.

### 58. Are temporal events mutually exclusive?
No. A cluster can inherit an identity and also be involved in a merge or split. Identity inheritance comes from the one-to-one matching, whereas event labels come from the full many-to-many overlap table.

### 59. What is lineage retention?
It is the fraction of nodes shared by two snapshots whose persistent super-node id is unchanged. It is an intuitive node-level continuity measure complementary to partition-level ARI and VI.

### 60. Why report both ARI and VI?
ARI is based on pairs of nodes and can be dominated by very large clusters. VI is information-theoretic and responds differently to small-cluster changes. The final 2024->2026 transition is a good example: static has higher ARI, but temporal has lower VI and higher lineage retention, so one metric alone would give an incomplete story.

---

## F. Labels and evaluation validity

### 61. How are label generation and label evaluation separated?
Each cluster is split roughly 70/30 by node type. The 70% generation set creates the label and gloss, while the held-out 30% is used for the faithfulness check. This prevents the labeller from being evaluated only on the same text it already saw.

### 62. Why use a hash-based split instead of random sampling?
Hash ordering is deterministic across runs and requires no mutable random state. Stratifying by node type also keeps both sets representative when possible.

### 63. How is the label phrase chosen?
A TF-IDF model over 1-3 grams is fitted on generation members. For each cluster, the phrase with highest mean TF-IDF weight over that cluster’s generation members is selected, excluding invalid/stopword-only phrases and limiting the label to six words.

### 64. How are representative examples chosen?
Non-author generation members are ranked by cosine similarity to the generation-only MiniLM centroid. The top two sufficiently short names are inserted into the gloss.

### 65. What does the gloss claim?
The template says, roughly, “This cluster groups evidence concerning LABEL, including REP1; REP2.” It deliberately makes a weak containment-style claim rather than claiming that every member proves some strong scientific statement.

### 66. What is the overclaim proxy?
Held-out member text is the NLI premise and the generated gloss is the hypothesis. If the model predicts neutral or contradiction rather than entailment, that label gets an overclaim indicator of 1. The overclaim rate is the mean of that indicator across evaluable labels.

### 67. Why is the NLI score only a proxy?
The DeBERTa NLI model is general-domain, not materials-science expert judgment. Vague glosses may be easy to entail, truncation can hide evidence, and entailment does not guarantee that a human would find a label useful. Therefore the report never treats it as a true scientific error rate.

### 68. How is temporal honesty enforced for labels?
Label generation only receives nodes and relations visible in the snapshot. The code asserts snapshot validity before generation. That prevents a 2020 label from using evidence that first appears in 2024, subject to the limitation of annual rather than monthly timestamps.

### 69. What is the main evaluation circularity hazard?
If MiniLM both creates the clusters and measures their semantic coherence, the method is graded with its own ruler and high coherence is partly guaranteed. Similarly, if labels are checked only against the same text used to generate them, faithfulness evaluation is circular.

### 70. How is coherence circularity neutralized?
Construction uses MiniLM, while coherence is measured with a separately trained BGE embedding family. The code also asserts that the construction and evaluation model names differ. This does not make BGE perfect, but it prevents exact reuse of the clustering signal.

### 71. What does the coherence null model preserve?
The null permutations preserve every cluster’s size and exact node-type composition. Labels are permuted only within node type. Therefore a cluster cannot beat the null merely by grouping all datasets or all methods together.

### 72. Why is a type-preserving null stricter than random labels?
Scientific node types already carry strong semantic signal. A completely random permutation would create obviously incoherent mixed-type clusters and make the observed method look artificially strong. Type preservation asks whether the clustering adds coherence **beyond** size and type composition.

### 73. Why are all empirical p-values 0.0099?
There are 100 null replicates, and no null replicate reaches the observed coherence. The empirical formula `(1 + exceedances)/(R+1)` therefore gives the minimum possible value `1/101 ≈ 0.0099`. It should not be interpreted as infinitely strong evidence.

### 74. How is perturbation stability evaluated?
For each variant, 10% of the 2026 hyperedges (143 edges) are removed using five fixed random seeds. The hierarchy is rebuilt with the same 2024 history, and ARI versus the unperturbed 2026 hierarchy is measured at all three levels. Mean, SD, and Student-t 95% intervals are reported.

### 75. Why is semantic-only perturbation ARI = 1 not impressive?
The semantic-only variant ignores hyperedges completely. Removing hyperedges therefore cannot change its construction input, so the hierarchy stays identical. ARI=1 here means edge blindness, not superior robustness.

---

## G. Retrieval and benchmark methodology

### 76. How is benchmark leakage prevented?
Retrieval predictions are written and hashed before the benchmark module is imported and ground truth is read. After evaluation, the prediction hash is checked again. Tests also verify that construction and retrieval source do not import or reference benchmark data.

### 77. What are EXACT, ALIAS, PARTIAL, COMPOSITE, and ABSENT targets?
EXACT means normalized full surface-form identity exists. ALIAS means inspected evidence supports another node as the same entity. PARTIAL means related components exist but do not justify full identity. COMPOSITE means the expected answer is a bundle of required components. ABSENT means no defensible mapping was found in the supplied graph.

### 78. What is the difference between direct identity and contextual evidence?
For EXACT and ALIAS targets, the target node itself must appear in the returned ranking. Merely retrieving a claim or problem document whose context mentions the method does not count as identity recovery. This deliberately separates “found evidence about X” from “retrieved X as the answer entity.”

### 79. Why are most expected claims marked unevaluable?
Automatic claim alignment is deliberately strict: exact claim text plus a resolved matching source article is accepted automatically. Semantic nearest neighbors are proposals only. Because the graph often phrases claims differently from the benchmark, only one of 224 expected claims was defensibly aligned.

### 80. What does source recovery measure?
It measures overlap between expected source articles and source articles attached to returned documents. It is a coarse topical/provenance signal. It does **not** prove that the required scientific claim was recovered or entailed.

### 81. How is a retrieval document constructed?
Every non-author node becomes a document containing its own `[type] surface_form` line plus at most 12 deduplicated direct neighbors through content relations, with claims prioritized. `authored_by` and broad `cites` are excluded. There is no recursive expansion, alias expansion, diffusion, or reranking.

### 82. Why are there 5,480 retrieval documents?
The final 2026 snapshot has 5,798 nodes, including 318 authors. Retrieval excludes author nodes, leaving 5,480 non-author documents.

### 83. What does the flat baseline do?
It embeds the question with BGE, computes cosine similarity to all 5,480 document vectors, and sorts every document by score. Its work is therefore exactly 5,480 cosine comparisons per question.

### 84. How does hierarchical beam retrieval work?
At P0, score all coarse cluster centroids. Keep the best b clusters, expand only their children at P1, repeat at P2, then score the leaves inside the surviving clusters. The same BGE document vectors and cosine scorer are used as the flat baseline.

### 85. Why use normalized cluster centroids?
A cluster centroid summarizes the average direction of its member document vectors. Normalizing it makes the cluster score a cosine-style similarity comparable across clusters. The risk is that a relevant leaf can be hidden inside a cluster whose average vector is not similar to the query.

### 86. What does “work” mean in retrieval?
Work is the number of cosine comparisons: cluster comparisons plus leaf comparisons. It is a deterministic algorithmic cost proxy. The report explicitly does not claim that a 69% reduction in comparisons means a 69% wall-clock latency speedup.

### 87. Why is beam 3 the primary setting?
Beam 3 was fixed in advance as a moderate work/coverage trade-off. Beams 1, 5, and 10 are diagnostics. The benchmark result was not used to choose a different beam afterward.

### 88. What was the top-10/top-25 retrieval result?
Every system recovered 0 of 63 Type-A target occurrences at both k=10 and k=25. Flat used 5,480 comparisons per question; beam 3 used about 1,712, a 68.8% reduction. Because flat recall is zero, the hierarchy cannot be said to preserve useful recall.

### 89. How do you know the evaluator is not simply broken?
A positive control forced each of the 49 EXACT/ALIAS target instances into rank 1 of an otherwise unchanged ranking. The evaluator recovered all 49/49. Rank inspection also shows real target nodes at finite but poor positions, such as MatterSim at 31 and DeepH-E3 at 57.

### 90. Why does flat retrieval fail so badly?
The benchmark questions are long descriptions of tasks and constraints, so problem, task, and claim documents often paraphrase them closely. Named method nodes such as “MACE” are much shorter and receive lower cosine scores even when related evidence appears in neighboring context. The representation is therefore good at finding topical regions but poor at ranking the exact answer identity.

### 91. Why not fix retrieval after seeing zero recall?
Obvious fixes existed: restrict candidates to method-like nodes, expand aliases, or rerank. Applying them after inspecting this benchmark would turn the benchmark into a tuning set. The clean report intentionally preserves the negative result and proposes such changes only as future pre-registered experiments on a held-out question set.

### 92. What does the retrieval experiment establish?
It establishes that the current hierarchy can reduce comparison work, but it does **not** establish downstream utility because flat identity recall is already zero. It also diagnoses a representation/ranking problem: relevant evidence is often found, but the exact target entity is not ranked highly enough.

---

## H. Results, interpretation, and defense

### 93. What is the key temporal-vs-static result?
Mean cross-snapshot ARI rises from about 0.725 for static native clustering to 0.892 for temporal native clustering. Mean 2026 coherence across levels drops only from about 0.8134 to 0.8114. On this corpus, that is a strong continuity gain for a small coherence cost, while the final-transition ARI caveat remains.

### 94. How should the coherence improvements over null be interpreted?
They are statistically clear within the 100-replicate permutation test but small in absolute cosine units. For the temporal variant, the 2026 absolute effects are about 0.0033, 0.0062, and 0.0113 at L0-L2. Large standardized effects mostly reflect an extremely narrow conditional null distribution, not huge practical differences.

### 95. What does the pairwise baseline show?
Compared with native hyperedges, the pairwise projection gives lower coherence, lower perturbation robustness, a more extreme giant cluster, and many unevaluable labels. This supports the decision to keep native edge identity and multiplicities. It is evidence about this chosen projection and dataset, not a theorem that every pairwise method must be worse.

### 96. What is the dominant-cluster problem?
In the 2026 temporal hierarchy, one L0 cluster contains 3,765 of 5,798 nodes, about 65%. The structural objective can favor many merges that reduce fragmentation around broad hubs, and the semantic term’s realized scale is weaker. This means satisfying the count budget does not guarantee balanced or equally useful macro concepts.

### 97. Are the labels good?
The answer is mixed. Temporal labels have an overall held-out overclaim proxy around 0.31, and some labels are useful, but several are generic, duplicated, or citation-like. A low overclaim score can reward vague text, so the report does not claim expert-quality labels.

### 98. Which variant would you ship, and why?
For a browsing hierarchy that must remain recognizable as the corpus grows, the temporal native variant is the defensible default because it offers much stronger continuity with coherence close to static and preserves native hyperedges. This recommendation is specifically about continuity. It is **not** based on successful retrieval or universally best label quality.

### 99. What are the biggest limitations?
The optimizer is greedy and proposal-dependent; one cluster dominates the hierarchy; the corpus is only 52 papers; time is annual; label evaluation uses a general-domain NLI proxy; the benchmark is heterogeneous, incomplete, and repeatedly inspected; only one expected claim is aligned; five perturbation seeds give wide intervals; and comparison counts are not latency measurements. Results are therefore evidence about this implementation and corpus, not universal claims.

### 100. If the interviewer asks for your one-minute summary and next steps, what do you say?
“I built a deterministic, hypergraph-native, temporally regularized agglomerative hierarchy with hard 12/40/120 display budgets. The method explicitly balances MiniLM semantic dispersion, native hyperedge fragmentation, and VI to the previous snapshot; collapse preserves edge identity, arity, multiplicity, and provenance. I evaluated it with an independent BGE encoder, type/size-preserving nulls, perturbation intervals, held-out label evidence, and frozen retrieval predictions. Temporal regularization substantially improves average continuity for very little coherence cost, but the hierarchy has a giant-cluster problem, labels are mixed, and the retrieval task fails even for the flat baseline. I would next do blinded expert audits and a genuinely held-out retrieval set, then pre-register fixes for balance and answer-entity retrieval rather than tuning on the current benchmark.”

---

## Fast interview reminders

- **Do not say:** “TSHC globally minimizes the objective.”  
  **Say:** “Each greedy step takes the minimum current candidate merge; global optimality is not claimed.”

- **Do not say:** “Temporal wins every stability metric.”  
  **Say:** “Temporal wins strongly on average and on VI/lineage, but static has higher ARI in the final transition.”

- **Do not say:** “p=0.0099 means a huge effect.”  
  **Say:** “It is the 100-replicate resolution floor; absolute effects are small.”

- **Do not say:** “Semantic-only is perfectly robust.”  
  **Say:** “Its perturbation ARI is 1 because it ignores the perturbed edges.”

- **Do not say:** “The hierarchy improves retrieval.”  
  **Say:** “It reduces comparisons, but no retrieval benefit is demonstrated because flat recall is already zero.”

- **Do not say:** “NLI proves labels are faithful.”  
  **Say:** “NLI is a held-out automatic proxy with known domain and vagueness limitations.”

- **Do not say:** “12/40/120 are natural communities.”  
  **Say:** “They are fixed display budgets.”

- **Do not say:** “Hungarian matching stabilizes the clusters.”  
  **Say:** “VI regularization stabilizes them; Hungarian matching only assigns persistent identities.”
