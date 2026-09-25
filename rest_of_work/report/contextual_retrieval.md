# Contextual retrieval: implementation and measured failure analysis

This pass preserves TKH construction and all hierarchy memberships. It recomputed the existing evaluation before changing retrieval, ran the required candidate audit, then ran incremental flat experiments. Historical v1 and v2 artifacts remain archived. All numbers below are generated from final JSON outputs.

## Benchmark reconciliation

The benchmark has **14 Type A questions, 4 Type B questions, 63 Type A target instances and 50 unique target labels**. “12 questions” means the subset with non-null canonical-target recall, not the benchmark size. In the legacy resolver, Q5 excludes Bead-mapping and GNN+GPP because neither has an accepted exact/alias node; Q11 excludes CE→MC→NN→PF and hybrid frameworks for the same reason. Composite phrases were not decomposed by that resolver. All 14 questions remain in metrics, with explicit evaluability status; unresolved instances remain in the raw denominator.

| Resolver / normalization | 2025 unique | 2026 unique | Interpretation |
| --- | --- | --- | --- |
| Original method-only | 21/50 | 26/50 | Limited node types |
| Basic NFKC/case/whitespace exact | 31/50 | 34/50 | All types; 34 is full-export literal coverage |
| V2 normalized exact | 32/50 | 35/50 | Also Unicode dashes/punctuation; parentheses/signs retained |
| V2 accepted exact + alias | 33/50 | 36/50 | 46/63 and 49/63 instances |
| Earlier approximately 37 | not final | preliminary | One unsafe future-topic containment was removed; see preserved initial run log |

These are different resolvers/cutoffs, not competing desired scores. V2 allows exact entity/claim names and containment only for method, technique, component and dataset nodes; its query-dependent routes can exclude other types. Final context retrieval has an all-type snapshot pool and optional soft priors. Exact duplicate node IDs remain available to evidence traversal; grouping changes retrieval slots only.

## Phase 0: why the original flat baseline failed

| Cutoff | Resolved instances | In flat pool | Excluded | Gold rank min / median / max | Top 10 |
| --- | --- | --- | --- | --- | --- |
| 2025 | 46 | 45 | 1 | 29.0 / 1463.0 / 2387.0 | 0 |
| 2026 | 49 | 48 | 1 | 33.0 / 1657.0 / 2748.0 | 0 |

The completely excluded primary instance is Q14 / JARVIS-DFT: the data route omitted its mapped `component` node type. The other 45 mapped instances entered exhaustive search but ranked poorly. Across target instances, 77 mapped node occurrences survived routing and 7 did not; these are occurrences, not unique graph nodes. Every node score/rank and exclusion reason is exported in `gold_candidate_audit.json`.

| Question | Expected | Mapped at 2025 | In pool | Best gold rank |
| --- | --- | --- | --- | --- |
| Q12 | 13 | 11 | 11 | 371 |
| Q13 | 8 | 8 | 8 | 217 |
| Q14 | 11 | 10 | 9 | 29 |

Q12 has 13 literal matches in the full export but only 11 at the conservative 2025 cutoff: PET and UMA are first visible in 2026. This is temporal exclusion, not a routing or lookup failure.

## Resolution and retrieval design

| Cutoff | Exact unique / instances | Alias | Composite | Evidence-backed | Ambiguous | Unresolved | Canonical coverage | Representable coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025 | 32 / 45 | 2 / 2 | 4 / 4 | 2 / 2 | 0 / 0 | 10 / 10 | 34/50; 47/63 | 36/50; 49/63 |
| 2026 | 35 / 48 | 2 / 2 | 4 / 4 | 2 / 2 | 0 / 0 | 7 / 7 | 37/50; 50/63 | 39/50; 52/63 |

Canonical coverage accepts exact names or conservative lexical identity evidence, never semantic similarity alone. Composite detection is not successful mapping: all components must resolve and share article provenance to enter separate representable recall. Shared provenance still does not establish functional composition. Evidence-backed phrases need repeated literal graph evidence and are not aliases. Semantic alias proposals were optional and are not used. “Unresolved” means unsupported by these rules, not proof that a concept is scientifically absent.

The generic resolver adds one conservative plural match (E(3)-equivariant GNNs); its annual 37/50 count is distinct from the earlier unsafe-containment 37/50 result. HamGNN uses generic unique identifier containment. Neither name is special-cased in production code.

Graph-only fields cover names, direct claims, lower-weight presenting-article claims, tasks, problems, technical context, datasets and titles. Strong relation endpoints are used only below the fixed arity cap; broad cites/extends co-members are excluded. Direct/explicit-name evidence precedes article context; equal-priority truncation uses stable node IDs and can lose useful context. R1_direct disables presenting-article expansion while retaining direct and explicit-name evidence. Entity groups use identical normalized names only. Query intent uses question text plus its supplied A/B type; views are exact substrings and always retain the original question. No expected answers or benchmark claims construct documents, queries, groups or ranking weights.

Weights, caps and max/mean pooling are explicit in `configs/contextual.yaml`. Each distinct text contributes to one field per result; all vectors are normalized. Missing fields contribute zero with fixed weights rather than renormalization, so sparsely described entities have a lower score ceiling. R5 reranks only the top 100 using fixed-stopword-filtered query overlap in direct/explicit-name claim/task/problem evidence. Parameters were not fitted to these answers.

The first checkpoint had a defect: some types repeated their own name text in a second semantic field. The read-only Ultralight reviewer identified this; removal was chosen on structural grounds before rerunning, not as a weight search. Pre-review R1 recovered 1/47 instances and R3–R5 recovered 3/47 at K=10. Those outputs and logs remain under `contextual/checkpoints/pre_review` and `notes/contextual`. The final tables use the corrected scorer. This is iterative benchmark development, not untouched held-out validation.

## Incremental ablations

R0: all-type surface names; R1: context/original query; R2: grouping; R3: soft priors; R4: extractive views; R5: deterministic reranking. R1_direct isolates presenting-article expansion. Original routed, type-prefixed flat remains the legacy reference, so its cost is not identical to R0.

| Legacy temporal surface baseline | Macro R@10 | A entity comparisons |
| --- | --- | --- |
| flat_exhaustive | 0.0000 | 2393.6429 |
| legacy_typed_fixed_beam | 0.0000 | 84.2857 |
| prototype_best_first | 0.0000 | 500.0000 |

### 2025 conservative primary

| Stage | Macro R@10 | Micro R@10 | Unique R@10 | Raw /63 | MRR | R-Prec = R@R | Candidate R@50 / 100 | A entity dots |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0004 | 0.0000 | 0.0000 / 0.0000 | 4792.0000 |
| R1 | 0.2028 | 0.0851 | 0.1176 | 0.0635 | 0.1296 | 0.0917 | 0.2327 / 0.3542 | 4792.0000 |
| R2 | 0.2028 | 0.0851 | 0.1176 | 0.0635 | 0.1316 | 0.0917 | 0.2431 / 0.3542 | 4792.0000 |
| R3 | 0.2028 | 0.0851 | 0.1176 | 0.0635 | 0.1589 | 0.0917 | 0.3264 / 0.3701 | 4792.0000 |
| R4 | 0.2028 | 0.0851 | 0.1176 | 0.0635 | 0.1465 | 0.0917 | 0.3264 / 0.3737 | 16772.0000 |
| R5 | 0.2028 | 0.0851 | 0.1176 | 0.0635 | 0.1935 | 0.0917 | 0.3264 / 0.3737 | 16772.0000 |
| R1_direct | 0.1944 | 0.0638 | 0.0882 | 0.0476 | 0.2013 | 0.1667 | 0.3049 / 0.3299 | 4792.0000 |

| Stage | Hit@1 / 5 / 10 / 20 | Recall@5 / 20 / 50 / 100 | Precision@5 / 10 | Median gold rank | Duplicate slots@10 |
| --- | --- | --- | --- | --- | --- |
| R0 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 | 4231.0000 | 0.0429 |
| R1 | 0.0833 / 0.1667 / 0.3333 / 0.3333 | 0.0917 / 0.2028 / 0.2327 / 0.3542 | 0.0286 / 0.0286 | 233.0000 | 0.0000 |
| R2 | 0.0833 / 0.1667 / 0.3333 / 0.3333 | 0.0917 / 0.2028 / 0.2431 / 0.3542 | 0.0286 / 0.0286 | 222.0000 | 0.0000 |
| R3 | 0.0833 / 0.1667 / 0.3333 / 0.3333 | 0.0917 / 0.2028 / 0.3264 / 0.3701 | 0.0286 / 0.0286 | 160.0000 | 0.0000 |
| R4 | 0.0833 / 0.1667 / 0.3333 / 0.4167 | 0.0917 / 0.2147 / 0.3264 / 0.3737 | 0.0286 / 0.0286 | 180.0000 | 0.0000 |
| R5 | 0.0833 / 0.2500 / 0.3333 / 0.4167 | 0.1750 / 0.2104 / 0.3264 / 0.3737 | 0.0429 / 0.0286 | 180.0000 | 0.0000 |
| R1_direct | 0.1667 / 0.1667 / 0.2500 / 0.5000 | 0.1667 / 0.2965 / 0.3049 / 0.3299 | 0.0286 / 0.0214 | 181.0000 | 0.0000 |

### 2026 annual sensitivity, potentially after February

| Stage | Macro R@10 | Micro R@10 | Unique R@10 | Raw /63 | MRR | R-Prec = R@R | Candidate R@50 / 100 | A entity dots |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0004 | 0.0000 | 0.0000 / 0.0000 | 5555.0000 |
| R1 | 0.1742 | 0.0600 | 0.0811 | 0.0476 | 0.1222 | 0.0909 | 0.2188 / 0.3245 | 5555.0000 |
| R2 | 0.1742 | 0.0600 | 0.0811 | 0.0476 | 0.1350 | 0.0909 | 0.2308 / 0.3245 | 5555.0000 |
| R3 | 0.1742 | 0.0600 | 0.0811 | 0.0476 | 0.1546 | 0.0909 | 0.3245 / 0.3245 | 5555.0000 |
| R4 | 0.1742 | 0.0600 | 0.0811 | 0.0476 | 0.1396 | 0.0909 | 0.3245 / 0.3245 | 19442.5000 |
| R5 | 0.2020 | 0.0800 | 0.1081 | 0.0635 | 0.2315 | 0.0909 | 0.3245 / 0.3245 | 19442.5000 |
| R1_direct | 0.1944 | 0.0600 | 0.0811 | 0.0476 | 0.1977 | 0.1667 | 0.3033 / 0.3336 | 5555.0000 |

| Stage | Hit@1 / 5 / 10 / 20 | Recall@5 / 20 / 50 / 100 | Precision@5 / 10 | Median gold rank | Duplicate slots@10 |
| --- | --- | --- | --- | --- | --- |
| R0 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 | 4934.5000 | 0.0571 |
| R1 | 0.0833 / 0.0833 / 0.2500 / 0.3333 | 0.0833 / 0.2020 / 0.2188 / 0.3245 | 0.0143 / 0.0214 | 261.5000 | 0.0000 |
| R2 | 0.0833 / 0.1667 / 0.2500 / 0.3333 | 0.0909 / 0.2020 / 0.2308 / 0.3245 | 0.0286 / 0.0214 | 235.0000 | 0.0000 |
| R3 | 0.0833 / 0.1667 / 0.2500 / 0.3333 | 0.0909 / 0.2020 / 0.3245 / 0.3245 | 0.0286 / 0.0214 | 181.5000 | 0.0000 |
| R4 | 0.0833 / 0.1667 / 0.2500 / 0.3333 | 0.0909 / 0.2020 / 0.3245 / 0.3245 | 0.0286 / 0.0214 | 191.0000 | 0.0000 |
| R5 | 0.1667 / 0.2500 / 0.3333 / 0.4167 | 0.1742 / 0.2084 / 0.3245 / 0.3245 | 0.0429 / 0.0286 | 191.0000 | 0.0000 |
| R1_direct | 0.1667 / 0.1667 / 0.2500 / 0.5000 | 0.1667 / 0.2958 / 0.3033 / 0.3336 | 0.0286 / 0.0214 | 206.0000 | 0.0000 |

Conditional macro recall averages only evaluable questions; raw micro recall retains all 63 instances. Primary recall and R use canonical scientific identities; synonymous supplied labels with overlapping resolved nodes count once, while raw label instances remain separate. Precision uses recovered distinct canonical targets divided by fixed K, even for short lists. MRR uses the first relevant position in the full exhaustive ranking. Unique-target recall means recovered in at least one associated question, not recovery on every occurrence. Representable evidence/composite recall is separate and cannot establish semantic support. Incomplete gold labels make precision a lower-bound relevance proxy. All K=1/5/10/20/50/100 values, including null statuses, are in JSON.

Candidate recall refers to the pre-reranking top 50/100, not membership in the full graph pool. An outcome `RESOLVED_BUT_NOT_CANDIDATE` is qualified by `in_pool`, `candidate_rank` and `candidate_failure_reason`; an in-pool rank below 100 is a first-stage ranking failure. `CANDIDATE_BUT_BELOW_K` concerns final K=10. All raw targets retain an explicit outcome, including ambiguous and unrepresentable targets.

## Type B evidence and cost

| Cutoff / stage | Source hit@10 | Source recall@10 | Source mapping coverage | Mean claim candidates | Entity + claim dots | Entity-search seconds |
| --- | --- | --- | --- | --- | --- | --- |
| 2025 / R0 | 1.0000 | 0.4167 | 0.6875 | 160.0000 | 4889.7500 | 0.1531 |
| 2025 / R1 | 0.2500 | 0.1250 | 0.6875 | 193.5000 | 4892.0000 | 0.4887 |
| 2025 / R2 | 0.2500 | 0.1250 | 0.6875 | 221.5000 | 4892.0000 | 0.4239 |
| 2025 / R3 | 0.2500 | 0.1875 | 0.6875 | 200.5000 | 4892.0000 | 0.4903 |
| 2025 / R4 | 0.2500 | 0.1250 | 0.6875 | 204.0000 | 15674.0000 | 0.4160 |
| 2025 / R5 | 0.2500 | 0.1250 | 0.6875 | 179.0000 | 15674.0000 | 0.4243 |
| 2025 / R1_direct | 0.5000 | 0.2083 | 0.6875 | 198.7500 | 4892.0000 | 0.5086 |
| 2026 / R0 | 0.7500 | 0.3333 | 0.8000 | 165.2500 | 5655.0000 | 0.2092 |
| 2026 / R1 | 0.7500 | 0.2708 | 0.8000 | 184.5000 | 5655.0000 | 0.5204 |
| 2026 / R2 | 0.7500 | 0.2708 | 0.8000 | 241.5000 | 5655.0000 | 0.5522 |
| 2026 / R3 | 0.7500 | 0.3333 | 0.8000 | 218.0000 | 5655.0000 | 0.5717 |
| 2026 / R4 | 0.7500 | 0.2708 | 0.8000 | 220.2500 | 18153.7500 | 0.5453 |
| 2026 / R5 | 0.5000 | 0.1875 | 0.8000 | 221.7500 | 18153.7500 | 0.5505 |
| 2026 / R1_direct | 0.5000 | 0.2083 | 0.8000 | 190.2500 | 5655.0000 | 0.5163 |

Source overlap measures article provenance, never entailment. Title/DOI or uniquely matching author/year resolve required references. Unmatched sources remain in source-resolution coverage and per-question raw lower bounds. Claim candidates come from retrieved entities, shared article provenance and direct graph relations; up to 100 receive question–claim scores, with deterministic direct-first/ID-order selection when capped. This ordering can exclude useful evidence. Strict required-claim alignment remains 0/224 in the separate legacy diagnostic, so claim recall is null. Required-source metrics are reported separately for Type B; entity-target metrics do not replace them. Source-at-K, returned claim texts/paths and per-question candidate counts are exported.

Context is not an across-the-board improvement: at 2025, Type B source hit falls from 1.00 in R0 to 0.25 in R3, and source recall falls from 0.4167 to 0.1875. The proposed Type A diagnostic configuration should therefore not be advertised as a better evidence retriever. The sample is only four Type B questions and these are provenance proxies.

A charged comparison is one computed query–text-vector dot product; identical text is reused across fields/entities, once per query view. Claim comparisons are charged separately, conservatively recomputed even when a context field used the same text. Offline encoding, representation/index preparation, graph enumeration, scalar aggregation and lexical reranking are outside dot-product cost; query time includes scalar aggregation/reranking but excludes claim traversal and offline model encoding. Index times include actual lazy grouped/direct index creation. Timings are local single-run observations, not controlled latency claims. Every flat run has zero cluster expansions.

## Failure analysis and eight conclusions

The best observed final primary top-10 micro recall is **4/47 = 0.0851** (R1); raw recovery is 0.0635. For R3, 38/47 canonical target instances have a fixed-field upper score bound below that question’s actual tenth result. Per-target field availability, bounds, ranks and thresholds are in `contextual/failure_analysis.json`. This diagnoses a representation/score-availability limitation; it is not a causal estimate of an alternative scorer. Sparse cited-work names often lack task-specific graph evidence; broad article context can add unrelated evidence. A better generic scorer needs independent validation, not retrospective benchmark-specific weights.

The four recovered R3 targets—D4FT, DeepH-E3, LiFlow, Matbench Discovery—were already accepted by the old resolver. Thus the top-10 gain is not merely the extra plural alias increasing mapping coverage. Median canonical target rank moves from 4231 in all-type R0 to 233 in R1 and 160 in R3; the old routed baseline median of 1463 used a different, smaller pool and target set.

1. **Why zero original flat recall?** Short name/type embeddings failed to connect long task questions to scientific identifiers; mapped targets ranked 29–2387 at 2025, median 1463. This is measured ranking failure, with representation inadequacy the supported interpretation.
2. **Were correct targets present?** Yes: 45/46 mapped target instances were in the routed flat pool. One complete exclusion came from type routing. New all-type pools remove that route restriction.
3. **What kind of mapping failures?** State and coverage tables distinguish exact/alias identities, incomplete composites, repeated evidence phrases, ambiguity and unsupported labels. No unresolved label is silently recast as an alias.
4. **Does context materially improve recovery?** R1 versus R0 micro R@10 is 0.0851 versus 0.0000. The table reports the gain without equating a small gain to reliable QA.
5. **Does deduplication help?** R2 minus R1 micro R@10 is 0.0000; duplicate-slot fractions and rank shifts reveal effects hidden by that aggregate.
6. **Do decomposition/reranking help?** R3/R4/R5 micro R@10 is 0.0851 / 0.0851 / 0.0851. MRR and candidate recall are shown separately; multiple views increase comparison cost.
7. **Is the 12/40/120 hierarchy justified after contextualization?** Not established. Flat quality remains poor, so R7/R8 were not implemented or claimed as measured improvements, following the explicit stop rule. Existing fixed-beam, prototype and flat surface baselines remain runnable across five variants and both cutoffs. Lower cost at near-zero recall is not a success.
8. **Are contextual prototypes useful?** Unknown, not tested because the prerequisite was not met. Existing surface prototypes still provide no Type A top-10 recall advantage; adding contextual traversal now would not resolve the dominant representation/ranking failure. Optional generative expansion was also omitted.

## Remaining primary noncanonical targets

| Target | State | Representable | Reason / components |
| --- | --- | --- | --- |
| Bead-mapping | UNRESOLVED | False | No canonical name; insufficient repeated evidence (semantic relatedness alone cannot certify identity) |
| CE→MC→NN→PF | COMPOSITE | False | CE: UNRESOLVED; MC: UNRESOLVED; NN: EVIDENCE_BACKED; PF: UNRESOLVED |
| DeePMD+AL | COMPOSITE | False | DeePMD: EXACT_NODE; AL: EXACT_NODE |
| GNN free energies | UNRESOLVED | False | No canonical name; insufficient repeated evidence (semantic relatedness alone cannot certify identity) |
| GNN stress fields | UNRESOLVED | False | No canonical name; insufficient repeated evidence (semantic relatedness alone cannot certify identity) |
| GNN+GPP | COMPOSITE | False | GNN: AMBIGUOUS; GPP: UNRESOLVED |
| GNNs + ACE descriptors | COMPOSITE | False | GNNs: EVIDENCE_BACKED; ACE descriptors: UNRESOLVED |
| Hessian training | UNRESOLVED | False | No canonical name; insufficient repeated evidence (semantic relatedness alone cannot certify identity) |
| MACE-F | UNRESOLVED | False | No canonical name; insufficient repeated evidence (semantic relatedness alone cannot certify identity) |
| MLIP descriptors | UNRESOLVED | False | No canonical name; insufficient repeated evidence (semantic relatedness alone cannot certify identity) |
| MPtrj | UNRESOLVED | False | No canonical name; insufficient repeated evidence (semantic relatedness alone cannot certify identity) |
| PET | UNRESOLVED | False | No canonical name; insufficient repeated evidence (semantic relatedness alone cannot certify identity) |
| UMA | UNRESOLVED | False | No canonical name; insufficient repeated evidence (semantic relatedness alone cannot certify identity) |
| hybrid frameworks | UNRESOLVED | False | No canonical name; insufficient repeated evidence (semantic relatedness alone cannot certify identity) |
| physics-informed | EVIDENCE_BACKED | True | Represented by multiple literal evidence texts; no canonical entity identity asserted |
| xDeepH | EVIDENCE_BACKED | True | Represented by multiple literal evidence texts; no canonical entity identity asserted |

## Files, tests and reproduction

Changed modules: `target_mapping.py` adds `TargetResolver`; `contextual.py` adds `entity_groups`, `build_representations`, `analyze_question`, `ContextIndex`; `evaluation/retrieval_v2.py` adds contextual metrics/summaries; `v2_pipeline.py` adds `run_context`; `cli.py` exposes ablations. Added `configs/contextual.yaml`, `tests/test_contextual.py`, `scripts/audit_retrieval_candidates.py`, and this generated-report/audit script; updated packaging, README and AI_USAGE. Original construction modules and hierarchy files remain hash-identical.

Synthetic tests cover Unicode/dashes, plural/identifier/acronym identity, unsafe/ambiguous aliases, composites, evidence-only and unresolved targets, exact grouping, broad-edge exclusion, per-field caps, no self-text duplication, vector normalization, soft priors, multi-query pooling/fusion, candidate cutoffs, multi-answer/R-precision, duplicate gold aliases, raw denominators, future provenance/edges, rerank bounds/stopwords and ground-truth independence. Existing hierarchy/flat surface contract and root-budget tests remain unchanged. Contextual hierarchy-specific tests are intentionally not claimed. JUnit and automated all-question/score/hash checks are shipped.

```powershell
$env:HF_HUB_OFFLINE='1'
.venv\Scripts\python.exe -m pytest -q --junitxml=artifacts/evaluation/tests.xml
.venv\Scripts\python.exe -m tkh_abstraction.cli evaluate-v2 --config configs/evaluation_v2.yaml
.venv\Scripts\python.exe scripts/audit_retrieval_candidates.py
.venv\Scripts\python.exe -m tkh_abstraction.cli evaluate-context --config configs/contextual.yaml
.venv\Scripts\python.exe scripts/audit_v2.py
.venv\Scripts\python.exe scripts/report_contextual.py
.venv\Scripts\python.exe scripts/package_submission.py
```

For one ablation, append `--stage R0`, `R1`, `R2`, `R3`, `R4`, `R5`, or `R1_direct`, and optionally `--cutoff 2025`. `--resolver exact` versus `generic` and `--candidate-k 50` versus `100` are available; the shipped complete run uses generic/100. Pooling, field weights, caps, fusion and top-K lists are in the same YAML system. Partial invocations replace contextual summary outputs with their requested subset; rerun the complete command to restore the complete report. For new graph/benchmark exports set data/output paths and `context_verify_reuse: false`; the graph-only retriever and evaluator use no current Q IDs. Baseline verification remains enabled for this assessment.

**Recommended default:** R3 grouped contextual flat with original questions and soft priors for Type A diagnostic work, not a certified answer engine. This was the predeclared contextual reference; selecting a best-on-benchmark system would require independent validation. Retain R0 as the stronger observed Type B source-retrieval reference; no automatic mixed-question deployment is claimed. The hierarchy remains useful for browsing under the separate intrinsic results, but its retrieval acceleration is not justified by these measurements. Priorities are better graph-attached entity evidence, auditing missing-field effects, independent query sets and human source/claim adjudication. No further hierarchy complexity or threshold fitting was used to hide failure.

## Query-seeded native hypergraph diffusion

The new Type A candidate experiment reuses the native sparse Zhou operator with atomic semantic seeds. At conservative 2025, macro Candidate Recall@100 is 0.3701 for historical R3, 0.3979 for R3 with the same answer pool, 0.4994 for pure native H0, 0.7146 for mention-assisted H2 fusion, and 0.7230 for relation-weighted H3. H2/H3 median gold ranks are 66.0/59.0, but both recover only 3/47 canonical instances in the top ten versus R3’s 4/47.

Real H0 exceeds its shuffled control (0.0917 C@100), while the weight-conserving pairwise control matches native C@100 (0.4994). This supports further candidate-mechanism research, not a native-hypergraph superiority or solved-QA claim. Type B and historical R0–R5 numbers remain unchanged. See [hypergraph_diffusion.md](hypergraph_diffusion.md) for all variants, costs, sensitivities, every target’s rank changes, approximate paths and thirteen explicit conclusions.
