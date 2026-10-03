# V2 flat retrieval report — DEVELOPMENT

## Material Passport

Computational retrieval development study; frozen supplied graph and questions; prior benchmark exposure acknowledged. Verification status: ANALYZED. Software invariants and fixed-input inference determinism tested; no held-out or statistical-superiority claim.


## Hypothesis and scope

The scientific question is whether explicit query structure, candidate evidence, and joint relevance scoring improve flat answer identity retrieval. This implementation tests a deterministic extractive parser, not perfect scientific understanding. It uses the frozen V1 2026 publication-visible graph and never builds or routes a hierarchy. All numbers are development results on an already-inspected benchmark, not held-out generalization or statistical superiority.

## Exact V1 differences

V1's document builder already attaches at most 12 direct neighbors, prioritizing claims. The raw-question V1-flat-like control reproduces this code rather than substituting identity-only documents. V2 changes retrieval representations and scoring only. Hierarchy construction, semantic/structural objectives, temporal VI, collapse, labels, null models, and perturbations are untouched.

## Question understanding

The independent V2 copy of the extractive parser detects an answer head, separates task from conditional clauses, splits coordinated requirements, preserves negations, and records temporal source spans. Q0 is the original question, Q1 concatenates extracted intent compactly, and Q2 labels the answer family, task, requirements, exclusions, and date. Unknown or coordinated answer heads fall back to all non-author entities with explicit warnings. Adjectival scientific modifiers before an answer head can be lost; no question-specific repair is made. OR clauses remain intact rather than being solved as Boolean logic.

A callable interface for an optional local instruction model accepts strict JSON and checks extraction constraints. It is not used in this run. No LLM result is fabricated. Year-level graph visibility cannot enforce February precision, so temporal clauses are recorded while the primary graph stays frozen at 2026.

## Answer families and evidence cards

The method-like family is method, technique, component, cited_work: all four exist in the graph schema and can name a scientific approach or its published source. This broad family is a modeling assumption, not proof every member is a method. Dataset, metric, and article families are mapped separately. Every exact mapping is in the frozen configuration. Hard filtering is compared with all non-authors and a fixed multiplicative soft prior of 0.8 outside the family.

Cards preserve name and type, followed by capped direct claims, tasks/problems, techniques/components, datasets/metrics, source titles, and other safe context. Native relations of arity at most three supply direct incidence evidence, with edge and node IDs and provenance. Incidence does not establish subject/predicate direction. Exact bounded candidate-name occurrences add claim evidence, with short generic strings excluded. Source titles can be attached by shared provenance, but an article's entire claim set is never dumped into a card. High-arity incidences are explicitly excluded and logged. This conservative policy can discard useful evidence and is itself a limitation.

## Retrieval and reranking

Pinned BGE independently embeds Q0/Q1/Q2 and V1 documents/cards. No query instruction prefix is added, matching the V1 reference. BM25 uses k1=1.5, b=0.75; dense and lexical full rankings fuse with RRF k=60. Ties break by node ID. The primary first stage is predeclared V2-B, not whichever system later scores best on gold. Candidate recall is examined at 50/100/200; relevance and requirement rerankers always use 100.

The genuine MS MARCO MiniLM-L6 relevance regressor jointly consumes the original question and identity-first card. It is not the V1 DeBERTa NLI model. Inputs never exceed 512 tokens, identity is retained, and omitted blocks, included blocks, token counts, raw logits, and amortized batch runtime are logged. The exact model ID, revision, weights, package versions, and CPU execution are recorded. A4 reranks A3; V2-C reranks V2-B. All remaining candidates retain their first-stage tail order. Raw relevance logits are not calibrated probabilities and are not arithmetically mixed with cosine scores.

V2-D uses max rectified BGE similarity to card blocks for the core and each requirement, a geometric mean across requirements, and a multiplicative exclusion penalty. A floor of 1e-6 inside the logarithm avoids undefined logs. Similarity to an exclusion is only a heuristic violation signal; this is not negation reasoning. Different blocks can maximize different requirements, so this score does not prove simultaneous satisfaction. Optional NLI verification is not run.

## Isolation and audit

Only graph data and questions are copied into prediction inputs. The prediction entry point does not import V1 benchmark or evaluation modules. A Python audit hook denies answer filenames, protected reference-source reads, and writes outside V2. It allows third-party packages from the recorded interpreter read-only. Rankings for every question and system are completed and written before the prediction SHA-256 and artifact freeze are saved. Evaluation runs in a separate process that verifies the freeze before importing its evaluator and loading gold. Phase events establish the order, and the final protected-tree hash comparison independently checks V1 immutability.

The audit hook is a cooperative-code tripwire, not hostile-code containment or an external timestamp authority. Source, data, configuration, and prediction hashes support reproducibility; exact repeated inference is tested on fixed synthetic input. A complete second benchmark rerun is not claimed. Pre-gold implementation failures and deterministic accumulation fixes are recorded in the run ledger. No parameters or scientific questions were tuned on target ranks.

## Ablation interpretation

A0→A1 isolates hard type filtering under identical V1 documents and BGE scores. A0→A2 isolates evidence-card construction with raw questions and all non-authors. A2→A3 isolates parsing with the same cards and pool. A3→A4 isolates top100 relevance reranking. A3→V2-A isolates filtering with parsed cards. V2-A→V2-B adds fixed lexical fusion; V2-B→V2-C changes only the prefix order using joint relevance. DenseRaw, DenseStructured, Lexical/All, SoftType, and V2-D provide additional diagnostic comparisons. These factors interact; the ladder is not a complete factorial causal study.


## Metrics and denominators

All 18 questions have frozen predictions. Direct identity metrics use 14 Type-A questions and 63 target occurrences: 48 EXACT, 1 ALIAS, 1 COMPOSITE, 11 PARTIAL, 2 ABSENT. The 49 EXACT/ALIAS occurrences are the direct-identity subset. PARTIAL and ABSENT never earn identity credit; COMPOSITE is reported separately.

The table reports **micro direct Recall@k over all 63 targets**, and MRR over all 14 Type-A questions. `metrics.json` also gives macro recall, conditional direct recall over 49 targets, V1-compatible composite bundle recovery, and V1 supported recall over 61 non-ABSENT targets (including PARTIAL in the denominator).

| System | R@1 | R@5 | R@10 | R@25 | R@50 | R@100 | MRR | Median direct rank | Mean direct rank | Missing /49 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.016 | 0.032 | 0.005 | 3122 | 3009.6 | 0 |
| A1 | 0.032 | 0.048 | 0.048 | 0.048 | 0.063 | 0.095 | 0.165 | 1293 | 1546.4 | 0 |
| A2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.032 | 0.032 | 0.006 | 2671 | 2759.8 | 0 |
| A3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.016 | 0.016 | 0.004 | 3446 | 3074.1 | 0 |
| A4 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.016 | 0.003 | 3446 | 3075.3 | 0 |
| DenseRaw | 0.000 | 0.000 | 0.000 | 0.032 | 0.048 | 0.048 | 0.013 | 1876 | 1652.4 | 0 |
| DenseStructured | 0.000 | 0.000 | 0.000 | 0.016 | 0.048 | 0.048 | 0.009 | 1863 | 1913.3 | 0 |
| V2-A | 0.000 | 0.000 | 0.000 | 0.016 | 0.032 | 0.063 | 0.011 | 1857 | 1954.8 | 0 |
| LexicalAll | 0.000 | 0.000 | 0.000 | 0.016 | 0.032 | 0.032 | 0.007 | 974 | 1098.6 | 0 |
| Lexical | 0.000 | 0.000 | 0.016 | 0.016 | 0.032 | 0.032 | 0.016 | 649 | 701.5 | 0 |
| V2-B | 0.000 | 0.000 | 0.016 | 0.016 | 0.048 | 0.048 | 0.017 | 1192 | 1108.5 | 0 |
| V2-C | 0.000 | 0.032 | 0.032 | 0.032 | 0.032 | 0.048 | 0.057 | 1192 | 1108.9 | 0 |
| V2-D | 0.000 | 0.032 | 0.032 | 0.032 | 0.032 | 0.048 | 0.053 | 1192 | 1108.3 | 0 |
| SoftType | 0.000 | 0.000 | 0.000 | 0.016 | 0.032 | 0.063 | 0.011 | 1857 | 1986.6 | 0 |

Rank means/medians exclude missing direct identities and report their counts. The machine-readable metrics also give a fixed missing-rank penalty of 5,481, preventing filtered pools from receiving a smaller penalty.

### Before-reranking candidate recall

| Generator | hits@50 /63 | hits@100 /63 | hits@200 /63 |
|---|---:|---:|---:|
| A3 | 1 | 1 | 3 |
| V2-A | 2 | 4 | 4 |
| Lexical | 2 | 2 | 7 |
| V2-B | 3 | 3 | 6 |

### Reranker retention

| Reranker | Targets in prefix | Still in prefix | Up | Down | Unchanged |
|---|---:|---:|---:|---:|---:|
| V2-C | 3 | 3 | 2 | 1 | 0 |
| A4 | 1 | 1 | 0 | 1 | 0 |

### Interpretation

A0 retrieves 0/63 direct identities at rank 10; primary V2-C retrieves 2/63. These are descriptive development counts. Individual regressions are retained below. A higher aggregate count does not establish that question parsing caused the difference; filtering, representation, and reranking controls must be read together.

The type-filter-only control A1 reaches 3/63 at top 10 and MRR 0.1645, above full V2-C's 2/63 and MRR 0.0566. Evidence cards alone (A2) and parsed cards (A3) recover no direct targets at top 10; A4's relevance reranking also recovers none. This run therefore does **not demonstrate that explicit parsing plus cards fixes the answer-identity failure**.

The primary generator admits only 3/49 mapped EXACT/ALIAS occurrences to top100; 46 cannot be rescued by its reranker. V2-C moves two present target occurrences upward and one downward. The main observed bottleneck is candidate generation, with sparse/misaligned evidence and parser losses plausible but not causally proven explanations.

The card audit finds 4,187/5,480 candidates without selected non-title evidence. Direct-family caps and static ID tie-breaking can omit query-relevant evidence; high-arity omission affects 34 mapped target occurrences. The parser loses pre-head scientific modifiers in some questions and deliberately leaves coordinated answer families unrestricted. These are documented limitations, not post-evaluation repairs.

## Per-question intent diagnostics

| Question | Original | Answer type | Core task | Requirements | Exclusions | Rewritten query |
|---|---|---|---|---|---|---|
| Q1 | Which methods (by Feb 2026) are best suited for predicting interatomic potentials when balancing accuracy close to DFT with computational efficiency for systems of millions of atoms? | method | predicting interatomic potentials | accuracy close to DFT; computational efficiency; systems of millions of atoms |  | method; predicting interatomic potentials; accuracy close to DFT; computational efficiency; systems of millions of atoms |
| Q2 | Which methods (by Feb 2026) are best suited for accelerating electronic structure calculations when self-consistent field iterations become prohibitively expensive for large systems (>10³ atoms)? | method | accelerating electronic structure calculations | self-consistent field iterations become prohibitively expensive; large systems (>10³ atoms) |  | method; accelerating electronic structure calculations; self-consistent field iterations become prohibitively expensive; large systems (>10³ atoms) |
| Q3 | Which methods (by Feb 2026) are best suited for learning DFT Hamiltonians when dealing with twisted van der Waals heterostructures containing thousands of magnetic atoms? | method | learning DFT Hamiltonians | dealing with twisted van der Waals heterostructures containing thousands of magnetic atoms |  | method; learning DFT Hamiltonians; dealing with twisted van der Waals heterostructures containing thousands of magnetic atoms |
| Q4 | Which methods (by Feb 2026) are best suited for phonon property prediction when universal MLIPs show inaccuracies despite good energy/force performance? | method | phonon property prediction | universal MLIPs show inaccuracies despite good energy/force performance |  | method; phonon property prediction; universal MLIPs show inaccuracies despite good energy/force performance |
| Q5 | Which methods (by Feb 2026) are best suited for coarse-graining molecular dynamics when crystalline symmetries and long-range elastic interactions must be preserved? | method | coarse-graining molecular dynamics | crystalline symmetries; long-range elastic interactions must be preserved |  | method; coarse-graining molecular dynamics; crystalline symmetries; long-range elastic interactions must be preserved |
| Q6 | Which methods (by Feb 2026) are best suited for predicting ionic conductivity when full ab initio MD is too expensive but accuracy must be maintained? | method | predicting ionic conductivity | full ab initio MD is too expensive; accuracy must be maintained |  | method; predicting ionic conductivity; full ab initio MD is too expensive; accuracy must be maintained |
| Q7 | Which methods (by Feb 2026) are best suited for predicting phase transitions when screening ~50,000 inorganic compounds at high throughput? | method | predicting phase transitions | screening ~50,000 inorganic compounds at high throughput |  | method; predicting phase transitions; screening ~50,000 inorganic compounds at high throughput |
| Q8 | Which methods (by Feb 2026) are best suited for predicting fracture patterns when generalizing from atomistic MD data to unseen bicrystalline structures? | method | predicting fracture patterns | generalizing from atomistic MD data to unseen bicrystalline structures |  | method; predicting fracture patterns; generalizing from atomistic MD data to unseen bicrystalline structures |
| Q9 | Which methods (by Feb 2026) are best suited for ensuring physical consistency when transferring information between scales without violating thermodynamic principles? | method | ensuring physical consistency | transferring information between scales | violating thermodynamic principles | method; ensuring physical consistency; transferring information between scales; without violating thermodynamic principles |
| Q10 | Which methods (by Feb 2026) are best suited for representing atomic environments when the system exhibits significant compositional diversity, defects, and disorder? | method | representing atomic environments | the system exhibits significant compositional diversity, defects; disorder |  | method; representing atomic environments; the system exhibits significant compositional diversity, defects; disorder |
| Q11 | Which methods (by Feb 2026) are best suited for integrating features from three or more distinct scales simultaneously within a unified framework? | method | integrating features from three or more distinct scales simultaneously within a unified framework |  |  | method; integrating features from three or more distinct scales simultaneously within a unified framework |
| Q12 | What are the leading (by Feb 2026) ML-based interatomic potential methods for large-scale atomistic simulation? | method | large-scale atomistic simulation |  |  | method; large-scale atomistic simulation |
| Q13 | What are the leading (by Feb 2026) GNN architectures for predicting properties of solid-state crystalline materials? | architecture | predicting properties of solid-state crystalline materials |  |  | architecture; predicting properties of solid-state crystalline materials |
| Q14 | What datasets and benchmarks are commonly used (by Feb 2026) to evaluate MLIPs and property prediction models for materials discovery? | None | evaluate MLIPs and property prediction models for materials discovery |  |  | evaluate MLIPs and property prediction models for materials discovery |

## Per-target direct rank diagnostics

An em dash means no direct identity rank. Composite completion and partial evidence mappings remain separate in `outputs/per_question_evaluation.json`. Every target record includes selected evidence, requirement components, relevance token packing when scored, and the candidate-top100 flag.

| Question | Target | Status | A0 | V2-A | V2-B | V2-C | V2-D | In primary top100 |
|---|---|---|---:|---:|---:|---:|---:|---|
| Q1 | MACE | EXACT | 5437 | 2377 | 1773 | 1773 | 1773 | False |
| Q1 | M3GNet | EXACT | 880 | 569 | 198 | 198 | 198 | False |
| Q1 | CHGNet | EXACT | 1340 | 939 | 1108 | 1108 | 1108 | False |
| Q1 | GAP | EXACT | 5357 | 1394 | 1383 | 1383 | 1383 | False |
| Q1 | NequIP | EXACT | 1536 | 2031 | 325 | 325 | 325 | False |
| Q1 | ACE | EXACT | 5374 | 2059 | 1601 | 1601 | 1601 | False |
| Q1 | EquiformerV2 | EXACT | 4656 | 1337 | 560 | 560 | 560 | False |
| Q2 | DeepH | EXACT | 5227 | 1596 | 1631 | 1631 | 1631 | False |
| Q2 | D4FT | EXACT | 370 | 1574 | 1321 | 1321 | 1321 | False |
| Q2 | HamGNN | PARTIAL | — | — | — | — | — | False |
| Q3 | xDeepH | PARTIAL | — | — | — | — | — | False |
| Q3 | DeepH-E3 | EXACT | 57 | 36 | 7 | 2 | 2 | True |
| Q4 | Hessian training | ALIAS | 634 | 54 | 111 | 111 | 111 | False |
| Q4 | VGNN | EXACT | 5193 | 1060 | 507 | 507 | 507 | False |
| Q4 | MACE-F | ABSENT | — | — | — | — | — | False |
| Q5 | Bead-mapping | ABSENT | — | — | — | — | — | False |
| Q5 | GNN+GPP | PARTIAL | — | — | — | — | — | False |
| Q6 | DeePMD+AL | PARTIAL | — | — | — | — | — | False |
| Q6 | LiFlow | EXACT | 199 | 350 | 128 | 128 | 128 | False |
| Q6 | MLIP descriptors | PARTIAL | — | — | — | — | — | False |
| Q7 | GNN free energies | PARTIAL | — | — | — | — | — | False |
| Q7 | symbolic regression | EXACT | 4259 | 2005 | 949 | 949 | 949 | False |
| Q8 | ConvLSTM | EXACT | 5206 | 88 | 28 | 4 | 5 | True |
| Q8 | GNN stress fields | PARTIAL | — | — | — | — | — | False |
| Q9 | E(3)-equivariant GNNs | PARTIAL | — | — | — | — | — | False |
| Q9 | physics-informed | PARTIAL | — | — | — | — | — | False |
| Q9 | NequIP | EXACT | 2625 | 2102 | 1257 | 1257 | 1257 | False |
| Q10 | GNNs + ACE descriptors | PARTIAL | — | — | — | — | — | False |
| Q10 | SchNet | EXACT | 4197 | 2599 | 1266 | 1266 | 1266 | False |
| Q11 | CE→MC→NN→PF | COMPOSITE | — | — | — | — | — | False |
| Q11 | hybrid frameworks | PARTIAL | — | — | — | — | — | False |
| Q12 | MACE | EXACT | 5360 | 2429 | 1313 | 1313 | 1313 | False |
| Q12 | NequIP | EXACT | 2476 | 2640 | 1201 | 1201 | 1201 | False |
| Q12 | CHGNet | EXACT | 1262 | 1086 | 908 | 908 | 908 | False |
| Q12 | M3GNet | EXACT | 748 | 1248 | 928 | 928 | 928 | False |
| Q12 | SevenNet | EXACT | 5432 | 2291 | 1300 | 1300 | 1300 | False |
| Q12 | GAP | EXACT | 5328 | 1854 | 1397 | 1397 | 1397 | False |
| Q12 | DeePMD | EXACT | 4621 | 1485 | 1289 | 1289 | 1289 | False |
| Q12 | MatterSim | EXACT | 31 | 570 | 268 | 268 | 268 | False |
| Q12 | Orb | EXACT | 4994 | 11 | 38 | 88 | 59 | True |
| Q12 | eSEN | EXACT | 539 | 302 | 284 | 284 | 284 | False |
| Q12 | PET | EXACT | 5452 | 2151 | 1250 | 1250 | 1250 | False |
| Q12 | UMA | EXACT | 5390 | 1857 | 1192 | 1192 | 1192 | False |
| Q12 | EquiformerV2 | EXACT | 4726 | 2365 | 1025 | 1025 | 1025 | False |
| Q13 | CGCNN | EXACT | 3322 | 887 | 355 | 355 | 355 | False |
| Q13 | SchNet | EXACT | 2625 | 2354 | 2242 | 2242 | 2242 | False |
| Q13 | MEGNet | EXACT | 3513 | 1501 | 949 | 949 | 949 | False |
| Q13 | M3GNet | EXACT | 2655 | 1948 | 492 | 492 | 492 | False |
| Q13 | NequIP | EXACT | 3198 | 2759 | 2410 | 2410 | 2410 | False |
| Q13 | MACE | EXACT | 5434 | 2621 | 1964 | 1964 | 1964 | False |
| Q13 | CHGNet | EXACT | 2770 | 1702 | 500 | 500 | 500 | False |
| Q13 | EquiformerV2 | EXACT | 4979 | 2516 | 1710 | 1710 | 1710 | False |
| Q14 | Matbench Discovery | EXACT | 447 | 838 | 555 | 555 | 555 | False |
| Q14 | Matbench | EXACT | 1596 | 2092 | 404 | 404 | 404 | False |
| Q14 | JARVIS-DFT | EXACT | 4681 | 4755 | 3777 | 3777 | 3777 | False |
| Q14 | Materials Project | EXACT | 591 | 864 | 1235 | 1235 | 1235 | False |
| Q14 | MPtrj | EXACT | 896 | 4264 | 1664 | 1664 | 1664 | False |
| Q14 | OQMD | EXACT | 1293 | 3545 | 765 | 765 | 765 | False |
| Q14 | AFLOW | EXACT | 5388 | 5107 | 2378 | 2378 | 2378 | False |
| Q14 | Alexandria | EXACT | 3122 | 4826 | 1792 | 1792 | 1792 | False |
| Q14 | GNoME | EXACT | 506 | 1585 | 923 | 923 | 923 | False |
| Q14 | WBM | EXACT | 217 | 4099 | 1985 | 1985 | 1985 | False |
| Q14 | OMat24 | EXACT | 1331 | 5064 | 1669 | 1669 | 1669 | False |

## Failure observations

| Observation | Target occurrences |
|---|---:|
| 1_correct_answer_not_in_candidate_top100 | 46 |
| 7_high_arity_evidence_excluded_ambiguity | 34 |
| not_a_resolved_direct_identity:PARTIAL | 11 |
| selected_card_lacks_non_title_evidence | 3 |
| 4_no_adjacent_claim_task_problem_evidence_in_graph | 3 |
| not_a_resolved_direct_identity:ABSENT | 2 |
| not_a_resolved_direct_identity:COMPOSITE | 1 |
| 2_in_top100_but_reranker_not_top10 | 1 |
| 3_target_type_excluded_by_answer_family | 0 |

Failure classes 5 (parser missed useful requirements), 6 (acronym representation), and 8 (requirements not jointly satisfied) are not automatically causally identified. The exported decompositions, evidence, and component scores support independent review; similarity alone cannot adjudicate these causes. No failures were manually patched.

## Limitations and next experiment


The benchmark is small, previously inspected, and domain-specific. Schema-based filtering may exclude valid identities; broad method-like families retain false positives. Sparse direct evidence and conservative arity caps can starve candidates. Literal name matching can be ambiguous. Extractive rewriting can omit scientific modifiers or fail on coordinated answer types. BGE and a web-passage reranker can misrepresent scientific acronyms, constraints, and negation. A frozen top100 prefix imposes a recall ceiling. Annual visibility is coarser than the questions' dates. No optional generative parsing or NLI result is available from this run.

The next experiment should pre-register a new, independently authored held-out question set and evaluate the frozen pipeline without retuning. Evidence coverage and missing query modifiers can be audited on a separate development set before that freeze. Compare explicit relation-role evidence against the present conservative incidence policy, using an independently verified schema and fixed evidence budget. Do not patch any individual question or answer from this run.


## Reproduction and audit artifacts

See [README](../README.md) for exact commands. Complete per-system rankings are in `outputs/retrieval/by_system/`; scored evidence is in `outputs/retrieval/`; all metrics and denominators are in `outputs/metrics.json`. The protected-tree comparison is `outputs/audit/protected_verification.json`; test output is `outputs/audit/tests_final.log`; every new file is listed in `outputs/audit/new_files.txt`.

Prediction SHA-256: `a6a1b828641a3471cd03f728054c92bdb42471b72274ed57886b64928249c90f`.

Phase ordering, input/source freeze, model weights, evaluation input hashes, and rank audit are saved under `outputs/audit/`. The implementation ledger records pre-gold retries. The scoped Claude consultation and 11-category descriptive validity review are in `notes/`.

Model sources: [BGE](https://huggingface.co/BAAI/bge-small-en-v1.5) and [MS MARCO relevance MiniLM](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2). The latter is a relevance model; optional DeBERTa NLI was not run.

## Final verification

19 tests passed. All prediction/source/input freeze hashes and cached model-file hashes verify. Copied graph and question files match the reference. Prediction freeze precedes evaluator import and gold loading. **CLEAN_TSHC UNCHANGED: YES** — all 1,056 file hashes and the complete before/after manifest bytes match. No git staging, commits, pushes, resets, or history changes were performed. The exhaustive file list, including generated artifacts and caches, is [new_files.txt](../outputs/audit/new_files.txt).
