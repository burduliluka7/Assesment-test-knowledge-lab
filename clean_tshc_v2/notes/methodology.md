# Methodology fixed before gold evaluation

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

## Limitations and next experiment

The benchmark is small, previously inspected, and domain-specific. Schema-based filtering may exclude valid identities; broad method-like families retain false positives. Sparse direct evidence and conservative arity caps can starve candidates. Literal name matching can be ambiguous. Extractive rewriting can omit scientific modifiers or fail on coordinated answer types. BGE and a web-passage reranker can misrepresent scientific acronyms, constraints, and negation. A frozen top100 prefix imposes a recall ceiling. Annual visibility is coarser than the questions' dates. No optional generative parsing or NLI result is available from this run.

The next experiment should pre-register a new, independently authored held-out question set and evaluate the frozen pipeline without retuning. Evidence coverage and missing query modifiers can be audited on a separate development set before that freeze. Compare explicit relation-role evidence against the present conservative incidence policy, using an independently verified schema and fixed evidence budget. Do not patch any individual question or answer from this run.
