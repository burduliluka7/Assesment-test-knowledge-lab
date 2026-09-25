# CR6 decisions and consultation record

The request is preserved in request.txt. Ultralight Opus planning and serial Sonnet review use the official first-party subscription helper. No native agent/account switching or API keys are used.

Before the first outcome run, Codex fixed the supplied structural caps (20/20/15/15/10/10, reserved path cap 10), top-three component-only selection, the old attribution weights, unchanged aggregation/RRF, and two evaluation controls: old parser/attribution on the new pool and CR6 with qualifiers removed. These choices are visible in the first run's protocol. No cap, weight, threshold or new model is chosen by score.

Pools reuse CandidateEvidenceIndex.pools, with graph-only stable ordering and normalized-text deduplication. No additional path expansion is introduced. Old query-seeded paths remain in diagnostic exports. Because structural caps differ from semantic caps and paths are excluded, a larger pool is not guaranteed to contain every old item; missing old items must be measured, not assumed absent or impossible.

Historical modules are unchanged. The new decomposition subclasses QuestionDecomposition; the new scorer extends RequirementCosineReranker and invokes aggregate_support verbatim. Qualifiers have exact source spans and shared passive/modal coordinated predicates remain contiguous. The participial correction is an additive adapter, leaving archived CR1 behavior intact.

The Opus planner completed successfully (claude-opus-5, claude.ai Pro). Accepted recommendations: preserve frozen modules; verify exact matrix-shape CR1 replay; cache query-independent attribution in pools; guard an ambiguous/discarded answer head; distinguish newly introduced components and pure pool-access effects; audit lost old evidence.

Not adopted: raising structural caps, adding three more scored variants, and running old report writers. The first would change the fixed experiment after outcomes, the second is unnecessary for this bounded hypothesis, and old report writers overwrite protected historical outputs and assert a historical CLI hash. The new audit verifies the protected artifacts directly.

Not adopted: treating every sentence-initial 'including' as non-list. An initial phrase can still introduce an illustrative list. The narrower finite-predicate heuristic is explicit, generic across domains, and tested with compilers, logs and seeds; it remains a limitation, not a general syntactic parser or scientific attribution validator.

After the preliminary run, generic correctness/metadata fixes added a dangling coordinated-head guard, cached pool attribution and normalized-text delta membership. None changes candidate discovery or fitted scoring parameters. A final clean rerun is required after these source changes. Preliminary results are not final timing evidence.

No shuffled control is added: merely permuting the same selected component scores is uninformative under symmetric geometric pooling; a meaningful misalignment control would need additional design. The report makes no causal claim from its absence.

Sonnet review completed successfully (claude-sonnet-5, claude.ai Pro), read-only, with no permission denials. Accepted substantive issues: narrow the participial correction to names after the first recognized finite predicate, reject comparative/negative contexts and introductory name-list commas; protect independent finite clauses from shared-predicate merging; separate exclusion-violation deltas from positive-component changes; explicitly describe development-motivated grammar and no misalignment null. Synthetic pipeline and archived parity tests were already being added while review ran and now cover the requested preservation/ground-truth cases. Additional unrelated negative tests reproduce the review's counterexamples. The final rerun retains the preliminary 2025 recovery; all final results remain subject to the final audit.
