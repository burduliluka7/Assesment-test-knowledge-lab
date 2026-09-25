# Diffusion implementation decisions

The user specified alpha .85, seed count 100, tolerance 1e-8, max iterations 100 and a fixed configured RRF constant. These defaults were frozen in `configs/diffusion.yaml` before the first run. H3 multipliers were also declared before outcomes; H0–H2 were measured before H3 execution. H0 and H3 one-at-a-time alpha/M sensitivities do not choose a new default.

The existing spectral incidence and `laplacian` implement the requested operator correctly at the fine graph. Retrieval reuses `x - L@x`, preserving core-source and hierarchy hashes. A local graph adapter honors explicit future native weights while the supplied data's absent weights mean 1. Zero relation weights are valid through this adapter; no construction constraint was relaxed.

The pairwise control applies the existing weight-conserving clique projection and the same arity-two Zhou normalization, including self retention. The Opus planner's proposal to multiply this operator by two omitted its identity term and was rejected after paper inspection and tests. This comparison measures one projection/normalization choice; linear native diffusion is not irreducible higher-order expressivity.

R3 remains the principal historical comparison. A separately exported matched scientific-answer pool control quantifies filtering gains; atomic answer-only ranking quantifies the no-propagation reference. All current canonical targets remain eligible. General new datasets retain out-of-pool targets in recall denominators and export missing-rank/censoring counts.

Safe mention links use no ground truth, reject numeric-only and stopword/generic short names, preserve identifier/version boundaries, and do not fuzzy-match. Mention support raises only one deterministic member of an exact-name group, retains evidence-node seed mass and then globally normalizes; this can enlarge seed support and downweight unaffected nodes. That change is disclosed, not disguised as an equal-seed-support experiment.

The shuffled control preserves each edge's arity, relation and weight, but not node degree/connectivity. It is one reproducible seed-42 sanity check; stronger semantics-only causal claims would require stronger nulls. No extra tuned null/weight variants were added.

Partial diffusion commands replace only diffusion aggregates. Both successful through-H2 and full pre-review snapshots were copied to `artifacts/evaluation/diffusion/checkpoints`. Final execution reran the complete stage sequence and both cutoffs after reviewer corrections. Historical contextual JSONs, full metrics sections and Type B outputs are preserved and checked separately.

Interpretation: candidate recall improves substantially with mention-assisted fusion, but top-ten ranking remains weak. Short seed-to-target paths demonstrate connectivity, not entailment or exact attribution. Zero-hop direct/mention seeds are explicitly separate from actual graph paths. The measured candidate mechanism warrants a later approximation study, while hierarchical diffusion remains outside this pass.
