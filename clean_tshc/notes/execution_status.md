# Execution provenance

The initial development process completed all 16 hierarchy builds and coherence/label generation, then stopped during NLI without a completion marker. Its tool process session became unavailable across a chat-turn transition. The first fresh-output attempt also stopped during embedding preparation. Neither is accepted as a complete run or a Gate-K reproduction. Their partial outputs and logs remain in the working directory for diagnosis.

Two new full executions were launched as hidden background processes with persistent stdout/stderr logs:

- `outputs_verified`, standard new-project `.cache`: `notes/run_verified.log` and `.err`.
- `outputs_reproduced`, a separate initially empty `.cache_reproduced`: `notes/run_reproduced.log` and `.err`.

Both use final source manifests and check that their source does not change during execution. The stricter reproduction regenerates every embedding as well as hierarchies, NLI, nulls, perturbations and retrieval. They can reuse exactly pinned downloaded model weights and the already installed dependency environment. Acceptance requires actual completion records and explicit scientific-result comparisons, not merely a running process or an existing directory.

The two accepted-run attempts execute concurrently on the same CPU. Their measured times include contention and different embedding-cache states; they are provenance, not standalone performance benchmarks. Retrieval efficiency is reported in cosine-comparison counts, independently of these wall-clock measurements.

Both attempts subsequently completed all hierarchy, label NLI, perturbation, sensitivity and frozen retrieval work, then failed in evaluation-only claim alignment because one asserting article has a null DOI. Added a guarded empty-string DOI comparison and a regression test. The graph, scorer, parameters and predictions were unaffected. These attempts are not represented as successful complete executions.

`notes/recover_evaluation.py outputs_verified` explicitly replays saved NLI and completed perturbation/sensitivity builds after checking their inputs, recomputes null summaries, and executes corrected claim/target evaluation. It asserts unchanged NLI records and frozen predictions and records `audit/recovered_execution.json`. Its runtime is not a full-experiment time.

An ordinary one-command run from the corrected source starts from the new empty `outputs_final` directory (`notes/run_final.log/.err`). It uses the fixed local model/embedding cache but recomputes every hierarchy, label inference, null, perturbation, sensitivity and prediction. Only this complete fresh-output workflow, followed by comparison against independently executed prior scientific artifacts, can satisfy Gate K. The earlier separate-cache run already regenerated all vectors; that additional evidence remains explicitly separate from final full-run completion.

Subsequent claim inspection found one defensible Q18.C16 paraphrase with matching source, recorded in `notes/claim_annotations.json`. The prior `outputs_final` process was deliberately stopped to freeze this required evaluation annotation before the last full execution. The final ordinary command uses `outputs_complete`, with `notes/run_complete.log/.err`. No hierarchy/scorer setting changed. The recovery is rerun with `notes/recover_complete.log/.err`, preserving saved NLI record order (deduplicating identical texts had incorrectly substituted tiny batch-dependent probability differences and was caught by an exact-equality assertion). Recovery remains explicitly distinct from full execution.

## Accepted execution

The ordinary fresh-output workflow completed successfully in 3039.78 measured seconds, with source unchanged throughout. The complete-output audit and Gate-K comparison passed: all scientific metric fields, hierarchy memberships and supernodes, label probabilities and frozen predictions exactly matched the recovered independently executed reference. Runtime and cache paths are excluded from scientific equality. All 18,008 vectors in the separately regenerated cache also matched byte-for-byte (`embedding_reproduction.json`).

Accepted `outputs_complete` was renamed to canonical `outputs` without changing its contents. The previous partial `outputs` remains as `outputs_development_partial`. The run logs and Gate K retain the original execution path; `outputs/audit/output_promotion.json` records promotion. No old experiment is a submission input.

After successful full execution, Sonnet's report review motivated presentation-only changes: clearer absolute null effects, perturbation uncertainty, all-variant label tables, Type-B limitations and readable plot annotations. Only plotting.py changed among the captured experiment sources. The executed version is retained in `notes/plotting_executed.py`, with before/after hashes in `outputs/audit/presentation_amendment.json`. Rendering was rerun and every scientific JSON was checked unchanged. This is not represented as a second full experiment using the revised renderer. Additional evaluator positive controls pass; the final suite has 24 tests.
