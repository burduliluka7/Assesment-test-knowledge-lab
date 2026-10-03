# Scoped Claude methodology consultation

Requested through the user's official Claude Code subscription helper, role `planner`, model `claude-opus-5`, first-party `claude.ai` Pro login. Initial attempt exited 1 because sandbox networking refused the connection; approved network retry exited 0 with `is_error: false`, `usedModels: [claude-opus-5]`. No billing provider, token, or API-key changes were made.

Scope: V2 methodology and source only; no gold, target annotations, or evaluated results. The consultant reported reading only V2 sources/model configurations and graph/question schema metadata. Its recommendations are advisory; Codex owns implementation and verification.

Accepted interpretations: distinguish low-arity incidence from entailment; report unknown-type fallback and annual temporal precision; treat MiniLM as a relevance regressor and preserve mixed-stage ranking semantics; expose uncalibrated requirement scores and the fixed soft prior; document Python-audit limitations and preserve a retry ledger; use a separate post-freeze evaluator with exact V1 semantics.

Not adopted: extra retrieval arms, query-cap tuning, ID-only controls, changing fusion cutoffs, or changing requirement floors. These would expand the predeclared experiment. The consultant's claim that A0→A1 changes document style was checked against the implementation and rejected: A1 filters exactly A0's already-scored V1 rows. A3→A4 changes only reranking; both relevance arms use the original question. No inferential significance tests will be used for this small, previously inspected development benchmark.

No method names or expected targets were supplied to the consultant. Benchmark-driven repairs are prohibited. Known parser limitations will be measured and reported after freeze without changing prediction logic.
