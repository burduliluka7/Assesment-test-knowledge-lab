# Final integration verification

- Corrected full run completed for both 2025 and 2026, with the original fixed budget grid and five hierarchy variants at500. No hierarchy rebuild or NLI inference.
- FINAL_FINE exact historical scores/ranks reproduced before hierarchy traversal:13/47 and13/50. Full rank hashes and independent historical regression pass.
- Full suite:245 passed in58.55 seconds, no failures/errors/skips. JUnit:artifacts/evaluation/final/tests.xml. Tests were run after the final source/test edits.
- Report writer verifies executed source hashes,655 frozen files,25 frozen hierarchy inputs via the tests, submission hierarchy memberships/parents, event fields, required sections, metric denominators, archived claim counts, and fresh passing JUnit.
- Unrestricted hierarchy and exhaustive flat rankings are exactly equal. Temporal hierarchy500 retains0/13 fine hits in both years; this is a negative result, not an experiment-completion failure.
- All18 questions have explicit required-claim evaluability at both cutoffs; Type B retrieval remains historical. Strict recall is null with zero alignable required claims.
- Final audit:24 PASS,5 PARTIAL,1 NOT_EVALUABLE across base requirements, detailed T6 checks and deliverables. P3/P6/T5/T6 remain partial; strict claim alignment is unevaluable.
- Both new figures were visually inspected. The tradeoff figure omits root-insufficient hierarchy budgets and shows the quality loss and unrestricted ceiling. The four-snapshot figure shows L0–L2 structure.
- Packaging verifies every uncompressed archive member against its SHA-256 manifest and current workspace. The package is local only. The packaging command's final success output is the archive-verification result.

All reported semantic/faithfulness measures remain automated proxies. No expert adjudication, unseen-question validation or production latency claim is made. The final pass stops here as requested.
