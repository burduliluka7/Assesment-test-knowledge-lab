# Codex verification of the Claude report draft

Claude Opus 5 returned both LaTeX drafts through Ultralight successfully (exit 0, no error, no permission denials). The untouched returned drafts remain in report_result.json. Codex applied the following editorial changes only, after prediction freeze; no predictive source/config/model was changed:

- Corrected LaTeX moving-argument/table errors, wrapping and long labels; condensed the companion to four readable pages.
- Corrected the graph description: direct paths are permitted; only a two-hop intermediate must be an article.
- Distinguished the two scoring models from the dense embedding model, and changed the support threshold wording to at least 0.7.
- Explained cross-component reciprocal-rank aggregation without claiming raw model calibration.
- Removed unsupported claims that A7 has the worst mean among all fusion arms, that aggregate effects are a wash, and that all large demotions are contradiction-driven.
- Clarified that equal multi-item selection/packing counts do not prove every block survived, and softened causal claims about NLI and timing.
- Corrected the chart caption and separated the best recall and best MRR arms.
- Replaced the composite target paraphrase with its exact arrow notation; all 63 occurrence rows and 29 metric-table rows match the handoff.
- Added a source-checked paragraph on separate evidence ownership/role validation and distinguished unsupported relations from rejected malformed supported relations.
- Replaced the old verification include with the current 57-test result, both protected-directory checks and active artifact paths.

The detailed report and companion explicitly preserve the assessment boundary: V2 is a flat-retrieval companion, not the temporal hierarchy submission. The results remain development diagnostics, not held-out evidence or expert scientific adjudication.
