# Ultralight review decisions

Official helper returned exit 0, requested/used `claude-sonnet-5`, `authMethod=claude.ai`, subscription Pro, no permission denials. Exact response is retained in `reviewer_result.json`. It was a read-only code consultation, not an executed independent experiment.

- Accepted mathematical check of VI and fragmentation deltas, unchanged-pair heap validity, stage remapping, disjoint label generation and gold-free construction. Codex also verified delta/heap claims with synthetic tests.
- Accepted first-hit limitation: deterministic node-ID scan order and conditional-on-hit medians are not valid efficiency rankings. Report total comparisons against recall as primary, retain the specifically requested first-hit diagnostic with its miss count and explicit definition.
- Accepted NLI caveat: generation-member names can be absent from disjoint held-out text; ordered truncation can omit types. Disclose this beside overclaim values. Kept the fixed gloss and evaluation rule rather than changing them after seeing outcomes.
- Accepted separation of recovery units: direct EXACT/ALIAS identity and COMPOSITE evidence bundles get separate counts; aligned-claim direct-only recall is added beside document-evidence recall. No change to rankings or existing recovery memberships.
- Accepted derived independence flags and exact encoder revisions. The models have different names/families, not merely different checkpoints of the same encoder. Prediction hash checks and source tests substantiate the execution order; flags alone do not prove isolation.
- Arity-one crash: supplied-data loader already rejects original arity <=1. Added an explicit Merger precondition; coarsened internal edges retain their original arity >1. No need to redefine the requested objective for unsupported inputs.
- Accepted hoisting the sensitivity membership dictionary; it changes work only, not results.

The planner retry failed with a 600-second timeout and supplied no advice. Its failure is disclosed separately. The first development execution was already running while export/reporting checks were added; the fresh Gate-K execution uses the final source. No outcome-tuned parameter, extra model or retrieval mechanism was introduced.

Actual full-data execution found a null DOI in article provenance that the source resolver attempted to normalize as text. The evaluation-only correction treats a missing DOI as an empty comparison string; a title match remains available. A regression test and processing of all 623 benchmark source-reference uses verified the fix. The latter yields 336 matched, 234 unmapped and 53 ambiguous uses (repeated sources across claims included); unresolved sources remain explicit. No gold target, query, hierarchy, representation or ranking rule changed.

## Final report review

The second read-only Sonnet 5 consultation also succeeded through the official helper (exit 0, claude.ai Pro, no permission denials). Its exact prompt and result are `final_report_review_prompt.txt` and `final_report_review_result.json`. The reviewer checked headline numbers against tables and found no mismatch. It explicitly did not inspect all tests or the existing output-inspection notes.

Accepted reporting improvements: show the small absolute null effects alongside the p-value resolution; disclose the wider temporal perturbation intervals; compare static and temporal label proxies; distinguish the last-transition ARI loss from VI/lineage gains; state the single aligned Type-B claim's recovery and the post-hoc nature of its annotation; summarize parameter sensitivity. These affect presentation only. The source-frozen full run must finish before applying rendering changes, and its executed renderer will be retained with an amendment record.

Accepted the need for stronger evaluator positive controls. Existing tests already recovered a synthetic composite. New EXACT/ALIAS tests recover a directly ranked identity while excluding context-only mentions. An additional evaluation-only check places each of 49 actual EXACT/ALIAS target instances at rank one and verifies recovery for all 49 without modifying saved predictions. The original best flat direct-target rank is 31. These controls demonstrate evaluator sensitivity, not model success. All 24 tests pass.

Do not interpret the reviewer's phrase "inflated standardized effects" as a numerical error: the z-like effects correctly divide by the very small conditional-null SD. They are not population effect sizes. Report the absolute magnitudes and this interpretation instead. No additional scorer, parameter selection or model is accepted.
