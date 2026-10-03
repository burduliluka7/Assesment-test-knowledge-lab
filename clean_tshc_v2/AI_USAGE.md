# AI usage: current V2 requirement-aware retrieval

This file discloses AI assistance for the current V2 implementation. It is a companion to the unchanged V1 assessment submission, not a claim of independent human scientific adjudication.

| Tool | Work | Major prompts and decisions | Verification |
|---|---|---|---|
| Codex coordinator | Code, integration, local execution, tests, freeze/evaluation and reports | User's supplied requirement-aware ranking specification; all changes confined to V2. No training or benchmark-driven tuning. | Synthetic tests, actual pinned model tests, process isolation attempts, input/model/source/prediction hashes, cached replay, protected-tree diff/content audit. |
| Claude Opus5 via Ultralight `planner` | Architecture consultation | [Prompt](notes/requirement_design/planner_prompt.txt), [response](notes/requirement_design/planner_review.md). Accepted separate ranking components, explicit tie/missing semantics, access auditing. Rejected per-sentence NLI because user specifies joint multi-evidence premises. | Codex inspected code/spec and implemented tests; advice was not applied blindly. |
| Claude Sonnet5 via Ultralight `reviewer` | Source-only correctness review | [Prompt](notes/requirement_design/reviewer_prompt.txt), [response](notes/requirement_design/reviewer_review.md). Identified supported-exclusion/tier1 ambiguity. | Corrected tier1 to require positive support before evaluation and added regression test. Reviewer overstated exclusion non-entailment: actual SUPPORT requires high contradiction of excluded property, not neutrality; that explanation was rejected. |
| Claude Opus5 via Ultralight `planner` | Original assessment compliance | [Prompt](notes/requirement_design/assessment_review_prompt.txt), [response](notes/requirement_design/assessment_review.md). | Codex verified brief and deliverable path existence, added explicit V1-versus-V2 scope matrix. No claim that V2 replaces hierarchy or demonstrates its benefit. |
| Claude Opus5 report author via Ultralight | Detailed LaTeX and concise companion from measured handoff | [Prompt](notes/requirement_design/report_prompt.txt), [returned drafts](notes/requirement_design/report_result.json); helper exit0, no error or permission denials. | Codex checked all 63 target rows and 29 metric rows, corrected factual/layout errors, condensed the companion and compiled PDFs. [Editorial changes](notes/requirement_design/report_corrections.md). Authorship is distinct from experimental execution. |

All Claude calls use the official Claude Code CLI with the user's subscription through the local `claude_delegate.py` helper. No API keys, token extraction, provider switching, recursive delegation or GitHub operations are used. Consultants were given scoped source/specification access; no gold or historical evaluation outputs were supplied for predictive design. Report authors receive evaluation summaries only after freeze.

## Accepted design constants

Primary A7 uses existing RRF pool500, top3 evidence per core/requirement/exclusion, fixed model revisions and NLI threshold0.7, four support/contradiction tiers, then geometric/weakest reciprocal positive-component ranks and global/RRF/identity tie-breaks. Missing evidence is unresolved; missing relevance has declared midpoint imputation. No fitted scalar weights. Top1/NoNLI/100/200 are controls, never promoted posthoc.

The positive-only tier1 clarification was made on semantic grounds before any current-run scoring. Previously computed pure model outputs were carried forward because their model inputs, evidence selection and packing were unchanged; no evaluated prediction or target rank was reused. The carry-forward file hashes and reason are documented in `notes/requirement_design/cache_carryforward.json`.

## Limits

The benchmark influenced earlier human design, so the experiment remains development-only. A cached replay checks byte-identical computation given model outputs; it does not repeat uncached neural inference. Python audit hooks enforce cooperative code behavior, not a hostile-native-code sandbox. NLI labels and agreement with target annotations do not establish scientific correctness. The user remains responsible for reviewing conclusions and assessment submission.

A second pre-evaluation syntactic correction preserves numeric commas during requirement atomization; a regression test covers it. Cache reuse remains keyed by the full intent, so changed component bundles recompute. No current-run gold had been opened at either correction.
