# Frozen design specification for Claude's final report

All results will be DEVELOPMENT diagnostics. Existing benchmark exposure motivated the user's redesign; new prediction/gold separation prevents execution-time leakage but does not make this held-out research. No inferential superiority claim is justified.

The active implementation stays in clean_tshc_v2. Previous V2 source/config/report/results are archived under archive/flat_20261001. V1 had two pre-existing tracked report changes and four untracked report files; user explicitly approved preserving them. Compare final V1 hashes and git diff to that starting state, not to HEAD.

## Identity and time

Canonical entities use NFKC, casefold, whitespace collapse and typographic-hyphen normalization; other punctuation is preserved. Only identical normalized surface names within compatible families merge: method/technique/component/cited_work share an approach family, while dataset, metric and article remain separate. Canonical IDs use the lowest member node ID, not a gold label. Actual graph aliases are retrieval strings, never inferred alias merges. Undated aliases are suppressed on nodes whose last observation exceeds an earlier cutoff. First_seen_year defines publication visibility; last_seen_year is not an expiry date.

All current questions use the frozen 2026 snapshot. The parser can derive earlier year cutoffs; edges must pass both edge-year and asserting-article-year checks. Month precision cannot be recovered from year-only evidence. The original question is retained verbatim, including every modifier. Extractive structure is supplementary, with preserved pre-head scientific modifiers and coordinated answer families. Parsed-only is an explicit diagnostic control.

## Indexed evidence and safe graph paths

Every evidence node is encoded separately: claim, task, problem, technique, component, dataset, metric, article. Each entity links to evidence with source IDs, relation paths, provenance and candidate-specific flags. Method-object roles are inferred from unique member type signatures, never member order. Supported patterns: one method plus typed tasks/problems/techniques/components/datasets/metrics; one article plus one claim and optionally one method; one article presenting one method; one article citing cited_work targets. Ambiguous extends or mismatched relations are omitted. This is schema inference, not annotated linguistic roles or entailment.

Safe exact bounded mentions attach claims to entity names/graph aliases. Short generic words are excluded; identifiers require digits, CamelCase, all caps, or a multiword name of at least10 characters. Article titles supply provenance context, not all same-article claims. The evidence index is uncapped for recall; per-family caps apply only at reranking selection.

Graph generation follows typed evidence→method in one hop or evidence→article→presented/cited entity in two hops. Only articles may be intermediate; no author traversal, unrestricted spreading, recursion, or hierarchy beam. Top200 evidence seeds, degree cap80, fanout64, max2 hops, path decay0.8. Citation-derived evidence is structural context and never eligible for requirement-verification premises.

## Retrieval and ablations

BGE uses exactly the existing revision and no query instruction prefix. The dense channel retains the old bounded V1 document representation, aggregates each canonical entity by its maximum member-node score, and scores original and structured questions independently. A1 uses maximum across those two views; TypedRaw isolates filtering alone. A0 uses raw-question V1-like dense over all non-author graph nodes, deduplicated when an answer entity exists. All baseline generators cap at500 candidates; conditional rank statistics must report misses, unlike prior full-corpus rank means.

BM25 k1=1.5/b=.75 independently scores names, graph aliases and all unique indexed entity evidence. Positive-overlap rows only enter its top500, preventing arbitrary zero-score RRF votes. Safe name matching is a separate channel. Evidence dense uses max cosine across original/structured views, selects500 evidence seeds, and maps any seed to supported answer entities, with max pooling per entity. Graph is separate. Each channel contributes up to500 entity IDs; RRF uses k60 and only rank contributions, with deterministic ID ties. Each candidate exports all contributing channel ranks and selected evidence/path reasons.

A2 round-robins original-dense, structured-dense, BM25 and name lists. A3 adds evidence; A4 adds graph. A5 uses the identical A4 union and changes only ordering to RRF. A6 reranks first500 with genuine MS MARCO relevance MiniLM, appending the unchanged RRF tail. A7 applies requirement verification to A6's top20 and leaves all other entities unchanged. Additional controls: TypedRaw, ParsedOnly, WithoutEvidenceRRF, WithoutGraphRRF and A6_100/200/500. The latter reuse identical pair scores from the500 run; their recorded prefix timings are amortized estimates, not separately timed production experiments. No diffusion arm is run.

## Query-dependent cross-encoder input

Within each evidence family, sort by max original/structured cosine and deterministic tie breaks, deduplicate text, apply caps, and preserve candidate identity first. Order: claims, explicit mentions, tasks, problems, techniques, components, datasets, metrics, article titles, structural path evidence. Token packing retains identity, then the strict block prefix that fits512 tokens. The full original question is the relevance query; only an overlong query would be truncated to preserve identity. Included/omitted blocks and token counts are exported. Relevance logits are not probabilities. Frozen pool membership cannot change during reranking.

## Requirement verification

For the top20, each core/requirement/exclusion retrieves the maximum-cosine item from the FULL candidate-specific index, not just the CE-truncated card. Article and structural-path-only evidence are ineligible. Pinned local DeBERTa NLI scores the selected premise and literal candidate-specific hypothesis. Record entailment/neutral/contradiction probabilities as uncalibrated proxies. Threshold .7: high entailment supports a positive requirement, high contradiction contradicts it; polarity reverses for an exclusion. Neutral/no clear result is NOT_SUPPORTED_BY_AVAILABLE_EVIDENCE. Missing evidence is the same unsupported status, never contradiction. If a model is unavailable, use NOT_EVALUABLE for existing evidence and skip NLI inference.

A7 uses stable tiers within the top20: all requested components proxy-supported first, unresolved/no-contradiction next, any proxy contradiction last; original CE order breaks ties. This is a heuristic reranking experiment, not scientific entailment or calibrated truth. Unsupported evidence alone is not treated as false. The model can misread conditions or negation; report harm as well as gains.

## Evaluation, efficiency and reproducibility

Predictions complete before SHA256 freeze. A separate entry point verifies every frozen digest, then imports evaluation and reads gold/target annotations. Prediction audit hooks deny gold filenames, historical outputs/reports/notes, protected project source reads and evaluator imports; permitted third-party packages are read-only. Hooks are not hostile-native-code containment. All local models have fixed revision/file hashes; model downloads require explicit --download. Current run uses the pre-existing interpreter's third-party installation with -B and no runtime project imports from rest_of_work.

V1 exact/alias/component mapping code is preserved evaluation-only. Direct identity metrics use EXACT/ALIAS and keep all target occurrences in the raw denominator; PARTIAL/ABSENT/COMPOSITE never count as direct identity. Composite completion is supplementary and separately labeled. Candidate recall uses mapped EXACT/ALIAS occurrences with raw-denominator figures alongside, depths50/100/200/500/1000, union recall and candidate size. Conditional reranker analysis uses only targets actually admitted to its prefix. Rank mean/median condition on observed targets and report misses; a fixed missing-rank penalty is also saved. Positive and empty-rank controls are isolated under evaluation_only, never real systems.

Efficiency counts dense dot products, lexical work, evidence comparisons, CE pairs and requirement/NLI pairs. The BM25 counter in the predictor is a token-count estimate; the report can derive an exact term-loop count from the frozen intent and tokenizer. Timings are descriptive CPU wall times, not production latency. The replay reuses cached model outputs and verifies full prediction bytes; synthetic model tests independently check fixed-input inference determinism. Distinguish those two checks.

Every missed target gets a documented primary failure at top10 and optional secondary observations. NO_RELEVANT_EVIDENCE operationally means no indexed candidate-specific evidence, not adjudicated scientific absence; NO_NAME_MATCH is absence of a lexical opportunity. Historical baseline comparisons are descriptive and differ in candidate truncation/grouping; matched new-run ablations are the primary contrasts.
