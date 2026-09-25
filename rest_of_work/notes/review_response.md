# Response to Ultralight's independent review

The reviewer JSON reports Claude Sonnet 5 (`claude-sonnet-5`), first-party subscription authentication, no permission denials and success. The reviewer confirmed the sparse VI and exact merge-delta logic within its inspected scope.

1. **Possible mutation of the native reference:** inspected `collapse.py`; it constructs fresh Unit/Edge/Hypergraph objects and does not mutate the input. Added an explicit regression test retaining the original graph and its fragmentation value.
2. **Cosine normalization assumption:** `Encoder.encode` uses `normalize_embeddings=True`; the synthetic encoder normalizes its rows too. Added an encoder normalization test and made flat/claim ranking explicitly normalize vectors, eliminating the retrieval dependency on upstream row norms. Hierarchical unit means are normalized by definition.
3. **Unmapped-target recall:** mapped recall and F1/precision/hit now carry unavailable values/status when no target maps. The all-expected statistic is retained, explicitly named `recall_all_expected_lower_bound` and documented: it counts only lexical recoveries over all expected names. It is not treated as measured zero recall. Summary denominators and unmappable question IDs are exported.
4. **Benchmark-cutoff events and caching:** the separate 2025 build now exports events, exact label inputs and shared corpus input and uses a run-key cache. It remains a branch from the previous main snapshot, not an extra predecessor inserted into the primary timeline.

The reviewer also noted already disclosed limitations: candidate restriction, zero results when whole child batches exceed a small budget, and the deliberate use of an independent evaluator for retrieval. These remain documented design choices. No review is represented as an expert audit of corpus scientific claims.
