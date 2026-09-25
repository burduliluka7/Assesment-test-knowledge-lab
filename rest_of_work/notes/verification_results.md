# Executed verification

This records assistant-executed checks, not candidate/manual review.

- Real supplied data: **5,798 nodes and 1,429 hyperedges**; 12 node types and 11 relations. Validation succeeded with recorded temporal warnings: 170 edges before a member's first appearance, 134 before the asserting paper, and three origins after first appearance.
- Conservative cumulative snapshots: **2020: 1,505 nodes / 374 edges; 2022: 2,164 / 526; 2024: 4,164 / 983; 2026: 5,798 / 1,429**.
- Final unit suite: **54 passed** in a fresh isolated Python 3.11 environment. Includes exact VI/hyperedge delta tests, cache parity, native collapse, empty-cluster rejection, temporal change, visibility, mapping distinctions, independent-null logic and scoring budgets.
- Clean dependency installation: all 43 pinned required distributions installed in `.repro-venv`; editable project installation succeeded. MiniLM and BGE loaded at their pinned commits, and the NLI model loaded with explicit contradiction/entailment/neutral labels.
- Full real-data pipeline: **completed**, approximately **1,075 seconds** in the recorded run. This is descriptive wall time on a shared machine, not a controlled runtime benchmark.
- Fast integration pipeline: **completed** on two real snapshots with five perturbation seeds.
- Artifact reload audit: **25 full-run hierarchies** (20 main plus five conservative benchmark branches), and **three fast-run hierarchies**. Coverage, laminarity, budgets, persistent IDs, multiplicity, parent/event references, visible label inputs and retrieval budgets passed.
- Evaluation: all five variants have independent BGE coherence, 50 size-preserving and within-type null permutations, five final-snapshot 10% edge-removal seeds with t intervals, all real cross-snapshot comparisons, and 96 sampled NLI labels per variant covering levels 0 and 1 across four cutoffs.
- Four intrinsic sensitivity configurations completed; no retrieval-based parameter tuning was performed.
- Retrieval was recomputed after correcting a mapping rule that conflated hyphenated/plus-suffixed model variants. The corrected 2025 benchmark maps **21 of 50 unique expected names**, leaving **10 questions** with mappable method targets. All question-level records and unavailable statuses remain visible.
- Source hashes distinguish the initial experiment source, the subsequent retrieval correction and delivered source in `environment.json`, `metrics.json` and `verification.json`. Later changes strengthened validation, made cache dtype consistent and revised report interpretation; no objective weights or hierarchy design were tuned against retrieval.

The final partition-rebuild check **passed for all 20 main hierarchies across five variants**: member partitions, parent IDs and persistent IDs exactly matched the delivered artifacts. Freshly inferred and cached real MiniLM vectors were also identical. Results are in `artifacts/full/reproduction_check.json`; the execution log is `notes/reproduction_check.log`.

`notes/full_run.log` is an earlier development run interrupted to apply review fixes; `notes/final_full_run.log` records the completed experiment. Final retrieval metrics additionally include the explicitly documented mapping correction. The report was then revised to discuss the actual mixed/negative findings.

Missing empirical checks: independent human label ratings, source-paper textual verification and adjudicated supporting-claim alignments. NLI non-entailment flags are computed but are not a substitute for those checks. No manual ratings were invented.
