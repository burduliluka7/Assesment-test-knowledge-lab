# Assessment alignment: V2 retrieval companion

The original [assessment brief](../../clean_tshc/data/ASSESSMENT.md) asks for a temporal multi-resolution hypergraph abstraction, not simply a question-answering retriever. **The main assessment submission remains `clean_tshc`. V2 is an additional flat-retrieval study relevant to the extrinsic portion of T6. It cannot replace the assessed hierarchy or prove its usefulness.** This scope follows the user's requirements that V1 remain unchanged and hierarchy traversal stay outside V2's critical answer path.

Claude Opus5 reviewed the original brief and V2 source through the subscription Ultralight helper. Its [complete review](requirement_design/assessment_review.md) is preserved. This matrix maps scope; it does not certify previously reported V1 numerical results anew.

| Requirement | Existing V1 responsibility | V2 contribution and limit |
|---|---|---|
| P1 laminar refinement, P2 display budgets | Nested hierarchy and bounded levels | No partitions are constructed; entity grouping is not a multilevel hierarchy. |
| P3 semantic coherence | Structural/semantic clustering trade-off and independent coherence evaluation | RRF combines retrieval signals. Retrieval fusion is not a cluster-coherence test. |
| P4 hyperedge fidelity / T4 collapse | Native hypergraph objective and implemented multiplicity-preserving coarsening | Bounded graph paths use hyperedge provenance, but do not implement coarsening. |
| P5 / T3 temporal stability | Multiple snapshots, temporal coupling and tracked events | Publication visibility filtering only; current retrieval benchmark uses2026. No new cross-snapshot stability claim. |
| P6 / T5 faithful labels | Automatic supernode label/gloss and measured over-claim proxy | Entity requirement NLI is a different proxy; it does not measure label faithfulness. |
| T1 graph description | Snapshot growth/type/arity/data-quality descriptions | Uses supplied graph and records index coverage; no replacement temporal descriptive study. |
| T2 / Deliverable0 formal method | Hierarchy objective, guarantees, costs and alternatives | Explicit ranking criterion, fixed missing-data convention and complexity, only for the retrieval extension. |
| T6 intrinsic evaluation | Independent coherence signal/nulls, five-seed perturbations/CIs, temporal stability and label evaluation | Does not rerun or claim these intrinsic measures. |
| T6 extrinsic evaluation | Coarse-to-fine versus flat comparison | A0-A7 are direct/flat retrieval arms. Their gains cannot show hierarchy benefit. No matched V1/V2 head-to-head is claimed. |
| T7 verified vs assumed | Assessment write-up | Frozen prediction/evaluation firewall, proxy caveats, measured versus assumed statements. |
| D1 reproducible repository | Main runnable assessment | V2 exact commands, pinned dependencies/models/config, tests and hashes. |
| D2 hierarchy/event files | Per-snapshot hierarchy and temporal events | Read-only links below; no duplicated/rebuilt hierarchy. |
| D3 complete T6 metrics | Main assessment metrics | V2 metrics contain retrieval and requirement diagnostics only. |
| D4 concise3-5page report | Main assessment report | A concise V2 companion accompanies the detailed V2 methods/results report; neither supersedes V1. |
| D5 AI disclosure | Original AI_USAGE.md | V2-specific AI_USAGE.md records prompts, accepted/rejected advice and verification. |
| D6 optional visualization | Existing static hierarchy views | Static retrieval figure only; no interactive UI. |

## Unchanged assessment deliverables

Paths were checked to exist; these outputs are not predictor inputs and their numbers are not imported into the V2 evaluation.

- [Main README](../../clean_tshc/README.md), [assessment report](../../clean_tshc/report.md), [detailed write-up](../../clean_tshc/report/full_report.tex).
- Hierarchies: [2020](../../clean_tshc/outputs/hierarchies/hierarchy_2020.json), [2022](../../clean_tshc/outputs/hierarchies/hierarchy_2022.json), [2024](../../clean_tshc/outputs/hierarchies/hierarchy_2024.json), [2026](../../clean_tshc/outputs/hierarchies/hierarchy_2026.json).
- [Temporal events](../../clean_tshc/outputs/temporal_events.json), [assessment metrics](../../clean_tshc/outputs/metrics.json), [original AI disclosure](../../clean_tshc/AI_USAGE.md).

## What is not claimed

V2 alone is not a complete assessment submission. Higher flat retrieval recall is not evidence of improved abstraction or hierarchy acceleration. Requirement support is an uncalibrated pretrained NLI proxy, not expert validation. Publication-year visibility is not temporal stability. This development benchmark has design-time exposure even though the prediction process is gold-blind. No models, ranking weights, thresholds or pool sizes are trained or optimized against benchmark labels.

A future comparison of hierarchy routing against this stronger flat arm would need identical eligible entities, evidence/scorers, snapshot, target mapping and cost accounting, declared before scoring. It would be a separate comparison baseline, not a required route for exact QA. The current work does not manufacture a favorable hierarchy result.
