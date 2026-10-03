# V2: entity-centric, requirement-aware scientific retrieval

The active V2 pipeline retrieves canonical answer entities with independent dense, BM25, safe-name, evidence and bounded-graph channels. RRF fuses their ranks. The fixed top500 pool feeds a global relevance baseline (A6) and the primary requirement-aware multi-evidence ranker (A7). Hierarchy traversal is outside the exact-answer path.

A7 preserves the full question, extracts literal atomic component spans, and treats the core task as essential. For each entity/component it selects up to three attached evidence items, packs identity first, scores requirement relevance with the pinned MS MARCO MiniLM cross-encoder, and scores support/contradiction with the pinned DeBERTa NLI proxy. Article titles and citation paths cannot serve as support premises.

Ordering is deterministic: contradiction/support tier, geometric mean reciprocal positive-component rank, weakest reciprocal rank, global relevance rank, RRF rank, entity ID. Exclusions affect support polarity and tiers, not positive relevance aggregation. Missing evidence is unresolved, never contradiction; missing relevance receives a neutral midpoint rank among scoreable entities. This is a declared missing-data heuristic, not an estimated scientific confidence.

No model is trained and no weights, thresholds, pool depths or evidence counts are fitted using benchmark targets. Primary pool500, evidencek3, RRF60 and NLI threshold0.7 are fixed before evaluation. A7_NoNLI and A7_SingleEvidence are controls, not candidates for post-hoc primary selection.

## Measured outcome

The frozen development run recovers 46/49 directly mapped target occurrences in the full candidate union, versus 19/49 for typed dense retrieval. The RRF top500 contains 44/49 (89.8%). Final recall uses all 63 target occurrences across 14 scored questions; requirement diagnostics include all 18 questions.

| Fixed arm | Recall@10 | Recall@100 | MRR |
|---|---:|---:|---:|
| A5: RRF | 9.52% | 58.73% | 0.1554 |
| A6: global relevance | 14.29% | 52.38% | 0.1480 |
| A7: requirement-aware, top3 evidence | 14.29% | 46.03% | 0.1102 |

High-recall generation is supported by this diagnostic comparison. Requirement-aware ranking does not improve aggregate performance over A6: among the 44 mapped targets in the pool, 18 move up, 25 move down, and one is unchanged. Four enter the top10 and four leave it. Sparse evidence is substantial: 13,558 of 20,349 candidate/component bundles have no selected support evidence. These results do not establish scientific correctness or generalization. The declared primary system and configuration were retained after evaluation.

## Execution and artifacts

### A6 technical explanation

The detailed [A6 technical explanation (PDF)](report/v2_a6_technical_explanation.pdf), with its [LaTeX source](report/v2_a6_technical_explanation.tex), explains entity representation, candidate generation and rank fusion, the query-dependent evidence card, and cross-encoder reranking. It includes a complete algorithm, a worked example, temporal integrity, leakage prevention, evaluation, and limitations. Its A6_100 configuration reranks the first 100 fused candidates and reports development results of 12/63 targets at top 10, 18/63 at top 25, 37/63 at top 100, and MRR 0.167. The development-informed depth choice and this configuration are distinct from the fixed top500 A6/A7 comparison above.

Only `clean_tshc_v2` is writable. V1 and `rest_of_work` are read-only. Old `outputs`, `archive` and reports are denied to prediction. Active artifacts are organized by phase inside this same V2 project:

- `artifacts/prediction/`: queries, entity index, channel provenance, global and requirement evidence, every candidate's ranking trace, predictions and freeze manifest.
- `artifacts/evaluation/`: gold-derived metrics, conditional movement, failure traces, requirement diagnostics and isolated positive controls.
- `artifacts/reproduction/`: full cached repeat of the predictor for hash comparison.
- `artifacts/run_logs/`: process logs and exit statuses.
- `artifacts/verification/`: protected-tree checks, tests, report validation and changed/added/deleted file inventory.
- `model_manifest.json`: prediction-independent pinned model availability and per-file hashes.

The predictor uses a strict input allowlist and logs all project-relative reads. Only source, graph/questions, fixed config, models and content-addressed prediction-independent caches are allowed. Historical outputs, gold mappings, annotations and evaluator imports are denied. These Python audit hooks are cooperative safeguards, not a sandbox against malicious native code.

Predictions and traces freeze before the separate evaluator imports scoring code or loads gold. The evaluator verifies all hashes, installs write/delete/rename/permission-change protection on frozen files, scores, and verifies the hashes again. Questions are legitimate inputs; expected answers are not.

## Reproduce

Use Python3.11 and the pinned requirements in an environment inside V2. The measured run uses an existing read-only interpreter's installed third-party packages with bytecode disabled; no project code is imported from that interpreter's parent project.

```powershell
python -m venv clean_tshc_v2/.venv
& clean_tshc_v2/.venv/Scripts/python.exe -m pip install -r clean_tshc_v2/requirements.txt
& clean_tshc_v2/.venv/Scripts/python.exe -B clean_tshc_v2/scripts/prepare_models.py
& clean_tshc_v2/.venv/Scripts/python.exe -B clean_tshc_v2/scripts/predict.py
& clean_tshc_v2/.venv/Scripts/python.exe -B clean_tshc_v2/scripts/evaluate.py
& clean_tshc_v2/.venv/Scripts/python.exe -B clean_tshc_v2/scripts/predict.py --output artifacts/reproduction
& clean_tshc_v2/.venv/Scripts/python.exe -B -m pytest clean_tshc_v2/tests -q -p no:cacheprovider
& clean_tshc_v2/.venv/Scripts/python.exe -B clean_tshc_v2/scripts/build_report_materials.py
python -B clean_tshc_v2/scripts/compile_report.py
python -B clean_tshc_v2/scripts/finalize.py
```

Model preparation is local-only unless explicitly run with `--download`. Revisions are pinned in `config.json`; preparation never silently selects a replacement. Prediction refuses to overwrite an existing freeze: use a fresh `--output` subdirectory for another run. Evaluating an alternate output uses `evaluate.py --output <same path>`; evaluation summaries are regenerated in `artifacts/evaluation`.

The numerical summary is [report/v2_retrieval_report.md](report/v2_retrieval_report.md); detailed methods/results are in the Claude-authored [LaTeX](report/v2_entity_retrieval_report.tex) and [PDF](report/v2_entity_retrieval_report.pdf). Local Ultralight consultation/report prompts and results are retained in `notes/requirement_design` and are forbidden prediction inputs.

The concise assessment companion is available as [LaTeX](report/v2_assessment_companion.tex) and [PDF](report/v2_assessment_companion.pdf). The [final verification](artifacts/verification/final_verification.json) and [exact changed/added/deleted file inventory](artifacts/verification/files_changed.json) document the completed local run.

## Interpretation and safety

This is a development benchmark whose earlier use influenced human design decisions. Execution-time isolation does not make it an unseen test. Sparse graph evidence, year-only time stamps, schema-inferred ownership, imperfect parsing, pretrained relevance domain mismatch and uncalibrated NLI remain limitations. Neither agreement with annotations nor an NLI SUPPORTED label establishes scientific truth.

V1 began with user-authored uncommitted report work. The user explicitly authorized preserving it. The final check therefore requires zero new changes relative to starting file hashes and binary git diffs, not an empty V1 diff against HEAD. The same file-hash/diff check covers all of `rest_of_work`. No GitHub operations are performed.

## Original assessment scope

V2 is a **flat-retrieval companion**, not a replacement for the temporal hypergraph hierarchy submission. The assessment's laminar hierarchy, display budgets, hyperedge collapse, temporal events, independent coherence/nulls, label faithfulness and coarse-to-fine experiment remain in the unchanged [V1 submission](../clean_tshc/README.md). V2 recall gains cannot demonstrate hierarchy benefit, and historical V1/V2 scores are not presented as a matched head-to-head.

See the [P1-P6/T1-T7 and deliverable mapping](notes/ASSESSMENT_ALIGNMENT.md), [V2 AI disclosure](AI_USAGE.md), [original hierarchies](../clean_tshc/outputs/hierarchies), [temporal events](../clean_tshc/outputs/temporal_events.json), and [assessment metrics](../clean_tshc/outputs/metrics.json). These links are for reviewers; they are forbidden prediction inputs.
