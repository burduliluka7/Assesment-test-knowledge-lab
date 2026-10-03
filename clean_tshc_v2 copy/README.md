# V2: entity-centric high-recall retrieval

This is the active V2 experiment. It replaces the previous V2 retrieval pipeline; the earlier source, configuration, report and results are preserved under `archive/flat_20261001/`. There is no V3. V1, its hierarchy, reports, configurations and tests remain unchanged from the user-approved task-start state. No GitHub operations were performed.

## Measured development outcome

Across 14 Type-A questions, there are 63 expected target occurrences, of which49 have EXACT/ALIAS identity mappings. All18 questions have predictions.

- A5 RRF candidate recovery: **37/49 at100**, **44/49 at500**, **46/49 in the complete union**. Previous V2 recovered3/49 at100.
- Primary A7 final direct Recall@10: **10/63 =15.87%**, versus2/63 in previous V2.
- The predeclared100-candidate reranking control recovered **12/63 at10**, better than the500-candidate primary. It was not substituted as the primary after evaluation.
- The500 reranker moved18 seen targets up and26 down; their median rank worsened39→50.5. A5 MRR0.1554 fell to A6 0.1480 and recovered slightly to A7 0.1516.

The candidate-generation hypothesis is supported on this development benchmark. Consistent precision improvement from larger reranking pools is not. These are not held-out generalization or statistical-superiority results.

## Architecture

1. Preserve the complete original question and supplement it with extractive task, requirements, exclusions, temporal conditions and scientific modifiers. Coordinated requested types produce a union of answer families.
2. Canonicalize identical normalized names within compatible schema families. Preserve member IDs and graph-supplied aliases; do not infer fuzzy aliases or use gold labels.
3. Index separate snapshot-visible evidence items and their entity owners. Derive directional roles from unique relation/type signatures, never from member order. Citation paths remain structural clues.
4. Generate independent original-dense, structured-dense, BM25, safe-name, evidence-dense and bounded-graph channels. Take up to500 candidates per channel; deduplicate by canonical entity ID.
5. Compare round-robin union with RRF k60. Keep every channel rank, contribution, evidence node and graph path.
6. Select evidence for each query and entity. Preserve identity first, cap evidence families, and pack at most512 model tokens.
7. Rerank500 candidates with the pinned MS MARCO relevance model. Compare100/200 controls using the same pair scores. Preserve the unreranked tail.
8. Verify the top20 using candidate-specific evidence and the pinned NLI model as a proxy. Missing evidence is unsupported, never contradiction. A7 stably reorders only these20 by predeclared support/uncertainty/contradiction tiers.

Hierarchy traversal and diffusion are not in the active QA path. See `notes/entity_redesign/method_specification.md` for exact choices and limitations, and `config.json` for every frozen parameter.

## Reproduction

Windows PowerShell, Python3.11. Run from the repository root. Code and data needed for prediction live entirely in V2. The recorded run used the existing read-only third-party installation at `rest_of_work/.repro-venv/Scripts/python.exe`, with `-B`; no project modules were imported from that tree.

For an independent environment:

```powershell
python -m venv clean_tshc_v2/.venv
& clean_tshc_v2/.venv/Scripts/python.exe -m pip install -r clean_tshc_v2/requirements.txt
& clean_tshc_v2/.venv/Scripts/python.exe -B clean_tshc_v2/scripts/prepare_models.py
```

`prepare_models.py` copies available pinned local snapshots. If any are missing, explicit download permission is provided through `--download`; model revisions are never silently changed. BGE and relevance models retain their earlier V2 pins. The NLI verifier uses `fa2804872c3b4bd748f38c0185cc85775361e735`. All model files are hashed. Inference runs offline.

The delivered primary predictions are frozen and cannot be overwritten by the runner. Use a fresh output directory to reproduce:

```powershell
& clean_tshc_v2/.venv/Scripts/python.exe -B clean_tshc_v2/scripts/predict.py --output outputs/replay
& clean_tshc_v2/.venv/Scripts/python.exe -B clean_tshc_v2/scripts/evaluate.py --output outputs/replay
& clean_tshc_v2/.venv/Scripts/python.exe -B -m pytest clean_tshc_v2/tests -q -o cache_dir=clean_tshc_v2/.cache/pytest
& clean_tshc_v2/.venv/Scripts/python.exe -B clean_tshc_v2/scripts/verify_protected.py
```

Choose another fresh output directory if `outputs/replay/` already contains predictions. Prediction generation rejects existing frozen files. The delivered `outputs/reproduction/` is an executed full cache replay with an identical prediction SHA256; it is not a second independent uncached inference run. Separate synthetic tests repeat real model inference with fixed inputs.

Evaluation verifies the entire prediction freeze before importing the evaluator and reading `clean_tshc/data/ground_truth.json` and `clean_tshc/notes/target_annotations.json` read-only. The gold files are not copied into prediction inputs. Prediction audit hooks deny gold reads, archived results/notes/reports, protected project sources and evaluator imports. This is a cooperative-code audit, not a security sandbox against hostile native code.

## Systems and outputs

| Arm | Change | Candidate ordering |
|---|---|---|
| A0 | Original V1-like dense baseline, canonical dedup, top500 | Raw-question dense |
| A1 | Type-filtered original+structured dense | Max across views |
| A2 | Add independent BM25 and safe names | Round-robin union |
| A3 | Add evidence-level dense | Round-robin union |
| A4 | Add controlled graph paths | Round-robin union |
| A5 | Same A4 candidate union | RRF60 |
| A6 | A5 top500 joint relevance | Reranked prefix + fixed tail |
| A7 | A6 top20 requirement verification | Stable proxy-status tiers |

`TypedRaw` isolates raw-query type filtering. `ParsedOnly`, `WithoutEvidenceRRF`, `WithoutGraphRRF`, and `A6_100/200/500` are predeclared controls. A0→A1 changes query views as well as filtering; it must not be presented as a pure filtering contrast.

- `outputs/entity_index/`: canonical entities, names/aliases, separate evidence, ownership, relation paths and rejected role patterns.
- `outputs/query_understanding/`: original and structured representations with source spans and warnings.
- `outputs/retrieval/predictions.json`: all frozen rankings.
- `outputs/retrieval/channels/`: individual channel rankings, actual candidate-entry sources and RRF contributions.
- `outputs/retrieval/selected_evidence/`, `cross_encoder/`, `requirement_verification/`: exact selected evidence, included/omitted blocks, scores, probabilities and statuses.
- `outputs/metrics.json`: candidate recall50/100/200/500/1000, union recall and size, final recall1/5/10/25/50/100, MRR, conditional ranks, missing counts, conditional reranker performance and channel contributions.
- `outputs/per_question_evaluation.json`, `failure_traces.json`: all expected-target ranks and deterministic explanations after gold loading.
- `outputs/evaluation_only/positive_controls.json`: isolated gold-informed oracle and empty-ranking checks; never a real retrieval result.
- `outputs/efficiency_report.json`: logical work counts and descriptive CPU timing. Prefix100/200 timing estimates reuse500-pool pair measurements; they are not independent latency benchmarks.
- `outputs/audit/`: input/model/source/prediction hashes, ordered phase events, tests, replay verification and V1 safety audit.
- `report/v2_retrieval_report.md`: generated numerical summary.
- `report/v2_entity_retrieval_report.tex` and `.pdf`: detailed Claude-authored methods/results report, integrated and checked by Codex.

The tests retain legacy behavior checks against the archived earlier experiment and add active entity-pipeline checks. Historical evaluation artifacts are not read until the new prediction freeze exists.

## V1 safety and report reproduction

At task start, V1 already contained modified tracked report files and four untracked report files. The user explicitly approved preserving them. `scripts/verify_protected.py` checks every V1 file hash, the exact starting git diff, and V1's starting git-status entries. Success means **zero new V1 changes**, not discarding existing work to make the diff empty. The baseline is in `notes/entity_redesign/`; the final checks and V2 change inventory are under `outputs/audit/`.

Claude Opus5 was consulted through the user's official Ultralight helper and authors the final LaTeX from a minimal verified results handoff. Codex implements, integrates, tests and verifies the report. The prompt and original Claude JSON response are retained under `notes/entity_redesign/`. To rebuild numerical report materials and compile the supplied report:

```powershell
& clean_tshc_v2/.venv/Scripts/python.exe -B clean_tshc_v2/scripts/build_report_materials.py
python -B clean_tshc_v2/scripts/compile_report.py
python -B clean_tshc_v2/scripts/finalize.py
```

Neither the cross-encoder nor NLI establishes scientific truth. Type-family exclusions, sparse candidate-specific evidence, query parsing limitations, citation ambiguity, domain shift and neutral/contradiction errors remain explicit limitations. Future improvements need an independent development set and a newly frozen held-out evaluation, not patches to individual benchmark answers.
