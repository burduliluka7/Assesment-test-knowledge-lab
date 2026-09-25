# Temporal knowledge hypergraph abstraction

This assessment implements temporal spectral–semantic native hypergraph coarsening, with five variants and explicit evidence limitations. Real data is in `data/data`; synthetic data is confined to tests. Start with [the report](report/report.md) and [requirements audit](artifacts/submission_checklist.md).

## Reproduce the final evaluation

From the repository root in PowerShell, using the existing Python 3.11 environment:

```powershell
.\.venv\Scripts\python.exe -m tkh_abstraction.cli evaluate-v2 --config configs/evaluation_v2.yaml
.\.venv\Scripts\python.exe -m pytest -q --junitxml=artifacts/evaluation/tests.xml
.\.venv\Scripts\python.exe scripts/audit_v2.py
.\.venv\Scripts\python.exe scripts/package_submission.py
```

`evaluate-v2` verifies source/config/data/artifact hashes against `artifacts/baseline_v1/manifest.json` before reusing 25 hierarchies. It evaluates 2025 and 2026, all five variants, typed fixed-beam, prototype best-first and flat retrieval, source-aware claim alignment, and held-out labels/NLI. It regenerates metrics, figures and report. Changed construction inputs cause an error; do not edit the manifest to force reuse. Unchanged coherence and perturbation experiments are retained, not rerun.

Original history remains under `artifacts/baseline_v1`, including the original archive. Hierarchy memberships and temporal events are unchanged. Revised labels are overlays in `artifacts/evaluation/labels/<variant>/labels_<year>.json`; original hierarchy labels remain historical.

## Clean installation and original construction

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe scripts/prepare_models.py
.\.venv\Scripts\python.exe -m tkh_abstraction.cli run-all --config configs/default.yaml
```

The last command regenerates original construction and v1 evaluation/report: four snapshots, five variants, five 10% edge-removal rebuilds per variant, independent coherence/nulls, original labels/retrieval and intrinsic sensitivities. Run v2 afterward for the revised report. CPU execution is supported. An isolated `.repro-venv` installation and exact rebuild of all 20 main hierarchies were executed; evidence is in `artifacts/full/reproduction_check.json` and `AI_USAGE.md`.

External prerequisites are three public Hugging Face models at revisions pinned in `configs/default.yaml`: MiniLM (construction/alignment), BGE (evaluation/retrieval), and DeBERTa NLI. `prepare_models.py` downloads weights once; the NLI evaluator then requires local weights. Corpus text stays local. Once models are present, `$env:HF_HUB_OFFLINE='1'` disables network checks. Derived `.cache/embeddings` files are optional and regenerate automatically. Models, environments and derived embedding caches are excluded from the archive. No hidden absolute machine path is required by the pipeline; environment metadata and consultation records disclose paths for provenance.

For a tiny integration build:

```powershell
.\.venv\Scripts\python.exe -m tkh_abstraction.cli build --config configs/synthetic.yaml
```

A shorter real-data v1 run is `run-all --fast`, writing only to `artifacts/fast`. Original commands are `validate-data`, `describe`, `build`, `label`, `evaluate`, `report`, `run-all`; the focused pass is `evaluate-v2 --config configs/evaluation_v2.yaml`. Avoid `--fast`, `--seed`, or `--snapshot` overrides for audited v2. The supplied dataset is never replaced or downloaded.

## Evaluation interpretation

Hierarchy budgets (12,40,120), weights and models remain fixed. Visibility uses corpus `first_seen_year`; edges wait for their asserting paper and every endpoint. The report explains raw edge-year versus visible-edge counts. February-2026 questions use end-2025 as primary and annual-2026 as potentially post-cutoff sensitivity.

The heterogeneous `expected_methods` field is evaluated as named targets across node types. Exact/whole-identifier mapping preserves DeepH versus DeepH-E3 and records ambiguity. Question-only routing is shared by flat and hierarchy. Strict recall, incomplete-label precision, extended precision (`valid_but_unlisted`), known-wrong and outdated-main rates are separate.

One query-centroid/prototype/leaf dot product costs one. Offline indexing/query encoding and graph enumeration are excluded; claim costs are separate. Budgets below root-layer cost are failure diagnostics. Exhaustive flat is the main quality reference; seeded budgeted scanning is a weaker secondary reference. Score traces and reused claim scores are exported.

Claim retrieval uses entities, relations and provenance without answers. Evaluation-only source/semantic/NLI alignment is conservative; unaligned claims have **null recall**, never fabricated zero recall. Source hits measure provenance overlap, not entailment. Mapping coverage accompanies results. Ground truth never affects clustering or ranking.

Coherence uses BGE separately from MiniLM construction, with size/type nulls. New label generation reserves deterministic held-out evidence and uses generation-local IDF. Label lexical support and gloss NLI are separate; threshold stays .50. Paired old/new NLI differs from historical shared-evidence NLI. Weaker glosses and proxy limitations are disclosed. Blind audit CSVs remain unfilled; no human judgment is claimed. Keep audit keys separate during review; blank rows are not ratings.

## Outputs

The latest experiment adds query-seeded native hypergraph diffusion. It preserves contextual and Type B results and writes its own artifact set:

```powershell
$env:HF_HUB_OFFLINE='1'
.\.venv\Scripts\python.exe -m tkh_abstraction.cli evaluate-diffusion --config configs/diffusion.yaml
.\.venv\Scripts\python.exe -m pytest -q --junitxml=artifacts/evaluation/diffusion/tests.xml
.\.venv\Scripts\python.exe scripts/report_diffusion.py
```

See [report/hypergraph_diffusion.md](report/hypergraph_diffusion.md) for all nine requested stages, the matched answer-pool controls, alpha/M sensitivities, costs and propagation paths. `--diffusion-stage H0` (repeatable), `--cutoff 2025` and `--no-sensitivity` select a partial experiment. A partial invocation replaces only the diffusion aggregate files; run the complete command before the full report audit. The historical evaluator audits validate their own captured versions; `report_diffusion.py` is the current audit for preservation plus the new run. No hierarchy diffusion or replacement of Type B retrieval is implemented.

Diffusion outputs under `artifacts/evaluation`: `hypergraph_diffusion_results.json`, `hypergraph_diffusion_diagnostics.json`, `diffusion_per_target.json`, `propagation_traces.json`, `diffusion_sensitivity.json`, plus `diffusion/` containing protocol, stage records, mention index, operator metadata, full node-score NPZ files, tests and audit. `artifacts/baseline_contextual` preserves pre-diffusion metrics and report. For new exports configure paths and `diffusion_verify_reuse: false`; benchmark-specific assertions reside in the assessment report/audit, not production retrieval.

The contextual follow-up is documented in [report/contextual_retrieval.md](report/contextual_retrieval.md). Reproduce its complete flat ablation sequence after `evaluate-v2`:

```powershell
.\.venv\Scripts\python.exe -m tkh_abstraction.cli evaluate-context --config configs/contextual.yaml
.\.venv\Scripts\python.exe scripts/report_contextual.py
```

`--stage R0|R1|R2|R3|R4|R5|R1_direct` selects surface, contextual original, grouping, soft priors, extractive views, deterministic reranking, or the direct-context sensitivity. Repeat `--stage` and `--cutoff` to select several. `--resolver exact|generic` and `--candidate-k 50|100` expose resolution and reranking-pool choices. Pooling, fusion, field caps/weights and metric K values use the same YAML configuration system. A partial invocation replaces contextual summaries with that subset; run the complete command before regenerating the full report.

The primary evaluation is conservative 2025; 2026 is annual sensitivity. Ground truth is used only in evaluation/resolution diagnostics. All questions remain in `metrics.json`, including explicit null evaluability and raw target denominators. Source overlap is not claim entailment. Contextual hierarchy stages are deliberately unavailable until useful flat recovery justifies them; the legacy surface hierarchy commands and outputs remain intact. For another graph/benchmark, set input/output paths and `context_verify_reuse: false` to avoid requiring this assessment's frozen construction archives.

New artifacts: `baseline_reconciliation.json`, `gold_candidate_audit.json`, `target_resolution.json`, `retrieval_ablation_results.json`, `per_question_diagnostics.json`, and `contextual/` under `artifacts/evaluation`. Historical v2 metrics/report are in `artifacts/baseline_v2`; pre-review contextual checkpoints are preserved separately. Run `scripts/audit_v2.py` before `scripts/report_contextual.py` to regenerate both audit sections, then `scripts/package_submission.py` to refresh the local archive.

- `report/report.md`, `report/appendix_v2.md`, `report/literature_notes.md`, `report/figures/`.
- `artifacts/full/metrics.json`: original metrics plus `extrinsic_v2` and `faithfulness_v2`.
- `artifacts/full/hierarchy_<year>.json`, `temporal_events.json`; all variants in `artifacts/full/variants/`.
- `artifacts/full/descriptive.json`, `data_quality.json`, `environment.json`, `models.json`, `reproduction_check.json`.
- `artifacts/evaluation/protocol.json`, `target_mapping_<year>.json`, `claim_alignment.json`, `extrinsic_v2.json`, `retrieval/`, `prototypes/`, `labels/`, `diagnostics.json`, `final_audit.json`, `tests.xml`, `run.log`.
- `artifacts/baseline_v1/`: immutable previous metrics, report, archive and reuse manifest.
- `artifacts/submission_checklist.md`: evidence-backed PASS/PARTIAL rubric.
- `AI_USAGE.md`, `notes/improvement/`: major prompts and Ultralight subscription consultation records.
- `submission.zip`: local archive with SHA-256 manifest; never automatically uploaded or published.

## Evidence-aware candidate reranking

Run the current experiment and audit (both cutoffs, frozen H2 pools 100/200):

```powershell
$env:HF_HUB_OFFLINE='1'
.\.venv\Scripts\python.exe -m tkh_abstraction.cli evaluate-reranking --config configs/reranking.yaml
.\.venv\Scripts\python.exe -m pytest -q --junitxml=artifacts/evaluation/reranker_tests.xml
.\.venv\Scripts\python.exe scripts/report_reranking.py
.\.venv\Scripts\python.exe scripts/package_submission.py
```

See [evidence_reranking.md](report/evidence_reranking.md). RR0 reproduces H2; RR1 uses capped candidate-specific evidence and availability-normalized BGE cosine; RR3 fuses RR1/H2 with RRF60. RR2 is explicitly unavailable because no local relevance cross-encoder exists (the NLI classifier is not substituted). Relevance models are never downloaded automatically. H2's configuration, graph, grouping, resolver, Type B and previous reports/metrics remain unchanged. The current audit is `scripts/report_reranking.py`; older audits verify their captured source versions and must not be used to regenerate historical reports with the extended CLI source tree.

New-question mode: append `--questions questions_new.csv --ground-truth ground_truth_new.json`; use separate `output`, `v2_output`, and `reranker_output` paths in a copied config. The same graph and H2 settings can retain history verification; a new graph requires `reranker_verify_history: false` and explicit input paths. `--cutoff 2025` produces a partial experiment; the assessment audit expects both cutoffs. Candidate K is configured by `reranker_candidate_ks`, not the older contextual command's `--candidate-k` flag.

Artifacts: `artifacts/evaluation/{reranker_results,reranker_per_target,candidate_evidence,reranker_failure_analysis,reranker_oracle}.json`, `reranker_tests.xml`, and `reranking/` protocols, per-stage outputs and audit. Pre-pass metrics and diffusion report are archived in `artifacts/baseline_diffusion`. All 14 Type A questions remain explicit, including Q5/Q11 with null canonical evaluability. Results remain development diagnostics; modest top-ten gains and a large oracle gap leave hierarchy approximation deferred.

## Requirement-aware candidate reranking

The current follow-up adds generic source-span question decomposition, comparison-aware attribution, requirement cosine (CR1), controlled local NLI (CR2), and fixed H2 rank fusion (CR3/CR4). CR0 reproduces the prior RR1 exactly. K100 is primary; only CR1/CR2 additionally run K200. H2, prior evidence bundles, resolver, oracle, Type B, hierarchy and previous reports remain unchanged.

```powershell
$env:HF_HUB_OFFLINE='1'
.\.venv\Scripts\python.exe -m tkh_abstraction.cli evaluate-constraints --config configs/constraints.yaml
.\.venv\Scripts\python.exe -m pytest -q --junitxml=artifacts/evaluation/constraint_reranking_tests.xml
.\.venv\Scripts\python.exe scripts/constraint_nli_probes.py
.\.venv\Scripts\python.exe scripts/report_constraints.py
.\.venv\Scripts\python.exe scripts/package_submission.py
```

See [constraint_reranking.md](artifacts/evaluation/constraint_reranking.md) for the results, source-span decompositions, all target movements, attribution/coverage diagnostics, actual NLI probes and limitations. `scripts/report_constraints.py` is the current audit; previous report writers remain historical and should not overwrite their old reports after source extensions. The NLI model is the pinned local `cross-encoder/nli-deberta-v3-small`, not a relevance reranker; no generative model or remote API is used for decomposition, and no model is downloaded. `.cache/nli_constraints` stores exact-input NLI probabilities; warm reruns report cache hits separately from new inference. Post-hoc probes never feed ranking.

New questions: add `--questions questions_new.csv --ground-truth ground_truth_new.json` and use separate configured output paths. Set `constraint_verify_history: false` for an isolated output directory without copied frozen baseline artifacts, or for a new graph; keep it true for the shipped assessment run. Cutoff overrides produce partial experiments; the assessment report audit expects both years. Required outputs are `constraint_reranking_results.json`, `question_decompositions.json`, `requirement_support.json`, `evidence_attribution.json`, `constraint_reranking_per_target.json`, `constraint_failure_analysis.json`, `constraint_reranking_tests.xml`, and the report under `artifacts/evaluation`. Detailed checkpoints/protocol/audit are under `constraints/`; pre-pass metrics/report are preserved under `artifacts/baseline_reranking`.

## Requirement-conditioned evidence follow-up (CR6)

The newest experiment searches a wider query-independent candidate evidence pool separately for each question component, reusing the CR1 aggregation and frozen H2 discovery. CR7 adds the existing RRF60 fusion. Generic answer qualifiers, shared-predicate preservation and a conservative participial-attribution correction are additive; historical CR1 stays unchanged. CR2/CR4 remain archived.

```powershell
$env:HF_HUB_OFFLINE='1'
.\.venv\Scripts\python.exe -m tkh_abstraction.cli evaluate-conditioned --config configs/conditioned.yaml
.\.venv\Scripts\python.exe -m pytest -q --junitxml=artifacts/evaluation/conditioned_evidence_tests.xml
.\.venv\Scripts\python.exe scripts/report_conditioned.py
.\.venv\Scripts\python.exe scripts/package_submission.py
```

See [conditioned_evidence.md](artifacts/evaluation/conditioned_evidence.md) for metrics, all target ranks, matched-component evidence changes, pool-only and qualifier controls, pool-size correlations, provenance and limitations. The current audit is `scripts/report_conditioned.py`; earlier commands above describe historical experiments and must not be rerun to overwrite their archived reports.

All eight requested CR6 artifacts are under `artifacts/evaluation`, with protocol/checkpoints/audit under `conditioned/`. `artifacts/full/metrics.json` adds `conditioned_evidence` and retains all 14 Type A questions, including explicit null canonical recall for Q5/Q11. Previous constraints metrics/report are copied to `artifacts/baseline_constraints`. No new model, NLI inference or hierarchy is introduced.

For unseen questions, use `--questions` and `--ground-truth` and separate output paths. If the new output directory has no copied historical baseline, use `conditioned_verify_history: false`. The shipped assessment audit expects both original cutoffs. Pools are cached per snapshot in memory and exported; BGE vectors reuse persistent caches. Results are development diagnostics, not independently validated scientific QA.
