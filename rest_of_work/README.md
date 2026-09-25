# Temporal knowledge hypergraph assessment

The final submission is [report.md](report.md), with [consolidated metrics](metrics.json) and the [requirement audit](artifacts/evaluation/final_requirement_audit.json). It evaluates the existing temporal native-hypergraph hierarchy as an approximation of a frozen, working fine retriever. Earlier experiments remain archived; their reports describe the state of the project at the time they were written.

## Final reproduction

From the repository root, with Python 3.11 and the pinned local models:

```powershell
$env:HF_HUB_OFFLINE='1'
.\.venv\Scripts\python.exe -m tkh_abstraction.cli evaluate-hierarchy-retrieval --config configs/final.yaml
.\.venv\Scripts\python.exe -m pytest -q --junitxml=artifacts/evaluation/final/tests.xml
.\.venv\Scripts\python.exe scripts/build_final_report.py
.\.venv\Scripts\python.exe scripts/package_submission.py
```

This is the complete final integration path over the supplied frozen artifacts. It first reproduces FINAL_FINE's exact archived scores/ranks and 13/47 (2025), 13/50 (2026) top-ten recovery, then executes the predeclared hierarchy/flat comparisons. The report writer audits the executed sources, preserved files, hierarchy invariants and passing tests before generating report.md, metrics.json and the requirement audit. It does not rebuild the hierarchy, rerun NLI or overwrite historical metric sections. The package is local; nothing is uploaded.

`--ground-truth <path>` changes evaluation only; the frozen graph and question text are required for this final replay. Use a separate `final_output` containing copies of `frozen_manifest.json` and `final_fine_regression.json` for evaluation-only perturbations. The original supplied answer key is required for the submission's 13/47 and 13/50 assertions and full regression tests. No new benchmark questions or mapping rules are introduced.

## Installation and offline models

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
```

[requirements.txt](requirements.txt) pins the environment. [configs/default.yaml](configs/default.yaml) and configuration defaults pin MiniLM (construction/alignment), BGE (coherence/retrieval) and DeBERTa NLI (historical label/claim proxies). The final integration needs the local pinned BGE weights and archived evaluations; it never calls NLI. For a first model installation, `python scripts/prepare_models.py` downloads the pinned public model snapshots. Then set `HF_HUB_OFFLINE=1`. Model weights, virtual environments and derived embedding caches are excluded from submission.zip. Embedding caches can regenerate; model downloads are not part of the final experiment.

Frozen input data is under `data/data`. The hierarchy, H2 node-score vectors, candidate evidence pools and claim-alignment archives are included in the submission. No private absolute path is required by production code. Local paths in AI consultation logs are provenance only.

## Final systems and interpretation

- **FINAL_FINE:** generic CR1-derived decomposition with the existing generic fixes; frozen H2 top100; wider query-independent evidence pools; component-conditioned BGE cosine and attribution; unchanged CR1 aggregation. No scored answer qualifiers, NLI, H2 fusion or primary K200. Historical `without_qualifiers` is its exact reference; CR stage names are unchanged.
- **Hierarchy:** frozen temporal hierarchy; all eligible L0 roots; fixed centroid/prototype score; deterministic best-first descent. An atomic member activates its existing exact-name group and evidence pool once. Only reached groups receive fine scores. Main search covers all eligible groups; H2-intersection controls isolate the restricted candidate universe.
- **Flat exhaustive:** all eligible entity groups with the identical fine scorer. **Flat equal-work:** seed42 random prefix with the same total comparison budget and scorer.
- Budgets25/50/100/200/500 plus unrestricted are fixed, not chosen from outcomes. A budget unable to score every root is `INSUFFICIENT_ROOT_BUDGET`. Evidence comparisons count against the budget. Reported work is a declared standalone logical count; no shared-machine latency speedup is claimed.
- Every stage retains all14 Type A questions. Q5/Q11 have null canonical recall; macro averages12 evaluable questions, micro uses47/50 canonical instances, and raw recall retains63 labels. Four Type B questions retain their previous output path and archived results.
- Source overlap measures provenance, not entailment. Required claims that cannot be aligned have null recall. Label NLI is a fallible held-out proxy; expert adjudication and versioned historical source text remain absent.

The temporal hierarchy at budget500 recovers 1/47 and 1/50 and retains 0/13 FINAL_FINE hits at both cutoffs. The final recommendation is a hierarchy for browsing, with partial label-faithfulness evidence; this search policy is not supported as a retrieval accelerator. H2-intersection controls presuppose H2 membership; their traversal-only costs must be augmented by the H2 prerequisite costs listed in the detailed report. No production latency speedup is claimed.

## Deliverables and artifact map

| Deliverable | Location |
|---|---|
| Formal statement and compact final report | [report.md](report.md) |
| Submission-facing metrics | [metrics.json](metrics.json) |
| Per-item P1–P6, T1–T7 and deliverable audit | [final_requirement_audit.json](artifacts/evaluation/final_requirement_audit.json) |
| Final labeled snapshots | `artifacts/evaluation/final/hierarchy_2020.json`, `hierarchy_2022.json`, `hierarchy_2024.json`, `hierarchy_2026.json` |
| Final temporal-event schema | `artifacts/evaluation/final/temporal_events.json` |
| Original frozen hierarchy/metrics | `artifacts/full/`, including all five variants |
| Exact FINAL_FINE full rankings and regression hashes | `artifacts/evaluation/final/final_fine_rankings.json`, `final_fine_regression.json`, `fine_parity.json` |
| Every search trace, evidence selection and per-question metric | `artifacts/evaluation/final/*_<year>_<budget>.json` |
| Detailed final tables/protocol | `artifacts/evaluation/final/integration_details.md`, `protocol.json`, `results.json` |
| Required-claim evaluability for all 18 questions | `artifacts/evaluation/final/claim_evaluability.json` (Type B alignment remains archived) |
| Growth/arity and coherence/stability figures | `report/figures/growth_arity.png`, `coherence_stability.png` |
| Final quality–cost figure | `report/figures/hierarchy_retrieval_tradeoff.png` and `.svg` |
| Optional static L0–L2 view across snapshots | `report/figures/hierarchy_levels.png` |
| Tests | `tests/`; executed JUnit `artifacts/evaluation/final/tests.xml` |
| AI disclosure | [AI_USAGE.md](AI_USAGE.md); exact prompts/results under `notes/` |
| Local submission package | `submission.zip`, with SHA-256 `MANIFEST.json` |

Final snapshot exports apply the already measured v2 label overlays at L0/L1. Original construction memberships, IDs, events and original-label exports remain byte-identical. No new label generator is introduced.

## Historical experiments and rebuilds

History is retained in `report/`, `artifacts/evaluation/`, and `artifacts/baseline_*`. The [old README](notes/final/README_before.md) records earlier commands and results; it is historical documentation. CR2/CR4 are negative local-NLI ablations; CR7 is a secondary retention/fusion ablation, not a final system.

Do not run historical report writers over the final workspace: they validate the source hashes captured at their own stage, and some overwrite older report paths. The additive CLI dispatch intentionally changes its historical hash; all underlying frozen core modules remain identical. `scripts/build_final_report.py` is the final report/audit entry point.

A tiny independent build remains available:

```powershell
.\.venv\Scripts\python.exe -m tkh_abstraction.cli build --config configs/synthetic.yaml
```

For original hierarchy reconstruction, create a separate config/output directory and run `python -m tkh_abstraction.cli build --config <isolated-config.yaml>` using the original weights, snapshots, models and budgets. Do not replace the frozen submission artifacts. Original independent clean-environment installation and exact20-hierarchy rebuild evidence is in `artifacts/full/reproduction_check.json`; the final pass verifies preserved hashes rather than repeating that expensive study.

The final stop rule is strict: no additional reranker, threshold tuning, benchmark construction, H2 modification or hierarchy redesign after this integration. Negative retrieval utility remains a valid reported result.
