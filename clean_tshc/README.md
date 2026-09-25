# Temporal Semantic Hypergraph Coarsening

**Full write-up:** [report/full_report.tex](report/full_report.tex) is where I wrote everything. The compiled [full_report.pdf](report/full_report.pdf) is included for reading. Earlier experiments and development runs are preserved separately in [../rest_of_work/](../rest_of_work/).

A small, standalone research implementation of the supplied evolving-knowledge-hypergraph assessment. Start with [report.md](report.md), then inspect [outputs/metrics.json](outputs/metrics.json). This directory replaces the earlier experimental stack as the submission; it does not import that stack or reuse its results.

Validation completed: 24 tests pass, the full fresh-output workflow completed in about 51 minutes on this machine, and all scientific metrics, memberships, label probabilities and predictions matched the independently executed reference artifacts. See [Gate K](outputs/audit/gate_K.json) for the recovery/comparison scope. All 18,008 embeddings also matched a separate-cache regeneration. Report/figure refinements after the full run changed no scientific JSON and are recorded in [the presentation amendment](outputs/audit/presentation_amendment.json).

The hierarchy uses normalized MiniLM embeddings of `type: surface_form` and native hyperedge endpoint entropy. Restricted greedy merges balance semantic dispersion, fragmentation and shared-node VI against the previous snapshot. The hierarchy is nested by construction and preserves hyperedge multiplicities at every level. Display budgets are 12/40/120, with singleton leaves. Hungarian matching tracks identities separately. Deterministic labels use generation evidence; BGE coherence, randomized nulls, edge perturbations and held-out NLI evaluate the result. Flat and beam retrieval share one bounded document representation and one cosine scorer. Negative results are retained.

## Reproduce

Python 3.11 and CPU execution are supported. Tested on Windows with Python 3.11.2; CUDA is not required. From this directory:

```bash
python -m venv .venv
# Activate .venv using the command appropriate for your shell.
python -m pip install -r requirements.txt
python scripts/run_all.py --config configs/default.yaml
python -m pytest -q
```

On PowerShell, activation is `.\.venv\Scripts\Activate.ps1`. The first run downloads the three exactly pinned Hugging Face models unless already cached. An internet connection is therefore needed initially. Subsequent runs may use `HF_HUB_OFFLINE=1`; the per-text embedding cache is in `.cache/`. No API key or paid inference service is needed for reproduction. NLI evaluates labels only and is never a retrieval component.

To reproduce into a fresh output directory without deleting any existing files:

```bash
python scripts/run_all.py --config configs/default.yaml --output outputs_clean
```

Add `--cache-dir .cache_clean` to regenerate all node, document and question vectors in a separate empty cache as well. Downloaded model weights may still be reused. The separate-cache validation matched vectors and frozen predictions; the final ordinary Gate-K workflow reuses those verified embedding caches and regenerates all experiments. A null-DOI evaluation correction and its recovery are disclosed in `notes/execution_status.md`.

The workflow recomputes hierarchies, all nulls/perturbations, label NLI, predictions and metrics. It may reuse only model files and content-addressed embeddings produced by this clean implementation. No historical metric file is an input. Runtime depends on CPU and caches; measured execution time is recorded in metrics. Plotting is outside that measured interval. A separate `scripts/verify_submission.py` checks actual deliverables and optionally compares two complete output directories.

## Inputs and outputs

Place the supplied unchanged `tkh_collection10.json`, semicolon-delimited `questions.csv`, `ground_truth.json` and `collection10_articles.csv` in `data/`. The supplied task and README are retained as `ASSESSMENT.md` and `SUPPLIED_README.md`. SHA256 hashes, schemas, date anomalies and strict snapshot counts are regenerated before clustering.

`outputs/` contains:

- `audit/`: input manifest, gates, benchmark support/claim mapping, independence and clean-reproduction checks.
- `snapshots/` and `hierarchies/`: four strict annual snapshots, four variants, nested supernodes and multiplicity-preserving edges. The top-level hierarchy files are the primary temporal variant.
- `temporal_events.json`: observed identity and overlap events across all variants.
- `labels/inputs/` and `labels/faithfulness.json`: exact generation evidence and held-out NLI/lexical diagnostics.
- `retrieval/`: frozen candidate documents, complete rankings and per-question metrics, including composite components and unevaluable claim alignments.
- `tables/`, `figures/`, `metrics.json`: complete machine-readable results, six PNG/PDF figures and consolidated metrics.

`configs/default.yaml` exposes years, budgets, exact model revisions, alpha/lambda, kNN proposals, seeds, perturbations, null replicates, label caps and retrieval beams. The fixed one-factor sensitivity study uses the final snapshot and fixed primary previous history. It is not benchmark tuning. Beam 3 is primary; beams 1/5/10 are diagnostics.

Ground truth and the evaluation annotations in `notes/target_annotations.json` and `notes/claim_annotations.json` are isolated from hierarchy construction and retrieval. Predictions are written and hashed before evaluation reads gold. Audit metadata may count input records without passing benchmark content to construction. Tests check this boundary, exact merge deltas, temporal slicing, native collapse, laminarity, label separation and retrieval parity. The claim annotation was added after inspecting frozen results and is explicitly post-hoc; it is not independent expert validation.

Annual timestamps cannot enforce “by February 2026.” Surface forms and pretrained knowledge are not historically versioned. The optimizer is greedy; labels and claim alignments need independent expert review. NLI is an imperfect proxy, the benchmark is incomplete and repeatedly used, and comparison reduction is not measured latency speedup. See the report for the computed limitations and negative results.

AI assistance, including the master directive and official Ultralight consultation outcomes, is documented in [AI_USAGE.md](AI_USAGE.md). Verified primary references and the exact ideas borrowed are in [references.md](references.md).
