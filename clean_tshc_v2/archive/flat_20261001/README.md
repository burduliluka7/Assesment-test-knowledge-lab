# V2 flat retrieval development experiment

This directory is an independent retrieval experiment. The copied 2026 publication-visibility rule and V1 document builder preserve the reference semantics. No hierarchy is built, routed, retuned, or evaluated here. `clean_tshc/` and `rest_of_work/` are read-only references.

All benchmark numbers are **DEVELOPMENT results**, not held-out generalization. The full benchmark has 18 questions; direct identity evaluation applies to the Type-A subset. Gold semantics and denominators are documented in the generated report.

## Reproduction (Windows PowerShell, Python 3.11)

Run from the repository root in a fresh copy of this directory, with no existing frozen predictions. Existing predictions are deliberately never silently overwritten.

```powershell
python clean_tshc_v2/scripts/verify_protected.py --initialize --before clean_tshc_v2/outputs/audit/reference_before.sha256
python -m venv clean_tshc_v2/.venv
& clean_tshc_v2/.venv/Scripts/python.exe -m pip install -r clean_tshc_v2/requirements.txt
& clean_tshc_v2/.venv/Scripts/python.exe -B clean_tshc_v2/scripts/prepare_models.py
& clean_tshc_v2/.venv/Scripts/python.exe -B -m pytest clean_tshc_v2/tests -q -o cache_dir=clean_tshc_v2/.cache/pytest
& clean_tshc_v2/.venv/Scripts/python.exe -B clean_tshc_v2/scripts/predict.py
& clean_tshc_v2/.venv/Scripts/python.exe -B clean_tshc_v2/scripts/evaluate.py
& clean_tshc_v2/.venv/Scripts/python.exe -B -m pytest clean_tshc_v2/tests -q -o cache_dir=clean_tshc_v2/.cache/pytest
& clean_tshc_v2/.venv/Scripts/python.exe -B clean_tshc_v2/scripts/diagnose_inputs.py
& clean_tshc_v2/.venv/Scripts/python.exe -B clean_tshc_v2/scripts/verify_protected.py --before clean_tshc_v2/outputs/audit/reference_before.sha256
```

The recorded delivery also includes `scripts/finalize.py`, which verifies frozen artifacts and model-file hashes, checks the saved final test log, appends the verification result to the report, and writes the exhaustive new-file inventory. Its saved-log check refers to this delivery's 19-test suite.

The first command creates a recursive SHA-256 baseline and refuses to overwrite an existing one. The executed experiment used `$env:TEMP/clean_tshc_before_v2.sha256`; its original before and final after manifests are preserved in `outputs/audit/`. The Python verifier preserves baseline ordering and newline style and compares complete manifest bytes. Do not regenerate the before manifest after a run has started. A PowerShell implementation is also supplied, but the Python command works when local PowerShell script execution is disabled.

`prepare_models.py` downloads only public pretrained model files into this directory. BGE is pinned to the V1 revision. The relevance model is pinned to `233902d25c440f23af6f7d6e94d2946bac0bee0a`. Predictions use only local model files with network access disabled. Missing models are reported as unavailable; lexical retrieval remains runnable with the installed numerical dependencies. No NLI classifier substitutes for the relevance model.

This machine's recorded run uses the existing `rest_of_work/.repro-venv/Scripts/python.exe` **only as a read-only third-party Python installation**, with bytecode writes disabled. No project modules or mutable runtime data are imported from that directory. The clean installation above has no such dependency. Inspect `outputs/audit/model_version_manifest.json` for the actual interpreter and package versions.

## Protocol and outputs

1. Predeclare configuration, answer-family mappings, ablations, top100 cutoff, and RRF60 in `config.json` before scoring.
2. Parse questions, construct cards using copied graph data, and complete all rankings in `scripts/predict.py`. A Python audit hook blocks answer-file reads, evaluator imports, reference-source reads, and writes outside V2.
3. Save full rankings to `outputs/retrieval/predictions.json`, write `predictions.sha256`, and freeze source, configuration, inputs, decompositions, cards, and score diagnostics.
4. `scripts/evaluate.py` verifies every frozen digest before importing evaluation code. Only then are strict V1 target mappings and gold files read. Evaluation never feeds back into predictions.

The audit is evidence of ordering in these cooperative local programs, not a tamper-proof external timestamp or an operating-system sandbox against hostile native code.

- `outputs/query_understanding/`: all Q0/Q1/Q2 representations, extractive source spans, and warnings.
- `outputs/candidate_cards/cards.json`: cards keyed by candidate ID, selected evidence, hyperedge IDs, provenance, and omitted evidence.
- `outputs/candidate_cards/v1_documents.json`: exact copied V1 document construction.
- `outputs/retrieval/`: complete system rankings, cross-encoder token packing and scores, requirement support and winning evidence blocks.
- `outputs/audit/`: configuration, model and input manifests, phase events, prediction hash, runtime, protected-file comparison, and rank audit.
- `outputs/metrics.json`: metrics with explicit target and question denominators.
- `report/v2_retrieval_report.md`: scientific interpretation and per-question diagnostics.

The optional `StrictJSONParser` accepts an explicitly supplied local model callable. It rejects invalid JSON or unsupported scientific text. No generative model was selected for this experiment, and no LLM parsing result is simulated. NLI verification is optional and not run.

## Fixed variants

| System | Query | Representation | Candidate pool | Final scoring |
|---|---|---|---|---|
| A0 / V1-flat-like | raw | exact V1 documents | all non-authors | BGE cosine |
| A1 | raw | V1 documents | hard type | BGE cosine |
| A2 | raw | cards | all non-authors | BGE cosine |
| A3 | parsed | cards | all non-authors | BGE cosine |
| A4 | raw for reranking | cards | A3 top100 | relevance logits |
| V2-A | parsed | cards | hard type | BGE cosine |
| V2-B | parsed | cards | hard type | BGE + BM25, RRF60 |
| V2-C (primary) | raw for reranking | cards | V2-B top100 | relevance logits |
| V2-D | parsed components | evidence blocks | V2-B top100 | requirement support |
| DenseRaw / DenseStructured | raw / structured | cards | hard type | BGE cosine |
| SoftType | parsed | cards | all non-authors | rectified cosine × prior (1 or 0.8) |
| Lexical / LexicalAll | parsed | cards | hard type / all | BM25 |

V2-B is predeclared as the primary generator; “best” is not chosen after looking at gold. Rerankers permute exactly the first 100 IDs and append the unchanged tail, so full-ranking MRR is available and Recall@100 is necessarily unchanged. Reranker logits and tail cosine/RRF scores are different scales; the stage label and list order define the ranking.

Evidence uses direct low-arity incidence (at most three members), bounded exact name mentions, and source titles. Undirected incidence is not interpreted as directional entailment. Higher-arity relations are logged and omitted. The evidence-family order is identity, claims, tasks/problems, techniques/components, datasets/metrics, articles, other evidence. Identity is preserved in full for relevance scoring; an overlong question is shortened first. Dense documents retain the same identity-first prefix up to 512 tokens.

The requirement diagnostic uses rectified BGE cosine per component, maximum over selected blocks, then `core × geometric_mean(requirements) × product(1 − exclusion_support) × type_prior`. These are similarity heuristics, not calibrated support probabilities or proof that requirements are jointly satisfied.

Model references: [BGE model card](https://huggingface.co/BAAI/bge-small-en-v1.5), [MS MARCO relevance model card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2).
