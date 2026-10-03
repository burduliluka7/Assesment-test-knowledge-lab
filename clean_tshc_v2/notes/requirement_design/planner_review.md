## Scope read
`config.json`, `src/tkh_abstraction_v2/{intent,question_understanding,extractive,entity_experiment,entity_index,entity_retrieval,models,verification,isolation}.py`, `scripts/{predict,evaluate}.py`. Nothing else opened â€” no metrics, gold, reports, archives, or old predictions. `entity_evaluation.py` was *not* read (out of scope), so all evaluator-side claims below are flagged as unverified.

---

## 1. What exists today vs. what A7 must become

Current A7 (`entity_experiment.py:88-99`, `verification.py:45-88`):
- Operates on **top-20** of A6 only (`cfg['verification_k']`).
- **One** evidence span per component, chosen by dense similarity argmax (`verification.py:57`).
- NLI only; no per-requirement cross-encoder.
- 3 tiers (`verification.py:80`): all-supported / other / any-contradicted.
- Reorder is stable-sort by tier within the 20-prefix (`verification.py:85-88`); A6 order is the only intra-tier signal.

Target A7:
- Pool = **500** = `cfg['rerank_k']`, identical to the A6 pool and identical candidate generation (`rankings['A5'][:rerank_k]`, `entity_experiment.py:69`).
- Per candidate Ã— per component bundle: **top-3** candidate-specific evidence, scored by the **same frozen** `cross-encoder/ms-marco-MiniLM-L6-v2` (relevance) **and** pinned `nli-deberta-v3-small` (proxy).
- 4 tiers, then a multi-key intra-tier order.
- A6's global original-question CE score stays exactly as-is and becomes a tie-break key only.

---

## 2. Spec corrections to make before coding

**(a) Exclusions must not enter the rank aggregation.** The aggregation key is "geometric mean of reciprocal per-component ranks", where higher relevance = better. For an exclusion, higher relevance to the excluded concept is *worse*. Including exclusion ranks in the geomean silently inverts their sign. Recommendation: the geomean/weakest-link keys range over **positive components only** (`core_task` + atomic requirements); exclusions influence **tiering only** (via polarity-swapped NLI). State this in the predeclaration.

**(b) Tier predicate needs an explicit "contradiction" definition.** Define: `contradicted = any positive component CONTRADICTED or any exclusion VIOLATED` (where VIOLATED is the polarity-swapped `CONTRADICTED` already produced by `status_from_probabilities(..., exclusion=True)`, `verification.py:13-19`). Then:
- T0: not contradicted and every positive component SUPPORTED
- T1: not contradicted and â‰¥1 positive SUPPORTED
- T2: not contradicted and 0 SUPPORTED
- T3: contradicted

**(c) Per-evidence â†’ per-component status aggregation (new, because of top-3).** With 3 spans you get 3 labels per bundle. Use a fixed, contradiction-dominant existential rule: map each span to {SATISFIED, VIOLATED, UNKNOWN} (exclusion polarity applied at the span level), then `VIOLATED > SATISFIED > UNKNOWN`. Conservative, deterministic, order-independent. Record all three span labels so the choice is auditable.

**(d) Two hypothesis templates currently disagree.** `verification.py:61` builds `name + ' supports '/' involves ' + text`, ignoring `candidate_hypothesis` in `extractive.py:102-108` (which is connector-aware: `when/where/while`). Single-source them on `candidate_hypothesis` and pass the real `component` dict (with `kind`, `connector`, span offsets) from `intent['components']`.

---

## 3. Atomization must not touch retrieval

`parse_intent` â†’ `structured_query` / `parsed_only_query` / `combined_query` feed every channel (`entity_retrieval.py:56-70`) and BM25. Requirements are already atomized in `extractive.py:69-87` (split on `and|nor|but|for|;`), and that atomization *already* flows into `structured_query` via `representations()` (`question_understanding.py:23-29`).

Therefore: **do not change `decompose_question` or `representations`.** Add a purely derived, ranking-only field:

```
intent['ranking_components'] = [  # from intent['components'], no new text
  {id, kind, connector, text, source_span, start, end}
]
```
built by filtering/deduping `intent['components']` (which already carries `kind`, `connector`, `id`). Enforce invariance with an assertion in `parse_intent`: `structured_query`, `parsed_only_query`, `combined_query`, `search_query`, `answer_types`, `temporal_cutoff` must be byte-identical to the values computed before the new field is attached (trivially true if you only append), plus a test that `EntityRetriever.generate` consumes no key named `ranking_components`.

Dedupe rule: key on `(kind, normalize_name(text), start, end)`. A substring appearing as both requirement and exclusion stays as two bundles (different `kind`); exact duplicates collapse so the geomean isn't double-weighted.

---

## 4. Concrete implementation plan

### `extractive.py`
No behavioral change. Optionally expose `candidate_hypothesis` unchanged (already literal-template).

### `intent.py`
Append `ranking_components` (derived, deduped, span-preserving) + the invariance assertion. ~10 lines.

### `verification.py` â†’ rewrite as the A7 core
New function, keep `status_from_probabilities` and `NliVerifier` as-is:

```
score_requirement_bundles(intent, entity_ids, index, retriever, cross, nli, cfg)
  -> per_entity: {component_id: bundle}, costs
```
Steps:
1. `components = intent['ranking_components']`; `queries = retriever.encoder.encode([c['text'] for c in components])` (cached via `Encoder.encode`, `models.py:17-26`); `matrix = retriever.evidence_vectors @ queries.T` â€” same as `verification.py:49-51`, computed **once per question**, now reused for 500 entities instead of 20. Cost `|evidence| Ã— C`, negligible.
2. Candidate evidence set per entity: keep the existing filter `candidate_specific and field not in ('articles','graph_evidence')` (`verification.py:55-56`). Take `sorted(candidates, key=lambda i: (-matrix[loc[i], j], i))[:cfg['a7']['evidence_k']]`. Note this deliberately uses the **full** candidate-specific evidence set, not the `select_evidence` card (which applies `query_evidence_caps`, `entity_index.py:206`) â€” A7 needs per-requirement recall, not a global card budget. Document that divergence explicitly.
3. **CE per bundle, one pair:** build `blocks = [identity] + [evidence blocks]` in the exact format of `entity_index.py:211-213` (identity first is mandatory, `models.py:31-32`), query = the component's **literal span text**. Reuse `pack()` so `query_truncated` / `omitted` / `token_count` are recorded.
4. **NLI per evidence span, 3 pairs:** premise = `'Candidate: ' + name + '. Relation: ' + support_kind + '. Evidence: ' + text` (as `verification.py:67`), hypothesis = `candidate_hypothesis(name, decomposition, component)`. Short premises are the key to runtime (see Â§6).
5. Status per span â†’ component status (rule (c)) â†’ tier (rule (b)). Keep `missing_evidence_is_not_contradiction=True` and emit it per component, not just per entity.

### `entity_retrieval.py`
Add a pure ranking function (no model calls, easily unit-tested):

```
def requirement_rerank(a5_rows, a6_rows, bundles, positive_ids, pool_k):
```
- `global_rank[e]` = 1-based index in A6 (which is itself the CE reorder of the A5 prefix, `entity_retrieval.py:110-115`).
- `rrf_rank[e]` = 1-based index in A5 (`fuse`, `entity_retrieval.py:92`).
- Per positive component `c`: rank all pool members by `(-ce_score, entity_id)` using **competition (min) ranks** for exact float ties; candidates with **no evidence** for `c` form the lowest score band â†’ all receive rank `n_scored + 1` (see Â§5 edge case 1).
- `geo[e] = exp(-fsum(log(rank_c[e]))/C)`  (== reciprocal of the geometric mean of ranks; monotone, higher better). Use `math.fsum` on logs.
- `weakest[e] = 1 / max_c rank_c[e]`.
- Sort key: `(tier, -geo, -weakest, global_rank, rrf_rank, entity_id)` â†’ **total order**, deterministic independent of dict iteration.
- Tail: `rows[pool_k:]` carried through unchanged with `stage='unreranked_fusion_tail'`, exactly as `rerank_prefix` does.

### `entity_experiment.py`
- Replace lines 88-99. Compute bundles over `[r['entity_id'] for r in rankings['A6'][:cfg['rerank_k']]]` (same 500-set as A6's pool; ordering irrelevant).
- `rankings['A7'] = requirement_rerank(...)`.
- `rankings['A7_NoNLI']`: identical pipeline with `nli=None` â†’ every component status is `NOT_EVALUABLE`/`NOT_SUPPORTED` â†’ all tier 2 by rule (b), so the system is pure CE rank aggregation. Define and document that collapse rather than letting it fall out accidentally (otherwise "NoNLI" and "all-tier-2" are indistinguishable). Recommendation: for `A7_NoNLI` force `tier=0` for all so tiering is fully disabled and the control isolates the aggregation.
- `rankings['A7_SingleEvidence']`: same bundles, `evidence_k=1` (reuse the top-1 span already in the top-3 list â€” **no extra model calls**, just recompute CE? No: the CE pack differs, so it needs its own CE pair. Budget 1 extra CE pair + 0 extra NLI pairs per bundle, since the top-1 NLI result is already a subset).
- Update `predeclaration.json` (`entity_experiment.py:29-33`) stage strings, add the two controls, add the tier policy string and aggregation policy string.
- Extend the freeze file list (`entity_experiment.py:115-118`): add `outputs/audit/efficiency.json`, `outputs/audit/prediction_accesses.json`, `outputs/audit/a7_diagnostics.json`, and the new `outputs/retrieval/requirement_rerank/*.json` (the `retrieval` rglob already covers the last if written before the freeze â€” ensure write ordering). `events.jsonl` cannot be hashed wholesale (evaluate appends to it, `evaluate.py:35`); hash the freeze-time **prefix** and record `event_lines=N` + `events_prefix_sha256`.
- Note the pre-existing `A6_500` duplicate: `cfg['rerank_controls']` includes 500, so `rankings['A6_500']` is byte-identical to `A6` (`entity_experiment.py:85-86`). Harmless, but it inflates the system list; user asked for 100/200 controls, so drop 500 from `rerank_controls` or keep and document.

### `config.json`
```
"a7": {
  "pool": 500, "evidence_k": 3,
  "aggregation": "geometric_mean_of_reciprocal_per_component_ranks",
  "tie_rank_policy": "competition_min_rank_then_entity_id",
  "missing_evidence_rank_policy": "lowest_score_band_shared_rank_n_scored_plus_1",
  "exclusions_in_aggregation": false,
  "nli_premise_max_tokens": 256,
  "controls": ["A7_NoNLI","A7_SingleEvidence"]
},
"verification_policy": "pool500_requirement_bundles_top3_evidence_4tier_then_geomean_reciprocal_rank"
```
`nli_threshold` stays 0.7 (pre-existing, not fitted). **No new numeric knob may be chosen by looking at any ranking quality signal** â€” `evidence_k=3`, `pool=500` come from the user's spec; `nli_premise_max_tokens` is a packing constraint, set from the token-length distribution of evidence text only.

### `isolation.py`
- Evaluation mode: accept the verified freeze manifest and **deny write/remove/rename/replace on every frozen path**. Current code only blocks mutation *outside* root (`isolation.py:26-27, 38-44`), which does not protect `outputs/retrieval/predictions.json` from the evaluator.
- Add audit events `os.replace` is delivered as `os.rename`; also hook `shutil.copyfile`, `shutil.copymode`, `shutil.move`, `os.truncate`.
- Add forbidden-read paths for prior results (extend `isolation.py:30` set) with the old-run artifact names, and forbid any path matching `outputs*/` other than the single active `--output` during prediction.
- **Access auditing of all project-relative reads** (user requirement): `isolation.py:36-37` records only `root/data`. Change to record every non-writing `open` whose resolved path `is_relative_to(root)`, bucketed by category (`data`, `config`, `src`, `models`, `cache`, `outputs`) with per-path counts, and emit that structure in `prediction_accesses.json` (`entity_experiment.py:109-111`). Keep the honest caveat string. Volume is a few thousand paths; store as `{category: {relpath: count}}` sorted.
- Ordering in `evaluate.py`: install the hook **before** `verify_freeze` reads anything, then re-arm it with the frozen path set once the manifest validates. The current `from tkh_abstraction_v2.io import ...` at `evaluate.py:8` happens with no hook installed at all.

### `scripts/`
- Keep predict/evaluate as **separate processes** (already true). Add a `scripts/run_all.ps1`/`.py` that `subprocess.run`s predict.py, asserts returncode 0 and `audit/freeze.json` exists, then runs evaluate.py in a fresh interpreter. Assert that `entity_experiment` is **not** in `sys.modules` inside evaluate.
- **In-place replacement** (the user already copied old V2, so `outputs/retrieval/predictions.json` exists and `entity_experiment.py:21-22` will raise). Add an explicit `--replace` flag to `predict.py` that:
  1. records `sha256` of the old `predictions.json`, old `freeze.json`, and old `predictions.sha256` into `outputs/audit/replaced_runs.json` (appending, preserved across the wipe);
  2. deletes only `outputs/{query_understanding,entity_index,retrieval}` and `outputs/audit/*` **except** `replaced_runs.json`;
  3. re-seeds `events.jsonl` with a `previous_run_replaced` event carrying the old hashes.
  No new experiment/archive directory is created. Without `--replace`, keep the current hard failure.

---

## 5. Edge cases (each needs an explicit, recorded decision)

1. **Missing evidence for a component.** Do *not* assign rank = `pool_k` (couples the key to pool padding) and do *not* drop the component from that candidate's geomean (makes geomeans of different dimensionality incomparable â€” the single worst failure mode of this aggregation). Use shared rank `n_scored + 1` for the whole missing band. Status = `NOT_SUPPORTED_BY_AVAILABLE_EVIDENCE`, never `CONTRADICTED` (`verification.py:9`), including for exclusions.
2. **Candidate with zero candidate-specific evidence** (only `articles` / `provenance_title_only` / `graph_evidence`). All components missing â†’ geomean identical across all such candidates â†’ order falls through to `global_rank` then `rrf_rank`. Correct, but log the count; if it is large, the `candidate_specific` filter is the dominant ranking factor and that must be stated, not tuned away.
3. **Float ties in CE scores.** Competition (min) ranks, then `entity_id` only in the *final* sort â€” not inside the per-component rank, otherwise tied candidates get asymmetric geomeans for no reason.
4. **Pool smaller than 500** (short unions, narrow `allowed_entity` type filter at `entity_index.py:57-60`): all ranks computed over the actual `n`; assert `len(set(ids)) == n`.
5. **Fallback parses** (`extractive.py:88-98`): `core_task` may be the full question or just the answer-type head; `requirements` may be empty. Then C=1 and T0 â‰¡ "core supported". Assert `C >= 1`.
6. **NLI token packing.** `NliVerifier.score` uses `truncation='only_first', max_length=512` (`verification.py:36`) â€” premise-only truncation, hypothesis always intact. Good. But truncation is silent; `full_tokens` is recorded (`verification.py:41`) â€” add an explicit `premise_truncated` boolean and count them in diagnostics. Cap premise construction at `nli_premise_max_tokens` by truncating the *evidence text* at a token boundary and recording the literal retained span offsets (literality requirement).
7. **CE packing.** `pack()` raises on an over-long identity (`models.py:37-38`) and on overflow (`models.py:52-53`). With a short per-requirement query these should never fire; wrap per bundle and record `pack_error` rather than aborting a 40-minute run.
8. **Batch-composition determinism.** `torch.use_deterministic_algorithms(True)` is set (`entity_experiment.py:27`) but padding width can still perturb logits in the last bits. If you length-bucket for speed, you must validate batch-invariance (Â§7) and freeze the bucketing policy, because scores are cached by content and a bucketing change silently invalidates nothing.
9. **Exclusion with `connector` not in the when/where/while set** falls through to `'{name} involves {text} for {core}.'` (`extractive.py:106`) â€” correct violation-framing; keep the inline comment.
10. **Duplicate (component, evidence) pairs across components** â€” very likely, since the same claim span is often top-3 for several requirements. Dedupe at the *pair* level before inference (Â§6).

---

## 6. Efficiency / caching (the main engineering risk)

Pair counts per question: `C Ã— 500` CE (+ `C Ã— 500` for SingleEvidence) and `3 Ã— C Ã— 500` NLI. With Câ‰ˆ4 and Q questions: â‰ˆ `4000Â·Q` CE and `6000Â·Q` NLI pairs. For Q=10 that's ~40k CE + ~60k NLI â€” consistent with the user's "couple tens of thousands".

The dominant cost is **sequence length, not pair count**. DeBERTa-v3-small at 512 tokens on 4 CPU threads is roughly an order of magnitude slower per pair than at 96 tokens. Therefore:
- Keep NLI premises to a single evidence span (short) rather than a packed 3-span premise. This is the reason to prefer per-span NLI over one packed premise, *in addition* to the span-level auditability the user requires: 3Ã— the pairs at ~1/4 the length is a net win and yields literal per-span labels. (Tradeoff if you disagree: one packed premise is ~1/3 the pairs but needs an evidence-dilution caveat, loses per-span labels, and makes the SingleEvidence diagnostic non-comparable.)
- **Length-bucketed batching**: sort pairs by `len(premise)+len(hypothesis)` into fixed buckets, batch within bucket. Replaces the fixed `range(0, len, 16)` loop (`verification.py:34`, `models.py:71`). Expect 2-4Ã— from padding elimination alone.
- **Pair-level in-process dedup**: hash `(model_revision, text_a, text_b)`, infer uniques only, fan results back out. Cheap and eliminates edge case 10.
- **Cache granularity**: do **not** write 100k per-pair files (NTFS will dominate the runtime). Follow the existing pattern â€” one blob per question per stage under `.cache/` keyed by `content_key([revision, policy_version, intent_fingerprint, index_fingerprint, ordered_pair_hashes])`, like `entity_experiment.py:74-79, 89-96`. Additionally keep a single append-only `.cache/a7_pairs/<model_revision>.jsonl` map `pair_sha256 â†’ scores` for cross-question reuse; load once into a dict at startup. All cached values are **gold-free model outputs**, and the key must include `fingerprints[cutoff]` (the index content key, `entity_experiment.py:58`) plus an explicit `a7_policy_version` string so any spec change invalidates the cache instead of silently reusing it.
- Existing A6 CE cache keys stay valid â†’ A6 is free on rerun. State that in the efficiency report so A7's cost isn't understated: report `*_inference_pairs` (0 on cache hit) separately from `*_pairs`, as the current code already does.
- Keep `show_progress_bar=True` off for the 500-entity loops or it will flood the log; `Encoder.encode` (`models.py:24`) prints progress for every call.

---

## 7. Validation plan

Unit tests (pure functions, no models, no gold):
1. `requirement_rerank` on synthetic score matrices: known geomean ordering; weakest-link only breaks geomean ties; `global_rank`/`rrf_rank`/`entity_id` cascade; output is a permutation of input with the tail byte-identical; **shuffling input dict order yields an identical ranking**.
2. Tier table: enumerate all status combinations over {core, 1 req, 1 excl} â†’ expected tier; assert T3 is reachable only via contradiction/violation and that missing evidence never reaches T3.
3. `status_from_probabilities` polarity: exclusion swap, `evidence_present=False`, `probabilities=None` (already covered by `verification.py:7-19` â€” add exclusion-specific cases).
4. Missing-rank policy: a component with 0 scored candidates â†’ all ranks equal, geomean well-defined, no `log(0)`.
5. Atomization invariance: for every row in `data/questions.csv`, assert the retrieval-facing intent keys are unchanged by `ranking_components` and that `original_question` passthrough still holds (`intent.py:39`).
6. Packing: for all generated bundles, `token_count <= limit`, identity block first, `premise_truncated` correctly flagged, retained evidence text is a literal prefix of the source span.
7. Isolation: evaluator process attempting `open(predictions.json,'w')`, `os.remove`, `os.rename`, `shutil.move` on frozen paths each raises `PermissionError`; prediction process attempting to read a forbidden old-result path raises.

Integration:
8. **Batch-invariance check**: score a 64-pair sample at batch sizes 1 / 8 / 32 and assert max abs delta below a recorded tolerance; if it exceeds float noise, pin batch size 1 for the final run and report the cost.
9. **Smoke run** to a scratch `outputs_smoke/` (deleted before the real run) with the full pipeline on a subset, to measure wall-clock per question and extrapolate. The final run uses `--output outputs` with `--replace`; assert at freeze time that no other `outputs*` directory exists.
10. **Freezeâ†’evaluate handshake**: run predict, then corrupt one byte of a frozen file and confirm `verify_freeze` (`evaluate.py:14-19`) fails; restore and confirm it passes.
11. **top-3 vs top-1 diagnostics** (`outputs/audit/a7_diagnostics.json`): Kendall Ï„ and top-k overlap between `A7` and `A7_SingleEvidence` (and `A7_NoNLI`) per question; tier-distribution shift; count of components whose status differs between 3-span and 1-span; count of candidates whose status is driven by the 2nd/3rd span only; tie-group size histogram; missing-evidence component counts; premise/query truncation counts. All rank-agreement and coverage statistics â€” **no gold, no quality metrics**, and none of it may feed back into any parameter.

---

## 8. Tradeoffs worth stating in the predeclaration

| Choice | Alternative | Why the recommendation |
|---|---|---|
| Per-span NLI (3 pairs) | one packed 3-span premise (1 pair) | literal span-level labels + shorter sequences; costs 3Ã— pairs but less total compute |
| Exclusions tier-only | exclusions in geomean | sign of "relevance to an excluded concept" is inverted; including it is a modeling error |
| Missing â†’ shared rank `n_scored+1` | rank = pool size, or drop component | avoids pool-padding dependence and dimensionality mismatch |
| Contradiction-dominant span aggregation | support-dominant | conservative, order-independent, keeps T3 meaningful |
| A7 draws from full candidate-specific evidence | reuse the capped `select_evidence` card | per-requirement recall; the caps were sized for one global card |
| `A7_NoNLI` forces tier 0 | let statuses collapse to T2 | otherwise the control confounds "no NLI" with "nothing supported" |

---

## 9. Assumptions

- `data/questions.csv` has ~10-100 rows; runtime estimates scale linearly and must be re-derived from the smoke run.
- `availability['relevance']` and `availability['nli']` models are present locally; `cross=None` currently skips A6/A7 entirely (`entity_experiment.py:72`) â€” A7 must keep that same fail-soft and record it in `model_load_failures`.
- The relevance model's 1-logit guard (`models.py:62-64`) and the NLI label guard (`verification.py:28-29`) remain the only model-identity checks; no fine-tuning, no threshold fitting, no re-use of any observed result.
- "Preserve candidate generation" means `entity_retrieval.generate` and `entity_index.build_index` are **not** edited at all.

## 10. Unresolved questions for the user

1. `entity_evaluation.py` was out of scope: does it enumerate systems from a hard-coded list? If so, `A7_NoNLI` / `A7_SingleEvidence` must be registered there, and that file is on the gold side of the freeze â€” confirm Codex may edit it.
2. CE query per bundle: the **bare literal span** (`"low-resource languages"`) or a literal-prefixed form (`"requirement: low-resource languages"`)? ms-marco MiniLM expects query-like text; the bare span is more literal, the prefix is more query-like. Both are deterministic and introduce no new vocabulary. Default: bare span.
3. For `A7` tie-breaks, should `global_rank` be A6's position within the 500-pool only, or A6 including its unreranked tail? (They coincide inside the pool; the question only matters if the pool shrinks below 500.) Default: position within A6 restricted to the pool.
4. May Codex produce a one-time pre-change fixture of A0â€“A5 rankings (freshly computed, gold-free) to golden-file-test retrieval invariance? It is not an old prediction or metric, but it is an output artifact â€” confirm it is allowed under the "no old predictions" rule.
5. `rerank_controls`: drop the redundant `500` entry, or keep `A6_500` as an explicit identity check?

No files were modified and no commands were run.