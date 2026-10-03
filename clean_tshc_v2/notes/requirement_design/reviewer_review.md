## Review summary

I reviewed the requirement-ranking/verification pipeline, the isolation/freeze layer, the predictor/evaluator entry points, config, and the allowed tests/plan. One concrete logic bug affects ranking correctness; one lower-confidence availability gap is worth flagging. Everything else I checked (evidence selection/dedup, token packing, competition ranking, tie-break ordering, isolation allowlist, freeze/protect-evaluation sequencing, gold-blind import guard) is internally consistent with the stated spec and the existing tests.

### Bug: exclusion "non-violation" incorrectly promotes candidates to tier 1 (requirement_ranking.py:77-87)

```python
def aggregation(records, epsilon=1e-12, use_nli=True):
    positives=[r for r in records if r['kind']!='exclusion']
    supported=sum(r['status']=='SUPPORTED' for r in records) if use_nli else 0          # line 79: ALL records
    contradicted=sum(r['status']=='CONTRADICTED' for r in records) if use_nli else 0
    positive_supported=sum(r['status']=='SUPPORTED' for r in positives) if use_nli else 0  # line 81: positives only
    tier=3 if contradicted else 0 if positives and positive_supported==len(positives) else 1 if supported else 2  # line 82
```

The tier-0 branch correctly uses `positive_supported` (positives only, excluding exclusions, matching the spec's "positive components core+literal atomicrequirements"). The tier-1 branch at line 82 instead falls back to `supported`, which is summed over **all** records including exclusions (line 79).

Consequence: an entity whose core task and every positive requirement are `NOT_SUPPORTED_BY_AVAILABLE_EVIDENCE` (zero real positive evidence) can still land in tier 1 — ahead of entities correctly placed in tier 2 — solely because one of its *exclusion* components resolved to `SUPPORTED`. For an exclusion, `SUPPORTED` means "the forbidden property was not entailed," which is the common/default outcome whenever evidence exists and is simply unrelated to the excluded property (not a signal of relevance to the positive requirements at all). This conflates "exclusion not violated" (a neutral, frequently-true outcome) with "some requirement is genuinely supported by evidence," inflating the rank of candidates that have no real positive support.

This directly contradicts the design intent stated in the spec and in the code's own comment pattern — exclusions are supposed to affect tier *only* via contradiction (the `tier=3` branch), never contribute a positive signal. The existing `positive_supported` variable already exists specifically to prevent exclusions from leaking into the positive-support determination (it's used for tier 0), but the same discipline wasn't applied to tier 1.

No existing test catches this: `test_four_tier_conjunction` only exercises all-positive records (no exclusions mixed with partial positive support), and `test_exclusions_do_not_reward_topical_similarity` only exercises the tier-3 (contradiction) path. There is no test combining an unsupported/absent positive with a `SUPPORTED` exclusion, which is exactly the scenario that trips this bug.

**Minimal fix:** change line 82 from
```python
tier=3 if contradicted else 0 if positives and positive_supported==len(positives) else 1 if supported else 2
```
to use `positive_supported` (or equivalently `any(r['status']=='SUPPORTED' for r in positives)`) in place of `supported` for the tier-1 condition:
```python
tier=3 if contradicted else 0 if positives and positive_supported==len(positives) else 1 if positive_supported else 2
```
This preserves all four existing test cases (none of which mix a supported exclusion with unsupported positives) while removing the exclusion leak. `supported` (all-records count) can still be kept/returned in the diagnostics dict for reporting — it's only the tier threshold at line 82 that needs the positive-only count.

This should be fixed before evaluation: it changes final_rank ordering (via `final_tier` in the sort key in `order_candidates`, requirement_ranking.py:108-113), so any run executed before the fix is not a faithful realization of the declared four-tier spec and would need to be re-predicted.

### Secondary note (lower confidence, not a confirmed bug): silent NLI unavailability

In `entity_experiment.py:56-63` and `:71-72`, the predictor aborts if the pinned relevance cross-encoder (`cross`) fails to load, but there is no equivalent guard for `nli`. If the pinned NLI model fails to load, `nli=None` is recorded only in `failures['nli']` (surfaced later in `model_version_manifest.json`), and the run proceeds, assigning every evidence-bearing requirement `NOT_EVALUABLE` (verification.py is never invoked). Given the primary system A7 is specified to rely on "same pinnedNLI jointly packed top3 premise," a silent fallback to NLI-less scoring would materially change what "A7" means for that run without an explicit failure. This isn't incorrect code, but given the declared spec treats "same pinnedNLI" as mandatory for the primary arm (not just the `A7_NoNLI` control), it merits an explicit assertion that `nli is not None` before proceeding, mirroring the existing assertion for `cross`, so a corrupted/missing NLI model triggers a hard failure instead of unannounced degradation into the no-NLI case under the "A7" label.

### Everything else checked — no issues found
- Evidence selection/dedup (`select_requirement_evidence`), packing (`pack`), hypothesis/status polarity swap for exclusions, and competition-rank tie handling all match their declared semantics and the provided unit tests.
- `order_candidates`'s missing-CE-logit handling assigns a uniform midpoint neutral rank to all unscored candidates per component (consistent with `config.json`'s `missing_requirement_rank: midpoint_of_scoreable_candidates`); since it's applied uniformly across the pool for that component it doesn't bias relative ordering.
- Freeze/protect sequencing (`freeze.py`) and the isolation allowlist/forbidden-path logic (`isolation.py`) correctly block gold/evaluation reads and writes outside V2 during prediction, and correctly protect frozen files (including recursively-captured `retrieval/*.json` trace files) during evaluation. The import audit hook blocks `entity_evaluation`/`strict_targets` regardless of import mechanism.
- `scripts/predict.py` / `scripts/evaluate.py` correctly order freeze-verification before any evaluation import, and `evaluate.py` re-verifies the freeze hash after writing metrics (entity_evaluation.py:227-228), guaranteeing deterministic, tamper-evident separation between prediction and evaluation.