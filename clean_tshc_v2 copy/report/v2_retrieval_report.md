# V2 entity-centric retrieval — DEVELOPMENT results

The detailed LaTeX report is authored by Claude through the user-requested Ultralight workflow. This summary is generated directly from frozen evaluation artifacts.

Direct recall counts EXACT/ALIAS target occurrences only. All-target denominator: 63; mapped denominator: 49. All results use14 Type-A questions;18 questions have predictions.

| Stage | Candidate@100 /mapped | Candidate@500 /mapped | Union /mapped | Avg pool | R@1 | R@5 | R@10 | R@25 | R@100 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 | 10 | 19 | 19 | 489.2 | 0.032 | 0.048 | 0.048 | 0.048 | 0.159 | 0.167 |
| TypedRaw | 10 | 19 | 19 | 489.2 | 0.032 | 0.048 | 0.048 | 0.048 | 0.159 | 0.167 |
| ParsedOnly | 6 | 19 | 19 | 489.2 | 0.000 | 0.032 | 0.048 | 0.048 | 0.095 | 0.084 |
| A2 | 20 | 33 | 43 | 913.6 | 0.032 | 0.048 | 0.079 | 0.095 | 0.317 | 0.187 |
| A3 | 31 | 43 | 45 | 1108.9 | 0.032 | 0.063 | 0.079 | 0.159 | 0.492 | 0.198 |
| A4 | 35 | 43 | 46 | 1295.6 | 0.032 | 0.063 | 0.079 | 0.159 | 0.556 | 0.201 |
| A5 | 37 | 44 | 46 | 1295.6 | 0.000 | 0.063 | 0.095 | 0.175 | 0.587 | 0.155 |
| WithoutEvidenceRRF | 29 | 45 | 46 | 1149.5 | 0.000 | 0.048 | 0.048 | 0.127 | 0.460 | 0.128 |
| WithoutGraphRRF | 26 | 44 | 45 | 1108.9 | 0.000 | 0.063 | 0.063 | 0.127 | 0.413 | 0.137 |
| A0 | 2 | 6 | 6 | 500.0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.032 | 0.004 |
| A6 | 37 | 44 | 44 | 489.2 | 0.016 | 0.079 | 0.143 | 0.222 | 0.524 | 0.148 |
| A6_100 | 37 | 37 | 37 | 100.0 | 0.016 | 0.095 | 0.190 | 0.286 | 0.587 | 0.167 |
| A6_200 | 37 | 43 | 43 | 200.0 | 0.016 | 0.079 | 0.143 | 0.254 | 0.540 | 0.152 |
| A6_500 | 37 | 44 | 44 | 489.2 | 0.016 | 0.079 | 0.143 | 0.222 | 0.524 | 0.148 |
| A7 | 37 | 44 | 44 | 489.2 | 0.016 | 0.079 | 0.159 | 0.222 | 0.524 | 0.152 |

Candidate metrics for A6/A7 use the actual pre-rerank input pool, not the reranked output. Larger union recovery is reported separately from its top500 ceiling. Conditional rank averages exclude misses; the JSON reports missing counts and a fixed missing-rank penalty.

## Conditional reranking

{
  "A6_100": {
    "pool_size": 100,
    "seen_target_occurrences": 37,
    "downward": 20,
    "upward": 17,
    "top10_after": 12,
    "new_top10": 7,
    "recall10_conditional": 0.32432432432432434,
    "median_before": 36,
    "median_after": 27,
    "strongest_improvements": [
      {
        "question_id": "Q12",
        "target": "CHGNet",
        "before": 94,
        "after": 20
      },
      {
        "question_id": "Q1",
        "target": "CHGNet",
        "before": 72,
        "after": 15
      },
      {
        "question_id": "Q12",
        "target": "DeePMD",
        "before": 62,
        "after": 8
      },
      {
        "question_id": "Q12",
        "target": "GAP",
        "before": 50,
        "after": 4
      },
      {
        "question_id": "Q12",
        "target": "EquiformerV2",
        "before": 57,
        "after": 19
      }
    ],
    "strongest_degradations": [
      {
        "question_id": "Q12",
        "target": "MACE",
        "before": 42,
        "after": 86
      },
      {
        "question_id": "Q12",
        "target": "eSEN",
        "before": 12,
        "after": 53
      },
      {
        "question_id": "Q14",
        "target": "WBM",
        "before": 29,
        "after": 68
      },
      {
        "question_id": "Q1",
        "target": "MACE",
        "before": 39,
        "after": 74
      },
      {
        "question_id": "Q12",
        "target": "PET",
        "before": 61,
        "after": 89
      }
    ]
  },
  "A6_200": {
    "pool_size": 200,
    "seen_target_occurrences": 43,
    "downward": 24,
    "upward": 19,
    "top10_after": 9,
    "new_top10": 5,
    "recall10_conditional": 0.20930232558139536,
    "median_before": 39,
    "median_after": 38,
    "strongest_improvements": [
      {
        "question_id": "Q13",
        "target": "CGCNN",
        "before": 118,
        "after": 15
      },
      {
        "question_id": "Q10",
        "target": "SchNet",
        "before": 156,
        "after": 56
      },
      {
        "question_id": "Q13",
        "target": "MACE",
        "before": 136,
        "after": 40
      },
      {
        "question_id": "Q13",
        "target": "CHGNet",
        "before": 113,
        "after": 26
      },
      {
        "question_id": "Q12",
        "target": "CHGNet",
        "before": 94,
        "after": 26
      }
    ],
    "strongest_degradations": [
      {
        "question_id": "Q12",
        "target": "MACE",
        "before": 42,
        "after": 157
      },
      {
        "question_id": "Q12",
        "target": "PET",
        "before": 61,
        "after": 161
      },
      {
        "question_id": "Q7",
        "target": "symbolic regression",
        "before": 48,
        "after": 140
      },
      {
        "question_id": "Q1",
        "target": "MACE",
        "before": 39,
        "after": 109
      },
      {
        "question_id": "Q4",
        "target": "Hessian training",
        "before": 44,
        "after": 113
      }
    ]
  },
  "A6": {
    "pool_size": 500,
    "seen_target_occurrences": 44,
    "downward": 26,
    "upward": 18,
    "top10_after": 9,
    "new_top10": 5,
    "recall10_conditional": 0.20454545454545456,
    "median_before": 39.0,
    "median_after": 50.5,
    "strongest_improvements": [
      {
        "question_id": "Q9",
        "target": "NequIP",
        "before": 256,
        "after": 60
      },
      {
        "question_id": "Q13",
        "target": "CGCNN",
        "before": 118,
        "after": 17
      },
      {
        "question_id": "Q13",
        "target": "MACE",
        "before": 136,
        "after": 50
      },
      {
        "question_id": "Q13",
        "target": "CHGNet",
        "before": 113,
        "after": 35
      },
      {
        "question_id": "Q10",
        "target": "SchNet",
        "before": 156,
        "after": 87
      }
    ],
    "strongest_degradations": [
      {
        "question_id": "Q12",
        "target": "MACE",
        "before": 42,
        "after": 302
      },
      {
        "question_id": "Q12",
        "target": "PET",
        "before": 61,
        "after": 317
      },
      {
        "question_id": "Q7",
        "target": "symbolic regression",
        "before": 48,
        "after": 276
      },
      {
        "question_id": "Q1",
        "target": "ACE",
        "before": 159,
        "after": 370
      },
      {
        "question_id": "Q4",
        "target": "Hessian training",
        "before": 44,
        "after": 229
      }
    ]
  },
  "A7": {
    "pool_size": 500,
    "seen_target_occurrences": 44,
    "downward": 26,
    "upward": 18,
    "top10_after": 10,
    "new_top10": 6,
    "recall10_conditional": 0.22727272727272727,
    "median_before": 39.0,
    "median_after": 50.5,
    "strongest_improvements": [
      {
        "question_id": "Q9",
        "target": "NequIP",
        "before": 256,
        "after": 60
      },
      {
        "question_id": "Q13",
        "target": "CGCNN",
        "before": 118,
        "after": 17
      },
      {
        "question_id": "Q13",
        "target": "MACE",
        "before": 136,
        "after": 50
      },
      {
        "question_id": "Q13",
        "target": "CHGNet",
        "before": 113,
        "after": 35
      },
      {
        "question_id": "Q10",
        "target": "SchNet",
        "before": 156,
        "after": 87
      }
    ],
    "strongest_degradations": [
      {
        "question_id": "Q12",
        "target": "MACE",
        "before": 42,
        "after": 302
      },
      {
        "question_id": "Q12",
        "target": "PET",
        "before": 61,
        "after": 317
      },
      {
        "question_id": "Q7",
        "target": "symbolic regression",
        "before": 48,
        "after": 276
      },
      {
        "question_id": "Q1",
        "target": "ACE",
        "before": 159,
        "after": 370
      },
      {
        "question_id": "Q4",
        "target": "Hessian training",
        "before": 44,
        "after": 229
      }
    ]
  }
}

## Efficiency

```json
{
  "prediction_phase_wall_seconds": 865.235102,
  "query_wall_seconds": 745.2575133000792,
  "dense_comparisons": 295920,
  "evidence_generation_comparisons": 153864,
  "bm25_document_term_lookups_exact": 1637900,
  "cross_encoder_pairs": 8849,
  "cross_encoder_actual_inference_pairs": 8849,
  "verification_candidates": 360,
  "verification_components": 800,
  "nli_pairs": 376,
  "requirement_evidence_comparisons": 170960,
  "verification_status_counts": {
    "SUPPORTED": 71,
    "NOT_SUPPORTED_BY_AVAILABLE_EVIDENCE": 694,
    "CONTRADICTED": 35
  },
  "prefix_timing_estimates": {
    "100": 170.12473662471893,
    "200": 325.14423435000936,
    "500": 675.0066569996998
  },
  "timing_note": "CPU descriptive wall times; prefix controls share scores from500 and are not separately timed; counters are logical work, not production latency."
}
```

## Failures

```json
{
  "counts": {
    "IN_POOL_CROSS_ENCODER_FAILED": 34,
    "RECOVERED_AT_10": 10,
    "GROUND_TRUTH_MAPPING_UNCERTAIN": 14,
    "IN_POOL_RRF_FAILED": 2,
    "WRONG_TYPE_FILTER": 3
  },
  "secondary_counts": {
    "NO_NAME_MATCH": 46,
    "NO_RELEVANT_EVIDENCE": 8
  }
}
```

## Reproducibility and interpretation

See README.md for commands; config.json for every fixed parameter; outputs/failure_traces.json for all target traces; outputs/evaluation_only/positive_controls.json for isolated oracle tests. The previous experiment is retained under archive/flat_20261001. The experiment does not route through or modify the hierarchy.

Prediction SHA-256: `5bc46c27e260572cc33630ff59b3dd142cc37ca5ef66d13eb77f29b1a2ad0eda`.

V1 was already dirty at task start. The user approved preserving it; the final audit must show zero additional V1 changes, including an unchanged starting git diff. No GitHub operations are performed.

## Final verification

42 tests passed. All frozen prediction/source/input and cached model-file hashes verify. The full cached replay produces byte-identical predictions. Prediction freeze precedes evaluator import and gold loading. **V1 has zero new changes**: all 1,056 file hashes, its starting git diff, and its starting git status match. Pre-existing V1 work is preserved as explicitly requested. No GitHub changes were made. Claude Opus 5 authored and revised the detailed [LaTeX report](v2_entity_retrieval_report.tex), with Codex factual corrections and compilation to [PDF](v2_entity_retrieval_report.pdf). See [file inventory](../outputs/audit/files_changed.json) for the active V2 changes and preserved archive.
