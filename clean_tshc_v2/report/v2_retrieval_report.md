# V2: entity-centric, requirement-aware retrieval

Frozen pretrained models; no benchmark training or weight/threshold fitting. Questions are prediction inputs; target annotations are evaluation-only. These are development diagnostics.

[Detailed method/results PDF](v2_entity_retrieval_report.pdf) | [LaTeX](v2_entity_retrieval_report.tex)

| Arm | Candidate@100 /49 | Candidate@500 /49 | Union /49 | R@1 | R@5 | R@10 | R@25 | R@50 | R@100 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 | 10 | 19 | 19 | 0.0317 | 0.0476 | 0.0476 | 0.0476 | 0.0794 | 0.1587 | 0.1666 |
| TypedRaw | 10 | 19 | 19 | 0.0317 | 0.0476 | 0.0476 | 0.0476 | 0.0794 | 0.1587 | 0.1667 |
| ParsedOnly | 6 | 19 | 19 | 0.0000 | 0.0317 | 0.0476 | 0.0476 | 0.0635 | 0.0952 | 0.0843 |
| A2 | 20 | 33 | 43 | 0.0317 | 0.0476 | 0.0794 | 0.0952 | 0.1746 | 0.3175 | 0.1872 |
| A3 | 31 | 43 | 45 | 0.0317 | 0.0635 | 0.0794 | 0.1587 | 0.2540 | 0.4921 | 0.1979 |
| A4 | 35 | 43 | 46 | 0.0317 | 0.0635 | 0.0794 | 0.1587 | 0.2698 | 0.5556 | 0.2013 |
| A5 | 37 | 44 | 46 | 0.0000 | 0.0635 | 0.0952 | 0.1746 | 0.4603 | 0.5873 | 0.1554 |
| WithoutEvidenceRRF | 29 | 45 | 46 | 0.0000 | 0.0476 | 0.0476 | 0.1270 | 0.2698 | 0.4603 | 0.1278 |
| WithoutGraphRRF | 26 | 44 | 45 | 0.0000 | 0.0635 | 0.0635 | 0.1270 | 0.2857 | 0.4127 | 0.1374 |
| A0 | 2 | 6 | 6 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0159 | 0.0317 | 0.0044 |
| A6 | 37 | 44 | 44 | 0.0159 | 0.0794 | 0.1429 | 0.2222 | 0.3492 | 0.5238 | 0.1480 |
| A6_100 | 37 | 37 | 37 | 0.0159 | 0.0952 | 0.1905 | 0.2857 | 0.3968 | 0.5873 | 0.1669 |
| A6_200 | 37 | 43 | 43 | 0.0159 | 0.0794 | 0.1429 | 0.2540 | 0.4127 | 0.5397 | 0.1523 |
| A6_500 | 37 | 44 | 44 | 0.0159 | 0.0794 | 0.1429 | 0.2222 | 0.3492 | 0.5238 | 0.1480 |
| A7 | 37 | 44 | 44 | 0.0000 | 0.0794 | 0.1429 | 0.2222 | 0.3651 | 0.4603 | 0.1102 |
| A7_NoNLI | 37 | 44 | 44 | 0.0000 | 0.0317 | 0.0317 | 0.1905 | 0.2698 | 0.4444 | 0.0651 |
| A7_SingleEvidence | 37 | 44 | 44 | 0.0000 | 0.0317 | 0.0952 | 0.1746 | 0.2698 | 0.3333 | 0.0801 |

Final recall denominator is all target occurrences; candidate denominator is directly mapped occurrences. A6/A7 candidate union refers to the actual500-prefix, while A5 reports the full retrieval union. Tails remain in final rankings.

## Conditional movement

```json
{
  "seen": 44,
  "upward": 18,
  "downward": 25,
  "unchanged": 1,
  "median_before": 50.5,
  "median_after": 46.5,
  "new_top10": 4,
  "removed_top10": 4,
  "records": [
    {
      "question_id": "Q1",
      "target": "MACE",
      "before": 174,
      "after": 43
    },
    {
      "question_id": "Q1",
      "target": "M3GNet",
      "before": 51,
      "after": 34
    },
    {
      "question_id": "Q1",
      "target": "CHGNet",
      "before": 31,
      "after": 107
    },
    {
      "question_id": "Q1",
      "target": "GAP",
      "before": 3,
      "after": 42
    },
    {
      "question_id": "Q1",
      "target": "NequIP",
      "before": 56,
      "after": 51
    },
    {
      "question_id": "Q1",
      "target": "ACE",
      "before": 370,
      "after": 415
    },
    {
      "question_id": "Q1",
      "target": "EquiformerV2",
      "before": 54,
      "after": 30
    },
    {
      "question_id": "Q2",
      "target": "DeepH",
      "before": 15,
      "after": 7
    },
    {
      "question_id": "Q2",
      "target": "D4FT",
      "before": 10,
      "after": 31
    },
    {
      "question_id": "Q3",
      "target": "DeepH-E3",
      "before": 7,
      "after": 5
    },
    {
      "question_id": "Q4",
      "target": "Hessian training",
      "before": 229,
      "after": 195
    },
    {
      "question_id": "Q4",
      "target": "VGNN",
      "before": 69,
      "after": 441
    },
    {
      "question_id": "Q6",
      "target": "LiFlow",
      "before": 42,
      "after": 5
    },
    {
      "question_id": "Q7",
      "target": "symbolic regression",
      "before": 276,
      "after": 495
    },
    {
      "question_id": "Q8",
      "target": "ConvLSTM",
      "before": 6,
      "after": 4
    },
    {
      "question_id": "Q9",
      "target": "NequIP",
      "before": 60,
      "after": 54
    },
    {
      "question_id": "Q10",
      "target": "SchNet",
      "before": 87,
      "after": 441
    },
    {
      "question_id": "Q12",
      "target": "MACE",
      "before": 302,
      "after": 490
    },
    {
      "question_id": "Q12",
      "target": "NequIP",
      "before": 65,
      "after": 486
    },
    {
      "question_id": "Q12",
      "target": "CHGNet",
      "before": 38,
      "after": 92
    },
    {
      "question_id": "Q12",
      "target": "M3GNet",
      "before": 108,
      "after": 420
    },
    {
      "question_id": "Q12",
      "target": "SevenNet",
      "before": 50,
      "after": 50
    },
    {
      "question_id": "Q12",
      "target": "GAP",
      "before": 5,
      "after": 7
    },
    {
      "question_id": "Q12",
      "target": "DeePMD",
      "before": 20,
      "after": 8
    },
    {
      "question_id": "Q12",
      "target": "MatterSim",
      "before": 24,
      "after": 12
    },
    {
      "question_id": "Q12",
      "target": "Orb",
      "before": 52,
      "after": 100
    },
    {
      "question_id": "Q12",
      "target": "eSEN",
      "before": 99,
      "after": 424
    },
    {
      "question_id": "Q12",
      "target": "PET",
      "before": 317,
      "after": 498
    },
    {
      "question_id": "Q12",
      "target": "UMA",
      "before": 161,
      "after": 67
    },
    {
      "question_id": "Q12",
      "target": "EquiformerV2",
      "before": 37,
      "after": 419
    },
    {
      "question_id": "Q13",
      "target": "CGCNN",
      "before": 17,
      "after": 22
    },
    {
      "question_id": "Q13",
      "target": "SchNet",
      "before": 198,
      "after": 447
    },
    {
      "question_id": "Q13",
      "target": "M3GNet",
      "before": 142,
      "after": 39
    },
    {
      "question_id": "Q13",
      "target": "NequIP",
      "before": 121,
      "after": 435
    },
    {
      "question_id": "Q13",
      "target": "MACE",
      "before": 50,
      "after": 462
    },
    {
      "question_id": "Q13",
      "target": "CHGNet",
      "before": 35,
      "after": 80
    },
    {
      "question_id": "Q14",
      "target": "Matbench Discovery",
      "before": 1,
      "after": 3
    },
    {
      "question_id": "Q14",
      "target": "Matbench",
      "before": 3,
      "after": 32
    },
    {
      "question_id": "Q14",
      "target": "Materials Project",
      "before": 49,
      "after": 2
    },
    {
      "question_id": "Q14",
      "target": "MPtrj",
      "before": 16,
      "after": 39
    },
    {
      "question_id": "Q14",
      "target": "OQMD",
      "before": 67,
      "after": 19
    },
    {
      "question_id": "Q14",
      "target": "GNoME",
      "before": 10,
      "after": 6
    },
    {
      "question_id": "Q14",
      "target": "WBM",
      "before": 86,
      "after": 14
    },
    {
      "question_id": "Q14",
      "target": "OMat24",
      "before": 4,
      "after": 17
    }
  ]
}
```

## Requirement diagnostics

```json
{
  "statuses": {
    "SUPPORTED": 664,
    "NOT_SUPPORTED_BY_AVAILABLE_EVIDENCE": 18369,
    "CONTRADICTED": 1316
  },
  "tiers": {
    "1": 268,
    "2": 7564,
    "0": 61,
    "3": 956
  },
  "missing_evidence": 13558,
  "multiple_selected_evidence": 4839,
  "multiple_packed_nli_evidence": 4839,
  "no_packed_nli_evidence": 13558,
  "imputed_relevance_ranks": 13558,
  "top3_vs_top1_status_differences": 1117,
  "candidates": 8849,
  "components": 20349
}
```

## Failure categories

```json
{
  "counts": {
    "REQUIREMENT_RANKER_FAILED": 22,
    "REQUIREMENT_EVIDENCE_MISSING": 7,
    "RECOVERED_AT_10": 9,
    "GROUND_TRUTH_MAPPING_UNCERTAIN": 14,
    "CONTRADICTION_PROXY_FAILURE": 6,
    "OUTSIDE_RERANK_POOL": 2,
    "WRONG_TYPE_FILTER": 3
  },
  "secondary_counts": {
    "NO_NAME_MATCH": 46,
    "NO_RELEVANT_EVIDENCE": 8
  }
}
```

## Efficiency

```json
{
  "dense_comparisons": 295920,
  "evidence_comparisons": 153864,
  "cross_encoder_pairs": 8849,
  "cross_encoder_inference_pairs": 1500,
  "cross_encoder_wall_seconds": 102.24639819993172,
  "total_wall_seconds": 471.1076621999964,
  "bm25_document_term_lookups_exact": 1637900,
  "requirement_arms": {
    "A7": {
      "requirement_dense_comparisons": 175234,
      "requirement_cross_encoder_pairs": 6791,
      "actual_requirement_cross_encoder_pairs": 1422,
      "nli_pairs": 6791,
      "actual_nli_pairs": 1422,
      "actual_wall_seconds": 232.66297929995926
    },
    "A7_SingleEvidence": {
      "requirement_dense_comparisons": 175234,
      "requirement_cross_encoder_pairs": 6791,
      "actual_requirement_cross_encoder_pairs": 1422,
      "nli_pairs": 6791,
      "actual_nli_pairs": 1422,
      "actual_wall_seconds": 119.58569269999862
    }
  },
  "wall_note": "Completed predictor invocation with mixed cold/cached inference; interrupted pre-evaluation invocations are not included. Logical totals describe all pairs, actual totals only cache misses. Single CPU run descriptive time; logical comparisons are not production latency. NoNLI reuses primary relevance scores;100/200 controls reuse500-pool global scores.",
  "prediction_process": {
    "exit_code": 0,
    "wall_seconds": 601.6485134999966
  }
}
```

Prediction SHA-256: `2d7485f1d0ecfb541d1d83413e2ac1d961519ce9c1e0870dbd74311462918570`.

Full candidate traces are in `artifacts/prediction/retrieval/requirement_ranking/`. Scored target traces are evaluation-only in `artifacts/evaluation/failure_traces.json`. No model or ranking policy was changed after evaluation.
