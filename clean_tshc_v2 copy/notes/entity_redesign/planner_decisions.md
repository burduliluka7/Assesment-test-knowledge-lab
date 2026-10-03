# Ultralight planning decisions

The official helper completed successfully with Claude Opus 5 through the user's first-party subscription. Full response is planner_review.md; original JSON is retained.

Adopted: infer roles from relation/type signatures rather than member position; do not call role inference annotated direction; keep evidence separate; preserve candidate and graph-path provenance; separate candidate recall from conditional reranking; preserve year visibility and raw query; missing evidence is not contradiction.

Checked and rejected several suggestions against the actual request/schema: last_seen_year means last observed, not valid-until, so old entities do not disappear; the source does contain explicit alias lists on some nodes; membership fields do not have the claimed general confidence/year fields; method/dataset homonyms remain separate compatible groups; hard answer-family filtering is required for the primary channel; top500 relevance reranking is retained rather than restricting the primary run back to100. Ambiguous extends edges are omitted, not made bidirectional. Graph expansion uses only evidence-to-article-to-presented/cited-entity templates, with citations explicitly structural only. No benchmark answers were used to settle these choices.
