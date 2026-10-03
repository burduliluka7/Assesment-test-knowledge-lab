## What the data actually supports (verified)

From `data/tkh_collection10.json` (schema/meta only):

- **Node**: `id`, `type`, `surface_form`, `year`, `origin_year`, `first_seen_year`, `last_seen_year`, `provenance{source, collection_id, articles[], arxiv_id, doi, status, ingested_at, extracted_at}`, `confidence`.
- **Hyperedge**: `id`, `relation_type`, `members[]`, `year`, `origin_year`, `first_seen_year`, `last_seen_year`, `provenance`, `confidence`.
- **Node types**: article, author, cited_work, claim, component, dataset, future_topic, method, metric, problem, task, technique.
- **Relations**: addresses, authored_by, cites, claims, evaluated_on, extends, presents, proposes_future_work, solves, uses_component, uses_technique.
- `meta.date_semantics` documents `node_year/node_origin_year/node_first_seen_year/node_last_seen_year/edge_year/provenance_article_year` — Codex must read it and quote it rather than assume.

**Decisive negative finding: hyperedges carry no role labels.** `members` is a flat list; there is no head/tail/role/qualifier/evidence-text field. Sampled edges put a single typed node first (`meth_00030` first in addresses/evaluated_on/extends/uses_component), so position *looks* conventional, but that is an unverified regularity, not schema. **So "direct explicit roles" as proposed is not available.** Replace it with *type-signature-derived roles*, and keep position as a logged diagnostic only.

## Role derivation (replaces "explicit roles")

For each content edge, define role by relation + member type, never by position:

| relation | subject role types | object role types |
|---|---|---|
| uses_component | method, technique | component |
| uses_technique | method, technique | technique |
| evaluated_on | method, technique, component | dataset, metric |
| addresses / solves | method, technique, component | task, problem |
| presents | article | method/technique/component/dataset/metric/cited_work |
| claims | claim | any entity |
| extends | method/technique/component/cited_work | method/technique/component/cited_work |
| proposes_future_work | article/method | future_topic |

Rules: (1) An edge yields a role assignment only if exactly one member matches the subject-type set; otherwise mark `role_ambiguous=true` and treat the edge as **co-membership** (current V2 semantics), not directional. (2) For symmetric-typed relations (`extends`, `uses_technique` method→technique where both sides can be technique), emit both directions with `direction=unresolved` and record `member_position_first` as a logged field; do not let it change scoring in the primary run. (3) Never relabel `support_kind`: keep the existing honest vocabulary (`low_arity_incidence_not_directional_entailment`, `literal_name_match_not_entailment`, `provenance_title_only`) and add `type_signature_role_not_annotated`.

## Canonical grouping (defensible, snapshot-valid)

New module `src/tkh_abstraction_v2/entities.py`.

Normalization (`normalize_name`): NFKC → strip → collapse internal whitespace → casefold → strip a single trailing `.`/`,`/`:`. **Nothing else.** No stemming, no hyphen/space equivalence, no parenthetical/acronym expansion, no plural folding, no edit distance. Each is an invented alias and is out of scope by your constraint; log counts of pairs that *would* have merged under each rejected rule as an appendix table instead of applying them.

Group key = `normalize_name(surface_form)` over nodes whose `type ∈ {method, technique, component, cited_work, dataset, metric, article}`. Keep **one group per key** with `member_types` as a multiset — do not merge-or-split by family at grouping time (cited_work legitimately belongs to two families; family eligibility is computed downstream as the union over member types). Store: `group_id` (stable `g_` + first 12 hex of sha256 of the key, so IDs are content-derived and reproducible), `display_name` (the surface form of the lowest `node_id` among members, deterministic), `member_node_ids` sorted, `member_types`, `first_seen_year = min`, `last_seen_year = max`, article provenance union, and `heterogeneous=True` when `member_types` spans more than one family partition.

Snapshot validity: a node is in-snapshot iff `first_seen_year <= cfg.snapshot`, with `last_seen_year` handling driven by `meta.date_semantics` (if it means "still present", require `last_seen_year >= snapshot`). Filter nodes **before** grouping, then drop empty groups, so groups are snapshot-pure. Record dropped counts. Prediction unit is the group; output must also carry `member_node_ids` so a node-level evaluator can map either way.

## Evidence items (separate retrieval units)

`src/tkh_abstraction_v2/evidence.py` produces one item per (group, evidence) pair, each with `group_id`, `kind`, `text`, `source_edge_ids`, `source_node_ids`, `provenance`, `hop_count`, `support_kind`. Kinds:

1. `role_edge` — one item per typed role assignment (1-hop), text = `"<relation> <role>: <other surface_form>"`.
2. `co_membership` — ambiguous/high-arity edges, arity logged, capped.
3. `name_mention` — exact bounded regex of the group's *literal member surface forms* in claim text (reuse the existing `len>=5 or (len>=3 and isupper())` guard from `candidate_cards.py:51-60`; extend the eligible type set to dataset/metric/article but keep the guard).
4. `source_title` — article titles from `provenance.articles` (title-only, no claim dump).
5. `two_hop_path` — see safety rules.

Unlike V1 cards, these are **not** capped into one blob; caps apply per-kind per-group for the card used by the cross-encoder, while the full item set feeds evidence-level dense retrieval.

## Graph safety rules (concrete, enforced in code + tests)

1. Max 2 hops, counted as edges traversed; `hop_count` stored per item.
2. Hop 1 must be a **non-ambiguous role edge**; ambiguous or `arity > direct_arity_cap` edges are terminal (cannot be extended).
3. Hop 2 must change relation type (no `uses_component → uses_component` chains) and must not revisit a node or group already on the path.
4. Allowed 2-hop templates only, whitelisted in config: `method -uses_component-> component -evaluated_on-> dataset`, `method -addresses-> task <-addresses- method`, `method -extends-> cited_work -presents- article`. Anything not whitelisted is dropped with a logged reason. No `authored_by`/`author` traversal ever; no `cites` as hop 2 from a `cites` hop 1.
5. Hub cap: any intermediate node with degree > `hub_degree_cap` (propose 50, tune on structure only, fixed before gold) cannot be an intermediate; log suppressed hubs.
6. Per-group cap: `two_hop_cap` (propose 8), selected deterministically by (template order, edge_id, node_id) — never by score.
7. Fan-in cap: a single intermediate contributes at most 2 items per group.
8. Snapshot purity: every node and edge on the path must be in-snapshot; path year = max of edge years and must be `<= snapshot`.
9. Transitive closure is forbidden: 2-hop items are labeled `support_kind='bounded_path_cooccurrence_not_entailment'` and never merged into `direct_claims`.
10. Determinism: all expansion iterates over sorted IDs; no dict-order dependence.

## Retrieval and fusion

Channels, each depth 500 over groups:
- **D1 dense(original question)** vs group card text.
- **D2 dense(structured_query)** — keep `question_understanding.representations` untouched; both representations retained per your constraint.
- **L1 BM25** independent index over group cards (reuse `ranking.BM25`).
- **L2 exact-name lexical**: score = guarded exact match of group member surface forms in the question (safe-name guard identical to evidence kind 3), binary + length tie-break. This is the main recall lever for named-entity questions.
- **E1 evidence-level dense**: encode evidence items, score group = **max** over its items; store argmax item id for explainability.
- **G1 graph expansion**: groups reachable by whitelisted bounded paths from seed groups matched by L2/top-k D1, ranked by (seed rank, template order, edge_id) — a deterministic, score-free channel.

Fusion: `rrf(..., k=60)` on the per-channel top-500 lists; then **union with a deterministic round-robin baseline** over channels so no channel can be starved by RRF ties. Ties broken by `node_id`/`group_id` ascending (already the convention in `ranking.rank`). Type-family filtering becomes a *soft prior* (`soft_type_prior=0.8`) applied after fusion, never a hard filter, because `answer_families` in `config.json` would hard-drop valid heterogeneous groups.

Rerank: freeze prefix 500, cross-encode top 100 via `models.RelevanceCrossEncoder` + `ranking.replace_prefix` (its exact-prefix assertion is the right guard — keep it). Card packing must keep identity first (`models.pack` enforces this) and now order blocks: identity → role_edge → name_mention → dataset/metric → two_hop → source_title.

## Ablations A0–A7

Implement as a single `--ablation` switch producing per-run `channels` config; each writes `predictions.jsonl` + `run_manifest.json` (config hash, seed, model revisions, data sha256, channel depths, code git sha). A0 dense only; A1 +typed/structured dense; A2 +L1/L2 union; A3 +evidence E1; A4 +G1; A5 RRF fusion of all; A6 cross-encoder on frozen 500 with rerank_k ∈ {100, 200} controls (200 is a *control*, not a tuning sweep — declare before running); A7 requirement verification (`ranking.requirement_score`) on top-20 only. Optional NLI: pinned local model, used **only** as a requirement-support signal, never as relevance; `models.RelevanceCrossEncoder.__init__` already refuses NLI heads for relevance — keep that check and add a mirrored one rejecting a regression head for the NLI slot. Absent evidence → `support=unknown`, never `contradiction`; make that an explicit enum with three values.

## Metrics that are actually meaningful

- **Candidate recall@{50,100,200,500}** at group level, plus `recall_ceiling` = fraction of questions whose gold group exists at all in the snapshot (report every downstream number against this ceiling).
- **Per-channel contribution**: unique-contribution recall (groups only this channel supplied) and marginal recall when the channel is removed from A5.
- **Conditional rerank metrics**: rerank MRR/Hit@1 computed **only** on the subset where the gold group is inside the frozen prefix, reported with subset size and alongside unconditional end-to-end Hit@1, so the conditional number can't be mistaken for system accuracy.
- **Rerank harm rate**: fraction where gold was in prefix and rank worsened.
- Report cost: cross-encoder amortized seconds (already emitted in `models.py:79-80`).

## Affected files

New: `entities.py`, `evidence.py`, `graph_expansion.py`, `fusion.py`, `scripts/run_ablation.py`, `tests/test_entities.py`, `tests/test_graph_safety.py`, `tests/test_fusion.py`.
Modified: `candidate_cards.py` (group-keyed cards, new field order), `ranking.py` (round-robin union, channel bookkeeping), `scripts/predict.py` (channel pipeline, manifest, freeze hash), `config.json` (channel depths, `hub_degree_cap`, `two_hop_cap`, whitelisted templates, `rerank_k` controls, `normalization: "nfkc_casefold_ws_only"`), `tests/test_invariants.py`, `scripts/finalize.py` (hash before evaluator import). Unchanged: `isolation.py`, `question_understanding.py`, `models.py` (verify `strict_targets.py`/`snapshots.py` — I was not permitted to read them; Codex must check whether they assume node-level prediction units).

## Test pitfalls to cover

Grouping is not transitive beyond exact keys (assert no pair in different groups has equal normalized names, and vice versa). Group IDs stable under node reordering and under input shuffling. Snapshot filter applied before grouping (a group whose only in-snapshot member is dropped must vanish entirely). Graph expansion: assert no path length 3, no author node on any path, no repeated relation type, cap enforcement, and that `two_hop` items never land in a `direct_claims`-equivalent field. Regex name matching must not match substrings (`(?<!\w)...(?!\w)` already) and must be tested against regex-metacharacter names (`re.escape` present) and short/generic names. RRF determinism under channel-order permutation. `replace_prefix` must raise on partial scoring. `pack` must raise rather than silently drop identity. Encoder cache key includes model fingerprint (`models.py:20`) — add revision, not just directory name, or ablations can collide in cache. Isolation: a test asserting that importing evaluation during prediction raises, and that writes outside V2 raise.

## Assumptions

Prediction/eval unit is the canonical group with node-ID expansion emitted; `last_seen_year` semantics per `meta.date_semantics`; `hub_degree_cap=50`, `two_hop_cap=8`, depth 500, `rrf_k=60` all frozen before any gold access; V1 directory and the dirty V1 reports untouched.

## Unresolved questions for you

1. Does the evaluator match on node IDs or names? If node IDs, does matching any group member count as correct (this materially changes recall)?
2. `last_seen_year` meaning — "last observed" vs "still valid"?
3. Are heterogeneous groups (same name as both dataset and metric) acceptable as a single prediction, or must they be split with a documented rule?
4. Confirm the whitelist of 2-hop templates is acceptable as declared-in-advance, since adding templates after seeing dev recall would be tuning.