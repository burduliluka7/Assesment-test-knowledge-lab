"""Evaluation-only entities, strict archived claim alignment and source overlap."""
from .evaluation.claim_alignment import benchmark_claims, SourceIndex
from .diffusion_pipeline import evaluate_ranking, summarize_diffusion
from .target_mapping import TargetResolver
import numpy as np

TOP_KS = [1, 5, 10, 20, 50, 100, 200]


def evidence_export(ranked, scorer, nodes, question_vector):
    """Only returned entities and question-selected evidence; no expected claims."""
    exported = []
    for row in ranked[:10]:
        ident = row['entity_id']; detail = scorer.details[ident]; pool = scorer.pools[ident]
        lookup = {i['node_id']:i for xs in pool['fields'].values() for i in xs}
        selected = {e['node_id']:e for r in detail['requirements'] for e in r['selected_evidence']}
        evidence = []
        for nid, e in sorted(selected.items()):
            item = lookup[nid]; n = nodes[nid]
            sources = set(n.get('provenance', {}).get('articles', []))
            links = item.get('source_links', [])
            for link in links:
                prov = link.get('provenance') or {}
                sources.update(prov.get('articles', []))
                if prov.get('article_id') is not None: sources.add(prov['article_id'])
            evidence.append(dict(e, node_type=n['type'], claim_text=n['surface_form'] if n['type']=='claim' else None,
                                 article_ids=sorted(sources), provenance=n.get('provenance', {}),
                                 graph_path=item.get('path'), source_links=links,
                                 question_similarity=float(scorer.node_vectors[nid] @ question_vector)))
        exported.append(dict(entity_id=ident, name=row['name'], node_ids=row['node_ids'], evidence=evidence,
                             requirement_selection=detail['requirements']))
    return exported


def claim_metrics(question_id, truth, snap, returned_evidence, archive):
    expected = [r for r in benchmark_claims(truth) if r['question_id']==question_id and r['kind']=='claim']
    prior = {r['item_id']:r for r in archive['items']}
    source_index = SourceIndex(snap); resolver = TargetResolver(snap)
    all_source = {a for r in returned_evidence for e in r['evidence'] for a in e['article_ids']}
    rows = []
    for item in expected:
        old = prior.get(item['item_id'])
        valid = old is not None and old['ground_truth_claim']==item['ground_truth_claim'] and old['sources']==item['sources']
        refs = [source_index.resolve(s) for s in item['sources']]
        source_ids = {a for r in refs if r['status']=='matched' for a in r['article_ids']}
        alignment = old['node_ids'] if valid and old['alignment_status']=='confident_proxy' else []
        target = resolver.resolve(item['target_entity']) if item['target_entity'] else None
        matching = [r for r in returned_evidence if target and set(r['node_ids']) & set(target['resolved_node_ids'])]
        retrieved = {e['node_id'] for r in matching for e in r['evidence']}
        candidate_sources = {a for r in matching for e in r['evidence'] for a in e['article_ids']}
        rows.append(dict(item_id=item['item_id'], target_entity=item['target_entity'], expected_claim=item['ground_truth_claim'],
                         aligned_node_ids=alignment, retrieved_aligned=bool(set(alignment)&retrieved),
                         source_article_ids=sorted(source_ids), matched_source_status=refs,
                         source_overlap=sorted(source_ids&all_source), candidate_source_overlap=sorted(source_ids&candidate_sources)))
    alignable = sum(bool(r['aligned_node_ids']) for r in rows)
    found = sum(r['retrieved_aligned'] for r in rows)
    source_ids = {a for r in rows for a in r['source_article_ids']}
    overlap = source_ids & all_source
    candidate_source_items = [r for r in rows if r['source_article_ids']]
    return dict(expected_claims_count=len(rows), automatically_alignable_claims_count=alignable,
                retrieved_aligned_claims_count=found,
                strict_claim_recall=found/alignable if alignable else None,
                claim_evaluation_status='NOT_EVALUABLE_ALIGNMENT' if not alignable else 'EVALUABLE' if alignable==len(rows) else 'PARTIALLY_EVALUABLE',
                source_count=len(source_ids), source_overlap=sorted(overlap), source_hit=bool(overlap) if source_ids else None,
                provenance_overlap=len(overlap)/len(source_ids) if source_ids else None,
                candidate_source_hit=sum(bool(r['candidate_source_overlap']) for r in candidate_source_items)/len(candidate_source_items) if candidate_source_items else None,
                source_resolution_coverage=sum(bool(r['source_article_ids']) for r in rows)/len(rows) if rows else None,
                items=rows, interpretation='Source overlap is not entailment. Strict alignments reuse the frozen proxy; unevaluable recall is null.')


def evaluate_result(result, fine, resolutions):
    metric = evaluate_ranking(result['ranked'], resolutions, dict(diffusion_top_ks=TOP_KS))
    prior = set(fine['at_k']['10']['found_targets']); found = set(metric['at_k']['10']['found_targets'])
    return metric, dict(fine_hits=len(prior), retained_fine_hits=len(found&prior),
                        fine_hit_retention=len(found&prior)/len(prior) if prior else None,
                        hierarchy_only_recovery=sorted(found-prior), lost_fine_hits=sorted(prior-found))


def summarize(rows):
    result = summarize_diffusion(rows, dict(diffusion_top_ks=TOP_KS))
    def mean(key, parent):
        values = [r[parent].get(key) for r in rows if r[parent].get(key) is not None]
        return float(np.mean(values)) if values else None
    fine = sum(r['retention']['fine_hits'] for r in rows)
    retained = sum(r['retention']['retained_fine_hits'] for r in rows)
    result['retention'] = dict(fine_hits=fine, retained_fine_hits=retained, fine_hit_retention=retained/fine if fine else None,
                               additional_hits=sum(len(r['retention']['hierarchy_only_recovery']) for r in rows))
    result['work'] = {k:mean(k, 'cost') for k in rows[0]['cost']}
    result['claims'] = dict(expected=sum(r['claim_metrics']['expected_claims_count'] for r in rows),
                           alignable=sum(r['claim_metrics']['automatically_alignable_claims_count'] for r in rows),
                           retrieved=sum(r['claim_metrics']['retrieved_aligned_claims_count'] for r in rows),
                           strict_claim_recall=mean('strict_claim_recall', 'claim_metrics'), source_hit=mean('source_hit', 'claim_metrics'),
                           provenance_overlap=mean('provenance_overlap', 'claim_metrics'), candidate_source_hit=mean('candidate_source_hit', 'claim_metrics'))
    result['status_counts'] = {s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})}
    return result
