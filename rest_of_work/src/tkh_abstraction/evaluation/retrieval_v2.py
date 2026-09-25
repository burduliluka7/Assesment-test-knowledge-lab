"""Evaluation-only judgments, with explicit incomplete-label denominators."""
import numpy as np

def expected_recall_evaluability(mappings):
    expected=[m for m in mappings if m['category']=='expected_targets']
    mapped=[m for m in expected if m['node_ids']]
    status='evaluable' if mapped else 'not_evaluable' if expected else 'not_applicable'
    reason=('At least one expected target has an accepted node mapping; recall uses only mapped targets.' if mapped else
        'No expected target has an accepted node mapping; mapped-target recall has a zero denominator and is null.' if expected else
        'No expected entity targets are supplied; entity recall is not applicable.')
    return dict(status=status,reason=reason,total_expected_targets=len(expected),mapped_expected_targets=len(mapped),
        excluded_targets=[dict(target=m['supplied_target'],mapping_status=m['status'],mapping_rule=m['mapping_rule']) for m in expected if not m['node_ids']])

def question_evaluability(questions,mappings):
    """Keep every CSV question, even when none of its targets can be resolved."""
    return [dict(question_id=q['question_id'],question_type=q['type'],
        expected_recall_evaluability=expected_recall_evaluability([m for m in mappings if m['question_id']==q['question_id']])) for q in questions]

def entity_metrics(returned,mappings,fp_annotation_available=True):
    rr=set(returned); expected=[m for m in mappings if m['category']=='expected_targets']; mapped=[m for m in expected if m['node_ids']]
    categories={c:{x for m in mappings if m['category']==c for x in m['node_ids']} for c in ['expected_targets','valid_but_unlisted','wrong_domain','outdated_as_main']}
    found=sum(bool(rr&set(m['node_ids'])) for m in mapped); n=len(returned)
    accepted=categories['expected_targets']|categories['valid_but_unlisted']
    return dict(expected_recall=found/len(mapped) if mapped else None,expected_found=found,mapped_expected=len(mapped),total_expected=len(expected),
        expected_recall_evaluability=expected_recall_evaluability(mappings),
        hit_rate=float(found>0) if mapped else None,strict_precision=len(rr&categories['expected_targets'])/n if n else 0.,
        extended_precision=(len(rr&accepted)/n if n else 0.) if fp_annotation_available else None,known_wrong_rate=(len(rr&categories['wrong_domain'])/n if n else 0.) if fp_annotation_available else None,
        outdated_main_rate=(len(rr&categories['outdated_as_main'])/n if n else 0.) if fp_annotation_available else None,returned=n,fp_annotation_available=fp_annotation_available,
        unjudged_rate=len(rr-set().union(*categories.values()))/n if n else 0.,
        all_expected_lower_bound=found/len(expected) if expected else None,
        judged_outputs={c:sorted(rr&ids) for c,ids in categories.items()},
        unjudged_ids=sorted(rr-set().union(*categories.values())),
        expected_ranks={m['supplied_target']:next((i+1 for i,x in enumerate(returned) if x in m['node_ids']),None) for m in mapped})

def claim_metrics(result,alignment_rows):
    claims=[r for r in alignment_rows if r['kind']=='claim']; mapped=[r for r in claims if r['node_ids']]
    retrieved=set(result['returned_ids']); sources={a for e in result['evidence'] for a in e['provenance'].get('articles',[])}
    sourced=[r for r in claims if r['source_article_ids']]
    found=sum(bool(retrieved&set(r['node_ids'])) for r in mapped)
    evidence=[r for r in alignment_rows if r['kind']=='evidence']; mapped_evidence=[r for r in evidence if r['node_ids']]
    return dict(claim_recall=found/len(mapped) if mapped else None,claims_found=found,mapped_claims=len(mapped),total_claims=len(claims),
        evidence_item_recall=sum(bool(retrieved&set(r['node_ids'])) for r in mapped_evidence)/len(mapped_evidence) if mapped_evidence else None,
        mapped_evidence_items=len(mapped_evidence),total_evidence_items=len(evidence),
        alignment_coverage=len(mapped)/len(claims) if claims else None,
        source_evidence_hit_rate=sum(bool(sources&set(r['source_article_ids'])) for r in sourced)/len(sourced) if sourced else None,
        source_evaluable_claims=len(sourced),source_mapping_coverage=len(sourced)/len(claims) if claims else None,
        interpretation='Source hit measures provenance overlap, not entailment of benchmark claims')

def summarize(records):
    keys=['expected_recall','hit_rate','strict_precision','extended_precision','known_wrong_rate','outdated_main_rate','expected_found',
        'claim_recall','evidence_item_recall','alignment_coverage','source_evidence_hit_rate','source_mapping_coverage','entity_comparisons','claim_comparisons','total_comparisons','returned','unjudged_rate','all_expected_lower_bound']
    groups={}
    for r in records:
        key=(r['variant'],r['cutoff'],r['method'],r['budget'],r['question_type']); groups.setdefault(key,[]).append(r)
    result=[]
    for (variant,cutoff,method,budget,typ),rows in groups.items():
        row=dict(variant=variant,cutoff=cutoff,method=method,budget=budget,question_subset=typ,questions=len(rows))
        row['expected_recall_excluded_questions']=[dict(question_id=r['question_id'],**r['expected_recall_evaluability']) for r in rows if r['expected_recall'] is None]
        row['fp_annotated_questions']=sum(r.get('fp_annotation_available',False) for r in rows)
        for key in keys:
            vals=[r[key] for r in rows if r.get(key) is not None]
            row[key]=float(np.mean(vals)) if vals else None; row[key+'_evaluable']=len(vals)
        row['mapping_coverage']=sum(r['mapped_expected'] for r in rows)/sum(r['total_expected'] for r in rows) if sum(r['total_expected'] for r in rows) else None
        result.append(row)
    return result

def contextual_metrics(result,resolutions,top_ks=(1,5,10,20,50,100),evaluation_k=10):
    """Distinct named targets, raw denominator retained; noncanonical scores separate."""
    rows=result['ranked']; candidate=set(result['candidate_entity_ids'][:100]); candidate_ranks={x:i+1 for i,x in enumerate(result['candidate_entity_ids'])}
    canonical=[]; representable=[r for r in resolutions if r['representable']]
    # Two supplied aliases for the same graph identity are one relevance unit.
    # Preserve the original label-instance denominator separately.
    for res in sorted((r for r in resolutions if r['canonical']),key=lambda r:r['label']):
        overlaps=[r for r in canonical if set(r['resolved_node_ids'])&set(res['resolved_node_ids'])]
        ids=set(res['resolved_node_ids']); labels=[res['label']]
        for other in overlaps:
            ids.update(other['resolved_node_ids']); labels.extend(other['supplied_labels']); canonical.remove(other)
        canonical.append(dict(res,label=min(labels),resolved_node_ids=sorted(ids),supplied_labels=sorted(labels)))
    def hit(res,returned):
        nodes={x for row in returned for x in row['node_ids']}
        if res['canonical']: return bool(nodes&set(res['resolved_node_ids']))
        if res['resolution_type']=='COMPOSITE': return res['representable'] and all(hit(c,returned) for c in res['component_resolutions'])
        if res['resolution_type']=='EVIDENCE_BACKED': return bool(nodes&set(res['evidence_node_ids']))
        return False
    def label_rank(res):
        if res['resolution_type']=='COMPOSITE':
            rr=[label_rank(c) for c in res['component_resolutions']]
            return max(rr) if rr and all(x is not None for x in rr) else None
        ids=res['evidence_node_ids'] if res['resolution_type']=='EVIDENCE_BACKED' else res['resolved_node_ids']
        return next((i+1 for i,row in enumerate(rows) if set(row['node_ids'])&set(ids)),None)
    ranks={r['label']:label_rank(r) for r in canonical}
    count=len(canonical); metrics={}; outcomes=[]
    for k in top_ks:
        selected=rows[:k]; found=[r['label'] for r in canonical if ranks[r['label']] is not None and ranks[r['label']]<=k]
        # One credit per distinct expected target; duplicate mentions/alias names cannot add precision credit.
        strict=len(found)/k
        from ..target_mapping import normalize
        unique_names={normalize(row['name']) for row in selected}
        metrics[str(k)]=dict(hit=float(bool(found)) if count else None,recall=len(found)/count if count else None,
            precision=strict,found_targets=found,found=len(found),raw_found=sum(hit(r,selected) for r in resolutions if r['canonical']),
            raw_end_to_end_lower_bound=sum(hit(r,selected) for r in resolutions if r['canonical'])/len(resolutions) if resolutions else None,
            representable_recall=sum(hit(r,selected) for r in representable)/len(representable) if representable else None,
            duplicate_slot_fraction=(len(selected)-len(unique_names))/len(selected) if selected else 0.)
    candidate_rows=[r for r in rows if r['entity_id'] in candidate]
    candidate_found=sum(hit(r,candidate_rows) for r in canonical)
    for res in resolutions:
        rank=label_rank(res) if res['representable'] else None; in_candidate=hit(res,candidate_rows)
        outcome=('AMBIGUOUS_MAPPING' if res['resolution_type']=='AMBIGUOUS' else 'UNREPRESENTABLE' if not res['representable'] else
            'RETRIEVED' if rank is not None and rank<=evaluation_k else 'CANDIDATE_BUT_BELOW_K' if in_candidate else 'RESOLVED_BUT_NOT_CANDIDATE')
        pool_hit=hit(res,rows)
        original_rank=min((candidate_ranks.get(row['entity_id'],float('inf')) for row in rows if hit(res,[row])),default=None)
        outcomes.append(dict(label=res['label'],resolution_type=res['resolution_type'],canonical=res['canonical'],representable=res['representable'],
            outcome=outcome,rank=rank,in_candidate_100=in_candidate,reason=res['reason'],node_bearing=res['label'] in ranks,
            in_pool=pool_hit,candidate_rank=original_rank,rank_percentile=rank/len(rows) if rank is not None and rows else None,outcome_k=evaluation_k,
            candidate_failure_reason=None if in_candidate else 'rank_below_top100' if pool_hit else 'not_in_scored_pool' if res['representable'] else 'mapping_unavailable',
            metric_scope='canonical_entity' if res['canonical'] else 'representable_evidence_or_composite'))
    first=min((r for r in ranks.values() if r is not None),default=None)
    at_r=sum(r is not None and r<=count for r in ranks.values())
    return dict(status='evaluable' if count else 'not_evaluable' if resolutions else 'not_applicable',
        evaluability_reason='Canonical entity targets available' if count else 'No canonical entity target; retained in raw denominator' if resolutions else 'No entity targets supplied; evaluate evidence separately',
        total_expected=len(resolutions),canonical_expected=count,canonical_label_instances=sum(r['canonical'] for r in resolutions),representable_expected=len(representable),
        canonical_evaluation_units=[dict(label=r['label'],supplied_labels=r['supplied_labels'],node_ids=r['resolved_node_ids']) for r in canonical],
        at_k=metrics,mrr=1/first if first else 0. if count else None,r_precision=at_r/count if count else None,recall_at_r=at_r/count if count else None,
        candidate_recall_50=sum(hit(r,[row for row in rows if row['entity_id'] in set(result['candidate_entity_ids'][:50])]) for r in canonical)/count if count else None,
        candidate_recall_100=candidate_found/count if count else None,retrieval_recall_given_candidate=sum(r is not None and r<=evaluation_k for r in ranks.values())/candidate_found if candidate_found else None,
        rank_quantiles=dict(zip(['min','q25','median','q75','max'],np.quantile([r for r in ranks.values() if r is not None],[0,.25,.5,.75,1]).tolist())) if any(r is not None for r in ranks.values()) else None,
        target_outcomes=outcomes)

def contextual_summary(records,top_ks=(1,5,10,20,50,100)):
    aa=[r for r in records if r['question_type']=='A']; bb=[r for r in records if r['question_type']=='B']; summary={}
    def mean(values):
        vv=[x for x in values if x is not None]; return float(np.mean(vv)) if vv else None
    for k in top_ks:
        mm=[r['entity_metrics'] for r in aa]; found=sum(m['at_k'][str(k)]['found'] for m in mm); total=sum(m['total_expected'] for m in mm); resolved=sum(m['canonical_expected'] for m in mm)
        unique={o['label'] for m in mm for o in m['target_outcomes'] if o['node_bearing']}; retrieved={x for m in mm for x in m['at_k'][str(k)]['found_targets']}
        summary[str(k)]=dict(macro_recall=mean([m['at_k'][str(k)]['recall'] for m in mm]),micro_recall=found/resolved if resolved else None,
            unique_target_recall=len(retrieved)/len(unique) if unique else None,unique_target_definition='Union of named targets recovered in any associated question divided by unique canonical targets',
            macro_hit=mean([m['at_k'][str(k)]['hit'] for m in mm]),macro_precision=mean([m['at_k'][str(k)]['precision'] for m in mm]),raw_end_to_end_lower_bound=sum(m['at_k'][str(k)]['raw_found'] for m in mm)/total if total else None,
            macro_representable_recall=mean([m['at_k'][str(k)]['representable_recall'] for m in mm]),found=found,canonical_target_instances=resolved,total_target_instances=total)
    ranks=[o['rank'] for r in aa for o in r['entity_metrics']['target_outcomes'] if o['node_bearing'] and o['rank'] is not None]
    return dict(type_a=dict(questions=len(aa),evaluable=sum(r['entity_metrics']['status']=='evaluable' for r in aa),at_k=summary,
        rank_quantiles=dict(zip(['min','q25','median','q75','max'],np.quantile(ranks,[0,.25,.5,.75,1]).tolist())) if ranks else None,
        mean_target_rank=float(np.mean(ranks)) if ranks else None,
        duplicate_slot_fraction_at10=mean([r['entity_metrics']['at_k'].get('10',{}).get('duplicate_slot_fraction') for r in aa]),
        mrr=mean([r['entity_metrics']['mrr'] for r in aa]),r_precision=mean([r['entity_metrics']['r_precision'] for r in aa]),recall_at_r=mean([r['entity_metrics']['recall_at_r'] for r in aa]),
        candidate_recall_50=mean([r['entity_metrics']['candidate_recall_50'] for r in aa]),candidate_recall_100=mean([r['entity_metrics']['candidate_recall_100'] for r in aa])),
        type_b=dict(questions=len(bb),source_recall=mean([r['evidence_metrics']['source_recall'] for r in bb]),source_hit=mean([r['evidence_metrics']['source_hit'] for r in bb]),
            source_resolution_coverage=mean([r['evidence_metrics']['source_resolution_coverage'] for r in bb]),claim_recall=mean([r['evidence_metrics']['claim_recall'] for r in bb]),
            mean_claim_candidates=mean([r['evidence_metrics']['claim_candidate_count'] for r in bb]),
            source_at_k={str(k):{key:mean([r['evidence_metrics']['source_at_k'][str(k)][key] for r in bb]) for key in ['hit','recall']} for k in top_ks}),
        costs_by_type={typ:dict(mean_entity_comparisons=mean([r['cost']['vector_comparisons'] for r in rr]),
            mean_claim_comparisons=mean([r['evidence_metrics']['claim_comparisons_new'] for r in rr]),
            mean_total_comparisons=mean([r['cost']['vector_comparisons']+r['evidence_metrics']['claim_comparisons_new'] for r in rr]),
            mean_entities_scored=mean([r['cost']['entity_scores'] for r in rr]),mean_latency_seconds=mean([r['latency_seconds'] for r in rr])) for typ,rr in [('A',aa),('B',bb)]},
        mean_vector_comparisons=mean([r['cost']['vector_comparisons'] for r in records]),mean_latency_seconds=mean([r['latency_seconds'] for r in records]))
