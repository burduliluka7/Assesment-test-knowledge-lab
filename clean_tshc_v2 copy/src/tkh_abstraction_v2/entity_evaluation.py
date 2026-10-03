"""Evaluation-only mapping, metrics, failure traces and positive controls."""
from collections import Counter
import statistics
from .io import read, write, sha, event
from .snapshots import snapshot
from .strict_targets import support_audit
from .entity_index import allowed_entity

DEPTHS = (50,100,200,500,1000)
KS = (1,5,10,25,50,100)


def hit(rank, k):
    return rank is not None and rank <= k


def rank_components(components, ranks):
    values = [min((ranks[v] for v in c if v in ranks), default=None) for c in components]
    return max(values) if values and all(v is not None for v in values) else None


def summarize(targets, qids, system, denominator, missing_penalty):
    direct = [t for t in targets if t['status'] in ('EXACT','ALIAS')]
    ranks = [t['direct_ranks'][system] for t in direct]
    observed = [r for r in ranks if r is not None]
    first = {q: min((t['direct_ranks'][system] for t in direct if t['question_id']==q and t['direct_ranks'][system] is not None),default=None) for q in qids}
    hits = {str(k):sum(hit(r,k) for r in ranks) for k in KS}
    return dict(direct_hits=hits, recall={k:n/denominator for k,n in hits.items()},
        exact_alias_conditional_recall={k:n/len(direct) for k,n in hits.items()},
        MRR=sum(1/r if r else 0 for r in first.values())/len(qids),
        median_gold_rank=statistics.median(observed) if observed else None,
        mean_gold_rank=statistics.mean(observed) if observed else None,
        observed_gold_ranks=len(observed), missing_gold_ranks=len(direct)-len(observed),
        mean_gold_rank_missing_penalized=statistics.mean(r or missing_penalty for r in ranks),
        first_direct_ranks=first)


def classify_failure(target, index, intent, cfg, channels, verification):
    if target['outside_snapshot']:
        return dict(primary='TEMPORAL_VISIBILITY_FAILURE', reason='Mapped identities exist only after the question cutoff', secondary=[])
    if target['status'] not in ('EXACT','ALIAS'):
        return dict(primary='GROUND_TRUTH_MAPPING_UNCERTAIN', reason='Not a directly resolvable EXACT/ALIAS identity: '+target['status'], secondary=[])
    eids={i for component in target['entity_components'] for i in component if i in index['entities']}
    if not eids:
        return dict(primary='TEMPORAL_VISIBILITY_FAILURE' if target['outside_snapshot'] else 'NOT_IN_ENTITY_INDEX', secondary=[])
    eligible={i for i in eids if allowed_entity(index['entities'][i],intent,cfg)}
    if not eligible:
        return dict(primary='WRONG_TYPE_FILTER',secondary=[])
    sources=channels['candidate_sources']
    secondary=[]
    if not any(sources.get(i,{}).get('name_matches') for i in eligible):secondary.append('NO_NAME_MATCH')
    if not any(r['candidate_specific'] for i in eligible for rows in index['entities'][i]['evidence'].values() for r in rows):
        secondary.append('NO_RELEVANT_EVIDENCE')
    ranks=target['direct_ranks']; final=ranks.get('A7',ranks.get('A6',ranks['A5']))
    if hit(final,10):primary='RECOVERED_AT_10'
    elif ranks['A5'] is None:primary='NOT_IN_CANDIDATE_POOL'
    elif ranks['A5']>cfg['rerank_k']:primary='IN_POOL_RRF_FAILED'
    elif hit(ranks.get('A6'),10) and not hit(final,10):primary='REQUIREMENT_VERIFICATION_FAILED'
    else:primary='IN_POOL_CROSS_ENCODER_FAILED'
    return dict(primary=primary,secondary=secondary,
        definitions='Top10 failure boundary; NO_RELEVANT_EVIDENCE means no indexed candidate-specific evidence, not proof of scientific absence; NO_NAME_MATCH is an absent lexical opportunity, not a cause.')


def evaluate(root, args):
    out=root/(args[0] if args else 'outputs'); cfg=read(root/'config.json')
    prediction_file=out/'retrieval/predictions.json'; before=sha(prediction_file)
    reference=root.parent/'clean_tshc'
    truth_path=reference/'data/ground_truth.json'; annotation_path=reference/'notes/target_annotations.json'
    truth=read(truth_path); annotations=read(annotation_path)
    event(out/'audit/events.jsonl','gold_loaded',gold_sha256=sha(truth_path),annotations_sha256=sha(annotation_path))
    predictions=read(prediction_file)
    graph=read(root/'data/tkh_collection10.json')
    indices={int(p.stem):read(p) for p in (out/'entity_index').glob('*.json')}
    supports={cutoff:support_audit(snapshot(graph,cutoff),truth,annotations) for cutoff in indices}
    full_support=support_audit(snapshot(graph,cfg['snapshot']),truth,annotations)
    full_targets={(t['question_id'],t['target']):t for t in full_support['records']}
    write(out/'audit/strict_target_support.json',supports)
    qids=[q for q,row in predictions.items() if row['question_type']=='A']
    systems=list(predictions[qids[0]]['rankings'])
    questions, targets, traces={},[],[]
    candidate_stage, all_channels, all_selected, all_verification = {}, {}, {}, {}
    rank_audit=[]
    for qid in qids:
        pred=predictions[qid]; index=indices[pred['cutoff']]
        intent=read(out/'query_understanding'/(qid+'.json'))
        channels=read(out/'retrieval/channels'/(qid+'.json'))
        selected=read(out/'retrieval/selected_evidence'/(qid+'.json'))
        vf=out/'retrieval/requirement_verification'/(qid+'.json')
        verification=read(vf) if vf.exists() else {}
        all_channels[qid]=channels;all_selected[qid]=selected;all_verification[qid]=verification
        ranks={name:{r['entity_id']:i for i,r in enumerate(rows,1)} for name,rows in pred['rankings'].items()}
        for name,rows in pred['rankings'].items():
            assert len(rows)==len(ranks[name])
            rank_audit.append(dict(question_id=qid,system=name,unique_candidates=len(rows),duplicate_ids=0))
        stage={name:rows for name,rows in pred['rankings'].items()}
        for name in ('A6','A6_100','A6_200','A6_500','A7'):
            if name in stage:
                k=int(name.split('_')[1]) if '_' in name else cfg['rerank_k']
                stage[name]=pred['rankings']['A5'][:k]
        candidate_stage[qid]=stage
        ts=[t for t in supports[pred['cutoff']]['records'] if t['question_id']==qid]
        questions[qid]=dict(question=pred['question'],intent=intent,targets=[],candidate_union_size=pred['candidate_union_size'],systems={})
        for original in ts:
            components=[[index['node_to_entity'].get(n,'node:'+n) for n in c] for c in original['components']]
            target=dict(question_id=qid,target=original['target'],status=original['status'],node_ids=original['node_ids'],
                node_components=original['components'],entity_components=components,direct_ranks={},composite_direct_ranks={},v1_compatible_bundle_ranks={},
                outside_snapshot=bool(full_targets[(qid,original['target'])]['node_ids']) and all(
                    n['first_seen_year']>pred['cutoff'] for n in graph['nodes'] if n['id'] in full_targets[(qid,original['target'])]['node_ids']))
            for name,rankmap in ranks.items():
                value=rank_components(components,rankmap)
                target['direct_ranks'][name]=value if target['status'] in ('EXACT','ALIAS') else None
                target['composite_direct_ranks'][name]=value if target['status']=='COMPOSITE' else None
                # Direct identities retain V1 semantics. Composite bundles use
                # actual exported query-selected evidence and are supplementary.
                if target['status']=='COMPOSITE':
                    accessible={}
                    for eid,position in rankmap.items():
                        members=index['entities'].get(eid,{}).get('node_ids',[])
                        for n in members:accessible.setdefault(n,position)
                        for item in selected.get(eid,{}).get('selected',[]):accessible.setdefault(item['node_id'],position)
                    bundle=rank_components(original['components'],accessible)
                else:bundle=target['direct_ranks'][name]
                target['v1_compatible_bundle_ranks'][name]=bundle if target['status'] in ('EXACT','ALIAS','COMPOSITE') else None
            target['failure']=classify_failure(target,index,intent,cfg,channels,verification)
            details={}
            for eid in sorted({i for c in components for i in c}):
                details[eid]=dict(candidate_sources=channels['candidate_sources'].get(eid),
                    rrf_rank=ranks['A5'].get(eid),cross_encoder_rank=ranks.get('A6',{}).get(eid),
                    in_reranker_pool=hit(ranks['A5'].get(eid),cfg['rerank_k']),
                    selected_evidence=selected.get(eid),requirement_support=verification.get(eid),
                    final_rank=ranks.get('A7',ranks.get('A6',ranks['A5'])).get(eid))
            target['trace']=details
            questions[qid]['targets'].append(target);targets.append(target)
            traces.append(dict(question_id=qid,question=pred['question'],structure=intent,target=original['target'],status=target['status'],trace=details,failure=target['failure']))
    direct=[t for t in targets if t['status'] in ('EXACT','ALIAS')]
    denom=len(targets); mapped=len(direct)
    metrics=dict(label='DEVELOPMENT',primary_system='A7',denominators=dict(all_questions=len(predictions),type_A_questions=len(qids),
        target_occurrences=denom,exact_alias_occurrences=mapped,status_counts=dict(Counter(t['status'] for t in targets))),
        definitions=dict(recall='Direct EXACT/ALIAS occurrences / all Type-A targets. PARTIAL/ABSENT/COMPOSITE never direct identity credit.',
            candidate_recall='Direct mapped occurrence recovery /49, plus raw /63. For reranked arms pool membership is measured BEFORE reranking.',
            MRR='First directly mapped identity per Type-A question; zero on misses; denominator14.',
            ranks='Conditional mean/median of observed direct ranks; missing count and fixed missing penalty are explicit.',
            composite='Complete component bundles reported separately; never mixed with direct identities.'),systems={},conditional_reranking={},channel_diagnostics={})
    penalty=len({n['id'] for n in graph['nodes']})+1
    for name in systems:
        m=summarize(targets,qids,name,denom,penalty)
        pool_sizes=[len(candidate_stage[q][name]) for q in qids]
        candidate_hits={}
        for k in DEPTHS:
            total=0
            for t in direct:
                rr={r['entity_id']:j for j,r in enumerate(candidate_stage[t['question_id']][name][:k],1)}
                total+=rank_components(t['entity_components'],rr) is not None
            candidate_hits[str(k)]=total
        union_hits=sum(rank_components(t['entity_components'],{r['entity_id']:j for j,r in enumerate(candidate_stage[t['question_id']][name],1)}) is not None for t in direct)
        m.update(candidate_hits=candidate_hits,candidate_recall={k:v/mapped for k,v in candidate_hits.items()},candidate_recall_raw={k:v/denom for k,v in candidate_hits.items()},
            candidate_union_hits=union_hits,candidate_union_recall=union_hits/mapped,average_candidate_set_size=statistics.mean(pool_sizes),
            candidate_size_range=[min(pool_sizes),max(pool_sizes)],
            composite_direct_hits={str(k):sum(hit(t['composite_direct_ranks'][name],k) for t in targets) for k in KS})
        metrics['systems'][name]=m
        for q in qids:questions[q]['systems'][name]=dict(first_direct_rank=m['first_direct_ranks'][q],direct_hits={str(k):sum(hit(t['direct_ranks'][name],k) for t in direct if t['question_id']==q) for k in KS})
    for name,k in [('A6_100',100),('A6_200',200),('A6',500),('A7',500)]:
        if name not in systems:continue
        pairs=[dict(question_id=t['question_id'],target=t['target'],before=t['direct_ranks']['A5'],after=t['direct_ranks'][name]) for t in direct if hit(t['direct_ranks']['A5'],k)]
        counts=Counter('upward' if p['after']<p['before'] else 'downward' if p['after']>p['before'] else 'unchanged' for p in pairs)
        metrics['conditional_reranking'][name]=dict(pool_size=k,seen_target_occurrences=len(pairs),**dict(counts),
            top10_after=sum(hit(p['after'],10) for p in pairs),new_top10=sum(not hit(p['before'],10) and hit(p['after'],10) for p in pairs),
            recall10_conditional=sum(hit(p['after'],10) for p in pairs)/len(pairs) if pairs else None,
            median_before=statistics.median(p['before'] for p in pairs) if pairs else None,median_after=statistics.median(p['after'] for p in pairs) if pairs else None,
            strongest_improvements=sorted([p for p in pairs if p['after']<p['before']],key=lambda p:(p['after']-p['before'],p['question_id']))[:5],
            strongest_degradations=sorted([p for p in pairs if p['after']>p['before']],key=lambda p:(p['before']-p['after'],p['question_id']))[:5])
    for channel in all_channels[qids[0]]['channels']:
        hits=unique=0
        for t in direct:
            data=all_channels[t['question_id']]
            ids={r['entity_id'] for r in data['channels'][channel]}
            if rank_components(t['entity_components'],{i:1 for i in ids}) is not None:
                hits+=1
                others={r['entity_id'] for c,rs in data['channels'].items() if c!=channel for r in rs}
                unique+=rank_components(t['entity_components'],{i:1 for i in others}) is None
        metrics['channel_diagnostics'][channel]=dict(mapped_occurrences_recovered=hits,unique_contribution=unique,recall=hits/mapped)
    failures=Counter(t['failure']['primary'] for t in targets)
    controls=[]
    for qid in qids:
        ts=[t for t in direct if t['question_id']==qid]
        oracle=sorted({c[0] for t in ts for c in t['entity_components'] if c})
        rr={eid:i for i,eid in enumerate(oracle,1)}
        controls.append(dict(question_id=qid,evaluation_only=True,forced_identities=oracle,
            positive_hits=sum(rank_components(t['entity_components'],rr) is not None for t in ts),expected_hits=len(ts),empty_ranking_hits=0))
    assert all(c['positive_hits']==c['expected_hits'] for c in controls)
    write(out/'evaluation_only/positive_controls.json',dict(label='GOLD-INFORMED EVALUATION-ONLY ORACLE; NEVER A RETRIEVAL RESULT',records=controls,passed=True))
    write(out/'metrics.json',metrics);write(out/'per_question_evaluation.json',questions)
    write(out/'failure_traces.json',traces);write(out/'audit/failure_classes.json',dict(counts=dict(failures),secondary_counts=dict(Counter(c for t in targets for c in t['failure']['secondary']))))
    write(out/'audit/rank_audit.json',rank_audit)
    assert sha(prediction_file)==before
    event(out/'audit/events.jsonl','evaluation_completed',prediction_sha256=before,metrics_sha256=sha(out/'metrics.json'))
    write(out/'audit/evaluation_manifest.json',dict(prediction_sha256=before,gold_sha256=sha(truth_path),annotations_sha256=sha(annotation_path),
        code_sha256=sha(root/'src/tkh_abstraction_v2/entity_evaluation.py'),positive_control_separate=True))
    print('Evaluation complete:',denom,'targets,',mapped,'direct mapped occurrences')
    for name in ['A0','A1','A2','A3','A4','A5','A6','A7']:
        if name in metrics['systems']:
            m=metrics['systems'][name]
            print(name,'candidate@500',m['candidate_hits']['500'],'/',mapped,'union',m['candidate_union_hits'],'top10',m['direct_hits']['10'],'/',denom,'MRR',round(m['MRR'],4))
