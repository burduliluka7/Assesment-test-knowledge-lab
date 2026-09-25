"""Additive CR6 experiment. Candidate discovery and all historical scorers are reused."""
from pathlib import Path
from copy import deepcopy
import time
import numpy as np
from .io import read_json, read_csv, write_json, locate, digest
from .config import read_config
from .snapshots import snapshot
from .target_mapping import TargetResolver, normalize
from .v2_pipeline import sha
from .candidate_evidence import CandidateEvidenceIndex
from .reranker_pipeline import FrozenH2Candidates, LocalEncoder, evaluate_reranked, summary
from .diffusion_pipeline import evaluate_ranking
from .evaluation.reranker_oracle import oracle_ceiling
from .reranking import EvidenceSemanticReranker, rerank_candidates
from .question_decomposition import decompose_question
from .constraint_scoring import RequirementCosineReranker, PreparedReranker
from .conditioned_decomposition import decompose_conditioned_question
from .conditioned_evidence import build_candidate_evidence_pool, ConditionedEvidenceReranker

EXPORTS = {'conditioned_evidence_results.json','candidate_evidence_pools.json','requirement_evidence_selection.json',
           'evidence_selection_delta.json','conditioned_ranks_per_target.json','conditioned_failure_analysis.json',
           'conditioned_evidence_tests.xml','conditioned_evidence.md'}
STAGES = ['CR6-100','CR7-100','CR6-200']


def component_key(c): return c['kind'], c['start'], c['end'], c['text']


def selection_delta(old, new, old_bundle, pool_only):
    lookup = {component_key(r['component']):r for r in old['requirements']}
    control = {component_key(r['component']):r for r in pool_only['requirements']}
    old_texts = {normalize(i['text']) for items in old_bundle['fields'].values() for i in items}
    rows=[]
    for req in new['requirements']:
        key=component_key(req['component']); prior=lookup.get(key); a=prior['best_evidence'] if prior else None; b=req['best_evidence']
        rows.append(dict(component=req['component'], comparison_status='matched' if prior else 'new_or_restructured_component',
            old_support=prior['cosine_support'] if prior else None, new_support=req['cosine_support'],
            support_delta=req['cosine_support']-prior['cosine_support'] if prior else None,
            pool_only_support=control[key]['cosine_support'] if key in control else None,
            old_evidence=a, new_evidence=b, changed=((a or {}).get('node_id') != (b or {}).get('node_id')) if prior else None,
            same_node=(a or {}).get('node_id')==(b or {}).get('node_id') if prior else None,
            higher_cosine=b['cosine']>a['cosine'] if a and b else None,
            stronger_attribution=b['weight']>a['weight'] if a and b else None,
            previously_unselected=normalize(b['text']) not in old_texts if b else False))
    new_keys={component_key(r['component']) for r in new['requirements']}
    removed=[r for r in old['requirements'] if component_key(r['component']) not in new_keys]
    return dict(components=rows, removed_or_restructured_old_components=[dict(component=r['component'],old_support=r['cosine_support'],old_evidence=r['best_evidence']) for r in removed])


def failure_category(rank, h2_rank, k, detail):
    if h2_rank is None or h2_rank>k: return f'NOT_IN_H2_{k}'
    if rank is not None and rank<=10: return 'RETRIEVED_TOP10'
    if not detail or not detail['pool_size']: return 'NO_CANDIDATE_EVIDENCE_POOL'
    if detail['no_direct_attributable_evidence']: return 'NO_ATTRIBUTABLE_EVIDENCE'
    if any(r['component']['kind']=='answer_qualifier' and r['cosine_support']<.5 for r in detail['requirements']): return 'ANSWER_QUALIFIER_FAILURE'
    if detail['requirement_coverage']==0: return 'REQUIREMENT_EVIDENCE_NOT_FOUND'
    if detail['requirement_coverage']<1: return 'PARTIAL_REQUIREMENT_SUPPORT'
    return 'SUPPORTED_BUT_RANKED_BELOW_10'


def run_conditioned(cfg, questions_path=None, truth_path=None):
    began=time.perf_counter(); root=Path(cfg['v2_output']); out=Path(cfg['conditioned_output']); out.mkdir(parents=True,exist_ok=True)
    ccfg=read_config(cfg['constraint_config']); rc=read_config(ccfg['reranking_config'])
    cc=read_config(rc['context_config']); dc=read_config(rc['diffusion_config']); scoring=dict(ccfg,**{k:v for k,v in cfg.items() if k.startswith('conditioned_')})
    data=read_json(locate(cfg['data_dir'],'tkh_collection10.json')); questions=read_csv(questions_path or locate(cfg['data_dir'],'questions.csv'))
    gt=read_json(truth_path or locate(cfg['data_dir'],'ground_truth.json')); aa=[q for q in questions if q['type']=='A']
    previous=read_json(root/'constraint_reranking_results.json') if (root/'constraint_reranking_results.json').exists() else None
    rr=read_json(root/'reranker_results.json') if (root/'reranker_results.json').exists() else None
    same=previous is not None and digest(data)==previous['protocol']['data_hash'] and digest(questions)==previous['protocol']['questions_hash']
    same_gold=same and digest(gt)==previous['protocol']['ground_truth_hash']
    if cfg['conditioned_verify_history']:
        if previous is None: raise ValueError('Historical constraints baseline required')
        for path,h in previous['protocol']['source_hashes'].items():
            if not path.endswith('/cli.py') and sha(path)!=h: raise ValueError('Frozen source changed: '+path)
        if ccfg!=previous['protocol']['config']: raise ValueError('Frozen CR1 config changed')
        if rc!=rr['protocol']['config'] or cc!=rr['protocol']['context_config'] or dc!=rr['protocol']['diffusion_config']: raise ValueError('Frozen H2/evidence config changed')
    history={p.as_posix():sha(p) for p in root.rglob('*') if p.is_file() and out not in p.parents and p.name not in EXPORTS}
    history.update({p.as_posix():sha(p) for p in Path('report').glob('*.md')})
    metrics_path=Path(cfg['output'])/'metrics.json'; metrics=read_json(metrics_path) if metrics_path.exists() else {}
    metric_hashes={k:digest(v) for k,v in metrics.items() if k!='conditioned_evidence'}
    protocol=dict(config=cfg, scoring_config=scoring, source_hashes={p.as_posix():sha(p) for p in Path('src').rglob('*.py')},
        historical_hashes=history,historical_metric_hashes=metric_hashes,data_hash=digest(data),questions_hash=digest(questions),ground_truth_hash=digest(gt),
        pool_policy='Reuse uncapped eligible CandidateEvidenceIndex pools, stable structural field caps and cross-field normalized deduplication. No query-seeded path expansion; historical paths retained separately.',
        selection='Component-only BGE query; top3 nonnegative cosine times frozen attribution weight; maximum support; unchanged aggregate_support.',
        diagnostic_controls='Pool-only: old decomposition and attribution on new pools. No-qualifiers: CR6 with only new qualifier components removed. Evaluation diagnostics, not selected defaults.',
        stages=STAGES, nli_calls=0, same_history=same, same_gold=same_gold)
    write_json(out/'protocol.json',protocol)
    encoder=LocalEncoder(cfg['evaluation_model'],cfg['evaluation_revision'])
    old_bundles={(r['cutoff'],r['question_id']):r for r in read_json(root/'candidate_evidence.json')} if same else {}
    results=[]; selections=[]; deltas=[]; targets=[]; pools_export=[]; inventory=[]; oracles=[]; pool_costs=[]
    references=[r for r in previous['results'] if r['stage'] in {'CR0-100','CR1-100','CR3-100'} and r['cutoff'] in cfg['conditioned_cutoffs']] if same_gold else []
    for year in cfg['conditioned_cutoffs']:
        snap=snapshot(data,year); nodes={n['id']:n for n in snap['nodes']}
        old_h2={r['question_id']:r for s in rr['results'] if s['cutoff']==year and s['stage']=='RR0' for r in s['records']} if same else {}
        old_ref={(s['stage'],r['question_id']):r for s in previous['results'] if s['cutoff']==year for r in s['records']} if same else {}
        source=FrozenH2Candidates(snap,encoder,cc,dc,old_h2,root/'diffusion'); resolver=TargetResolver(snap)
        t=time.perf_counter(); index=CandidateEvidenceIndex(snap,source.groups,source.operator,source.mentions,rc)
        pools={g['id']:build_candidate_evidence_pool(index,g['id'],cfg['conditioned_pool_caps']).to_dict() for g in source.groups}
        pool_seconds=time.perf_counter()-t
        pools_export.extend(pools.values()); pool_costs.append(dict(cutoff=year,seconds=pool_seconds,candidates=len(pools),items=sum(sum(map(len,p['fields'].values())) for p in pools.values())))
        stage_rows={s:[] for s in STAGES}; reference_rows={s:[] for s in ['H2','CR0-100','CR1-100','CR3-100']}
        for q in aa:
            qstart=time.perf_counter(); qid=q['question_id']; h2=source.get(q['question'],qid); frozen=h2['ranked']; frozen_hash=digest(frozen)
            gold=[resolver.resolve(label) for label in gt.get(qid,{}).get('expected_methods',[])]
            baseline=evaluate_ranking(frozen,gold,dict(diffusion_top_ks=rc['reranker_top_ks']))
            if same_gold and baseline!=old_h2[qid]['entity_metrics']: raise ValueError('Frozen H2 metrics differ')
            if same: original=old_bundles[year,qid]; bundles={b['entity_id']:b for b in original['candidates']}
            else:
                paths,_=index.support_paths(h2['seeds'])
                bundles={r['entity_id']:index.build_candidate_evidence(q['question'],r,h2['similarities'],paths,i+1) for i,r in enumerate(frozen[:200])}
            d0=decompose_question(q['question']); d=decompose_conditioned_question(q['question']); noqual=deepcopy(d)
            noqual.components=[c for c in d.components if c['kind']!='answer_qualifier']; noqual.answer_qualifiers=[]
            # Reproduce the historical matrix shape and scalar values first.
            t=time.perf_counter(); texts0=[c['text'] for c in d0.components]
            X0=encoder.encode(texts0,'constraint-components:'+digest(texts0)) if texts0 else np.zeros((0,source.context.X.shape[1]))
            ids0=sorted({i['node_id'] for b in bundles.values() for field in b['fields'].values() for i in field})
            Z0=X0@source.context.X[[source.context.text_idx[nodes[n]['surface_form']] for n in ids0]].T
            sims0={c['id']:dict(zip(ids0,map(float,Z0[i]))) for i,c in enumerate(d0.components)}
            prior_scorer=RequirementCosineReranker(scoring,d0,sims0,source.mentions.names)
            details0={r['entity_id']:prior_scorer.score(q['question'],r['name'],bundles[r['entity_id']]) for r in frozen[:200]}
            reference_rankings={'H2':frozen}
            for stage in ['CR0-100','CR1-100','CR3-100']:
                scorer=EvidenceSemanticReranker(rc) if stage.startswith('CR0') else PreparedReranker(details0)
                ranked=rerank_candidates(frozen,bundles,scorer,q['question'],100,stage.startswith('CR3'),rc['reranker_rrf_k'])['ranked']
                metric=evaluate_reranked(ranked,frozen,gold,rc,100)
                if same and ranked[:200]!=old_ref[stage,qid]['returned']: raise ValueError('Historical ranking changed: '+stage)
                if same_gold and metric!=old_ref[stage,qid]['entity_metrics']: raise ValueError('Historical metric changed: '+stage)
                reference_rankings[stage]=ranked
            replay_seconds=time.perf_counter()-t
            t=time.perf_counter(); pool_ids=sorted({i['node_id'] for r in frozen[:200] for field in pools[r['entity_id']]['fields'].values() for i in field})
            texts=list(dict.fromkeys(texts0+[c['text'] for c in d.components])); new_texts=[x for x in texts if x not in texts0]
            Xnew=encoder.encode(new_texts,'conditioned-components:'+digest(new_texts)) if new_texts else np.zeros((0,source.context.X.shape[1]))
            vectors={text:X0[i] for i,text in enumerate(texts0)}; vectors.update({text:Xnew[i] for i,text in enumerate(new_texts)})
            cache={c['text']:dict(sims0[c['id']]) for c in d0.components}; new_dots=0; reused=0
            for text in texts:
                existing=cache.setdefault(text,{}); missing=[n for n in pool_ids if n not in existing]; reused+=len(pool_ids)-len(missing)
                if missing:
                    dots=vectors[text]@source.context.X[[source.context.text_idx[nodes[n]['surface_form']] for n in missing]].T
                    existing.update(zip(missing,map(float,dots))); new_dots+=len(missing)
            sims={c['id']:cache[c['text']] for c in d.components}; old_pool_sims={c['id']:cache[c['text']] for c in d0.components}
            preparation_seconds=time.perf_counter()-t
            main=ConditionedEvidenceReranker(scoring,d,sims,source.mentions.names)
            control=ConditionedEvidenceReranker(scoring,d0,old_pool_sims,source.mentions.names,False)
            qualifier_control=ConditionedEvidenceReranker(scoring,noqual,sims,source.mentions.names)
            details={}; pool_only={}; noqual_details={}; times=[]
            for r in frozen[:200]:
                ident=r['entity_id']; t=time.perf_counter(); details[ident]=main.score(q['question'],r['name'],pools[ident]); times.append(time.perf_counter()-t)
                pool_only[ident]=control.score(q['question'],r['name'],pools[ident]); noqual_details[ident]=qualifier_control.score(q['question'],r['name'],pools[ident])
            rankings=dict(reference_rankings)
            for stage in STAGES:
                k=int(stage.split('-')[1]); ranked=rerank_candidates(frozen,pools,PreparedReranker(details),q['question'],k,stage.startswith('CR7'),rc['reranker_rrf_k'])
                rankings[stage]=ranked['ranked']; m=evaluate_reranked(ranked['ranked'],frozen,gold,rc,k)
                seconds=preparation_seconds+sum(times[:k])+ranked['runtime_seconds']; total=h2['candidate_generation_runtime']+seconds
                cost=dict(vector_comparisons=new_dots,entity_scores=min(k,len(frozen)),runtime_seconds=seconds,reranker_seconds=seconds,
                    reranker_comparisons=min(k,len(frozen)),total_query_seconds=total,candidate_generation_seconds=h2['candidate_generation_runtime'],
                    evidence_selection_dots_new=new_dots,evidence_selection_scores_reused=reused,evidence_selection_seconds=sum(times[:k]),
                    preparation_seconds=preparation_seconds,semantic_support_comparisons=sum(details[r['entity_id']]['semantic_support_comparisons'] for r in frozen[:k]),
                    historical_reproduction_seconds=replay_seconds,evidence_embeddings_reused=len(pool_ids),evidence_encodings_new=0,new_component_strings=len(new_texts),nli_calls=0)
                stage_rows[stage].append(dict(cutoff=year,stage=stage,candidate_k=k,question_id=qid,question_type='A',question=q['question'],entity_metrics=m,
                    evidence_metrics=dict(claim_comparisons_new=0),cost=cost,latency_seconds=total,candidate_preserved=ranked['candidate_preserved'],returned=ranked['ranked'][:200]))
            for stage,rs in reference_rankings.items():
                m=baseline if stage=='H2' else evaluate_reranked(rs,frozen,gold,rc,100)
                cost=deepcopy(old_h2[qid]['cost'] if stage=='H2' and same else old_ref[stage,qid]['cost'] if same else dict(vector_comparisons=0,entity_scores=0,runtime_seconds=0,total_query_seconds=0))
                reference_rows[stage].append(dict(cutoff=year,stage=stage,question_id=qid,question_type='A',question=q['question'],entity_metrics=m,cost=cost,evidence_metrics=dict(claim_comparisons_new=0),latency_seconds=cost.get('total_query_seconds',0),candidate_preserved=True,returned=rs[:200]))
            diagnostic_rankings={name:rerank_candidates(frozen,pools,PreparedReranker(ds),q['question'],100)['ranked'] for name,ds in [('pool_only',pool_only),('without_qualifiers',noqual_details)]}
            rank_maps={s:{r['entity_id']:i+1 for i,r in enumerate(rs)} for s,rs in {**rankings,**diagnostic_rankings}.items()}
            if digest(frozen)!=frozen_hash: raise ValueError('H2 mutated')
            inv=dict(cutoff=year,question_id=qid,prefix100_hash=digest(frozen[:100]),prefix200_hash=digest(frozen[:200]),H2_rank_hash=frozen_hash,reference_parity=same,reference_metric_parity=same_gold)
            inventory.append(inv)
            for k in [100,200]:
                oracle=dict(cutoff=year,question_id=qid,**oracle_ceiling(baseline,frozen,k))
                if same_gold and oracle!=next(o for o in previous['oracle'] if o['cutoff']==year and o['question_id']==qid and o['candidate_k']==k): raise ValueError('Oracle changed')
                oracles.append(oracle)
            # Gold membership starts here: exports/evaluation only.
            candidates=[]
            for i,r in enumerate(frozen[:200]):
                ident=r['entity_id']; is_gold=any(g['canonical'] and set(g['resolved_node_ids'])&set(r['node_ids']) for g in gold)
                ranks={s:rm[ident] for s,rm in rank_maps.items()}
                pool_texts={normalize(x['text']) for xs in pools[ident]['fields'].values() for x in xs}
                missing_old=[dict(node_id=x['node_id'],text=x['text'],kind=x['kind'],attribution=x['attribution']['category'],weight=x['attribution_weight'],
                    reason='query_seeded_path_excluded' if x['kind']=='short_path_source' else 'structural_cap_or_deduplication')
                    for x in details0[ident]['attributions'] if x['attribution_weight']>0 and normalize(x['text']) not in pool_texts]
                candidates.append(dict(entity_id=ident,candidate=r['name'],node_ids=r['node_ids'],h2_rank=i+1,gold=is_gold,ranks=ranks,
                    CR6=details[ident],pool_only=pool_only[ident],without_qualifiers_score=noqual_details[ident]['score'],CR1_score=details0[ident]['score'],
                    weighted_degree=pools[ident]['weighted_degree'],old_pool_size=sum(map(len,bundles[ident]['fields'].values())),
                    historical_selected_paths=bundles[ident].get('selected_paths',[]),old_positive_items_missing_from_pool=missing_old))
                deltas.append(dict(cutoff=year,question_id=qid,entity_id=ident,candidate=r['name'],h2_rank=i+1,gold=is_gold,**selection_delta(details0[ident],details[ident],bundles[ident],pool_only[ident])))
            selections.append(dict(cutoff=year,question_id=qid,question=q['question'],decomposition=d.to_dict(),historical_decomposition=d0.to_dict(),candidates=candidates))
            r3rank={n:i+1 for i,r in enumerate(h2['r3']) for n in r['node_ids']}
            for unit in baseline['canonical_evaluation_units']:
                ids=set(unit['node_ids']); target=dict(cutoff=year,question_id=qid,label=unit['label'],node_ids=unit['node_ids'],R3_rank=min((r3rank[n] for n in ids if n in r3rank),default=None),ranks={},failures={})
                for stage,rs in {**rankings,**diagnostic_rankings}.items(): target['ranks'][stage]=next((i+1 for i,r in enumerate(rs) if ids&set(r['node_ids'])),None)
                matching=next((r['entity_id'] for r in frozen[:200] if ids&set(r['node_ids'])),None)
                for stage in STAGES: target['failures'][stage]=failure_category(target['ranks'][stage],target['ranks']['H2'],int(stage.split('-')[1]),details.get(matching))
                targets.append(target)
            write_json(out/f'checkpoint_{year}_{qid}.json',dict(inventory=inv,records={s:stage_rows[s][-1] for s in STAGES}))
            print('CONDITIONED',year,qid,'seconds',round(time.perf_counter()-qstart,2),flush=True)
        for stage,rows in {**reference_rows,**stage_rows}.items():
            results.append(dict(cutoff=year,stage=stage,summary=summary(rows,rc),records=rows))
            print('RESULT',year,stage,results[-1]['summary']['at_k']['10']['micro_recall'],flush=True)
    if any(sha(p)!=h for p,h in history.items()): raise ValueError('Historical artifact changed')
    run=dict(status='computed',protocol=protocol,results=results,inventory=inventory,oracle=oracles,pool_build_costs=pool_costs,elapsed_seconds=time.perf_counter()-began)
    write_json(root/'conditioned_evidence_results.json',run); write_json(root/'candidate_evidence_pools.json',pools_export)
    write_json(root/'requirement_evidence_selection.json',selections); write_json(root/'evidence_selection_delta.json',deltas)
    write_json(root/'conditioned_ranks_per_target.json',targets)
    write_json(root/'conditioned_failure_analysis.json',dict(targets=targets,policy='Mutually exclusive operational categories; decomposition/attribution failure require qualitative adjudication, never inferred from gold rank alone.'))
    current=read_json(metrics_path) if metrics_path.exists() else {}
    if any(digest(current.get(k))!=h for k,h in metric_hashes.items()): raise ValueError('Historical metric section changed')
    current['conditioned_evidence']=dict(protocol=protocol,results=[{k:r[k] for k in ('cutoff','stage','summary')} for r in results],
        question_metrics=[{k:q[k] for k in ('cutoff','stage','question_id','entity_metrics','cost')} for r in results for q in r['records']])
    write_json(metrics_path,current)
    return run
