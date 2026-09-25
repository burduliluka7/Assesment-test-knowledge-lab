"""Additive controlled experiments around existing frozen candidate and metric adapters."""
from pathlib import Path
import time
import numpy as np
from .io import read_json,read_csv,write_json,locate,digest
from .config import read_config
from .snapshots import snapshot
from .target_mapping import TargetResolver
from .v2_pipeline import sha,verify_reuse
from .candidate_evidence import CandidateEvidenceIndex
from .reranker_pipeline import FrozenH2Candidates,LocalEncoder,evaluate_reranked,summary
from .diffusion_pipeline import evaluate_ranking
from .evaluation.reranker_oracle import oracle_ceiling
from .reranking import EvidenceSemanticReranker,rerank_candidates
from .question_decomposition import decompose_question
from .constraint_scoring import ControlledLocalNLI,RequirementCosineReranker,RequirementNLIReranker,PreparedReranker,uniform_nli_fallback

NEW_EXPORTS={'constraint_reranking_results.json','question_decompositions.json','requirement_support.json','evidence_attribution.json',
    'constraint_reranking_per_target.json','constraint_failure_analysis.json','constraint_reranking_tests.xml','constraint_reranking.md'}

def classify_failure(rank,h2_rank,k,detail,stage):
    flags=[]
    if h2_rank is None or h2_rank>k: return f'NOT_IN_H2_{k}',flags
    if rank is not None and rank<=10: return 'RETRIEVED_TOP10',flags
    if stage=='CR0': return 'BASELINE_REFERENCE',flags
    if detail is None: return 'SEMANTIC_SCORER_FAILURE',flags
    if detail.get('status')=='cosine_fallback_nli_failed': flags.append('NLI_EXECUTION_FAILURE')
    if detail.get('no_direct_attributable_evidence'): flags.append('NO_DIRECT_ATTRIBUTABLE_EVIDENCE')
    if detail.get('contradicted_requirements'): flags.append('MODEL_CONTRADICTION_DETECTED')
    if detail.get('no_direct_attributable_evidence'): return 'NO_ATTRIBUTABLE_EVIDENCE',flags
    if detail.get('contradicted_requirements'): return 'CONTRADICTORY_EVIDENCE',flags
    if detail['requirement_coverage']==0: return 'NLI_FAILURE' if stage in {'CR2','CR4'} else 'SEMANTIC_SCORER_FAILURE',flags
    if detail['requirement_coverage']<1: return 'PARTIAL_REQUIREMENT_SUPPORT',flags
    return 'SUPPORTED_BUT_RANKED_BELOW_10',flags

def run_constraints(cfg,questions_path=None,truth_path=None):
    began=time.perf_counter(); root=Path(cfg['v2_output']); out=Path(cfg['constraint_output']); out.mkdir(parents=True,exist_ok=True)
    rc=read_config(cfg['reranking_config']); cc=read_config(rc['context_config']); dc=read_config(rc['diffusion_config'])
    data=read_json(locate(cfg['data_dir'],'tkh_collection10.json')); questions=read_csv(questions_path or locate(cfg['data_dir'],'questions.csv'))
    gt=read_json(truth_path or locate(cfg['data_dir'],'ground_truth.json')); aa=[q for q in questions if q['type']=='A']
    old_path=root/'reranker_results.json'; old=read_json(old_path) if old_path.exists() else None
    same=old is not None and digest(data)==old['protocol']['data_hash'] and digest(questions)==old['protocol']['questions_hash']
    if cfg['constraint_verify_history']:
        if old is None: raise ValueError('Frozen reranking baseline missing')
        for path,h in old['protocol']['source_hashes'].items():
            if path.endswith('/cli.py'): continue
            if sha(path)!=h: raise ValueError('Frozen implementation changed: '+path)
        if rc!=old['protocol']['config'] or cc!=old['protocol']['context_config'] or dc!=old['protocol']['diffusion_config']: raise ValueError('Frozen candidate/evidence configuration changed')
        verify_reuse(cc,data)
    history={p.as_posix():sha(p) for p in root.rglob('*') if p.is_file() and out not in p.parents and p.name not in NEW_EXPORTS}
    for p in [Path('report/hypergraph_diffusion.md'),Path('report/evidence_reranking.md')]:
        if p.exists(): history[p.as_posix()]=sha(p)
    metric_path=Path(cfg['output'])/'metrics.json'; metrics=read_json(metric_path) if metric_path.exists() else {}
    prior={k:digest(v) for k,v in metrics.items() if k!='constraint_reranking'}
    source_hashes={p.as_posix():sha(p) for p in Path('src').rglob('*.py')}
    protocol=dict(config=cfg,reranking_config=rc,data_hash=digest(data),questions_hash=digest(questions),ground_truth_hash=digest(gt),
        source_hashes=source_hashes,historical_hashes=history,historical_metrics_hashes=prior,
        decomposition='Deterministic extractive linguistic boundaries; contiguous original spans; no local generative model',
        aggregation='core_support * geometric_mean(all positive components) * product(1-exclusion_violation) * soft_type_prior; .5 diagnostic threshold does not enter scores',
        nli='Controlled local NLI, not relevance cross-encoder; top1 attribution-weighted cosine item per component; max(weight*max(0,entailment-contradiction)); positive exclusion-violation hypotheses',
        fallback='Failed/unavailable NLI explicitly labelled; entire query uses deterministic CR1 fallback to avoid mixed score scales; no invented probabilities',
        oracle='Existing unchanged evaluation-only exact subset coverage',primary_k=100,diagnostic_k=200,
        stages=['CR0-100','CR1-100','CR2-100','CR3-100','CR4-100','CR1-200','CR2-200'],
        evidence='Identical prior selected bundles; no graph/path/mention/selection changes; new attribution weights are final scoring only',
        costs='Historical H2 and evidence timings plus current decomposition/support/NLI/rank times; component-sum not cold end-to-end; NLI request/forward/cache work separately logged')
    write_json(out/'protocol.json',protocol)
    nli=None; model_status=dict(status='unavailable',reason='disabled in configuration')
    if cfg['constraint_nli_enabled']:
        try:
            nli=ControlledLocalNLI(cfg); model_status=dict(status='available',metadata=nli.metadata,local_only=True)
        except (ImportError,ValueError,RuntimeError,OSError) as error: model_status=dict(status='unavailable',reason=type(error).__name__+': '+str(error),local_only=True)
    encoder=LocalEncoder(cfg['evaluation_model'],cfg['evaluation_revision']); cr0=EvidenceSemanticReranker(rc)
    old_evidence={(q['cutoff'],q['question_id']):q for q in read_json(root/'candidate_evidence.json')} if same else {}
    old_oracle={(r['cutoff'],r['question_id'],r['candidate_k']):r for r in read_json(root/'reranker_oracle.json')['records']} if same else {}
    decompositions=[]
    for q in questions: decompositions.append(dict(question_id=q['question_id'],question_type=q['type'],**decompose_question(q['question'],q['type']).to_dict()))
    write_json(root/'question_decompositions.json',decompositions)
    results=[]; support_exports=[]; attribution_exports=[]; targets=[]; failures=[]; inventory=[]; oracles=[]
    for year in cfg['constraint_cutoffs']:
        snap=snapshot(data,year); old_h2={r['question_id']:r for s in old['results'] if s['cutoff']==year and s['stage']=='RR0' for r in s['records']} if same else {}
        old_cr0={r['question_id']:r for s in old['results'] if s['cutoff']==year and s['stage']=='RR1-100' for r in s['records']} if same else {}
        source=FrozenH2Candidates(snap,encoder,cc,dc,old_h2,root/'diffusion'); resolver=TargetResolver(snap)
        evidence_index=None if same else CandidateEvidenceIndex(snap,source.groups,source.operator,source.mentions,rc)
        stage_rows={s:[] for s in protocol['stages']}
        for q in aa:
            qstart=time.perf_counter(); qid=q['question_id']; h2=source.get(q['question'],qid); frozen=h2['ranked']; rank_hash=digest(frozen)
            gold=[resolver.resolve(label) for label in gt.get(qid,{}).get('expected_methods',[])]
            baseline=evaluate_ranking(frozen,gold,dict(diffusion_top_ks=rc['reranker_top_ks']))
            if same and digest(gt)==old['protocol']['ground_truth_hash'] and baseline!=old_h2[qid]['entity_metrics']: raise ValueError('Frozen H2 metrics differ')
            if same:
                original=old_evidence[year,qid]; bundles={b['entity_id']:b for b in original['candidates']}; evidence_seconds=original['evidence_seconds']
            else:
                paths,_=evidence_index.support_paths(h2['seeds']); r3r={x:i+1 for i,r in enumerate(h2['r3']) for x in r['node_ids']}
                bundles={r['entity_id']:evidence_index.build_candidate_evidence(q['question'],r,h2['similarities'],paths,i+1,min((r3r[x] for x in r['node_ids']),default=None)) for i,r in enumerate(frozen[:200])}
                evidence_seconds=time.perf_counter()-qstart-h2['replay_runtime_seconds']
            baseline_result=rerank_candidates(frozen,bundles,cr0,q['question'],100)
            cr0metric=evaluate_reranked(baseline_result['ranked'],frozen,gold,rc,100)
            if same:
                if baseline_result['ranked'][:200]!=old_cr0[qid]['returned']: raise ValueError('CR0 ranking differs from RR1')
                if digest(gt)==old['protocol']['ground_truth_hash'] and cr0metric!=old_cr0[qid]['entity_metrics']: raise ValueError('CR0 metrics differ from RR1')
            inventory.append(dict(cutoff=year,question_id=qid,H2_rank_hash=rank_hash,prefix100_hash=digest(frozen[:100]),prefix200_hash=digest(frozen[:200]),
                CR0_rank_hash=digest(baseline_result['ranked'][:200]),H2_parity=same,CR0_parity=same,bundle_hash=digest(bundles)))
            t=time.perf_counter(); decomposition=decompose_question(q['question']); component_texts=[c['text'] for c in decomposition.components]
            X=encoder.encode(component_texts,'constraint-components:'+digest(component_texts)) if component_texts else np.zeros((0,source.context.X.shape[1]))
            selected_nodes=sorted({item['node_id'] for b in bundles.values() for values in b['fields'].values() for item in values})
            node_map={n['id']:n for n in snap['nodes']}; idx=[source.context.text_idx[node_map[n]['surface_form']] for n in selected_nodes]
            values=X@source.context.X[idx].T
            similarities={c['id']:dict(zip(selected_nodes,map(float,values[i]))) for i,c in enumerate(decomposition.components)}
            requirement_encoding_seconds=time.perf_counter()-t
            cosine=RequirementCosineReranker(cfg,decomposition,similarities,source.mentions.names)
            details1={}; detail_seconds=[]
            for row in frozen[:200]:
                t=time.perf_counter(); details1[row['entity_id']]=cosine.score(q['question'],row['name'],bundles[row['entity_id']]); detail_seconds.append(time.perf_counter()-t)
            # Bounded NLI work in H2 prefix order; diagnostic tail reuses primary prefix outputs.
            predictions={}; nli_costs={}; details2={}
            for k,start in [(100,0),(200,100)]:
                pairs=[]
                for row in frozen[start:k]:
                    for requirement in details1[row['entity_id']]['requirements']:
                        pairs.extend((item['text'],requirement['hypothesis']) for item in requirement['nli_selected'])
                print('CONSTRAINT NLI',year,qid,'pool',k,'pair_requests',len(pairs),flush=True)
                probs,cost=nli.predict_controlled(pairs,cfg['constraint_nli_batch_size']) if nli else ([dict(status='unavailable',entailment=None,neutral=None,contradiction=None)]*len(pairs),dict(nli_pair_requests=len(pairs),nli_forward_pairs=0,nli_cache_hits=0,nli_seconds=0.))
                predictions.update({digest(list(pair)):p for pair,p in zip(pairs,probs)})
                nli_costs[k]={key:cost[key]+(nli_costs[100][key] if k==200 else 0) for key in cost}
            nli_scorer=RequirementNLIReranker(cfg,decomposition,details1,predictions)
            t=time.perf_counter()
            for row in frozen[:200]: details2[row['entity_id']]=nli_scorer.score(q['question'],row['name'],bundles[row['entity_id']])
            details2=uniform_nli_fallback(details1,details2)
            nli_aggregation_seconds=time.perf_counter()-t
            rankings={}; rows_this={}
            for stage in protocol['stages']:
                name,ks=stage.split('-'); k=int(ks); is_nli=name in {'CR2','CR4'}; detail=details2 if is_nli else details1
                if name=='CR0': ranked=baseline_result; metric=cr0metric
                else:
                    ranked=rerank_candidates(frozen,bundles,PreparedReranker(detail),q['question'],k,name in {'CR3','CR4'},rc['reranker_rrf_k'])
                    metric=evaluate_reranked(ranked['ranked'],frozen,gold,rc,k)
                cosine_seconds=sum(detail_seconds[:k]); supporting_seconds=requirement_encoding_seconds+cosine_seconds
                ncost=nli_costs[k] if is_nli else dict(nli_pair_requests=0,nli_forward_pairs=0,nli_cache_hits=0,nli_seconds=0.)
                selection_seconds=old_cr0[qid]['cost']['evidence_selection_seconds'] if same and k==100 else evidence_seconds
                new_seconds=(supporting_seconds+ncost['nli_seconds']+nli_aggregation_seconds if is_nli else supporting_seconds)+ranked['runtime_seconds']
                if name=='CR0': new_seconds=ranked['runtime_seconds']
                total=h2['candidate_generation_runtime']+selection_seconds+new_seconds
                cost=dict(vector_comparisons=len(component_texts)*len(selected_nodes) if name!='CR0' else 0,entity_scores=min(k,len(frozen)),
                    runtime_seconds=new_seconds,reranker_seconds=new_seconds,reranker_comparisons=min(k,len(frozen)),
                    semantic_support_comparisons=sum(details1[r['entity_id']]['semantic_support_comparisons'] for r in frozen[:k]) if name!='CR0' else 0,
                    requirement_encoding_seconds=requirement_encoding_seconds if name!='CR0' else 0,
                    candidate_generation_seconds=h2['candidate_generation_runtime'],evidence_selection_seconds=selection_seconds,total_query_seconds=total,**ncost)
                record=dict(cutoff=year,stage=name,candidate_k=k,question_id=qid,question_type='A',question=q['question'],entity_metrics=metric,
                    evidence_metrics=dict(claim_comparisons_new=0),cost=cost,latency_seconds=total,candidate_preserved=ranked['candidate_preserved'],
                    returned=ranked['ranked'][:200],nli_fallback_candidates=sum(detail[r['entity_id']]['status']!='computed' for r in frozen[:k]) if is_nli else 0)
                stage_rows[stage].append(record); rankings[stage]=ranked['ranked']; rows_this[stage]=record
            if digest(frozen)!=rank_hash: raise ValueError('Constraint reranking mutated H2')
            for k in [100,200]:
                o=dict(cutoff=year,question_id=qid,**oracle_ceiling(baseline,frozen,k))
                if same and digest(gt)==old['protocol']['ground_truth_hash'] and o!=old_oracle[year,qid,k]: raise ValueError('Oracle semantics changed')
                oracles.append(o)
            # Below this boundary: diagnostic gold membership only, never an input to scorers.
            exported=[]
            for row in frozen[:200]:
                ident=row['entity_id']; is_gold=any(g['canonical'] and set(g['resolved_node_ids'])&set(row['node_ids']) for g in gold)
                record=dict(entity_id=ident,candidate=row['name'],node_ids=row['node_ids'],h2_rank=bundles[ident]['h2_rank'],
                    ranks={stage:next(i+1 for i,r in enumerate(ranking) if r['entity_id']==ident) for stage,ranking in rankings.items()},
                    CR1=details1[ident],CR2=details2[ident])
                record['CR1']={k:v for k,v in record['CR1'].items() if k!='attributions'}
                exported.append(record)
                attribution_exports.append(dict(cutoff=year,question_id=qid,entity_id=ident,candidate=row['name'],h2_rank=bundles[ident]['h2_rank'],gold=is_gold,
                    items=details1[ident]['attributions'],CR1_coverage=details1[ident]['requirement_coverage'],CR2_coverage=details2[ident]['requirement_coverage'],
                    no_direct_attributable_evidence=details1[ident]['no_direct_attributable_evidence'],contradiction_detected=bool(details2[ident]['contradicted_requirements'])))
            support_exports.append(dict(cutoff=year,question_id=qid,original_question=q['question'],decomposition=decomposition.to_dict(),candidates=exported))
            r3rank={n:i+1 for i,row in enumerate(h2['r3']) for n in row['node_ids']}
            for unit in baseline['canonical_evaluation_units']:
                ids=set(unit['node_ids']); h2rank=next((i+1 for i,row in enumerate(frozen) if ids&set(row['node_ids'])),None)
                matching=next((r['entity_id'] for r in frozen[:200] if ids&set(r['node_ids'])),None)
                target=dict(cutoff=year,question_id=qid,label=unit['label'],node_ids=unit['node_ids'],R3_rank=min((r3rank[n] for n in ids if n in r3rank),default=None),H2_rank=h2rank,rerankings={})
                for stage,ranking in rankings.items():
                    name,ks=stage.split('-'); rank=next((i+1 for i,row in enumerate(ranking) if ids&set(row['node_ids'])),None)
                    detail=(details2 if name in {'CR2','CR4'} else details1).get(matching)
                    category,flags=classify_failure(rank,h2rank,int(ks),detail,name)
                    target['rerankings'][stage]=dict(rank=rank,rank_gain=h2rank-rank if h2rank is not None and rank is not None else None,category=category,flags=flags)
                    failures.append(dict(cutoff=year,question_id=qid,label=unit['label'],stage=stage,category=category,flags=flags))
                targets.append(target)
            write_json(out/f'checkpoint_{year}_{qid}.json',dict(source_hash=digest(source_hashes),records=rows_this,support=support_exports[-1],inventory=inventory[-1]))
            print('CONSTRAINT CHECKPOINT',year,qid,'seconds',round(time.perf_counter()-qstart,2),flush=True)
        for stage,records in stage_rows.items():
            summ=summary(records,rc)
            summ['constraint_cost']={key:float(np.mean([r['cost'][key] for r in records])) for key in ['semantic_support_comparisons','nli_pair_requests','nli_forward_pairs','nli_cache_hits','nli_seconds','runtime_seconds','total_query_seconds']}
            result=dict(cutoff=year,stage=stage,summary=summ,records=records); results.append(result); write_json(out/f'{stage}_{year}.json',result)
            print('CONSTRAINT RESULT',year,stage,'R10',summ['at_k']['10']['micro_recall'],flush=True)
    if any(sha(p)!=h for p,h in history.items()): raise ValueError('Historical artifacts changed')
    write_json(root/'constraint_reranking_results.json',dict(status='computed',protocol=protocol,nli_model=model_status,elapsed_seconds=time.perf_counter()-began,results=results,inventory=inventory,oracle=oracles))
    write_json(root/'requirement_support.json',support_exports); write_json(root/'evidence_attribution.json',attribution_exports)
    write_json(root/'constraint_reranking_per_target.json',targets); write_json(root/'constraint_failure_analysis.json',dict(target_outcomes=failures))
    current=read_json(metric_path) if metric_path.exists() else {}
    if any(digest(current.get(k))!=h for k,h in prior.items()): raise ValueError('Historical metric sections changed')
    current['constraint_reranking']=dict(protocol=protocol,nli_model=model_status,results=[{k:r[k] for k in ['cutoff','stage','summary']} for r in results],
        question_metrics=[{k:q[k] for k in ['cutoff','stage','candidate_k','question_id','entity_metrics','cost']} for r in results for q in r['records']])
    write_json(metric_path,current)
    print('Constraint evaluation complete',round(time.perf_counter()-began,1),'seconds',flush=True)
