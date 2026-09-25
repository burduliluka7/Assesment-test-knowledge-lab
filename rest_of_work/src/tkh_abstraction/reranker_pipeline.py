"""Frozen H2 candidate reranking and evaluation, preserving previous experiment files."""
from pathlib import Path
from collections import Counter
import time
import numpy as np
from .io import read_json,read_csv,write_json,locate,digest
from .config import read_config
from .snapshots import snapshot
from .semantics import Encoder
from .evaluation.reranker_oracle import oracle_ceiling
from .contextual import ContextIndex,build_representations,analyze_question
from .retrieval import normalize as unit
from .target_mapping import TargetResolver
from .diffusion import ThetaOperator,retrieval_graph,MentionIndex,semantic_seeds,diffuse,rank_entities,rrf_fuse
from .diffusion_pipeline import evaluate_ranking,summarize_diffusion
from .v2_pipeline import sha,verify_reuse
from .candidate_evidence import CandidateEvidenceIndex
from .reranking import inspect_local_reranker,EvidenceSemanticReranker,LocalCrossEncoderReranker,rerank_candidates


class LocalEncoder(Encoder):
    """Same BGE vectors and cache identity, with explicit offline loading."""
    def load(self):
        if self.name=='synthetic-hash': return super().load()
        if self.model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            torch.set_num_threads(4); torch.manual_seed(0)
            self.model=SentenceTransformer(self.name,revision=self.revision,device='cpu',local_files_only=True)
            self.model.eval()
            self.metadata['resolved_revision']=getattr(self.model[0].auto_model.config,'_commit_hash',None)


class FrozenH2Candidates:
    """Replay archived node relevance, or compose the unchanged H2 functions for new queries."""
    def __init__(self,snap,encoder,context_cfg,diffusion_cfg,historical_rows=None,archive_dir=None):
        self.snap=snap; self.cfg=diffusion_cfg; self.encoder=encoder; self.archive_dir=Path(archive_dir) if archive_dir else None
        self.rows=historical_rows or {}; self.nodes=snap['nodes']; self.ids=[n['id'] for n in self.nodes]
        self.context=ContextIndex(snap,build_representations(snap,context_cfg),encoder,context_cfg,True)
        self.groups=[g for g in self.context.groups if set(g['types'])&set(diffusion_cfg['diffusion_answer_types'])]
        self.answer_ids={g['id'] for g in self.groups}; self.operator=ThetaOperator(retrieval_graph(snap))
        self.mentions=MentionIndex(self.nodes,self.groups,diffusion_cfg['diffusion_mention_sources'])

    def get(self,question,qid):
        began=time.perf_counter(); intent=analyze_question(question,'A'); cfg=self.cfg
        q=unit(self.encoder.encode(intent['extractive_views'],f'context-query:{digest(intent["extractive_views"])}')[0])
        r3=self.context.search([q],intent,type_enabled=True); filtered=[r for r in r3['ranked'] if r['entity_id'] in self.answer_ids]
        historic=self.rows.get(qid); source='computed_for_new_query'
        if historic is not None:
            if historic['question']!=question: raise ValueError('Frozen H2 question text mismatch')
            path=self.archive_dir/f'node_scores_H2_{self.snap["snapshot"]}_{qid}.npz'
            with np.load(path) as data:
                if data['node_ids'].tolist()!=self.ids: raise ValueError('Frozen H2 snapshot node-order mismatch')
                f=data['f'].copy(); scores=data['semantic_scores'].copy(); y=data['y'].copy()
            source='frozen_historical_H2_node_scores'
            _,seed_info=semantic_seeds(scores,self.nodes,cfg['diffusion_seed_top_m'],cfg['diffusion_seed_types'])
        else:
            text_ids=sorted({self.context.text_idx[n['surface_form']] for n in self.nodes if n['type'] in cfg['diffusion_seed_types']})
            sims=self.context.X[text_ids]@q; lookup=dict(zip(text_ids,map(float,sims)))
            scores=np.array([lookup.get(self.context.text_idx[n['surface_form']],0.) for n in self.nodes])
            own,seed_info=semantic_seeds(scores,self.nodes,cfg['diffusion_seed_top_m'],cfg['diffusion_seed_types'])
            y,_=self.mentions.assist(own,self.groups,self.ids)
            f,run=diffuse(self.operator,y,cfg['diffusion_alpha'],cfg['diffusion_tolerance'],cfg['diffusion_max_iterations'],[n['type'] for n in self.nodes])
        propagated=rank_entities(f,self.groups,self.ids)
        ranking=filtered if seed_info['fallback'] else rrf_fuse(filtered,propagated,cfg['diffusion_rrf_k'])
        if historic:
            if ranking[:200]!=historic['returned']: raise ValueError('Frozen H2 prefix/rank/score changed')
        elapsed=time.perf_counter()-began
        return dict(ranked=ranking,r3=r3['ranked'],similarities=dict(zip(self.ids,map(float,scores))),seeds=seed_info['top_seeds'],
            y=y,source=source,candidate_generation_runtime=historic['cost']['runtime_seconds'] if historic else elapsed,
            replay_runtime_seconds=elapsed,rank_hash=digest(ranking),historic=historic)


def evaluate_reranked(ranking,frozen,resolutions,cfg,candidate_k=100):
    from .evaluation.retrieval_v2 import contextual_metrics
    metric=evaluate_ranking(ranking,resolutions,dict(diffusion_top_ks=cfg['reranker_top_ks']))
    frozen_metric=evaluate_ranking(frozen,resolutions,dict(diffusion_top_ks=cfg['reranker_top_ks']))
    candidate_correct=contextual_metrics(dict(ranked=ranking,candidate_entity_ids=[r['entity_id'] for r in frozen]),resolutions,cfg['reranker_top_ks'])
    for key in ['candidate_recall_50','candidate_recall_100','retrieval_recall_given_candidate','target_outcomes']:
        metric[key]=candidate_correct[key]
    metric['candidate_at_k']=frozen_metric['candidate_at_k']
    metric['frozen_candidate_recall_100']=frozen_metric['candidate_at_k']['100']
    metric['frozen_candidate_recall_200']=frozen_metric['candidate_at_k']['200']
    metric['retrieval_recall_given_candidate_k']=candidate_k
    pool_nodes={x for row in frozen[:candidate_k] for x in row['node_ids']}
    top_nodes={x for row in ranking[:10] for x in row['node_ids']}
    units=metric['canonical_evaluation_units']; present=[u for u in units if set(u['node_ids'])&pool_nodes]
    metric['retrieval_recall_given_candidate']=sum(bool(set(u['node_ids'])&top_nodes) for u in present)/len(present) if present else None
    metric['candidate_recall_at_reranker_k']=len(present)/len(units) if units else None
    original={o['label']:o for o in frozen_metric['target_outcomes']}
    for outcome in metric['target_outcomes']:
        prior=original[outcome['label']]; present=prior['rank'] is not None and prior['rank']<=candidate_k
        outcome['reranker_candidate_k']=candidate_k; outcome['in_reranker_candidates']=present
        outcome['legacy_outcome_at_candidate100']=outcome['outcome']
        if outcome['representable']:
            outcome['outcome']='RETRIEVED' if outcome['rank'] is not None and outcome['rank']<=10 else 'CANDIDATE_BUT_BELOW_K' if present else 'RESOLVED_BUT_NOT_CANDIDATE'
        outcome['legacy_candidate100_failure_reason']=outcome['candidate_failure_reason']
        outcome['candidate_failure_reason']=None if present else 'rank_below_reranker_pool' if prior['rank'] is not None else 'not_in_scored_pool' if outcome['representable'] else 'mapping_unavailable'
    return metric


def summary(rows,cfg):
    result=summarize_diffusion(rows,dict(diffusion_top_ks=cfg['reranker_top_ks']))
    def mean_defined(values):
        values=[v for v in values if v is not None]
        return float(np.mean(values)) if values else None
    result['candidate_recall']={str(k):mean_defined([r['entity_metrics']['candidate_at_k'][str(k)] for r in rows]) for k in [20,50,100,200]}
    result['reranking_cost']={key:float(np.mean([r['cost'].get(key,0.) for r in rows])) for key in
        ['reranker_comparisons','evidence_selection_dots_new','evidence_selection_scores_reused','evidence_selection_seconds','reranker_seconds','total_query_seconds','candidate_generation_seconds']}
    result['candidate_preservation_rate']=float(np.mean([r.get('candidate_preserved',True) for r in rows]))
    return result


def failure_category(h2_rank,new_rank,candidate_k,specific_count,r3_rank=None):
    if h2_rank is None or h2_rank>candidate_k: return f'NOT_IN_H2_{candidate_k}'
    if new_rank is not None and new_rank<=10: return 'RETRIEVED_TOP10'
    if (h2_rank<=10 or r3_rank is not None and r3_rank<=10) and new_rank is not None and new_rank>10: return 'RERANK_DEGRADED_EXISTING_GOOD_RESULT'
    if specific_count==0: return 'IN_CANDIDATES_NO_SPECIFIC_EVIDENCE'
    if new_rank is not None and new_rank<h2_rank: return 'RERANK_IMPROVED_BUT_BELOW_10'
    return 'IN_CANDIDATES_EVIDENCE_PRESENT_RERANK_FAILED'


def run_reranking(cfg,questions_path=None,truth_path=None):
    began=time.perf_counter(); root=Path(cfg['v2_output']); out=Path(cfg['reranker_output']); out.mkdir(parents=True,exist_ok=True)
    cc=read_config(cfg['context_config']); dc=read_config(cfg['diffusion_config']); data=read_json(locate(cfg['data_dir'],'tkh_collection10.json'))
    questions=read_csv(questions_path or locate(cfg['data_dir'],'questions.csv')); gt=read_json(truth_path or locate(cfg['data_dir'],'ground_truth.json'))
    aa=[q for q in questions if q['type']=='A']; metrics_path=Path(cfg['output'])/'metrics.json'
    metrics=read_json(metrics_path) if metrics_path.exists() else {}; prior_metrics={k:digest(v) for k,v in metrics.items() if k!='candidate_reranking'}
    if cfg['reranker_verify_history']: verify_reuse(cc,data)
    historical_sources=['diffusion.py','diffusion_pipeline.py','contextual.py','target_mapping.py','snapshots.py','spectral.py','hypergraph.py','baselines.py']
    old=read_json(root/'hypergraph_diffusion_results.json') if (root/'hypergraph_diffusion_results.json').exists() else None
    same_benchmark=old is not None and digest(data)==old['protocol']['data_hash'] and digest(questions)==old['protocol']['questions_hash']
    if cfg['reranker_verify_history']:
        if old is None: raise ValueError('Expected frozen diffusion baseline missing')
        for name in historical_sources:
            path='src/tkh_abstraction/'+name
            if sha(path)!=old['protocol']['source_hashes'][path]: raise ValueError('Frozen H2 source changed: '+name)
        if dc!=old['protocol']['config']: raise ValueError('Frozen H2 configuration changed')
        if cc!=old['protocol']['context_config']: raise ValueError('Frozen R3 configuration changed')
    fresh_names={'reranker_results.json','reranker_per_target.json','candidate_evidence.json','reranker_failure_analysis.json','reranker_oracle.json'}
    historical_paths=[p for p in root.rglob('*') if p.is_file() and out not in p.parents and p.name not in fresh_names|{'reranker_tests.xml'}]
    history={p.as_posix():sha(p) for p in historical_paths}
    models=inspect_local_reranker(cfg['reranker_cross_encoder_path']); scorer=EvidenceSemanticReranker(cfg)
    ce=LocalCrossEncoderReranker(models['selected_path'],cfg['reranker_document_tokens'],cfg['reranker_cross_encoder_relevance_label']) if models['status']=='available' else None
    fusion_scorer=ce or scorer; fusion_source='RR2' if ce else 'RR1'
    protocol=dict(config=cfg,diffusion_config=dc,context_config=cc,data_hash=digest(data),questions_hash=digest(questions),ground_truth_hash=digest(gt),
        historical_hashes=history,historical_metrics_hashes=prior_metrics,source_hashes={p.as_posix():sha(p) for p in sorted(Path('src').rglob('*.py'))},
        frozen_h2_sources={n:sha('src/tkh_abstraction/'+n) for n in historical_sources},cross_encoder=models,rr3_source=fusion_source,
        rules=dict(h2='Exact current alpha.85/M100/mention/uniform-weight/R3 fusion; replayed or recomposed unchanged. Prefix hashes checked before scoring.',
            fields='Fixed weights; available-field normalized nonnegative cosine, multiplied by .5+.5*sqrt(available_weight/total_weight).',
            evidence='Direct/explicit-name > task/problem > technical/data > presents title > short-path source > provenance title fallback. Generic bridges never copied as semantic evidence.',
            oracle='Evaluation-only ceiling; no expected target labels enter candidate generation, evidence or reranker.',
            costs='Atomic similarities reused from H2; evidence lookups, path expansion and scalar field scoring timed separately. Reranker comparisons count candidate score calls, not embedding dots.',
            tail='Only top-K order changes; complete H2 tail remains untouched. Candidate recalls always use original H2 order.',
            type_b='Historical Type B evidence retrieval unchanged; no mixed-type configuration selection.',rr2='Only local relevance-trained model; no downloads or NLI substitution',
            development='Fixed design choices, no fitting or sweeping on the repeatedly used benchmark; new-question CLI supported'))
    write_json(out/'protocol.json',protocol); encoder=LocalEncoder(cfg['evaluation_model'],cfg['evaluation_revision'])
    all_results=[]; bundles_export=[]; target_records=[]; oracle_records=[]; failures=[]; evidence_diagnostics=[]; inventory=[]
    for year in cfg['reranker_cutoffs']:
        snap=snapshot(data,year); year_rows={r['question_id']:r for result in (old['results'] if same_benchmark else []) if result['cutoff']==year and result['stage']=='H2' for r in result['records']}
        source=FrozenH2Candidates(snap,encoder,cc,dc,year_rows,root/'diffusion')
        evidence_index=CandidateEvidenceIndex(snap,source.groups,source.operator,source.mentions,cfg)
        write_json(out/f'genericness_{year}.json',dict(hub_threshold=evidence_index.hub_threshold,nodes=evidence_index.generic))
        resolver=TargetResolver(snap); stage_rows={}
        for q in aa:
            qid=q['question_id']; h2=source.get(q['question'],qid); frozen=h2['ranked']; before_hash=digest(frozen)
            gold=[resolver.resolve(x) for x in gt.get(qid,{}).get('expected_methods',[])]
            base_metric=evaluate_ranking(frozen,gold,dict(diffusion_top_ks=cfg['reranker_top_ks']))
            if h2['historic']:
                if base_metric!=h2['historic']['entity_metrics']: raise ValueError('RR0 fails exact historical H2 metric parity')
            inventory.append(dict(cutoff=year,question_id=qid,rank_hash=before_hash,prefix100_hash=digest(frozen[:100]),prefix200_hash=digest(frozen[:200]),h2_source=h2['source'],historical_metric_parity=h2['historic'] is not None))
            evidence_start=time.perf_counter(); paths,path_cost=evidence_index.support_paths(h2['seeds']); path_seconds=time.perf_counter()-evidence_start; bundle_seconds=[]; bundles={}; rank_r3={x:i+1 for i,r in enumerate(h2['r3']) for x in r['node_ids']}
            for i,candidate in enumerate(frozen[:max(cfg['reranker_candidate_ks'])]):
                bundle_start=time.perf_counter()
                bundle=evidence_index.build_candidate_evidence(q['question'],candidate,h2['similarities'],paths,i+1,min((rank_r3[x] for x in candidate['node_ids']),default=None))
                bundles[candidate['entity_id']]=bundle
                bundle_seconds.append(time.perf_counter()-bundle_start)
            evidence_seconds=time.perf_counter()-evidence_start
            bundles_export.append(dict(cutoff=year,question_id=qid,candidate_rank_hash=before_hash,path_cost=path_cost,evidence_seconds=evidence_seconds,candidates=list(bundles.values())))
            base_cost=dict(h2['historic']['cost']) if h2['historic'] else {}
            base_cost.update(entity_scores=len(frozen),runtime_seconds=h2['candidate_generation_runtime'],vector_comparisons=base_cost.get('vector_comparisons',0),
                candidate_generation_seconds=h2['candidate_generation_runtime'],total_query_seconds=h2['candidate_generation_runtime'])
            baseline_row=dict(cutoff=year,stage='RR0',candidate_k=None,question_id=qid,question_type='A',question=q['question'],entity_metrics=base_metric,evidence_metrics=dict(claim_comparisons_new=0),
                cost=base_cost,latency_seconds=base_cost['runtime_seconds'],candidate_preserved=True,returned=frozen[:200])
            stage_rows.setdefault('RR0',[]).append(baseline_row)
            evaluated={('RR0',None):(baseline_row,frozen)}
            for k in cfg['reranker_candidate_ks']:
                oracle_records.append(dict(cutoff=year,question_id=qid,**oracle_ceiling(base_metric,frozen,k)))
                pool_evidence_seconds=path_seconds+sum(bundle_seconds[:k])
                for name,engine,fuse in [('RR1',scorer,False)]+([('RR2',ce,False)] if ce else [])+[('RR3',fusion_scorer,True)]:
                    result=rerank_candidates(frozen,bundles,engine,q['question'],k,fuse,cfg['reranker_rrf_k'])
                    metric=evaluate_reranked(result['ranked'],frozen,gold,cfg,k)
                    selected_ids={node for cand in frozen[:k] for node in bundles[cand['entity_id']]['similarity_ids_consulted']}
                    cost=dict(vector_comparisons=0,entity_scores=k,runtime_seconds=result['runtime_seconds'],reranker_comparisons=result['comparisons'],
                        evidence_selection_dots_new=0,evidence_selection_scores_reused=len(selected_ids),evidence_selection_seconds=pool_evidence_seconds,
                        reranker_seconds=result['runtime_seconds'],candidate_generation_seconds=h2['candidate_generation_runtime'],
                        total_query_seconds=h2['candidate_generation_runtime']+pool_evidence_seconds+result['runtime_seconds'])
                    row=dict(cutoff=year,stage=name,candidate_k=k,question_id=qid,question_type='A',question=q['question'],entity_metrics=metric,evidence_metrics=dict(claim_comparisons_new=0),
                        cost=cost,latency_seconds=cost['total_query_seconds'],candidate_preserved=result['candidate_preserved'],returned=result['ranked'][:200],score_details=result['details'])
                    stage_rows.setdefault(f'{name}-{k}',[]).append(row); evaluated[(name,k)]=(row,result['ranked'])
            if digest(frozen)!=before_hash: raise ValueError('Reranker mutated frozen H2 candidates')
            # Everything below is evaluation-only; candidate evidence above never saw gold.
            for cand in frozen[:200]:
                b=bundles.get(cand['entity_id'])
                if b is None: continue
                canonical_gold=any(r['canonical'] and set(r['resolved_node_ids'])&set(cand['node_ids']) for r in gold)
                evidence_diagnostics.append(dict(cutoff=year,question_id=qid,entity_id=cand['entity_id'],name=cand['name'],gold=canonical_gold,
                    h2_rank=b['h2_rank'],specific_evidence_count=b['specific_evidence_count'],counts={f:len(ii) for f,ii in b['fields'].items()},
                    available_counts=b['available_counts'],best_similarity_by_field=b['best_similarity_by_field']))
            for res in gold:
                if not res['canonical']: continue
                nodes=set(res['resolved_node_ids']); matching=[r for r in frozen if nodes&set(r['node_ids'])]
                h2rank=next((i+1 for i,r in enumerate(frozen) if nodes&set(r['node_ids'])),None)
                r3rank=min((rank_r3[x] for x in nodes if x in rank_r3),default=None)
                bundle_ids=[r['entity_id'] for r in matching if r['entity_id'] in bundles]
                specific=sum(bundles[x]['specific_evidence_count'] for x in bundle_ids)
                target=dict(cutoff=year,question_id=qid,label=res['label'],node_ids=res['resolved_node_ids'],entity_groups=[r['entity_id'] for r in matching],R3_rank=r3rank,H2_rank=h2rank,
                    in_h2_100=h2rank is not None and h2rank<=100,in_h2_200=h2rank is not None and h2rank<=200,specific_evidence_count=specific,rerankings={})
                for (name,k),(row,ranking) in evaluated.items():
                    if name=='RR0': continue
                    rank=next((i+1 for i,r in enumerate(ranking) if nodes&set(r['node_ids'])),None)
                    pool_specific=sum(bundles[x]['specific_evidence_count'] for x in bundle_ids if bundles[x]['h2_rank']<=k)
                    category=failure_category(h2rank,rank,k,pool_specific,r3rank)
                    target['rerankings'][f'{name}-{k}']=dict(rank=rank,rank_gain=h2rank-rank if h2rank is not None and rank is not None else None,category=category)
                    failures.append(dict(cutoff=year,question_id=qid,label=res['label'],stage=name,candidate_k=k,h2_rank=h2rank,r3_rank=r3rank,rank=rank,category=category))
                target_records.append(target)
        for stage,rows in stage_rows.items():
            result=dict(cutoff=year,stage=stage,summary=summary(rows,cfg),records=rows); all_results.append(result)
            write_json(out/f'{stage}_{year}.json',result)
            print('RERANK CHECKPOINT',year,stage,'microR5/10/20',[result['summary']['at_k'][str(k)]['micro_recall'] for k in [5,10,20]],'MRR',result['summary']['mrr'],flush=True)
    if any(sha(p)!=h for p,h in history.items()): raise ValueError('Historical evaluation artifact changed')
    write_json(root/'reranker_results.json',dict(status='computed',protocol=protocol,elapsed_seconds=time.perf_counter()-began,results=all_results,RR2_status=models,inventory=inventory))
    write_json(root/'candidate_evidence.json',bundles_export); write_json(root/'reranker_per_target.json',target_records)
    write_json(root/'reranker_oracle.json',dict(status='evaluation_only',records=oracle_records))
    write_json(root/'reranker_failure_analysis.json',dict(target_outcomes=failures,evidence_presence=evidence_diagnostics))
    current=read_json(metrics_path) if metrics_path.exists() else {}
    if any(digest(current.get(k))!=h for k,h in prior_metrics.items()): raise ValueError('Historical metric section changed')
    current['candidate_reranking']=dict(protocol=protocol,results=[{k:r[k] for k in ['cutoff','stage','summary']} for r in all_results],
        question_metrics=[{k:row[k] for k in ['cutoff','stage','candidate_k','question_id','question_type','entity_metrics','cost']} for r in all_results for row in r['records']],RR2_status=models)
    write_json(metrics_path,current); print('Reranking evaluation complete',round(time.perf_counter()-began,1),'seconds',flush=True)
