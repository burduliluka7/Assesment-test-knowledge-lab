"""Diffusion experiments integrated with snapshots, contextual scoring and evaluation."""
from pathlib import Path
from collections import Counter
import hashlib,time
import numpy as np
from .io import read_json,read_csv,write_json,locate,digest
from .config import read_config
from .snapshots import snapshot
from .semantics import Encoder
from .retrieval import normalize as unit
from .contextual import ContextIndex,build_representations,analyze_question
from .target_mapping import TargetResolver
from .evaluation.retrieval_v2 import contextual_metrics,contextual_summary
from .v2_pipeline import verify_reuse,sha
from .diffusion import (retrieval_graph,ThetaOperator,shuffled_graph,clique_projection,semantic_seeds,
    diffuse,MentionIndex,rank_entities,rrf_fuse,propagation_diagnostics,connecting_paths)


def evaluate_ranking(ranked,resolutions,cfg):
    result=dict(ranked=ranked,candidate_entity_ids=[r['entity_id'] for r in ranked])
    metric=contextual_metrics(result,resolutions,cfg['diffusion_top_ks'])
    # No reranker: candidates are exactly the ranked prefix for every K.
    metric['candidate_at_k']={str(k):metric['at_k'][str(k)]['recall'] for k in [20,50,100,200]}
    ranks=[o['rank'] for o in metric['target_outcomes'] if o['node_bearing']]
    metric['rank_coverage']=dict(ranked=sum(r is not None for r in ranks),canonical=len(ranks),outside_scored_pool=sum(r is None for r in ranks),pool_size=len(ranked),
        censored_ranks=[r if r is not None else len(ranked)+1 for r in ranks],missing_rank_policy='pool_size+1; missing targets remain in recall denominator')
    return metric


def summarize_diffusion(rows,cfg):
    base=contextual_summary(rows,cfg['diffusion_top_ks'])['type_a']
    base['candidate_recall']={str(k):base['at_k'][str(k)]['macro_recall'] for k in [20,50,100,200]}
    base['cost']={key:float(np.mean([r['cost'].get(key,0.) for r in rows])) for key in
        ['seed_comparisons','r3_comparisons','vector_comparisons','theta_applications','incidence_entry_visits_upper','iterations','runtime_seconds']}
    base['unconverged_queries']=sum(r.get('converged') is False for r in rows)
    censored=[rank for r in rows for rank in r['entity_metrics']['rank_coverage']['censored_ranks']]
    base['rank_coverage']=dict(ranked=sum(r['entity_metrics']['rank_coverage']['ranked'] for r in rows),canonical=len(censored),
        censored_median=float(np.median(censored)) if censored else None,censored_mean=float(np.mean(censored)) if censored else None)
    return base


def run_diffusion(cfg,stages=None,sensitivity=True):
    stages=stages or cfg['diffusion_stages']
    expected=['R0','R1','R3','H0','H0_shuffled','H0_pairwise','H1','H2','H3']
    if any(s not in expected for s in stages): raise ValueError('Unknown diffusion stage')
    data=read_json(locate(cfg['data_dir'],'tkh_collection10.json')); gt=read_json(locate(cfg['data_dir'],'ground_truth.json'))
    all_questions=read_csv(locate(cfg['data_dir'],'questions.csv')); questions=[q for q in all_questions if q['type']=='A']
    context_cfg=read_config(cfg['context_config']); context_cfg.update(data_dir=cfg['data_dir'])
    if cfg['diffusion_verify_reuse']: verify_reuse(context_cfg,data)
    out=Path(cfg['diffusion_output']); out.mkdir(parents=True,exist_ok=True); root=Path(cfg['v2_output'])
    new_names={'hypergraph_diffusion_results.json','hypergraph_diffusion_diagnostics.json','propagation_traces.json','diffusion_per_target.json','diffusion_sensitivity.json'}
    historical_paths=[p for p in root.glob('*.json') if p.name not in new_names]+list((root/'contextual').rglob('*'))
    historical_paths=[p for p in historical_paths if p.is_file()]
    historical={p.as_posix():sha(p) for p in historical_paths if p.exists()}
    metrics_path=Path(cfg['output'])/'metrics.json'
    prior_metrics=read_json(metrics_path) if metrics_path.exists() else {}
    prior_metric_hashes={k:digest(v) for k,v in prior_metrics.items() if k!='hypergraph_diffusion'}
    protocol=dict(config=cfg,context_config=context_cfg,requested_stages=stages,data_hash=digest(data),questions_hash=digest(all_questions),ground_truth_hash=digest(gt),historical_artifact_hashes=historical,
        source_hashes={p.as_posix():sha(p) for p in sorted(Path('src').rglob('*.py'))},
        paper='https://proceedings.neurips.cc/paper_files/paper/2006/file/dff8e9c2ac33381546d96deea9922999-Paper.pdf',
        formulation='Our query-semantic restart adaptation; Zhou section 7 uses initial labels, not query retrieval. f=(1-alpha)*(I-alpha*Theta)^-1*y.',
        defaults='alpha=.85, M=100, tolerance=1e-8, max_iterations=100, RRF k=60 with average ranks for exact score ties; sensitivities do not select defaults',historical_metric_hashes=prior_metric_hashes,
        answer_pool='Fixed configured scientific entity types; all node types remain propagation intermediates. H0 has no post-hoc soft prior; H1/H2/H3 inherit R3 soft prior only via RRF.',
        comparison='Historical R0/R1/R3 are recomputed unchanged, plus R3_answer_pool control and atomic answer-only control to separate filtering from propagation.',
        weights='Native exports without explicit weight use 1; H0 uniform relation multipliers. H3 semantic priors fixed before outcomes; vertex degrees recomputed with weighted W.',
        costs='Seed dot products and sparse applications separate. H1/H2/H3 standalone cost sums measured components; actual incremental cache reuse also logged. Diagnostics/paths and model encoding excluded from search time.',
        sensitivity='One-at-a-time alpha with M100, and M50/200 with alpha.85, on H0 and H3. No retrospective optimum selected.',
        type_b='Existing source/evidence pipeline and all historical Type B outputs unchanged; diffusion run contains Type A only.',
        null='One arity/relation/weight-preserving membership shuffle with seed42; degrees, connectivity and type mixes are not preserved.',
        pairwise='Existing weight-conserving clique projection; same Zhou arity2 operator = half(identity + symmetric normalized adjacency) on nonisolates; not ordinary non-lazy PPR. Native Theta itself is linear and has an equivalent weighted pair expansion, so success is not proof of irreducible higher-order expressivity.')
    write_json(out/'protocol.json',protocol)
    encoder=Encoder(cfg['evaluation_model'],cfg['evaluation_revision'])
    results=[]; diagnostics=[]; targets=[]; traces=[]; sensitivities=[]; start=time.perf_counter()
    for year in cfg['diffusion_cutoffs']:
        snap=snapshot(data,year); nodes=snap['nodes']; ids=[n['id'] for n in nodes]; node_idx={x:i for i,x in enumerate(ids)}
        graph=retrieval_graph(snap); operators={}; t=time.perf_counter()
        reps=build_representations(snap,context_cfg); ungrouped=ContextIndex(snap,reps,encoder,context_cfg,False); grouped=ContextIndex(snap,reps,encoder,context_cfg,True)
        answers=[g for g in grouped.groups if set(g['types'])&set(cfg['diffusion_answer_types'])]; answer_ids={g['id'] for g in answers}
        mentions=MentionIndex(nodes,answers,cfg['diffusion_mention_sources']); index_seconds=time.perf_counter()-t
        write_json(out/f'mention_index_{year}.json',dict(names=mentions.names,mentions=mentions.mentions,index_seconds=index_seconds,
            evidence_counts={x:len(mm) for x,mm in mentions.mentions.items()},total_links=sum(map(len,mentions.mentions.values()))))
        resolver=TargetResolver(snap); resolutions={q['question_id']:[resolver.resolve(x) for x in gt[q['question_id']].get('expected_methods',[])] for q in questions}
        eligible=[i for i,n in enumerate(nodes) if n['type'] in cfg['diffusion_seed_types']]
        text_ids=sorted({grouped.text_idx[nodes[i]['surface_form']] for i in eligible})
        queries={}; stage_cache={}; saved_rows={}; filtered_rows=[]; atomic_rows=[]
        for q in questions:
            qid=q['question_id']; intent=analyze_question(q['question'],'A')
            vector=unit(encoder.encode(intent['extractive_views'],f'context-query:{digest(intent["extractive_views"])}')[0])
            t=time.perf_counter(); sims=grouped.X[text_ids]@vector; lookup=dict(zip(text_ids,map(float,sims)))
            scores=np.array([lookup.get(grouped.text_idx[n['surface_form']],0.) for n in nodes]); seed_seconds=time.perf_counter()-t
            y,seed_info=semantic_seeds(scores,nodes,cfg['diffusion_seed_top_m'],cfg['diffusion_seed_types'])
            queries[qid]=dict(question=q,intent=intent,vector=vector,scores=scores,y=y,seed_info=seed_info,seed_seconds=seed_seconds)

        def get_operator(mode):
            if mode not in operators:
                if mode=='H0_shuffled': gg=shuffled_graph(graph,cfg['diffusion_null_seed'])
                elif mode=='H0_pairwise': gg=clique_projection(graph)
                elif mode=='H3': gg=retrieval_graph(snap,cfg['diffusion_relation_weights'])
                else: gg=graph
                began=time.perf_counter(); operators[mode]=ThetaOperator(gg)
                write_json(out/f'operator_{mode}_{year}.json',dict(**operators[mode].metadata,build_seconds=time.perf_counter()-began,graph_hash=digest([(e.id,e.weight,e.original_members) for e in gg.edges])))
            return operators[mode]

        def direct(qid,stage):
            key=(qid,stage)
            if key not in stage_cache:
                qq=queries[qid]; ix=grouped if stage=='R3' else ungrouped
                stage_cache[key]=ix.search([qq['vector']],qq['intent'],surface=stage=='R0',type_enabled=stage=='R3')
            return stage_cache[key]

        def propagated(qid,stage,alpha=None,top_m=None,record_diagnostics=True):
            alpha=cfg['diffusion_alpha'] if alpha is None else alpha; top_m=cfg['diffusion_seed_top_m'] if top_m is None else top_m
            mode='H0' if stage=='H1' else stage
            cache_key=(qid,mode,alpha,top_m)
            reused=cache_key in stage_cache; qq=queries[qid]
            if not reused:
                op=get_operator(mode if mode in {'H0_shuffled','H0_pairwise','H3'} else 'H0')
                prepare_start=time.perf_counter()
                y,seed_info=semantic_seeds(qq['scores'],nodes,top_m,cfg['diffusion_seed_types']); transfer=dict(transfers=[],normalizer=1.)
                if mode in {'H2','H3'}: y,transfer=mentions.assist(y,answers,ids)
                preparation_seconds=time.perf_counter()-prepare_start
                f,run=diffuse(op,y,alpha,cfg['diffusion_tolerance'],cfg['diffusion_max_iterations'],[n['type'] for n in nodes])
                rank_start=time.perf_counter()
                ranked=rank_entities(f,answers,ids)
                stage_cache[cache_key]=dict(ranked=ranked,f=f,y=y,run=run,seed_info=seed_info,transfer=transfer,op=op,
                    preparation_seconds=preparation_seconds,rank_seconds=time.perf_counter()-rank_start)
            stored=stage_cache[cache_key]; ranked=stored['ranked']; r3_seconds=0.; r3_comparisons=0
            fusion_start=time.perf_counter(); fallback_fusion=False
            if stage in {'H1','H2','H3'}:
                r3=direct(qid,'R3'); filtered=[r for r in r3['ranked'] if r['entity_id'] in answer_ids]
                fusion_start=time.perf_counter(); fallback_fusion=stored['seed_info']['fallback'] is not None
                ranked=filtered if fallback_fusion else rrf_fuse(filtered,ranked,cfg['diffusion_rrf_k']); r3_seconds=r3['latency_seconds']; r3_comparisons=r3['cost']['vector_comparisons']
            run=stored['run']; cost=dict(seed_comparisons=len(text_ids),r3_comparisons=r3_comparisons,vector_comparisons=len(text_ids)+r3_comparisons,
                entity_scores=len(answers),theta_applications=run['theta_applications'],iterations=run['iterations'],incidence_entry_visits_upper=run['incidence_entry_visits_upper'],
                runtime_seconds=qq['seed_seconds']+stored['preparation_seconds']+run['runtime_seconds']+stored['rank_seconds']+r3_seconds+time.perf_counter()-fusion_start,
                actual_new_theta_applications=0 if reused else run['theta_applications'],shared_seed_scores_reused=True,diagnostic_extra_incidence_multiplies=2 if record_diagnostics else 0,
                seed_fallback=stored['seed_info']['fallback'],fusion_fallback_to_R3=fallback_fusion)
            if record_diagnostics:
                score_path=out/f'node_scores_{stage}_{year}_{qid}.npz'
                np.savez_compressed(score_path,node_ids=np.asarray(ids),y=stored['y'],f=stored['f'],semantic_scores=qq['scores'],degree=stored['op'].degree)
                diagnostics.append(dict(cutoff=year,question_id=qid,stage=stage,alpha=alpha,seed_top_m=top_m,semantic_seeds=stored['seed_info'],mention_assistance=stored['transfer'],
                    final_seed_support=int(sum(stored['y']>0)),post_assist_seed_l1=float(stored['y'].sum()),node_scores_artifact=score_path.as_posix(),
                    **propagation_diagnostics(stored['op'],stored['f'],nodes,run)))
            return ranked,cost,stored

        for stage in stages:
            rows=[]
            for q in questions:
                qid=q['question_id']; qq=queries[qid]
                if stage.startswith('R'):
                    result=direct(qid,stage); ranked=result['ranked']; cost=dict(result['cost'],seed_comparisons=0,r3_comparisons=result['cost']['vector_comparisons'],theta_applications=0,iterations=0,runtime_seconds=result['latency_seconds'])
                    converged=None
                else:
                    ranked,cost,stored=propagated(qid,stage); converged=stored['run']['converged']
                metric=evaluate_ranking(ranked,resolutions[qid],cfg)
                row=dict(cutoff=year,stage=stage,question_id=qid,question_type='A',question=q['question'],entity_metrics=metric,evidence_metrics=dict(claim_comparisons_new=0),
                    cost=cost,latency_seconds=cost['runtime_seconds'],converged=converged,returned=ranked[:200])
                rows.append(row); saved_rows[(qid,stage)]=(row,ranked)
            summary=summarize_diffusion(rows,cfg); results.append(dict(cutoff=year,stage=stage,summary=summary,records=rows))
            write_json(out/f'{stage}_{year}.json',results[-1])
            print('DIFFUSION CHECKPOINT',year,stage,'C50/100/200',summary['candidate_recall'],'R10',summary['at_k']['10']['micro_recall'],'median',summary['rank_quantiles']['median'] if summary['rank_quantiles'] else None,flush=True)
            write_json(root/'hypergraph_diffusion_results.json',dict(status='in_progress',protocol=protocol,results=results))

        # Output-pool and atomic no-propagation controls, not extra tuned variants.
        for q in questions:
            qid=q['question_id']; r3=direct(qid,'R3'); filtered=[r for r in r3['ranked'] if r['entity_id'] in answer_ids]
            atomic=rank_entities(queries[qid]['scores'],answers,ids)
            for name,ranking,collection in [('R3_answer_pool',filtered,filtered_rows),('atomic_answer_pool',atomic,atomic_rows)]:
                cost=dict(vector_comparisons=r3['cost']['vector_comparisons'] if name.startswith('R3') else len(text_ids),entity_scores=len(answers),runtime_seconds=0.)
                collection.append(dict(cutoff=year,stage=name,question_id=qid,question_type='A',entity_metrics=evaluate_ranking(ranking,resolutions[qid],cfg),evidence_metrics=dict(claim_comparisons_new=0),cost=cost,latency_seconds=0.))
        write_json(out/f'pool_controls_{year}.json',{name:dict(summary=summarize_diffusion(rows,cfg),records=rows) for name,rows in [('R3_answer_pool',filtered_rows),('atomic_answer_pool',atomic_rows)]})

        # Evaluation-only analysis: canonical labels do not enter any search operation.
        for q in questions:
            qid=q['question_id']; qq=queries[qid]
            for res in resolutions[qid]:
                if not res['canonical']: continue
                matching=set(res['resolved_node_ids']); target=dict(cutoff=year,question_id=qid,gold_label=res['label'],resolved_node_ids=res['resolved_node_ids'],
                    entity_group_ids=[g['id'] for g in answers if matching&set(g['node_ids'])],ranks={},scores={},support={})
                for stage in stages:
                    row,ranking=saved_rows[(qid,stage)]; hits=[(i+1,r) for i,r in enumerate(ranking) if matching&set(r['node_ids'])]
                    target['ranks'][stage]=hits[0][0] if hits else None; target['scores'][stage]=hits[0][1]['score'] if hits else None
                filtered=[r for r in direct(qid,'R3')['ranked'] if r['entity_id'] in answer_ids]
                atomic=rank_entities(qq['scores'],answers,ids)
                target['ranks']['R3_answer_pool']=next((i+1 for i,r in enumerate(filtered) if matching&set(r['node_ids'])),None)
                target['ranks']['atomic_answer_pool']=next((i+1 for i,r in enumerate(atomic) if matching&set(r['node_ids'])),None)
                for mode in ['H0','H2','H3']:
                    key=(qid,mode,cfg['diffusion_alpha'],cfg['diffusion_seed_top_m'])
                    if key not in stage_cache: continue
                    stored=stage_cache[key]; seeds=sorted([dict(node_id=n['id'],type=n['type'],surface_form=n['surface_form'],semantic_score=float(qq['scores'][i]),y=float(stored['y'][i])) for i,n in enumerate(nodes) if stored['y'][i]>0],key=lambda r:(-r['y'],r['node_id']))
                    paths=connecting_paths(stored['op'],seeds,matching)
                    seed_ranked=rank_entities(stored['y'],answers,ids)
                    seed_rank=next((i+1 for i,r in enumerate(seed_ranked) if r['score']>0 and matching&set(r['node_ids'])),None)
                    support=dict(paths=paths,mention_transfers=[t for t in stored['transfer']['transfers'] if t['entity_id'] in target['entity_group_ids']],
                        entity_seed_before_diffusion=max(float(stored['y'][node_idx[x]]) for x in matching),entity_diffusion_score=max(float(stored['f'][node_idx[x]]) for x in matching),
                        top_connecting_relations=dict(Counter(step['relation'] for p in paths for step in p['path'])),path_found_within_four_hops=any(p['hops']>0 for p in paths),seed_is_target=any(p['hops']==0 for p in paths),
                        rank_seed_only_if_positive=seed_rank,rank_atomic_similarity=target['ranks']['atomic_answer_pool'],
                        raw_diffusion_rank=next((i+1 for i,r in enumerate(stored['ranked']) if matching&set(r['node_ids'])),None))
                    target['support'][mode]=support
                    traces.append(dict(cutoff=year,question_id=qid,gold_label=res['label'],stage=mode,rank_before=target['ranks']['R3_answer_pool'],rank_r3_all_types=target['ranks'].get('R3'),rank_after=target['ranks'].get(mode),**support))
                rank=target['ranks'].get('R3'); fields={f for x in matching for f,items in reps[x]['fields'].items() if items}
                target['categories']=[('R3_top10' if rank is not None and rank<=10 else 'R3_11_100' if rank is not None and rank<=100 else 'R3_above100_or_missing')]
                if any(nodes[node_idx[x]]['type']=='cited_work' for x in matching) and len(fields)<=2: target['categories'].append('sparse_cited_work')
                if target['support'].get('H0',{}).get('path_found_within_four_hops'): target['categories'].append('positive_seed_path_within_four_hops')
                else: target['categories'].append('no_top_seed_path_within_four_hops')
                targets.append(target)

        if sensitivity:
            pairs=sorted({(a,cfg['diffusion_seed_top_m']) for a in cfg['diffusion_alpha_sensitivity']}|{(cfg['diffusion_alpha'],m) for m in cfg['diffusion_seed_sensitivity']})
            for mode in cfg['diffusion_sensitivity_stages']:
                if mode not in stages: continue
                for alpha,top_m in pairs:
                    rows=[]
                    for q in questions:
                        qid=q['question_id']; ranked,cost,stored=propagated(qid,mode,alpha,top_m,False)
                        rows.append(dict(question_id=qid,question_type='A',entity_metrics=evaluate_ranking(ranked,resolutions[qid],cfg),evidence_metrics=dict(claim_comparisons_new=0),cost=cost,latency_seconds=cost['runtime_seconds'],converged=stored['run']['converged']))
                    sensitivities.append(dict(cutoff=year,stage=mode,alpha=alpha,seed_top_m=top_m,default=alpha==cfg['diffusion_alpha'] and top_m==cfg['diffusion_seed_top_m'],summary=summarize_diffusion(rows,cfg),question_metrics=rows))
    assert all(sha(p)==h for p,h in historical.items()),'Historical contextual artifact changed'
    bundle=dict(status='computed',protocol=protocol,elapsed_seconds=time.perf_counter()-start,results=results)
    write_json(root/'hypergraph_diffusion_results.json',bundle)
    write_json(root/'hypergraph_diffusion_diagnostics.json',diagnostics)
    write_json(root/'diffusion_per_target.json',targets); write_json(root/'propagation_traces.json',traces)
    write_json(root/'diffusion_sensitivity.json',dict(protocol=protocol['sensitivity'],records=sensitivities))
    metrics=read_json(metrics_path) if metrics_path.exists() else {}
    assert all(digest(metrics.get(k))==h for k,h in prior_metric_hashes.items()),'Pre-existing metrics changed during diffusion run'
    metrics['hypergraph_diffusion']=dict(protocol=protocol,elapsed_seconds=bundle['elapsed_seconds'],results=[dict(cutoff=r['cutoff'],stage=r['stage'],summary=r['summary']) for r in results],
        question_metrics=[{k:r[k] for k in ['cutoff','stage','question_id','question_type','entity_metrics','cost']} for result in results for r in result['records']],
        type_b_status='unchanged historical evidence pipeline; not included in diffusion experiments')
    write_json(metrics_path,metrics)
    print('Diffusion evaluation complete',round(bundle['elapsed_seconds'],1),'seconds',flush=True)
