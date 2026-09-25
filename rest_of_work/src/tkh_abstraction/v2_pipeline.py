"""Audited evaluation pass; never modifies hierarchy construction or memberships."""
from pathlib import Path
import hashlib,os,time
from .io import read_json,write_json,read_csv,locate,digest
from .snapshots import snapshot
from .hierarchy import validate_hierarchy
from .semantics import Encoder,node_texts
from .target_mapping import map_benchmark,mapping_summary,route_question
from .search_v2 import SearchIndex,flat_search,retrieve_claims
from .entailment import NLI
from .evaluation.claim_alignment import align,coverage
from .evaluation.retrieval_v2 import entity_metrics,claim_metrics,summarize,question_evaluability
from .labels_v2 import label_overlay,evaluate_labels

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify_reuse(cfg,data):
    manifest=read_json('artifacts/baseline_v1/manifest.json')
    for p,h in {**manifest['core_sources'],**manifest['artifact_sha256']}.items():
        if sha(p)!=h: raise ValueError(f'Baseline reuse rejected: changed {p}')
    if digest(data)!=manifest['data_hash']: raise ValueError('Baseline data changed')
    keys=['snapshots','budgets','seed','spectral_dim','neighbors','weights','relation_weights','embedding_model','embedding_revision','include_type','match_threshold','event_threshold']
    for key in keys:
        if cfg[key]!=manifest['construction_config'][key]: raise ValueError(f'Construction config changed: {key}')
    for p in manifest['partition_hashes']:
        h=read_json(p); validate_hierarchy(h,{n['id'] for n in snapshot(data,h['snapshot'])['nodes']})
    return dict(status='passed',validated_hierarchies=len(manifest['partition_hashes']),baseline_manifest_sha256=sha('artifacts/baseline_v1/manifest.json'),construction_keys_verified=keys)

def run_v2(cfg):
    start=time.time(); out=Path(cfg['v2_output']); out.mkdir(parents=True,exist_ok=True)
    data=read_json(locate(cfg['data_dir'],'tkh_collection10.json')); gt=read_json(locate(cfg['data_dir'],'ground_truth.json')); questions=read_csv(locate(cfg['data_dir'],'questions.csv'))
    reuse=verify_reuse(cfg,data); print('Validated reuse:',reuse['validated_hierarchies'],flush=True)
    protocol=dict(schema_version='evaluation-v2',config=cfg,reuse=reuse,data_hash=digest(data),ground_truth_hash=digest(gt),questions_hash=digest(questions),
        code_hashes={p.as_posix():sha(p) for p in sorted(Path('src').rglob('*.py'))},
        definitions=dict(entity_cost='Every query-centroid and query-prototype dot product, including leaves, costs one. Offline indexing excluded and timed separately.',
            claim_cost='Each new question-claim dot product costs one; already-paid claim leaf scores reused and logged. Graph enumeration excluded.',
            flat='Exhaustive same BGE leaf scorer is primary reference; seeded query-independent scan is secondary equal-budget reference.',
            precision='Output-node fraction matching expected names; extended adds valid_but_unlisted. Empty output precision=0; unmapped recall=null.',
            recall='Macro question recall over mapped named targets; identical-surface mentions share a target. Claim recall only conservative automated alignments.',
            cutoff='2025 primary; 2026 annual sensitivity may include post-February material.',
            label='Paired legacy/improved gloss NLI on fixed held-out chunks; generation-only independence. Original historical NLI retained separately.'))
    write_json(out/'protocol.json',protocol) # frozen before scoring; no answer-based parameter selection
    retrieval=Encoder(cfg['evaluation_model'],cfg['evaluation_revision']); alignment_encoder=Encoder(cfg['embedding_model'],cfg['embedding_revision'])
    nli=NLI(cfg['nli_model'],cfg['nli_revision']); records=[]; mappings={}; alignments={}; evaluability={}; index_seconds=0.
    Q=retrieval.encode([q['question'] for q in questions],'v2-questions')
    for year in cfg['v2_cutoffs']:
        snap=snapshot(data,year); nodes=snap['nodes']; X=retrieval.encode(node_texts(nodes,cfg['include_type']),f'v2-retrieval:{year}')
        mapping=map_benchmark(gt,nodes,year); mappings[str(year)]=mapping_summary(mapping); write_json(out/f'target_mapping_{year}.json',mapping)
        evaluability[str(year)]=question_evaluability(questions,mapping)
        alignment=align(gt,snap,alignment_encoder,nli,cfg); alignments[str(year)]=alignment; write_json(out/'claim_alignment.json',alignments)
        print('Cutoff',year,'mapping',mappings[str(year)],'claim alignment',coverage(alignment),flush=True)
        for variant in cfg['variants']:
            path=Path(cfg['output'])/'variants'/variant/f'hierarchy_{year}{"_benchmark" if year==2025 else ""}.json'; h=read_json(path)
            indices={}; exports={}; then=time.time()
            for question in questions:
                route=route_question(question['question'],question['type']); name=route['profile']
                if name not in indices:
                    indices[name]=SearchIndex(h,nodes,X,route['types'],cfg['v2_prototypes']); exports[name]=indices[name].export()
            index_seconds+=time.time()-then; write_json(out/'prototypes'/f'{variant}_{year}.json',exports)
            detail=[]
            for question,q in zip(questions,Q):
                qid=question['question_id']; route=route_question(question['question'],question['type']); index=indices[route['profile']]
                mr=[r for r in mapping if r['question_id']==qid]; ar=[r for r in alignment['items'] if r['question_id']==qid]
                runs=[('flat_exhaustive',None)]+[(method,b) for b in cfg['v2_budgets'] for method in ['legacy_typed_fixed_beam','prototype_best_first','flat_query_independent_scan']]
                for method,budget in runs:
                    if method.startswith('flat'):
                        result=flat_search(q,nodes,X,route['types'],cfg['v2_output_k'],budget,cfg['seed'])
                    else: result=index.search(q,budget,method,cfg['v2_centroid_weight'],cfg['v2_prototype_top_r'],cfg['v2_output_k'],cfg['retrieval_beam'])
                    claims=retrieve_claims(q,result['returned_ids'],snap,X,cfg['v2_claim_budget'],cfg['v2_claim_k'],result['leaf_scores'])
                    row=dict(variant=variant,cutoff=year,method=method,budget=budget,question_id=qid,question_type=question['type'],route=route,
                        **entity_metrics(result['returned_ids'],mr,'fp_categories' in gt[qid]),**claim_metrics(claims,ar),entity_comparisons=result['score_comparisons'],claim_comparisons=claims['score_comparisons'],
                        total_comparisons=result['score_comparisons']+claims['score_comparisons'],root_layer_complete=result.get('root_layer_complete',True))
                    if method=='flat_exhaustive':
                        reference=retrieve_claims(q,[],snap,X,k=cfg['v2_claim_k'],exhaustive=True)
                        row['flat_claim_corpus_reference']=dict(**claim_metrics(reference,ar),comparisons=reference['score_comparisons'],returned_ids=reference['returned_ids'])
                    records.append(row); detail.append(dict(metrics=row,entities=result,claims=claims))
            write_json(out/'retrieval'/f'{variant}_{year}.json',detail); print('Retrieval finished',variant,year,flush=True)
    extrinsic=dict(status='computed',schema_version='extrinsic-v2',benchmark_schema=dict(questions=len(questions),types={t:sum(q['type']==t for q in questions) for t in ['A','B']},expected_targets_field='expected_methods'),
        definitions=protocol['definitions'],mapping_coverage=mappings,question_evaluability=evaluability,claim_alignment_coverage={y:coverage(a) for y,a in alignments.items()},cost_curves=summarize(records),records=records,
        index_seconds=index_seconds,retrieval_model=retrieval.metadata,primary_cutoff=2025,cutoff_sensitivity=2026)
    write_json(out/'extrinsic_v2.json',extrinsic)
    faithfulness={}
    for variant in cfg['variants']:
        logs=[]; folder=out/'labels'/variant
        for year in cfg['snapshots']:
            h=read_json(Path(cfg['output'])/'variants'/variant/f'hierarchy_{year}.json'); overlay,items=label_overlay(h,snapshot(data,year),cfg)
            write_json(folder/f'labels_{year}.json',overlay); write_json(folder/f'generation_inputs_{year}.json',items); logs.extend(items)
        result=evaluate_labels(logs,nli,cfg,folder); write_json(folder/'faithfulness.json',result); faithfulness[variant]={k:v for k,v in result.items() if k not in {'rows','controls'}}
        print('Labels finished',variant,result['summaries'],flush=True)
    metrics=read_json('artifacts/baseline_v1/metrics.json'); metrics['extrinsic_v2']=extrinsic; metrics['faithfulness_v2']=faithfulness
    write_json(Path(cfg['output'])/'metrics.json',metrics)
    write_json(out/'verification.json',dict(status='passed',reuse=reuse,protocol_hash=digest(protocol),elapsed_seconds=time.time()-start,original_metrics_preserved=all(metrics[k]==v for k,v in read_json('artifacts/baseline_v1/metrics.json').items())))
    from .report_v2 import generate_report
    generate_report(cfg,metrics)
    print('Evaluation v2 complete',round(time.time()-start,1),'seconds',flush=True)

def run_context(cfg,stages=None):
    """Incremental evaluation using existing graph, encoders, claim paths and metrics."""
    from collections import Counter
    import numpy as np
    from .contextual import build_representations,ContextIndex,analyze_question,entity_groups
    from .target_mapping import TargetResolver
    from .evaluation.retrieval_v2 import contextual_metrics,contextual_summary
    from .evaluation.claim_alignment import SourceIndex,benchmark_claims
    out=Path(cfg['context_output']); out.mkdir(parents=True,exist_ok=True); start=time.time()
    stages=stages or cfg['context_stages']; data=read_json(locate(cfg['data_dir'],'tkh_collection10.json'))
    gt=read_json(locate(cfg['data_dir'],'ground_truth.json')); questions=read_csv(locate(cfg['data_dir'],'questions.csv'))
    if cfg.get('context_verify_reuse',True): verify_reuse(cfg,data)
    unsupported=set(stages)-{'R0','R1','R1_direct','R2','R3','R4','R5'}
    if unsupported: raise ValueError(f'Unimplemented contextual stages (hierarchy gate not passed): {sorted(unsupported)}')
    protocol=dict(config=cfg,stages=stages,data_hash=digest(data),questions_hash=digest(questions),ground_truth_hash=digest(gt),
        code_hashes={p.as_posix():sha(p) for p in sorted(Path('src').rglob('*.py'))},
        definitions=dict(candidate_pool='All snapshot nodes, including unusual types; soft priors never filter. R0 is all-type surface-only to isolate the R1 context increment; legacy routed type-prefixed flat remains separately archived.',
            cost='Each actually computed query/text-vector dot product costs one; identical stored text vectors are computed once per query view and reused across fields/entities. Offline encoding/index construction excluded and timed.',
            grouping='Exact normalized names only, no fuzzy merges. Same resolved target denominator in every stage; report number of underlying nodes covered by top-K groups.',
            missing_fields='Contribute zero with fixed weights, without renormalizing available fields; this favors entities with documented context and is a design limitation.',
            evidence='Context shared through a presenting article is a separate lower-weight claim_article field; ordinary cited-work provenance is not evidence that every article claim concerns that work.',
            resolution='Canonical exact/alias coverage distinct from representable evidence/composites. Composite representability requires all canonical components plus shared article provenance; this is not proof of functional composition.',
            metrics='Primary entity recall uses canonical exact/alias targets only. Representable recall separately includes evidence and fully graph-supported composites (all components required). Raw lower bound retains all expected targets. Precision gives one credit per recovered canonical target at fixed K; duplicate mentions cannot add credit. MRR is reciprocal first relevant rank over the exhaustive ranking.',
            source='Source hit/recall measures provenance overlap, never semantic entailment. Required sources outside the snapshot remain in resolution coverage and raw lower-bound denominators.',
            hierarchy_gate='Implement contextual hierarchy only after the flat original-question checkpoint demonstrates nontrivial recovery; original hierarchy construction remains unchanged.'))
    write_json(out/'protocol.json',protocol)
    encoder=Encoder(cfg['evaluation_model'],cfg['evaluation_revision']); results={}; resolution_exports={}; total_records=[]
    for year in cfg['context_cutoffs']:
        snap=snapshot(data,year); then=time.time(); representations=build_representations(snap,cfg)
        write_json(out/f'representations_{year}.json',representations)
        resolver=TargetResolver(snap); names=sorted({x for g in gt.values() for x in g.get('expected_methods',[])})
        resolutions={name:resolver.resolve(name,cfg['context_resolver']) for name in names}
        counts=Counter(r['resolution_type'] for r in resolutions.values()); instances=[resolutions[x] for g in gt.values() for x in g.get('expected_methods',[])]
        resolution_exports[str(year)]=dict(targets=list(resolutions.values()),unique=dict(total=len(names),by_state=dict(counts),canonical=sum(r['canonical'] for r in resolutions.values()),representable=sum(r['representable'] for r in resolutions.values())),
            instances=dict(total=len(instances),by_state=dict(Counter(r['resolution_type'] for r in instances)),canonical=sum(r['canonical'] for r in instances),representable=sum(r['representable'] for r in instances)))
        write_json(Path(cfg['v2_output'])/'target_resolution.json',resolution_exports)
        indices={False:ContextIndex(snap,representations,encoder,cfg,False)}
        index_times={False:time.time()-then}
        intents=[analyze_question(q['question'],q['type']) for q in questions]
        Q=[encoder.encode(i['extractive_views'],f'context-query:{digest(i["extractive_views"])}') for i in intents]
        source_index=SourceIndex(snap); claim_nodes_X=encoder.encode(node_texts(snap['nodes'],False),f'real:{year}:independent-evaluation')
        source_items=benchmark_claims(gt); source_rows={}
        for question in questions:
            refs=[s for r in source_items if r['question_id']==question['question_id'] for s in r['sources']]
            refs=list({digest(s):s for s in refs}.values()); source_rows[question['question_id']]=[source_index.resolve(s) for s in refs]
        for stage in stages:
            grouped=stage not in {'R0','R1','R1_direct'}
            if grouped not in indices:
                then=time.time(); indices[grouped]=ContextIndex(snap,representations,encoder,cfg,grouped); index_times[grouped]=time.time()-then
            index=indices[grouped]; rows=[]
            index_seconds=index_times[grouped]
            if stage=='R1_direct':
                then=time.time(); direct_cfg=dict(cfg,context_article_expansion=False)
                index=ContextIndex(snap,build_representations(snap,direct_cfg),encoder,direct_cfg,False)
                index_seconds=time.time()-then
            for question,intent,vectors in zip(questions,intents,Q):
                qid=question['question_id']; views=vectors if stage in {'R4','R5'} else vectors[:1]
                result=index.search(views,intent,type_enabled=stage in {'R3','R4','R5'},surface=stage=='R0',rerank=stage=='R5')
                rr=[resolutions[x] for x in gt[qid].get('expected_methods',[])]; metrics=contextual_metrics(result,rr,cfg['context_top_ks'])
                seed=[x for r in result['ranked'][:10] for x in r['node_ids']]
                claims=retrieve_claims(vectors[0],seed,snap,claim_nodes_X,budget=100,k=100)
                refs=source_rows[qid]; matched=[r for r in refs if r['status']=='matched']; required={a for r in matched for a in r['article_ids']}
                got={a for r in claims['evidence'][:10] for a in r['provenance'].get('articles',[])}
                evidence_metrics=dict(source_recall=len(required&got)/len(required) if required else None,source_hit=float(bool(required&got)) if required else None,
                    source_resolution_coverage=len(matched)/len(refs) if refs else None,source_raw_lower_bound=sum(bool(set(r['article_ids'])&got) for r in matched)/len(refs) if refs else None,
                    required_source_articles=sorted(required),articles_touched=sorted(got),source_alignment=refs,claim_recall=None,
                    claim_recall_status='unavailable_without_conservative_alignment; source overlap is not entailment',claim_candidate_count=claims['candidate_count'],
                    claim_comparisons_new=claims['score_comparisons'],claim_comparisons_reused=len(claims['reused_entity_scores']),source_at_k={})
                for k in cfg['context_top_ks']:
                    touched={a for r in claims['evidence'][:k] for a in r['provenance'].get('articles',[])}
                    evidence_metrics['source_at_k'][str(k)]=dict(recall=len(required&touched)/len(required) if required else None,hit=float(bool(required&touched)) if required else None,articles_touched=len(touched))
                row=dict(question_id=qid,question=question['question'],question_type=question['type'],stage=stage,cutoff=year,intent=intent,
                    temporal_policy='year <= 2025 conservative primary' if year==2025 else 'annual 2026 potentially post-February sensitivity',
                    entity_metrics=metrics,evidence_metrics=evidence_metrics,cost=result['cost'],latency_seconds=result['latency_seconds'],expansions=result['expansions'],
                    nodes_covered_by_top10=sum(len(r['node_ids']) for r in result['ranked'][:10]),
                    gold=[dict(label=r['label'],resolution_type=r['resolution_type'],resolved_nodes=r['resolved_node_ids'],entity_groups=[g['id'] for g in index.groups if set(g['node_ids'])&set(r['resolved_node_ids'])],reason=r['reason']) for r in rr],
                    retrieved=[dict(rank=i+1,**r) for i,r in enumerate(result['ranked'][:100])],claims=claims,
                    candidate_diagnostics=metrics['target_outcomes'])
                rows.append(row)
            summary=contextual_summary(rows,cfg['context_top_ks']); results[f'{year}:{stage}']=dict(cutoff=year,stage=stage,index_seconds=index_seconds,summary=summary)
            write_json(out/f'{stage}_{year}.json',dict(protocol_hash=digest(protocol),summary=summary,records=rows)); total_records.extend(rows)
            print('CHECKPOINT',year,stage,'Type A',summary['type_a']['at_k']['10'],'MRR',summary['type_a']['mrr'],'comparisons',summary['mean_vector_comparisons'],flush=True)
            write_json(Path(cfg['v2_output'])/'retrieval_ablation_results.json',dict(status='in_progress',results=results,definitions=protocol['definitions']))
    write_json(Path(cfg['v2_output'])/'retrieval_ablation_results.json',dict(status='computed',results=results,definitions=protocol['definitions'],elapsed_seconds=time.time()-start))
    write_json(Path(cfg['v2_output'])/'per_question_diagnostics.json',total_records)
    metrics_path=Path(cfg['output'])/'metrics.json'; metrics=read_json(metrics_path) if metrics_path.exists() else {}
    metrics['contextual_retrieval']=dict(protocol=protocol,resolution_coverage={y:{k:v for k,v in r.items() if k!='targets'} for y,r in resolution_exports.items()},results=results,
        question_metrics=[{k:r[k] for k in ['question_id','question_type','stage','cutoff','entity_metrics','evidence_metrics','cost']} for r in total_records],
        hierarchy_status='not_implemented: apply flat-quality decision gate before adding contextual hierarchy',generative_expansion_status='optional_not_run')
    write_json(Path(cfg['output'])/'metrics.json',metrics)
