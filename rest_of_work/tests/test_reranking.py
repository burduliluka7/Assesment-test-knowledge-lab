import inspect
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from tkh_abstraction.config import read_config
from tkh_abstraction.io import digest,read_json,write_json
from tkh_abstraction.diffusion import ThetaOperator,retrieval_graph,MentionIndex
from tkh_abstraction.contextual import entity_groups
from tkh_abstraction.snapshots import snapshot
from tkh_abstraction.target_mapping import TargetResolver,normalize
from tkh_abstraction.candidate_evidence import CandidateEvidenceIndex,FIELD_ORDER
from tkh_abstraction.reranking import EvidenceSemanticReranker,LocalCrossEncoderReranker,inspect_local_reranker,rerank_candidates
from tkh_abstraction.reranker_pipeline import FrozenH2Candidates,LocalEncoder,evaluate_reranked,failure_category,run_reranking
from tkh_abstraction.diffusion_pipeline import evaluate_ranking
from tkh_abstraction.evaluation.reranker_oracle import oracle_ceiling

def node(i,t='method',text=None,year=2020):
    return dict(id=i,type=t,surface_form=text or i,first_seen_year=year,provenance=dict(articles=['paper']))

def edge(i,m,rel='claims',year=2020):
    return dict(id=i,members=m,relation_type=rel,year=year,provenance=dict(article_year=year))

def fixture():
    cfg=read_config('configs/reranking.yaml')
    ns=[node('a',text='AlphaNet'),node('b',text='BetaNet'),node('c','claim','AlphaNet predicts elastic forces'),
        node('d','claim','AlphaNet predicts elastic forces'),node('z','claim','Broad article context unrelated to candidate'),
        node('t','task','Predict elastic forces'),node('paper','article','Scientific paper'),node('hub','task','machine learning'),
        node('future','claim','AlphaNet predicts future findings',2026)]
    es=[edge('direct',['a','c']),edge('task',['a','t'],'solves'),edge('present',['a','paper'],'presents'),
        edge('broad',['paper','z']),edge('future',['a','future'],year=2026)]
    es += [edge('h'+str(i),['hub',n['id']],'addresses') for i,n in enumerate(ns) if n['id']!='hub']
    snap=snapshot(dict(nodes=ns,hyperedges=es),2025); groups=entity_groups(snap['nodes'][:2])
    op=ThetaOperator(retrieval_graph(snap)); mentions=MentionIndex(snap['nodes'],groups,['claim'])
    idx=CandidateEvidenceIndex(snap,groups,op,mentions,cfg)
    a=next(g for g in groups if g['name']=='AlphaNet')
    candidate=dict(entity_id=a['id'],name=a['name'],node_ids=a['node_ids'],score=.2)
    sims={n['id']:.4+i*.01 for i,n in enumerate(snap['nodes'])}
    return cfg,idx,candidate,sims

def test_deterministic_specific_evidence_priority_mentions_caps_and_dedup():
    cfg,idx,candidate,sims=fixture()
    a=idx.build_candidate_evidence('forces?',candidate,sims,{},1)
    assert a==idx.build_candidate_evidence('forces?',candidate,sims,{},1)
    assert idx.mentions.mentions[candidate['entity_id']]==['c','d']
    pieces=[x for f in a['fields'].values() for x in f]
    assert len({normalize(x['text']) for x in pieces})==len(pieces)
    assert a['document'].index('direct_claims [')<a['document'].index('article [')
    assert 'Broad article context' not in a['document'] and 'future findings' not in a['document']
    assert all(len(v)<=cfg['reranker_field_caps'][f] for f,v in a['fields'].items())

def test_generic_bridge_remains_in_paths_but_not_semantic_document():
    cfg,idx,candidate,sims=fixture(); assert idx.generic['hub']['generic_bridge']
    paths,_=idx.support_paths([dict(node_id='hub',surface_form='machine learning',semantic_score=.9,y=1.)])
    b=idx.build_candidate_evidence('forces?',candidate,sims,paths,1)
    assert b['selected_paths'] and 'machine learning' not in b['document']
    assert not idx.generic['a']['generic_bridge']

def evidence(value=.8):
    return dict(name_similarity=value,fields={f:[] for f in FIELD_ORDER})

def test_availability_normalization_and_score_component_sum():
    cfg=read_config('configs/reranking.yaml'); rr=EvidenceSemanticReranker(cfg); b=evidence()
    out=rr.score('','',b)
    assert out['available_semantic_score']==pytest.approx(.8)
    assert out['score']==pytest.approx(.8*(.5+.5*np.sqrt(.1)))
    assert out['score']==pytest.approx(sum(out['score_components'].values()))
    b['fields']['direct_claims']=[dict(similarity=.8)]
    second=rr.score('','',b)
    assert second['available_semantic_score']==pytest.approx(.8) and second['score']>out['score']

def test_coverage_monotone_and_lucky_field_bounded():
    rr=EvidenceSemanticReranker(read_config('configs/reranking.yaml')); b=evidence(1.)
    values=[rr.score('','',b)['score']]
    for f in FIELD_ORDER:
        b['fields'][f]=[dict(similarity=1.)]; values.append(rr.score('','',b)['score'])
    assert values==sorted(values) and values[0]<.7 and values[-1]==pytest.approx(1.)

def test_top_two_pooling_and_negative_similarities():
    cfg=read_config('configs/reranking.yaml'); cfg['reranker_pooling']='top2_mean'
    b=evidence(-.2); b['fields']['direct_claims']=[dict(similarity=x) for x in [-.1,.4,.8]]
    result=EvidenceSemanticReranker(cfg).score('','',b)
    assert result['field_scores']['name']==0 and result['field_scores']['direct_claims']==pytest.approx(.6)

@pytest.mark.parametrize('k',[100,200])
@pytest.mark.parametrize('fuse',[False,True])
def test_prefix_preservation_tail_grouping_no_mutation_and_deterministic_rrf(k,fuse):
    rows=[dict(entity_id=str(i),name=str(i),node_ids=[str(i),'alias'+str(i)],score=1/(i+1)) for i in range(230)]
    original=deepcopy(rows); bundles={r['entity_id']:evidence((int(r['entity_id'])%13)/13) for r in rows}
    rr=EvidenceSemanticReranker(read_config('configs/reranking.yaml'))
    out=rerank_candidates(rows,bundles,rr,'query',k,fuse)
    again=rerank_candidates(rows,bundles,rr,'query',k,fuse)
    assert out['ranked']==again['ranked'] and rows==original
    assert out['ranked'][k:]==rows[k:] and out['comparisons']==k
    assert {r['entity_id'] for r in out['ranked'][:k]}=={r['entity_id'] for r in rows[:k]}
    assert all(len(r['node_ids'])==2 for r in out['ranked'])

def test_multianswer_oracle_exact_coverage_and_frozen_candidate_metrics():
    cfg=read_config('configs/reranking.yaml')
    ns=[node(str(i),text='Model'+str(i)) for i in range(220)]
    resolver=TargetResolver(dict(nodes=ns,hyperedges=[]))
    gold=[resolver.resolve('Model'+str(i)) for i in range(12)]
    rows=[dict(entity_id=str(i),node_ids=[str(i)],name='Model'+str(i),score=220-i) for i in range(220)]
    rotated=rows[100:200]+rows[:100]+rows[200:]
    base=evaluate_ranking(rows,gold,dict(diffusion_top_ks=cfg['reranker_top_ks']))
    result=evaluate_reranked(rotated,rows,gold,cfg)
    assert result['candidate_recall_100']==1 and result['at_k']['10']['recall']==0
    oracle=oracle_ceiling(base,rows,100)
    assert oracle['oracle_recall_at10']==pytest.approx(10/12)
    merged=[dict(node_ids=[str(i) for i in range(12)])]
    assert oracle_ceiling(base,merged,100)['oracle_recall_at10']==1

def test_oracle_alias_units_and_null_evaluability():
    metric=dict(canonical_evaluation_units=[dict(node_ids=['a'],supplied_labels=['A','Alias'])],total_expected=2)
    out=oracle_ceiling(metric,[dict(node_ids=['a'])],100)
    assert out['oracle_found_at10']==1 and out['oracle_raw_found']==2
    assert oracle_ceiling(dict(canonical_evaluation_units=[],total_expected=1),[],100)['oracle_recall_at10'] is None

def test_production_sources_no_answer_key_or_oracle_inputs():
    import tkh_abstraction.candidate_evidence as ce
    import tkh_abstraction.reranking as rr
    for obj in [ce,rr,FrozenH2Candidates]:
        code=inspect.getsource(obj)
        for forbidden in ['expected_methods','method_claims','method_evidence','fp_categories','required_citations','oracle_ceiling']:
            assert forbidden not in code

def test_local_nli_is_rejected_and_missing_model_is_deterministic(tmp_path):
    (tmp_path/'config.json').write_text(json.dumps(dict(architectures=['DebertaV2ForSequenceClassification'],id2label={'0':'contradiction','1':'entailment','2':'neutral'})))
    (tmp_path/'model.safetensors').write_bytes(b'')
    a=inspect_local_reranker(tmp_path); assert a['status']=='unavailable' and a==inspect_local_reranker(tmp_path)
    assert not a['network_access']
    assert inspect_local_reranker(tmp_path/'absent')['status']=='unavailable'

def test_model_loaders_force_local_files():
    assert inspect.getsource(LocalCrossEncoderReranker).count('local_files_only=True')==2
    assert 'local_files_only=True' in inspect.getsource(LocalEncoder)

def test_bin_only_reranker_not_reported_loadable(tmp_path):
    (tmp_path/'config.json').write_text(json.dumps(dict(architectures=['BertForSequenceClassification'],id2label={'0':'relevance'})))
    (tmp_path/'pytorch_model.bin').write_bytes(b'')
    assert inspect_local_reranker(tmp_path)['status']=='unavailable'

def test_ce_token_packing_preserves_priority_and_logs_omitted():
    import torch
    from types import SimpleNamespace
    class Tokenizer:
        def __call__(self,q,d,**kw):
            tokens=list(range(len((q+' '+d).split())+2))
            return dict(input_ids=torch.tensor([tokens]) if kw.get('return_tensors') else tokens)
    rr=LocalCrossEncoderReranker.__new__(LocalCrossEncoderReranker)
    rr.tokenizer=Tokenizer(); rr.token_budget=9; rr.model=lambda **kw:SimpleNamespace(logits=torch.tensor([[.3]])); rr.labels={0:'relevance'}
    b=dict(document_blocks=[dict(field='name',text='Candidate AlphaNet'),dict(field='direct_claims',text=' '.join(['long']*20)),dict(field='article',text='short')])
    result=rr.score('question','AlphaNet',b)
    assert len(result['included'])==1 and len(result['omitted'])==2 and result['token_count']==5

def test_intermediate_hub_is_flagged_but_its_text_is_not_scored():
    cfg,idx,candidate,sims=fixture()
    snap=deepcopy(idx.snap); snap['hyperedges']=[e for e in snap['hyperedges'] if e['id']!='broad']
    idx=CandidateEvidenceIndex(snap,list(idx.groups.values()),ThetaOperator(retrieval_graph(snap)),idx.mentions,cfg)
    paths,_=idx.support_paths([dict(node_id='z',surface_form=idx.nodes['z']['surface_form'],semantic_score=.9,y=1.)])
    b=idx.build_candidate_evidence('query',candidate,sims,paths,1)
    assert any(p['via_generic_bridge'] for p in b['selected_paths'])
    assert 'machine learning' not in b['document']
    assert all(p['path'][-1]['to_role']=='target_entity' for p in b['selected_paths'])

def test_k200_conditional_recall_counts_the_frozen_200_pool():
    cfg=read_config('configs/reranking.yaml')
    ns=[node(str(i),text='Model'+str(i)) for i in range(220)]; resolver=TargetResolver(dict(nodes=ns,hyperedges=[]))
    gold=[resolver.resolve('Model'+str(i)) for i in [0,150,151]]
    rows=[dict(entity_id=str(i),node_ids=[str(i)],name='Model'+str(i),score=220-i) for i in range(220)]
    reordered=[rows[i] for i in [0,150,151]]+[r for i,r in enumerate(rows) if i not in {0,150,151}]
    metric=evaluate_reranked(reordered,rows,gold,cfg,200)
    assert metric['candidate_recall_100']==pytest.approx(1/3)
    assert metric['retrieval_recall_given_candidate']==1 and metric['retrieval_recall_given_candidate_k']==200
    assert all(o['in_reranker_candidates'] for o in metric['target_outcomes'])
    assert not next(o for o in metric['target_outcomes'] if o['label']=='Model150')['in_candidate_100']

def test_evidence_dedup_prefers_low_arity_even_with_later_id():
    cfg,idx,candidate,sims=fixture(); pool=idx.pools[candidate['entity_id']]['direct_claims']
    text=idx.nodes['c']['surface_form']
    pool[:]=[dict(idx.item('c','direct:claims',1,['broad']),support_kind='comembership_only'),
             dict(idx.item('d','direct:claims',1,['narrow']),support_kind='low_arity_direct')]
    b=idx.build_candidate_evidence('query',candidate,sims,{},1)
    item=next(x for x in b['fields']['direct_claims'] if x['text']==text)
    assert item['support_kind']=='low_arity_direct'

@pytest.mark.parametrize('args,expected',[
    ((201,201,200,0,None),'NOT_IN_H2_200'),((101,101,100,3,None),'NOT_IN_H2_100'),
    ((40,3,100,1,None),'RETRIEVED_TOP10'),((4,70,100,1,None),'RERANK_DEGRADED_EXISTING_GOOD_RESULT'),
    ((40,50,100,0,None),'IN_CANDIDATES_NO_SPECIFIC_EVIDENCE'),
    ((40,50,100,1,None),'IN_CANDIDATES_EVIDENCE_PRESENT_RERANK_FAILED'),
    ((40,20,100,1,None),'RERANK_IMPROVED_BUT_BELOW_10')])
def test_failure_taxonomy(args,expected): assert failure_category(*args)==expected

def test_new_benchmark_pipeline_no_gold_leakage_preserves_history_and_type_b(tmp_path,monkeypatch):
    import yaml
    import tkh_abstraction.reranker_pipeline as pipeline
    monkeypatch.setattr(pipeline,'inspect_local_reranker',lambda _:dict(status='unavailable',selected_path=None))
    cfg=read_config('configs/reranking.yaml'); data=tmp_path/'data'; data.mkdir(); out=tmp_path/'evaluation'; out.mkdir()
    cfg.update(data_dir=str(data),output=str(tmp_path/'full'),v2_output=str(out),reranker_output=str(out/'reranking'),
        reranker_cutoffs=[2025],reranker_verify_history=False,evaluation_model='synthetic-hash')
    ns=[node('a',text='AlphaNet'),node('b',text='BetaNet'),node('c','claim','AlphaNet predicts forces')]
    write_json(data/'tkh_collection10.json',dict(nodes=ns,hyperedges=[edge('e',['a','c'])]))
    questions=tmp_path/'new.csv'; questions.write_text('question_id,type,question\nNEW,A,Which model predicts forces?\nEMPTY,A,Which absent method?\nBNEW,B,Why?\n')
    truth=tmp_path/'new.json'; write_json(truth,dict(NEW=dict(expected_methods=['AlphaNet']),EMPTY=dict(expected_methods=['Absent']),BNEW={}))
    history=out/'historical.json'; history.write_text('{"sentinel":true}'); original=history.read_bytes()
    write_json(tmp_path/'full'/'metrics.json',dict(type_b_sentinel={'source_hit':.25}))
    run_reranking(cfg,str(questions),str(truth))
    first=read_json(out/'candidate_evidence.json'); first_hash=digest([{k:v for k,v in q.items() if k!='evidence_seconds'} for q in first])
    assert history.read_bytes()==original
    metrics=read_json(tmp_path/'full'/'metrics.json'); assert metrics['type_b_sentinel']=={'source_hit':.25}
    records=metrics['candidate_reranking']['question_metrics']; assert {r['question_id'] for r in records}=={'NEW','EMPTY'}
    assert all(r['entity_metrics']['status']=='not_evaluable' for r in records if r['question_id']=='EMPTY')
    write_json(truth,dict(NEW=dict(expected_methods=[]),EMPTY=dict(expected_methods=[]),BNEW={}))
    run_reranking(cfg,str(questions),str(truth)); second=read_json(out/'candidate_evidence.json')
    assert first_hash==digest([{k:v for k,v in q.items() if k!='evidence_seconds'} for q in second])

def test_rr0_archived_h2_metric_regression():
    path=Path('artifacts/evaluation/reranker_results.json')
    if not path.exists(): pytest.skip('Full artifact parity checked after experiment')
    old=read_json('artifacts/evaluation/hypergraph_diffusion_results.json'); new=read_json(path)
    old_rows={(r['cutoff'],q['question_id']):q['entity_metrics'] for r in old['results'] if r['stage']=='H2' for q in r['records']}
    checked=0
    for r in new['results']:
        if r['stage']!='RR0': continue
        for q in r['records']:
            assert q['entity_metrics']==old_rows[(r['cutoff'],q['question_id'])]; checked+=1
    assert checked==28

def test_new_query_h2_recompute_exactly_matches_historical_archive():
    path=Path('artifacts/evaluation/hypergraph_diffusion_results.json')
    if not path.exists(): pytest.skip('Assessment artifact unavailable')
    old=read_json(path); cfg=old['protocol']['config']; cc=old['protocol']['context_config']
    historic=next(r for r in old['results'] if r['cutoff']==2025 and r['stage']=='H2')['records'][0]
    data=read_json('data/data/tkh_collection10.json'); snap=snapshot(data,2025)
    source=FrozenH2Candidates(snap,LocalEncoder(cfg['evaluation_model'],cfg['evaluation_revision']),cc,cfg)
    result=source.get(historic['question'],historic['question_id'])
    assert result['source']=='computed_for_new_query' and result['ranked'][:200]==historic['returned']
