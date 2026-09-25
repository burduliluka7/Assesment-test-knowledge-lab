from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path
import inspect
import numpy as np
import pytest
from tkh_abstraction.config import read_config
from tkh_abstraction.io import read_json, write_json, digest
from tkh_abstraction.conditioned_decomposition import decompose_conditioned_question, classify_conditioned_attribution
from tkh_abstraction.question_decomposition import decompose_question
from tkh_abstraction.evidence_attribution import classify_attribution
from tkh_abstraction.conditioned_evidence import build_candidate_evidence_pool, ConditionedEvidenceReranker
from tkh_abstraction.constraint_scoring import aggregate_support, PreparedReranker
from tkh_abstraction.reranking import rerank_candidates
from tkh_abstraction.candidate_evidence import FIELD_ORDER

def cfg(): return dict(read_config('configs/constraints.yaml'),conditioned_evidence_top_k=3)
def item(ident,text,kind='safe_explicit_name',node_type='claim'):
    return dict(node_id=ident,text=text,kind=kind,node_type=node_type,priority=1,path=[],first_seen_year=2020,provenance={'articles':['paper']})
def bundle(items):
    return dict(entity_id='a',name='ToolAlpha',node_ids=['a'],types=['method'],fields={'direct_claims':items})

@pytest.mark.parametrize('text,expected',[
 ('What are the leading memory-safe concurrent tools for compiling programs?', ['memory-safe','concurrent']),
 ('Which lightweight distributed algorithms classify images?', ['lightweight distributed']),
 ('What are the current solar-powered irrigation systems for growing crops?', ['solar-powered','irrigation']),
])
def test_generic_answer_modifiers_exact_spans(text,expected):
    d=decompose_conditioned_question(text)
    assert d.answer_qualifiers==expected
    for c in d.components: assert text[c['start']:c['end']]==c['text']==c['source_span']
    assert decompose_question(text).to_dict()==decompose_question(text).to_dict()

def test_temporal_head_never_scored():
    q='What are the leading (by May 2024) memory-safe compilation tools for building programs?'
    d=decompose_conditioned_question(q)
    assert d.answer_qualifiers==['memory-safe','compilation']
    assert all('2024' not in c['text'] for c in d.components)

def test_ambiguous_coordinate_head_has_no_qualifiers():
    d=decompose_conditioned_question('Which fast methods and datasets are used for compilation?')
    assert d.answer_qualifiers==[] and d.warnings

@pytest.mark.parametrize('clause',['logs and audit records must be retained','seeds and roots should remain dry','inputs and outputs can be encrypted'])
def test_shared_predicate_contiguous(clause):
    d=decompose_conditioned_question('Which tools operate when '+clause+'?')
    assert d.requirements==[clause]

def test_independent_conditions_and_exclusions_stay_separate():
    d=decompose_conditioned_question('Which tools compile code with low memory and fast startup without remote services?')
    assert d.requirements==['low memory','fast startup'] and d.exclusions==['remote services']

def test_independent_clause_with_modal_is_not_shared_predicate():
    d=decompose_conditioned_question('Which tools operate when supports fast training and the output must be deterministic?')
    assert d.requirements==['supports fast training','the output must be deterministic']

@pytest.mark.parametrize('text',[
 'Including logs in training improves ToolAlpha accuracy.',
 'Including extra seeds increases ToolAlpha harvest.',
 'Including redundant packets enables ToolAlpha recovery.',
])
def test_participial_including_is_not_list(text):
    e=item('e',text)
    assert classify_attribution(e,['ToolAlpha'])['category']=='MENTION_ONLY'
    assert classify_conditioned_attribution(e,['ToolAlpha'])['category']=='EXPLICIT_NAME_SUPPORT'

@pytest.mark.parametrize('text',[
 'Several tools, including ToolAlpha, improve compilation.',
 'Including ToolAlpha and ToolBeta, these tools improve compilation.',
 'Including logs does not improve ToolAlpha accuracy.',
 'Including ToolAlpha and ToolBeta, this method improves compilation.',
 'Including ToolBeta improves accuracy relative to ToolAlpha.',
 'Including ToolAlpha in the ensemble improves ToolBeta accuracy.',
 'Including logs without timestamps improves ToolAlpha accuracy.',
])
def test_lists_and_ambiguous_negation_unchanged(text):
    e=item('e',text)
    assert classify_conditioned_attribution(e,['ToolAlpha'])==classify_attribution(e,['ToolAlpha'])

def test_structural_pool_query_independence_caps_dedup_and_provenance():
    values=[item(str(i),'ToolAlpha supports feature '+str(i)) for i in range(25)]
    duplicate=dict(values[0],node_id='duplicate',path=['edge'])
    fields={f:[] for f in FIELD_ORDER}; fields['direct_claims']=values; fields['explicit_mentions']=[duplicate]
    index=SimpleNamespace(groups={'a':dict(id='a',name='ToolAlpha',types=['method'],node_ids=['a'])},nodes={'a':{'surface_form':'ToolAlpha'}},
        pools={'a':fields},snap={'snapshot':2025},mentions=SimpleNamespace(names={'a':['ToolAlpha']}),op=SimpleNamespace(degree=np.array([4]),idx={'a':0}))
    caps=read_config('configs/conditioned.yaml')['conditioned_pool_caps']
    before=deepcopy(fields); pool=build_candidate_evidence_pool(index,'a',caps).to_dict()
    assert pool==build_candidate_evidence_pool(index,'a',caps).to_dict() and fields==before
    assert len(pool['fields']['direct_claims'])==20>3
    assert not pool['fields']['explicit_mentions'] and len(pool['fields']['direct_claims'][0]['source_links'])==2
    assert all(i['first_seen_year']<=pool['cutoff'] for xs in pool['fields'].values() for i in xs)
    assert not {'question','requirements','gold','ground_truth'}&set(inspect.signature(build_candidate_evidence_pool).parameters)

def test_distinct_component_evidence_top_three_and_frozen_aggregate():
    d=decompose_conditioned_question('Which tools compile code with low memory and fast startup?')
    evidence=bundle([item('x','ToolAlpha compiles code.'),item('y','ToolAlpha needs low memory.'),item('z','ToolAlpha starts quickly.')])
    sims={c['id']:{n:(.9 if i==j else .2) for j,n in enumerate(['x','y','z'])} for i,c in enumerate(d.components)}
    scorer=ConditionedEvidenceReranker(cfg(),d,sims,{'a':['ToolAlpha']}); score=scorer.score('', 'ToolAlpha',evidence)
    assert [r['best_evidence']['node_id'] for r in score['requirements']]==['x','y','z']
    assert all(len(r['selected_evidence'])==3 for r in score['requirements'])
    assert score['score']==aggregate_support(d.components,[.9]*3,[],['method'],d.answer_type,cfg())['score']

def test_qualifiers_participate_without_hard_filter():
    d=decompose_conditioned_question('Which memory-safe tools compile code?')
    sims={c['id']:{'e':.9 if c['kind']=='core_task' else .2} for c in d.components}
    evidence=bundle([item('e','ToolAlpha compiles code.')]); score=ConditionedEvidenceReranker(cfg(),d,sims,{'a':['ToolAlpha']}).score('','',evidence)
    assert 0<score['score']<.81 and score['positive_components']==2

def test_baseline_paths_mentions_and_exclusion_weights():
    d=decompose_conditioned_question('Which tools compile code without remote services?')
    es=[item('base','ToolBeta outperforms ToolAlpha.'),item('path','Fast compilation works.','short_path_source'),
        item('list','Many tools including ToolAlpha compile code.')]
    sims={c['id']:{e['node_id']:1. for e in es} for c in d.components}
    score=ConditionedEvidenceReranker(cfg(),d,sims,{'a':['ToolAlpha']}).score('','',bundle(es))
    assert all(r['best_evidence']['node_id']=='list' and r['cosine_support']==.15 for r in score['requirements'])
    assert score['exclusion_factor']==.85

@pytest.mark.parametrize('k',[100,200])
def test_prefix_and_deterministic_rrf_preserve_tail(k):
    rows=[dict(entity_id=str(i),name=str(i),score=1/(i+1),node_ids=[str(i)],score_components={}) for i in range(220)]
    details={r['entity_id']:dict(score=0.,score_components={}) for r in rows}; bs={r['entity_id']:dict(entity_id=r['entity_id']) for r in rows}
    a=rerank_candidates(rows,bs,PreparedReranker(details),'',k,True)
    b=rerank_candidates(rows,bs,PreparedReranker(details),'',k,True)
    assert a['ranked']==b['ranked'] and a['ranked'][k:]==rows[k:]
    assert {r['entity_id'] for r in a['ranked'][:k]}=={r['entity_id'] for r in rows[:k]}

def test_no_benchmark_names_or_neural_calls_in_production_rules():
    for name in ['conditioned_decomposition.py','conditioned_evidence.py']:
        text=(Path('src/tkh_abstraction')/name).read_text(encoding='utf8')
        for forbidden in ['Q1','MACE','SchNet','M3GNet','MLIP','ground_truth','expected_methods','ControlledLocalNLI','predict_controlled']:
            assert forbidden not in text

def test_ground_truth_cannot_enter_component_scoring():
    d=decompose_conditioned_question('Which tools compile code?'); b=bundle([item('e','ToolAlpha compiles code.')])
    scorer=ConditionedEvidenceReranker(cfg(),d,{'component_0':{'e':.8}},{'a':['ToolAlpha']})
    before=scorer.score('','',b)
    b['ground_truth']={'expected_methods':['ToolBeta']}
    assert scorer.score('','',b)==before

def test_synthetic_pipeline_gold_perturbation_temporal_and_history(tmp_path,monkeypatch):
    import tkh_abstraction.conditioned_pipeline as pipeline
    from tkh_abstraction.reranker_pipeline import LocalEncoder
    monkeypatch.setattr(pipeline,'LocalEncoder',lambda *args:LocalEncoder('synthetic-hash'))
    config=read_config('configs/conditioned.yaml'); data=tmp_path/'data'; data.mkdir(); root=tmp_path/'eval'; root.mkdir()
    config.update(data_dir=str(data),v2_output=str(root),output=str(tmp_path/'full'),conditioned_output=str(root/'conditioned'),
                  conditioned_verify_history=False,conditioned_cutoffs=[2025])
    nodes=[dict(id=i,type=typ,surface_form=text,first_seen_year=year,provenance=dict(articles=[])) for i,typ,text,year in [
        ('a','method','ToolAlpha',2020),('b','method','ToolBeta',2020),('c','claim','ToolAlpha compiles programs',2020),
        ('future','claim','ToolAlpha needs less memory',2026)]]
    write_json(data/'tkh_collection10.json',dict(nodes=nodes,hyperedges=[dict(id='e',members=['a','c'],relation_type='claims',year=2020,provenance=dict(article_year=2020))]))
    (data/'questions.csv').write_text('question_id,type,question\nNEW,A,Which tools compile programs with low memory?\nBNEW,B,Why?\n')
    write_json(data/'ground_truth.json',dict(NEW=dict(expected_methods=['ToolAlpha']),BNEW={}))
    write_json(tmp_path/'full'/'metrics.json',dict(type_b={'source_hit':.3})); write_json(root/'history.json',dict(sentinel=True)); before=(root/'history.json').read_bytes()
    pipeline.run_conditioned(config)
    first=read_json(root/'requirement_evidence_selection.json'); pools=read_json(root/'candidate_evidence_pools.json')
    scores=[(c['entity_id'],c['CR6'],c['ranks']) for q in first for c in q['candidates']]
    assert all(i['node_id']!='future' for p in pools for xs in p['fields'].values() for i in xs)
    write_json(data/'ground_truth.json',dict(NEW=dict(expected_methods=['Missing']),BNEW={}))
    pipeline.run_conditioned(config)
    second=read_json(root/'requirement_evidence_selection.json')
    assert scores==[(c['entity_id'],c['CR6'],c['ranks']) for q in second for c in q['candidates']]
    assert pools==read_json(root/'candidate_evidence_pools.json')
    assert before==(root/'history.json').read_bytes() and read_json(tmp_path/'full'/'metrics.json')['type_b']=={'source_hit':.3}
    results=read_json(root/'conditioned_evidence_results.json')['results']
    assert all(r['summary']['candidate_preservation_rate']==1 and r['summary']['evaluable']==0 for r in results)

def test_shipped_frozen_h2_cr1_oracle_and_all_question_statuses():
    new=read_json('artifacts/evaluation/conditioned_evidence_results.json'); old=read_json('artifacts/evaluation/constraint_reranking_results.json')
    baseline={(r['cutoff'],r['stage']):r for r in old['results']}
    inv={(r['cutoff'],r['question_id']):r for r in old['inventory']}
    for r in new['inventory']:
        for k in ['prefix100_hash','prefix200_hash']: assert r[k]==inv[r['cutoff'],r['question_id']][k]
    assert new['oracle']==old['oracle']
    for r in new['results']:
        assert len(r['records'])==14 and r['summary']['evaluable']==12
        if r['stage'] in ['CR0-100','CR1-100','CR3-100']:
            for a,b in zip(r['records'],baseline[r['cutoff'],r['stage']]['records']):
                assert a['returned']==b['returned'] and a['entity_metrics']==b['entity_metrics']

def test_historical_sources_reports_and_metric_sections_preserved():
    from tkh_abstraction.v2_pipeline import sha
    run=read_json('artifacts/evaluation/conditioned_evidence_results.json'); old=read_json('artifacts/evaluation/constraint_reranking_results.json')
    for path,h in old['protocol']['source_hashes'].items():
        if not path.endswith('/cli.py'): assert sha(path)==h
    for path,h in run['protocol']['historical_hashes'].items(): assert sha(path)==h
    metrics=read_json('artifacts/full/metrics.json')
    for k,h in run['protocol']['historical_metric_hashes'].items(): assert digest(metrics[k])==h
