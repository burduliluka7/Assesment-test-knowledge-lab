import inspect
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from tkh_abstraction.config import read_config
from tkh_abstraction.io import digest,read_json,write_json
from tkh_abstraction.question_decomposition import decompose_question,candidate_hypothesis
from tkh_abstraction.evidence_attribution import classify_attribution
from tkh_abstraction.constraint_scoring import RequirementCosineReranker,RequirementNLIReranker,ControlledLocalNLI,aggregate_support

def cfg(): return read_config('configs/constraints.yaml')
def item(text,kind='safe_explicit_name',support='name_match'):
    return dict(node_id='e',text=text,node_type='claim',kind=kind,support_kind=support)

@pytest.mark.parametrize('question',[
    'Which methods predict rainfall while balancing precision with throughput for regional forecasts?',
    'Which datasets contain magnetic materials and spin-resolved properties?',
    'Which tools compile programs without remote services by February 2026?',
    'Which polymers tolerate cold with flexibility?',
    'Explain the behavior of a novel optical device.',
])
def test_decomposition_generic_exact_spans_and_determinism(question):
    d=decompose_question(question); assert d==decompose_question(question)
    assert d.original_question==question and d.core_task
    for c in d.components: assert question[c['start']:c['end']]==c['text']==c['source_span']
    assert 'interpretability' not in ' '.join(d.requirements)

def test_multiple_requirements_temporal_and_exclusion_separated():
    d=decompose_question('Which tools are suitable for compiling code while balancing precision with speed for large projects without remote services by February 2026?')
    assert d.core_task=='compiling code'
    assert d.requirements==['precision','speed','large projects'] and d.exclusions==['remote services']
    assert d.temporal_condition=='by February 2026'
    assert all('2026' not in c['text'] for c in d.components)

@pytest.mark.parametrize('marker',['without','excluding','except','does not','neither'])
def test_generic_negative_constraints_never_positive(marker):
    d=decompose_question('Which tools compile programs '+marker+' remote services?')
    assert d.exclusions==['remote services'] and 'remote services' not in d.requirements
    exclusion=next(c for c in d.components if c['kind']=='exclusion')
    assert candidate_hypothesis('ToolAlpha',d,exclusion).startswith('ToolAlpha involves remote services')

def test_different_candidates_hypotheses_and_requirements():
    d=decompose_question('Which tools compile programs with low memory and fast startup?')
    hs=[candidate_hypothesis('ToolAlpha',d,c) for c in d.components]
    assert len(set(hs))==3 and all('ToolAlpha' in h for h in hs)
    assert hs!=[candidate_hypothesis('ToolBeta',d,c) for c in d.components]

@pytest.mark.parametrize('question',['Which tools are not cloud-based?','Which tools do not require remote services?'])
def test_negative_only_question_does_not_promote_exclusion_to_core(question):
    d=decompose_question(question)
    assert d.exclusions and d.core_task=='tools' and d.fallback
    assert all(c['kind']=='exclusion' for c in d.components if 'cloud' in c['text'] or 'remote' in c['text'])

def test_coordinated_core_and_prepositional_complement_remain_complete():
    d=decompose_question('Which methods classify and segment images with support for missing data?')
    assert d.core_task=='classify and segment images' and d.requirements==['support for missing data']

def test_partial_nli_execution_failure_falls_back_whole_question():
    from tkh_abstraction.constraint_scoring import uniform_nli_fallback
    good=dict(score=.02,status='computed',requirements=[])
    bad=dict(score=.8,status='cosine_fallback_nli_failed',requirements=[],nli_requirements=[])
    cosine={'a':dict(score=.7,requirements=[]),'b':dict(score=.8,requirements=[])}
    result=uniform_nli_fallback(cosine,{'a':good,'b':bad})
    assert [result[x]['score'] for x in ['a','b']]==[.7,.8]
    assert all(x['status']=='cosine_fallback_nli_failed' for x in result.values())

def test_controlled_nli_batch_probabilities_counts_and_failure(monkeypatch):
    model=ControlledLocalNLI.__new__(ControlledLocalNLI); model.cache={}
    pairs=[('premise'+str(i),'hypothesis') for i in range(257)]
    def fake_predict(batch,batch_size):
        if len(batch)==1: raise RuntimeError('synthetic batch failure')
        return [dict(entailment=.6,neutral=.3,contradiction=.1,prediction='entailment') for _ in batch]
    monkeypatch.setattr(model,'predict',fake_predict)
    probs,cost=model.predict_controlled(pairs)
    assert len(probs)==257 and cost['nli_forward_pairs']==256 and probs[-1]['entailment'] is None
    assert probs[-1]['status']=='failed' and probs[0]['neutral']==.3

def test_exact_zero_ties_reuse_existing_entity_id_policy():
    from tkh_abstraction.reranking import rerank_candidates
    from tkh_abstraction.constraint_scoring import PreparedReranker
    rows=[dict(entity_id=i,name=i,node_ids=[i],score=s) for i,s in [('z',3),('a',2),('m',1)]]
    details={r['entity_id']:dict(score=0.,score_components=dict(requirement_score=0.)) for r in rows}
    bundles={r['entity_id']:dict(entity_id=r['entity_id']) for r in rows}
    ranked=rerank_candidates(rows,bundles,PreparedReranker(details),'query',100)
    assert [r['entity_id'] for r in ranked['ranked']]==['a','m','z']

def test_shipped_constraint_baseline_parity():
    path=Path('artifacts/evaluation/constraint_reranking_results.json')
    if not path.exists(): pytest.skip('Run full experiment for shipped-artifact check')
    new=read_json(path); old=read_json('artifacts/evaluation/reranker_results.json')
    prior={(r['cutoff'],q['question_id']):q for r in old['results'] if r['stage']=='RR1-100' for q in r['records']}
    count=0
    for r in new['results']:
        if r['stage']!='CR0-100': continue
        for q in r['records']:
            assert q['entity_metrics']==prior[r['cutoff'],q['question_id']]['entity_metrics']
            assert q['returned']==prior[r['cutoff'],q['question_id']]['returned']; count+=1
    assert count==28

@pytest.mark.parametrize('text,who,expected',[
    ('AlphaNet outperforms BetaNet.','AlphaNet','COMPARISON_WINNER'),
    ('AlphaNet outperforms BetaNet.','BetaNet','COMPARISON_BASELINE'),
    ('AlphaNet is outperformed by BetaNet.','AlphaNet','COMPARISON_BASELINE'),
    ('AlphaNet is worse than BetaNet.','BetaNet','COMPARISON_WINNER'),
    ('AlphaNet does not outperform BetaNet.','AlphaNet','UNCERTAIN'),
    ('AlphaNet is compared with BetaNet.','AlphaNet','UNCERTAIN'),
    ('Available models include AlphaNet and BetaNet.','AlphaNet','MENTION_ONLY'),
    ('AlphaNet predicts rainfall accurately.','AlphaNet','DIRECT_SUBJECT_SUPPORT'),
    ('AlphaNet does not predict rainfall.','AlphaNet','UNCERTAIN'),
    ('AlphaNet outperforms BetaNet but loses to GammaNet which outperforms AlphaNet.','AlphaNet','UNCERTAIN'),
])
def test_comparative_roles_and_mentions(text,who,expected): assert classify_attribution(item(text),[who])['category']==expected

def test_path_only_and_comembership_never_full_support():
    a=classify_attribution(item('Some model predicts rainfall','short_path_source','indirect_path_source'),['AlphaNet'])
    assert a['category']=='INDIRECT_GRAPH_NEIGHBOR' and cfg()['constraint_attribution_weights'][a['category']]==0
    a=classify_attribution(item('Some model predicts rainfall','direct:claims','comembership_only'),['AlphaNet'])
    assert a['category']=='UNCERTAIN' and cfg()['constraint_attribution_weights'][a['category']]<1
    assert cfg()['constraint_attribution_weights']['MENTION_ONLY']<cfg()['constraint_attribution_weights']['DIRECT_SUBJECT_SUPPORT']

def prepared():
    d=decompose_question('Which tools compile programs with fast startup?')
    items=[dict(item('AlphaNet compiles programs.'),node_id='a'),dict(item('AlphaNet has fast startup.'),node_id='b')]
    b=dict(entity_id='alpha',name='AlphaNet',types=['method'],fields={'direct_claims':items})
    similarities={d.components[0]['id']:{'a':.9,'b':.2},d.components[1]['id']:{'a':.1,'b':.8}}
    scorer=RequirementCosineReranker(cfg(),d,similarities,{'alpha':['AlphaNet']})
    return d,b,scorer.score('',b['name'],b)

def test_requirements_select_different_evidence_and_missing_core_gate():
    d,b,score=prepared()
    assert [r['best_evidence']['node_id'] for r in score['requirements']]==['a','b']
    assert score['score']==pytest.approx(.9*np.sqrt(.9*.8))
    assert aggregate_support(d.components,[0.,1.],[],['method'],d.answer_type,cfg())['score']==0
    all_strong=aggregate_support(d.components,[.8,.8],[],['method'],d.answer_type,cfg())['score']
    one_strong=aggregate_support(d.components,[1.,.1],[],['method'],d.answer_type,cfg())['score']
    assert all_strong>one_strong

def test_nli_probabilities_preserved_with_candidate_specific_pairs():
    d,b,score=prepared(); predictions={}
    for r in score['requirements']:
        i=r['nli_selected'][0]; predictions[digest([i['text'],r['hypothesis']])]=dict(status='computed',entailment=.8,neutral=.15,contradiction=.05,prediction='entailment',truncated=False,input_tokens=20)
    rr=RequirementNLIReranker(cfg(),d,{'alpha':score},predictions); result=rr.score('',b['name'],b)
    assert result['status']=='computed' and result['core_support']==pytest.approx(.75)
    assert result['requirements'][0]['best_evidence']['nli']['neutral']==.15

def test_nli_failure_explicit_deterministic_cosine_fallback():
    d,b,score=prepared(); rr=RequirementNLIReranker(cfg(),d,{'alpha':score},{})
    result=rr.score('',b['name'],b)
    assert result['status']=='cosine_fallback_nli_failed' and result['score']==score['score']
    assert result==rr.score('',b['name'],b)
    assert result['nli_requirements'][0]['nli_evidence'][0]['nli'] is None

def test_exclusion_violation_reduces_aggregate_score_without_rewarding_neutral():
    d=decompose_question('Which tools compile programs without remote services?')
    a=aggregate_support(d.components,[.8,0.],[],['method'],d.answer_type,cfg())
    b=aggregate_support(d.components,[.8,.9],[.9],['method'],d.answer_type,cfg())
    assert b['score']==pytest.approx(a['score']*.1)

def test_nli_inherits_strict_local_loader_and_exact_pair_cache():
    from tkh_abstraction.entailment import NLI
    assert issubclass(ControlledLocalNLI,NLI)
    assert inspect.getsource(NLI).count('local_files_only=True')==2

def test_production_no_benchmark_rules_or_evaluation_inputs():
    import re
    import tkh_abstraction.question_decomposition as decomp
    import tkh_abstraction.evidence_attribution as attribution
    import tkh_abstraction.constraint_scoring as scoring
    names={label for q in read_json('data/data/ground_truth.json').values() for label in q.get('expected_methods',[])}
    for module in [decomp,attribution,scoring]:
        code=inspect.getsource(module)
        assert not re.search(r'\bQ(?:1[0-4]|[1-9])\b',code)
        for forbidden in ['expected_methods','method_claims','method_evidence','fp_categories','required_citations','oracle_ceiling']:
            assert forbidden not in code
        for name in names:
            if len(name)>=3: assert not re.search(r'(?<!\w)'+re.escape(name)+r'(?!\w)',code),name

def test_synthetic_pipeline_ground_truth_invariance_history_and_candidate_preservation(tmp_path,monkeypatch):
    import tkh_abstraction.constraint_pipeline as pipeline
    from tkh_abstraction.reranker_pipeline import LocalEncoder
    monkeypatch.setattr(pipeline,'LocalEncoder',lambda *args:LocalEncoder('synthetic-hash'))
    c=cfg(); data=tmp_path/'data'; data.mkdir(); root=tmp_path/'eval'; root.mkdir()
    c.update(data_dir=str(data),v2_output=str(root),output=str(tmp_path/'full'),constraint_output=str(root/'constraints'),constraint_verify_history=False,
        constraint_cutoffs=[2025],constraint_nli_enabled=False)
    nodes=[dict(id=i,type=typ,surface_form=text,first_seen_year=2020,provenance=dict(articles=[])) for i,typ,text in [('a','method','AlphaNet'),('b','method','BetaNet'),('c','claim','AlphaNet compiles programs')]]
    write_json(data/'tkh_collection10.json',dict(nodes=nodes,hyperedges=[dict(id='e',members=['a','c'],relation_type='claims',year=2020,provenance=dict(article_year=2020))]))
    (data/'questions.csv').write_text('question_id,type,question\nNEW,A,Which tools compile programs?\nBNEW,B,Why?\n')
    write_json(data/'ground_truth.json',dict(NEW=dict(expected_methods=['AlphaNet']),BNEW={}))
    write_json(tmp_path/'full'/'metrics.json',dict(type_b={'source_hit':.3})); write_json(root/'history.json',dict(sentinel=True)); before=(root/'history.json').read_bytes()
    pipeline.run_constraints(c); one=read_json(root/'requirement_support.json')
    assert all(r['summary']['candidate_preservation_rate']==1 for r in read_json(root/'constraint_reranking_results.json')['results'])
    write_json(data/'ground_truth.json',dict(NEW=dict(expected_methods=['Missing']),BNEW={}))
    pipeline.run_constraints(c); assert one==read_json(root/'requirement_support.json')
    assert before==(root/'history.json').read_bytes() and read_json(tmp_path/'full'/'metrics.json')['type_b']=={'source_hit':.3}

def test_prior_source_freeze_and_oracle_unchanged():
    from tkh_abstraction.v2_pipeline import sha
    old=read_json('artifacts/evaluation/reranker_results.json')
    for path,h in old['protocol']['source_hashes'].items():
        if path.endswith('/cli.py'): continue
        assert sha(path)==h,path
