"""Synthetic requirement-ranker and adversarial process-boundary tests; no gold."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tkh_abstraction_v2.intent import parse_intent
from tkh_abstraction_v2.requirement_ranking import (atomic_components,select_requirement_evidence,
    bundle,competition_ranks,aggregation,order_candidates)
from tkh_abstraction_v2.verification import status_from_probabilities
from tkh_abstraction_v2.models import pack
from test_invariants import TinyTokenizer


def test_atomic_source_spans_and_core():
    q='Which methods are suited for predicting fields when balancing high accuracy with low cost for systems of many particles without iterative updates?'
    d=parse_intent(q);before=copy.deepcopy(d);cs=atomic_components(d)
    assert d==before and d['original_question']==q
    assert [c['text'] for c in cs]==['predicting fields','high accuracy','low cost','systems of many particles','iterative updates']
    assert cs[0]['kind']=='core_task' and cs[-1]['polarity']=='excluded_property'
    for c in cs:
        span=c['source_span'];assert q[span['start']:span['end']]==span['text']==c['text']


def test_atomic_commas_and_disjunction_preservation():
    cs=atomic_components(parse_intent('Which methods predict values with low cost, high accuracy, and low memory?'))
    assert [c['text'] for c in cs[1:]]==['low cost','high accuracy','low memory']
    cs=atomic_components(parse_intent('Which methods predict values with low cost or high accuracy?'))
    assert any('or' in c['text'] for c in cs)


def index_fixture():
    def link(i,field='claims',specific=True):return dict(evidence_id=i,field=field,candidate_specific=specific,paths=[],support_kind='direct')
    return dict(entities={'e':dict(entity_id='e',name='ToyEngine',types=['method'],evidence={
        'claims':[link('a'),link('b'),link('dup')],'tasks':[link('t','tasks')],
        'articles':[link('art','articles')],'graph_evidence':[link('path','graph_evidence',False)]})},
        items={i:dict(text=text,provenance={'source':i}) for i,text in {
            'a':'Accurate estimates.','b':'Fast estimates.','dup':'Accurate estimates.',
            't':'Estimate fields.','art':'A title','path':'Cites another method','unrelated':'Unrelated other entity claim'}.items()})


def test_top3_owned_deduplicated_evidence_only():
    idx=index_fixture();scores={i:1 for i in idx['items']};scores['unrelated']=100;scores['art']=99
    a=select_requirement_evidence(idx,'e',scores,3)
    assert [r['evidence_id'] for r in a]==['a','b','t']
    assert a==select_requirement_evidence(idx,'e',scores,3)
    assert all(r['provenance'] and r['candidate_specific'] for r in a)


def test_requirement_specific_selection():
    idx=index_fixture();scores={i:0 for i in idx['items']};scores['b']=1
    assert select_requirement_evidence(idx,'e',scores,1)[0]['evidence_id']=='b'
    scores['a']=2;assert select_requirement_evidence(idx,'e',scores,1)[0]['evidence_id']=='a'


def test_multi_evidence_packing_identity_and_limits():
    idx=index_fixture();ev=select_requirement_evidence(idx,'e',{i:1 for i in idx['items']},3)
    c=dict(id='r',text='low cost');card=bundle(idx['entities']['e'],c,ev)
    a=pack(TinyTokenizer(),card['query'],card['blocks'],40)
    assert a==pack(TinyTokenizer(),card['query'],card['blocks'],40)
    assert len(a[2]['included'])==4 and a[2]['token_count']<=40
    assert a[2]['included'][0]['field']=='identity' and 'ToyEngine' in a[1]


def rows(statuses,ranks):
    return [dict(kind='core_task' if j==0 else 'requirement',status=s,requirement_rank=r) for j,(s,r) in enumerate(zip(statuses,ranks))]


@pytest.mark.parametrize('statuses,tier',[
    (['SUPPORTED','SUPPORTED'],0),(['SUPPORTED','NOT_SUPPORTED_BY_AVAILABLE_EVIDENCE'],1),
    (['NOT_EVALUABLE','NOT_SUPPORTED_BY_AVAILABLE_EVIDENCE'],2),(['SUPPORTED','CONTRADICTED'],3)])
def test_four_tier_conjunction(statuses,tier):
    assert aggregation(rows(statuses,[1,2]))['final_tier']==tier


def test_geometric_and_weakest_scores():
    a=aggregation(rows(['SUPPORTED']*3,[1,4,16]),epsilon=0)
    assert a['requirement_geometric_score']==pytest.approx(1/4)
    assert a['weakest_requirement_score']==1/16
    balanced=aggregation(rows(['SUPPORTED']*3,[3,3,3]),epsilon=0)
    assert balanced['requirement_geometric_score']>a['requirement_geometric_score']


def test_exclusions_do_not_reward_topical_similarity():
    rec=rows(['SUPPORTED'],[5]);rec.append(dict(kind='exclusion',status='CONTRADICTED',requirement_rank=1))
    assert aggregation(rec)['final_tier']==3
    assert aggregation(rec,epsilon=0)['requirement_geometric_score']==pytest.approx(.2)
    p={'entailment':.98,'neutral':.01,'contradiction':.01}
    assert status_from_probabilities(p,True,exclusion=True)=='CONTRADICTED'
    assert status_from_probabilities(p,False,exclusion=True)=='NOT_SUPPORTED_BY_AVAILABLE_EVIDENCE'
    assert status_from_probabilities({'entailment':.01,'neutral':.01,'contradiction':.98},True,exclusion=True)=='SUPPORTED'


def test_shared_rank_ties_and_missing_midpoint():
    assert competition_ranks({'b':1,'a':1,'c':0})=={'b':1,'a':1,'c':3}
    rrf=[dict(entity_id=i,score=1) for i in 'abcd']
    details={i:dict(relevance_score=1) for i in 'abc'}
    records={i:[dict(kind='core_task',status='NOT_SUPPORTED_BY_AVAILABLE_EVIDENCE',requirement_relevance_logit=v)] for i,v in zip('abc',[2,1,None])}
    cfg={'rerank_k':3,'requirement_epsilon':1e-12}
    a,t=order_candidates(rrf,details,[{'id':'r'}],records,cfg)
    assert t['c']['requirements'][0]['requirement_rank']==1.5 and t['c']['requirements'][0]['rank_imputed']
    assert [r['entity_id'] for r in a]==list('acbd')
    assert order_candidates(rrf,details,[{'id':'r'}],dict(reversed(list(records.items()))),cfg)[0]==a


def test_all_missing_falls_back_to_global_then_rrf():
    rrf=[dict(entity_id=i,score=1) for i in 'abc'];details={i:dict(relevance_score=v) for i,v in zip('abc',[1,2,2])}
    records={i:[dict(kind='core_task',status='NOT_SUPPORTED_BY_AVAILABLE_EVIDENCE',requirement_relevance_logit=None)] for i in 'abc'}
    ranked,t=order_candidates(rrf,details,[{'id':'r'}],records,{'rerank_k':3,'requirement_epsilon':1e-12})
    assert [r['entity_id'] for r in ranked]==list('bca')
    assert all(v['final_tier']==2 for v in t.values())


def subprocess_code(code):
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1')
    result=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_every_historical_and_gold_category_denied():
    subprocess_code('''from pathlib import Path
from tkh_abstraction_v2.isolation import install
r=Path.cwd();audit=install(r)
for relative in ['data/ground_truth.json','data/target_annotations.json','outputs/retrieval/predictions.json',
 'outputs/audit/strict_target_support.json','outputs/failure_traces.json','report/v2_retrieval_report.md',
 'archive/old.json','notes/old.txt','artifacts/evaluation/metrics.json','data/gold_aliases.json']:
 try:(r/relative).open()
 except PermissionError:pass
 else:raise AssertionError(relative)
try:__import__('tkh_abstraction_v2.entity_evaluation')
except (PermissionError,ModuleNotFoundError):pass
else:raise AssertionError('import allowed')
assert len([a for a in audit if not a['allowed']])==10
(r/'data/questions.csv').read_bytes()
assert any(a['path']=='clean_tshc_v2/data/questions.csv' and a['allowed'] for a in audit)
''')


def test_evaluator_cannot_write_remove_rename_or_chmod_frozen_file():
    subprocess_code('''from pathlib import Path
import os
from tkh_abstraction_v2.isolation import install
r=Path.cwd();p=r/'.cache/test_readonly/prediction.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text('frozen')
install(r,prediction=False,frozen_files=[p])
for action in [lambda:p.write_text('bad'),lambda:p.unlink(),lambda:p.rename(p.with_suffix('.other')),lambda:os.chmod(p,0o777)]:
 try:action()
 except PermissionError:pass
 else:raise AssertionError('mutation allowed')
assert p.read_text()=='frozen'
for base in ['clean_tshc','rest_of_work']:
 try:(r.parent/base/'DO_NOT_CREATE').write_text('bad')
 except PermissionError:pass
 else:raise AssertionError('protected write allowed')
''')


def test_evaluator_refuses_missing_freeze():
    from tkh_abstraction_v2.freeze import verify_freeze
    with pytest.raises(FileNotFoundError):verify_freeze(ROOT,ROOT/'.cache/nonexistent_freeze')


def test_no_answer_rules_training_or_fitted_weights():
    source=(ROOT/'src/tkh_abstraction_v2/requirement_ranking.py').read_text()
    for forbidden in ('ground_truth','target_annotations','strict_targets','MACE','DeepH','D4FT','.fit(','.backward(','.train(', 'optimizer'):
        assert forbidden not in source
    cfg=json.loads((ROOT/'config.json').read_text())
    assert cfg['rerank_k']==500 and cfg['top_k_requirement_evidence']==3 and cfg['primary_system']=='A7'


def test_candidate_generator_source_unchanged():
    original=json.loads((ROOT/'notes/requirement_design/v2_initial.json').read_text())
    for p in ['src/tkh_abstraction_v2/entity_index.py','src/tkh_abstraction_v2/entity_retrieval.py','src/tkh_abstraction_v2/intent.py','src/tkh_abstraction_v2/extractive.py']:
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==original[p]

def test_requirement_artifact_ownership_and_conjunction_when_frozen():
    pred=ROOT/'artifacts/prediction'
    if not (pred/'audit/freeze.json').exists():pytest.skip('Prediction must freeze before artifact inspection')
    index=json.loads((pred/'entity_index/2026.json').read_text(encoding='utf8'))
    for path in (pred/'retrieval/requirement_ranking/A7').glob('*.json'):
        trace=json.loads(path.read_text(encoding='utf8'))
        intent=json.loads((pred/'query_understanding'/path.name).read_text(encoding='utf8'))
        for eid,t in trace.items():
            if t.get('final_tier') is None:continue
            owned={r['evidence_id'] for f,rs in index['entities'][eid]['evidence'].items() if f not in ('articles','graph_evidence') for r in rs if r['candidate_specific']}
            for c in t['requirements']:
                assert set(c['selected_evidence_ids'])<=owned and len(c['selected_evidence_ids'])<=3
                sp=c['source_span'];assert intent['original_question'][sp['start']:sp['end']]==c['text']
                for key in ('relevance_input','nli_input'):
                    if c[key]:assert c[key]['token_count']<=512 and c[key]['included'][0]['field']=='identity'
                if not c['selected_evidence_ids']:assert c['status']=='NOT_SUPPORTED_BY_AVAILABLE_EVIDENCE'
            expected=aggregation(t['requirements'])
            assert all(t[k]==v for k,v in expected.items())


def test_z_both_protected_trees_unchanged():
    # Full content audit, including the read-only third-party interpreter tree.
    sys.path.insert(0,str(ROOT/'scripts'))
    from audit_reference import verify_reference
    result=verify_reference()
    assert result['unchanged']
    assert set(result['trees'])=={'clean_tshc','rest_of_work'}


def test_supported_exclusion_alone_cannot_replace_core_support():
    records=rows(['NOT_SUPPORTED_BY_AVAILABLE_EVIDENCE'],[1])
    records.append(dict(kind='exclusion',status='SUPPORTED',requirement_rank=1))
    assert aggregation(records)['final_tier']==2
    records[0]['status']='SUPPORTED'
    assert aggregation(records)['final_tier']==0

def test_bundle_scoring_uses_joint_owned_premise_and_literal_requirement():
    import numpy as np
    from tkh_abstraction_v2.requirement_ranking import score_requirements
    idx=index_fixture()
    idx['entities']['empty']=dict(entity_id='empty',name='EmptyEngine',types=['method'],evidence={})
    class Encoder:
        def encode(self,texts):return np.ones((len(texts),2))
    class Retriever:
        encoder=Encoder()
        evidence_ids=sorted(idx['items'])
        evidence_vectors=np.ones((len(evidence_ids),2))
    class Cross:
        def score_pairs(self,cards,batch=32):
            result={}
            for c in cards:
                assert 'Candidate ToyEngine:' in c['query']
                _,doc,detail=pack(TinyTokenizer(),c['query'],c['blocks'],512)
                assert 'Unrelated other entity claim' not in doc and 'A title' not in doc
                result[c['node_id']]=dict(detail,relevance_score=2.)
            return result
    class Nli:
        tokenizer=TinyTokenizer()
        calls=[]
        def score(self,pairs):
            self.calls.extend(pairs)
            for premise,hyp in pairs:
                assert 'ToyEngine' in premise and 'ToyEngine' in hyp
                assert all(t in premise for t in ('Accurate estimates.','Fast estimates.','Estimate fields.'))
            return [dict(probabilities={'entailment':.95,'neutral':.04,'contradiction':.01}) for _ in pairs]
    intent=parse_intent('Which methods predict fields with low cost without labels?')
    intent['ranking_components']=atomic_components(intent);nli=Nli()
    rr=[dict(entity_id=i,score=1) for i in ('e','empty')]
    records,costs=score_requirements(intent,rr,idx,Retriever(),Cross(),nli,
        dict(rerank_k=2,batch_size=32,cross_encoder_max_tokens=512,nli_threshold=.7),3)
    assert costs['nli_pairs']==3 and costs['requirement_cross_encoder_pairs']==3
    assert [r['status'] for r in records['e']]==['SUPPORTED','SUPPORTED','CONTRADICTED']
    assert all(r['status']=='NOT_SUPPORTED_BY_AVAILABLE_EVIDENCE' and r['probabilities'] is None for r in records['empty'])
    assert all(len(r['nli_input']['included'])==4 for r in records['e'])


def test_atomic_numeric_commas_preserve_quantitative_requirement():
    q='Which methods estimate values when screening 50,000 examples with low cost, high accuracy, and low memory?'
    cs=atomic_components(parse_intent(q))
    assert any(c['text']=='screening 50,000 examples' for c in cs)
    assert [c['text'] for c in cs[-3:]]==['low cost','high accuracy','low memory']
