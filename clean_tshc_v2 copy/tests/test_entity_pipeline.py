import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from tkh_abstraction_v2.entity_index import build_index, group_entities, allowed_entity, name_matches, expand_graph, select_evidence
from tkh_abstraction_v2.entity_retrieval import fuse, round_robin, rerank_prefix, EntityRetriever
from tkh_abstraction_v2.intent import parse_intent
from tkh_abstraction_v2.snapshots import snapshot
from tkh_abstraction_v2.verification import status_from_probabilities, verification_order
from tkh_abstraction_v2.v1_documents import documents


@pytest.fixture
def cfg():
    return json.loads((ROOT/'config.json').read_text())


def n(ident, typ, text, year=2020, **kwargs):
    return dict(id=ident,type=typ,surface_form=text,first_seen_year=year,last_seen_year=year,provenance={'articles':[1]},**kwargs)


def e(ident, rel, members, year=2020):
    return dict(id=ident,relation_type=rel,members=members,year=year,provenance={'article_year':year,'article_id':1})


@pytest.fixture
def graph():
    return dict(nodes=[n('m','method','ToyEngine',aliases=['TE-7']),n('a','article','A source title'),
        n('c','claim','ToyEngine performs fast prediction.'),n('t','task','fast prediction'),n('u','task','accurate prediction'),
        n('x','cited_work','OtherEngine'),n('f','claim','future evidence',2027)],
        hyperedges=[e('ec','claims',['a','m','c']),e('ep','presents',['a','m']),e('et','addresses',['m','t','u']),
                    e('ex','cites',['a','x']),e('ef','claims',['a','m','f'],2027)])


def test_grouping_deterministic_and_no_fuzzy_alias_merges(cfg,graph):
    nodes=graph['nodes']+[n('dup','technique','toyengine'),n('near','method','Toy-Engine'),n('different','dataset','ToyEngine')]
    a,m=group_entities(nodes,cfg,2026)
    random.Random(42).shuffle(nodes)
    b,_=group_entities(nodes,cfg,2026)
    assert a==b
    assert m['m']==m['dup'] and m['m']!=m['near'] and m['m']!=m['different']
    assert 'TE-7' in a[m['m']]['aliases']


def test_temporal_visibility_applies_to_nodes_edges_aliases(cfg,graph):
    graph['nodes'][0]['last_seen_year']=2028
    index=build_index(snapshot(graph,2026),cfg)
    assert 'f' not in index['items']
    assert not index['entities'][index['node_to_entity']['m']]['aliases']
    assert all(step['year']<=2026 for steps in index['arcs'].values() for step in steps)


def test_safe_names_match_boundaries_and_punctuation():
    assert name_matches('We use TE-7 for prediction.','TE-7')
    assert not name_matches('We use TE-70.','TE-7')
    assert not name_matches('a general model','model')
    assert name_matches('Uses AbC(X)+ in this case','AbC(X)+')
    assert not name_matches('Uses AbCX in this case','AbC(X)+')


def test_original_question_and_modifiers_are_retained():
    q='What are the leading ML-based potential methods for large-scale prediction with low cost?'
    d=parse_intent(q)
    assert d['original_question']==q and q in d['combined_query']
    assert 'ML-based potential' in d['preserved_modifiers']
    assert 'ML-based potential' in d['structured_query']
    assert 'large-scale' in d['structured_query']


def test_coordinate_answer_families_and_cutoff():
    d=parse_intent('What datasets and benchmarks are used for prediction by 2022?')
    assert set(d['answer_types'])=={'dataset','benchmark'}
    assert d['temporal_cutoff']==2022


def test_answer_type_and_direct_evidence_mapping(cfg,graph):
    index=build_index(snapshot(graph,2026),cfg)
    entity=index['entities'][index['node_to_entity']['m']]
    assert allowed_entity(entity,parse_intent('Which methods predict values?'),cfg)
    assert not allowed_entity(entity,parse_intent('Which datasets are used?'),cfg)
    assert {'c'}=={x['evidence_id'] for x in entity['evidence']['claims']}
    assert {'t','u'}=={x['evidence_id'] for x in entity['evidence']['tasks']}
    assert any(o['entity_id']==entity['entity_id'] for o in index['evidence_owners']['c'])


def test_role_inference_does_not_depend_on_member_order(cfg,graph):
    index=build_index(snapshot(graph,2026),cfg)
    for edge in graph['hyperedges']: edge['members'].reverse()
    assert index==build_index(snapshot(graph,2026),cfg)


def test_ambiguous_method_roles_are_rejected(cfg,graph):
    graph['hyperedges'].append(e('ambiguous','addresses',['m','x','t']))
    index=build_index(snapshot(graph,2026),cfg)
    assert {'hyperedge_id':'ambiguous','reason':'role_schema_mismatch'} in index['rejected_edges']


def test_graph_bounded_paths_and_hubs(cfg,graph):
    index=build_index(snapshot(graph,2026),cfg)
    scores,traces=expand_graph(index,[('c',.9)],set(index['entities']),cfg)
    target=index['node_to_entity']['x']
    assert target in scores
    assert any(p['path_length']==2 for p in traces[target])
    assert all(p['path_length']<=2 for ts in traces.values() for p in ts)
    assert all(p['interpretation']=='structural_path_not_entailment' for ts in traces.values() for p in ts)
    capped=dict(cfg,graph_article_degree_cap=0)
    scores,_=expand_graph(index,[('c',.9)],set(index['entities']),capped)
    assert target not in scores


def test_query_dependent_selection_and_deduplication(cfg,graph):
    index=build_index(snapshot(graph,2026),cfg); eid=index['node_to_entity']['m']
    cfg=copy.deepcopy(cfg);cfg['query_evidence_caps']['tasks']=1
    a=select_evidence(index,eid,{'t':.9,'u':.1},cfg)
    b=select_evidence(index,eid,{'t':.1,'u':.9},cfg)
    assert [r['node_id'] for r in a['selected'] if r['field']=='tasks']==['t']
    assert [r['node_id'] for r in b['selected'] if r['field']=='tasks']==['u']
    assert len([r for r in a['selected'] if r['node_id']=='c'])==1
    assert a['blocks'][0]['field']=='identity' and 'ToyEngine' in a['blocks'][0]['text']


def test_rrf_is_rank_only_and_order_independent():
    channels={'a':[{'entity_id':'x','score':1000},{'entity_id':'y','score':2}],
              'b':[{'entity_id':'y','score':.7},{'entity_id':'z','score':.6}]}
    rows,parts=fuse(channels,60)
    assert rows[0]['entity_id']=='y'
    assert rows[0]['score']==pytest.approx(1/62+1/61)
    assert sum(x['contribution'] for x in parts['y'].values())==rows[0]['score']
    assert fuse(dict(reversed(list(channels.items()))),60)==(rows,parts)
    assert set(r['entity_id'] for r in round_robin(channels))=={'x','y','z'}


def test_reranker_only_changes_frozen_prefix():
    rows=[dict(entity_id=i,score=4-j) for j,i in enumerate('abcd')]
    details={i:dict(relevance_score=j) for j,i in enumerate('ab')}
    out=rerank_prefix(rows,details,2)
    assert [r['entity_id'] for r in out]==list('bacd')
    with pytest.raises(KeyError):rerank_prefix(rows,details,3)


def test_requirement_statuses_never_equate_missing_and_contradiction():
    con={'entailment':.01,'neutral':.01,'contradiction':.98}
    ent={'entailment':.98,'neutral':.01,'contradiction':.01}
    assert status_from_probabilities(con,False)=='NOT_SUPPORTED_BY_AVAILABLE_EVIDENCE'
    assert status_from_probabilities(None,True)=='NOT_EVALUABLE'
    assert status_from_probabilities(ent,True)=='SUPPORTED'
    assert status_from_probabilities(con,True)=='CONTRADICTED'
    assert status_from_probabilities(ent,True,exclusion=True)=='CONTRADICTED'
    rows=[dict(entity_id=i,score=1) for i in 'abcd']
    result=verification_order(rows,{'a':{'tier':2},'b':{'tier':1},'c':{'tier':0}},3)
    assert [r['entity_id'] for r in result]==list('cbad')


class TestEncoder:
    def encode(self,texts):
        values=[]
        for t in texts:
            v=np.array([t.lower().count(k)+.01 for k in ('fast','accurate','prediction','toy','source')])
            values.append(v/np.linalg.norm(v))
        return np.array(values)


def test_multichannel_generation_deterministic_with_real_index(cfg,graph):
    snap=snapshot(graph,2026);index=build_index(snap,cfg)
    retriever=EntityRetriever(index,documents(snap,cfg),TestEncoder(),cfg)
    intent=parse_intent('Which methods perform fast prediction?')
    a=retriever.generate(intent);b=retriever.generate(intent)
    assert a==b
    assert len(a['systems']['A4'])==len(a['systems']['A5'])
    union={r['entity_id'] for rows in a['channels'].values() for r in rows}
    assert union==set(a['candidate_sources'])
    assert all(allowed_entity(index['entities'][i],intent,cfg) for i in union)


def test_runtime_refuses_historical_answers_and_evaluator_imports():
    code="""
from pathlib import Path
from tkh_abstraction_v2.isolation import install
r=Path.cwd();install(r)
for p in [r/'archive/anything.json',r/'outputs/metrics.json',r/'data/target_annotations.json']:
 try:p.open()
 except PermissionError:pass
 else:raise AssertionError(p)
try:__import__('tkh_abstraction_v2.entity_evaluation')
except (PermissionError,ModuleNotFoundError):pass
else:raise AssertionError('evaluation import')
"""
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1')
    result=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_v1_matches_task_start_manifest():
    import hashlib
    before=json.loads((ROOT/'notes/entity_redesign/v1_before.json').read_text())
    actual={p.relative_to(ROOT.parent/'clean_tshc').as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT.parent/'clean_tshc').rglob('*') if p.is_file()}
    assert actual==before
