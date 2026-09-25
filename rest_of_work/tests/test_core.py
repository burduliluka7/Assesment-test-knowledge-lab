from collections import Counter
import numpy as np
import pytest
from tkh_abstraction.hypergraph import Unit,Edge,Hypergraph,fragmentation,from_snapshot
from tkh_abstraction.collapse import collapse
from tkh_abstraction.spectral import incidence,laplacian,coordinates
from tkh_abstraction.snapshots import snapshot
from tkh_abstraction.schema import validate_data
from tkh_abstraction.temporal import vi_delta,normalized_vi,track
from tkh_abstraction.hierarchy import build,validate_hierarchy
from tkh_abstraction.coarsening import agglomerate
from tkh_abstraction.labeling import label
from tkh_abstraction.retrieval import hierarchical,flat
from tkh_abstraction.evaluation.extrinsic import map_methods
from tkh_abstraction.evaluation.coherence import evaluate as coherence

def graph(n=5):
    return Hypergraph([Unit(str(i),(str(i),),np.array([float(i),1.])) for i in range(n)],
        [Edge('edge','relation',dict.fromkeys(map(str,range(n)),1),tuple(map(str,range(n))),{'article_id':7})])

@pytest.mark.parametrize('groups,mults',[
    ([['0','1','2','3','4']],[5]),
    ([['0','1','2'],['3','4']],[2,3]),
    ([['0','1'],['2'],['3','4']],[1,2,2])])
def test_collapse_cases(groups,mults):
    g,_=collapse(graph(),groups); e=g.edges[0]
    assert len(g.edges)==1 and sorted(e.multiplicities.values())==mults
    assert e.arity==5 and e.provenance=={'article_id':7} and e.original_members==tuple(map(str,range(5)))
    assert e.internal==(len(groups)==1)
    B,_=incidence(g); assert B.shape[1]==(0 if e.internal else 1)
    gg,_=collapse(g,[[u.id for u in g.units]])
    assert list(gg.edges[0].multiplicities.values())==[5]

def test_fragmentation_multiplicity():
    e=graph(4).edges[0]
    assert fragmentation(e)==1
    a=fragmentation(e,dict(zip(map(str,range(4)),[0,0,0,1])))
    b=fragmentation(e,dict(zip(map(str,range(4)),[0,0,1,1])))
    assert a<b

@pytest.mark.parametrize('seed',range(5))
def test_hyper_delta_trace_equals_full_objective(seed,cfg):
    from tkh_abstraction.hypergraph import objective_hyper
    g=graph(8)
    g.edges.append(Edge('other','relation',{'1':1,'3':1,'5':1},('1','3','5'),{},2.))
    X=np.random.default_rng(seed).normal(size=(8,2))
    for u,x in zip(g.units,X): u.centroid=x
    cfg['weights']=dict(hyper=.8,spectral=0,semantic=.2,temporal=0)
    groups,trace=agglomerate(g,np.zeros((8,1)),2,cfg,'static')
    mapping={x:i for i,group in enumerate(groups) for x in group}
    assert trace['sum_merge_deltas']['hyper']==pytest.approx(objective_hyper(g,mapping)-objective_hyper(g))

def test_collapse_rejects_overlap():
    with pytest.raises(ValueError): collapse(graph(3),[['0','1'],['1','2']])

def test_collapse_does_not_mutate_native_reference():
    g=graph(4); ids=[u.id for u in g.units]; members=dict(g.edges[0].multiplicities)
    collapsed,_=collapse(g,[['0','1'],['2','3']])
    assert collapsed is not g and [u.id for u in g.units]==ids
    assert g.edges[0].multiplicities==members and fragmentation(g.edges[0])==1

def test_flat_uses_cosine_even_with_unnormalized_rows():
    nodes=[dict(id='correct',type='method'),dict(id='large_norm',type='method')]
    X=np.array([[1.,0.],[10.,10.]])
    assert flat(np.array([5.,0.]),nodes,X)['returned_ids'][0]=='correct'

def test_encoder_normalizes_rows(tmp_path):
    from tkh_abstraction.semantics import Encoder
    e=Encoder('synthetic-hash',cache=tmp_path)
    X=e.encode(['quantum atoms','graph neural graph'],'unit-test')
    assert np.allclose(np.linalg.norm(X,axis=1),1.)

def test_fresh_and_cached_embeddings_identical(tmp_path):
    from tkh_abstraction.semantics import Encoder
    e=Encoder('synthetic-hash',cache=tmp_path)
    fresh=e.encode(['quantum quantum atoms','neural graph model'],'cache-parity')
    cached=e.encode(['quantum quantum atoms','neural graph model'],'cache-parity')
    assert fresh.dtype==cached.dtype==np.float32
    assert np.array_equal(fresh,cached)

def test_supporting_claim_budget():
    from tkh_abstraction.retrieval import supporting_claims
    nodes=[dict(id='m',type='method',provenance={'articles':[1]}),dict(id='c',type='claim',surface_form='supported claim',provenance={'articles':[1]}),dict(id='unrelated',type='claim',surface_form='other claim',provenance={'articles':[2]})]
    s=dict(nodes=nodes,hyperedges=[]); X=np.array([[1.,0.],[1.,0.],[1.,0.]])
    assert supporting_claims(X[0],['m'],s,X,0)['inspections']==0
    r=supporting_claims(X[0],['m'],s,X,1)
    assert r['inspections']==1 and [x['id'] for x in r['claims']]==['c']

def test_spectral():
    g=graph(); g.units.append(Unit('iso',('iso',),np.zeros(2)))
    L,B,dv=laplacian(g); dense=L@np.eye(6)
    assert B.shape==(6,1)
    assert np.allclose(dense,dense.T) and np.isfinite(dense).all()
    assert np.linalg.eigvalsh(dense).min()>-1e-9
    x,_=coordinates(g,3); assert np.isfinite(x).all()

def test_temporal_honesty(fixture_data,cfg):
    data=fixture_data
    data['hyperedges'].append(dict(id='future',members=['n0','n10'],year=1999,relation_type='claims',provenance={'article_year':2022}))
    s=snapshot(data,2020)
    assert 'n10' not in {n['id'] for n in s['nodes']}
    assert 'future' not in {e['id'] for e in s['hyperedges']}
    assert 'future' in s['deferred_edge_ids']
    assert all('last_seen_year' not in n for n in s['nodes'])
    X=np.random.default_rng(0).normal(size=(len(s['nodes']),8)); h,_=build(s,X,cfg)
    logs=label(h,s,X)
    assert all(n['first_seen_year']<=2020 for r in logs for n in r['input'])
    assert all(n['id']!='n10' for r in logs for n in r['input'])

def test_quality_records_conflicts(fixture_data):
    fixture_data['hyperedges'][0]['year']=1900
    r=validate_data(fixture_data)
    assert r['issue_counts']['edge_before_members_seen']>=1

@pytest.mark.parametrize('seed',range(20))
def test_vi_delta_matches_brute_force(seed):
    rng=np.random.default_rng(seed); q={i:int(rng.integers(0,4)) for i in range(30)}
    p={i:int(rng.integers(0,6)) for i in range(35)}
    a=Counter(q[i] for i in q if p[i]==0); b=Counter(q[i] for i in q if p[i]==1)
    merged={i:0 if c==1 else c for i,c in p.items()}
    assert vi_delta(a,b,30)==pytest.approx(normalized_vi(merged,q)-normalized_vi(p,q))
    assert vi_delta(a,Counter(),30)==0

def test_temporal_changes_construction(cfg):
    g=Hypergraph([Unit(str(i),(str(i),),np.zeros(1)) for i in range(4)],[])
    cfg['weights']=dict(hyper=0,spectral=0,semantic=0,temporal=1)
    prev={'0':'a','2':'a','1':'b','3':'b'}
    groups,diag=agglomerate(g,np.zeros((4,1)),2,cfg,'temporal',prev)
    assert {frozenset(g) for g in groups}=={frozenset(['0','2']),frozenset(['1','3'])}
    assert diag['nonzero_temporal_merges']>0
    g.units[0].centroid=np.array([0.]); g.units[1].centroid=np.array([0.]); g.units[2].centroid=np.array([10.]); g.units[3].centroid=np.array([10.])
    cfg['weights']=dict(hyper=0,spectral=0,semantic=.999,temporal=.001)
    groups,_=agglomerate(g,np.zeros((4,1)),2,cfg,'temporal',prev)
    assert {frozenset(g) for g in groups}=={frozenset(['0','1']),frozenset(['2','3'])}

def test_recursive_hierarchy(fixture_data,cfg,monkeypatch):
    import tkh_abstraction.hierarchy as module
    s=snapshot(fixture_data,2022); X=np.random.default_rng(0).normal(size=(12,6)); seen=[]
    original=module.coordinates
    def spy(g,*args): seen.append((len(g.units),[dict(e.multiplicities) for e in g.edges])); return original(g,*args)
    monkeypatch.setattr(module,'coordinates',spy)
    h,_=build(s,X,cfg); validate_hierarchy(h,{n['id'] for n in s['nodes']})
    assert [r[0] for r in seen]==[12,4]
    assert any(max(m.values())>1 for m in seen[1][1])
    assert [len(l) for l in h['levels']]==[2,4,12]
    h['levels'][1][0]['parent_id']='invalid'
    with pytest.raises(ValueError): validate_hierarchy(h,{n['id'] for n in s['nodes']})

def test_validator_rejects_empty_clusters(fixture_data,cfg):
    from copy import deepcopy
    s=snapshot(fixture_data,2020); X=np.zeros((len(s['nodes']),2)); h,_=build(s,X,cfg)
    h['budgets'][0]+=1
    empty=deepcopy(h['levels'][0][0]); empty['id']='empty'; empty['member_ids']=[]
    h['levels'][0].append(empty)
    with pytest.raises(ValueError,match='empty cluster'): validate_hierarchy(h,{n['id'] for n in s['nodes']})

def cl(ident,members): return dict(id=ident,member_ids=members,persistent_id=ident)

@pytest.mark.parametrize('old,new,events',[
    ([cl('a',['1','2'])],[cl('b',['1','2'])],{'continuation'}),
    ([cl('a',['1','2'])],[cl('b',['1','2','3'])],{'growth'}),
    ([cl('a',['1','2','3','4'])],[cl('b',['1','2']),cl('c',['3','4'])],{'split'}),
    ([cl('a',['1','2']),cl('b',['3','4'])],[cl('c',['1','2','3','4'])],{'merge'}),
    ([cl('a',['1'])],[cl('b',['2'])],{'birth','death'})])
def test_events(old,new,events):
    records=track(old,new,2020,2022,0)
    assert events<={r['event_type'] for r in records}

def test_retrieval_accounting():
    nodes=[dict(id=x,type='method') for x in ['a','b','c','d']]; X=np.array([[1.,0.],[.9,.1],[0.,1.],[.1,.9]])
    h=dict(levels=[[dict(id='left',member_ids=['a','b'],parent_id=None),dict(id='right',member_ids=['c','d'],parent_id=None)],
        [dict(id=x,member_ids=[x],parent_id='left' if x in 'ab' else 'right') for x in 'abcd']])
    q=np.array([1.,0.]); r=hierarchical(q,h,nodes,X,4,1)
    assert r['inspections']==4 and r['returned_ids'][0]=='a'
    assert len(r['inspected_objects'])==len(set(r['inspected_objects']))
    assert hierarchical(q,h,nodes,X,3,1)['inspections']==2
    assert flat(q,nodes,X)['inspections']==4
    assert flat(q,nodes,X,budget=2)['inspections']==2

def test_mapping_no_acronym_substring():
    nodes=[dict(id='a',type='method',surface_form='MACE'),dict(id='b',type='method',surface_form='ACE (atomic cluster expansion)')]
    mapping=map_methods({'Q':{'expected_methods':['ACE','MACE','Missing']}},nodes)
    assert mapping['ACE']['node_ids']==['b'] and mapping['Missing']['node_ids']==[]

def test_mapping_preserves_model_variants():
    nodes=[dict(id='deep',type='method',surface_form='DeepH-E3'),dict(id='mace',type='method',surface_form='MACE-MP-0'),dict(id='dime',type='method',surface_form='DimeNet++')]
    gt={'Q':{'expected_methods':['DeepH','DeepH-E3','MACE','DimeNet','DimeNet++']}}
    r=map_methods(gt,nodes)
    assert r['DeepH']['node_ids']==[] and r['MACE']['node_ids']==[] and r['DimeNet']['node_ids']==[]
    assert r['DeepH-E3']['node_ids']==['deep'] and r['DimeNet++']['node_ids']==['dime']

def test_coherence_null_preserves_sizes():
    nodes=[dict(id=str(i),type='method') for i in range(4)]
    h=dict(levels=[[dict(member_ids=['0','1']),dict(member_ids=['2','3'])]])
    X=np.array([[1.,0.],[1.,0.],[0.,1.],[0.,1.]])
    r=coherence(h,nodes,X,50)['0']
    assert r['node_weighted']==1 and r['null']['difference']>0
