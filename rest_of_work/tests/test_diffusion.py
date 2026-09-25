import inspect
import numpy as np
import pytest
from scipy import sparse
from tkh_abstraction.diffusion import (retrieval_graph,ThetaOperator,apply_theta,diffuse,semantic_seeds,
    shuffled_graph,MentionIndex,rank_entities,rrf_fuse,safe_mention_name,connecting_paths,propagation_diagnostics)
from tkh_abstraction.baselines import clique_projection
from tkh_abstraction.contextual import entity_groups
from tkh_abstraction.snapshots import snapshot

def node(ident,typ='method',text=None,year=2020):
    return dict(id=ident,type=typ,surface_form=text or ident,first_seen_year=year,provenance=dict(articles=[]))

def edge(ident,members,weight=1.,relation='claims',year=2020):
    return dict(id=ident,members=members,weight=weight,relation_type=relation,year=year,provenance=dict(article_year=year))

def tiny():
    return dict(snapshot=2025,nodes=[node(x) for x in ['A','B','C','D','isolated']],
        hyperedges=[edge('e1',['A','B','C'],2.),edge('e2',['C','D'],.5)])

def dense_theta(snap,rw=None):
    ids=[n['id'] for n in snap['nodes']]; edges=snap['hyperedges']; H=np.zeros((len(ids),len(edges)))
    weights=np.array([e.get('weight',1.)*(rw or {}).get(e['relation_type'],1.) for e in edges])
    for j,e in enumerate(edges):
        for x in e['members']: H[ids.index(x),j]=1
    de=H.sum(axis=0); dv=H@weights; inv=np.zeros(len(ids)); inv[dv>0]=1/np.sqrt(dv[dv>0])
    return np.diag(inv)@H@np.diag(weights/de)@H.T@np.diag(inv)

@pytest.mark.parametrize('rw',[None,{'claims':.25},{'claims':0.}])
def test_dense_sparse_theta_weight_degree_and_arity(rw):
    snap=tiny(); op=ThetaOperator(retrieval_graph(snap,rw)); x=np.array([.2,.7,.1,.5,.4])
    assert np.allclose(op.apply_theta(x),dense_theta(snap,rw)@x,atol=1e-14)
    assert np.allclose(apply_theta(x,snap,rw),dense_theta(snap,rw)@x)
    if rw is None:
        assert op.degree.tolist()==[2.,2.,2.5,.5,0.]
        assert op.arity.tolist()==[3.,2.]
        assert dense_theta(snap)[0,1]==pytest.approx(1/3)
    assert op.apply_theta(x)[-1]==0
    assert op.edge_contributions(x).sum()==pytest.approx(op.apply_theta(x).sum())

@pytest.mark.parametrize('alpha',[.5,.7,.85,.95])
def test_iteration_matches_dense_solve_with_restart_scaling(alpha):
    snap=tiny(); op=ThetaOperator(retrieval_graph(snap)); y=np.array([.6,0.,0.,.2,.2])
    f,run=diffuse(op,y,alpha=alpha,tolerance=1e-12,max_iterations=1000)
    expected=(1-alpha)*np.linalg.solve(np.eye(5)-alpha*dense_theta(snap),y)
    assert np.allclose(f,expected,atol=3e-11) and run['converged']
    assert f[-1]==pytest.approx((1-alpha)*y[-1])
    assert run['theta_applications']==run['iterations'] and run['last_delta_l1']<1e-12

def test_iteration_cap_and_invalid_parameters():
    op=ThetaOperator(retrieval_graph(tiny())); y=np.array([1.,0,0,0,0])
    _,run=diffuse(op,y,max_iterations=1)
    assert not run['converged'] and run['iterations']==1
    with pytest.raises(ValueError): diffuse(op,y,alpha=1.)

def test_deterministic_nonnegative_l1_seeds_and_authors():
    nodes=[node('b'),node('a'),node('c'),node('person','author')]
    y,info=semantic_seeds([.4,.4,-.1,1.],nodes,top_m=2)
    assert [x['node_id'] for x in info['top_seeds']]==['a','b']
    assert y.sum()==pytest.approx(1.) and y[2]==y[3]==0
    fallback,info=semantic_seeds([-.1,-.2,-.3,1.],nodes,top_m=2)
    assert fallback.tolist()==[.5,.5,0,0] and info['fallback']
    z,info=semantic_seeds([.1],[node('author','author')])
    assert z.sum()==0 and info['fallback'].startswith('empty')

def test_entity_max_does_not_reward_duplicate_nodes():
    nodes=[node('a',text='AlphaNet'),node('b','cited_work','ALPHANET'),node('c',text='BetaNet')]
    groups=entity_groups(nodes); ranked=rank_entities([.2,.2,.3],groups,[n['id'] for n in nodes])
    assert ranked[0]['name']=='BetaNet' and ranked[1]['score']==.2

def test_claim_article_intermediates_and_paths_survive_answer_filter():
    nodes=[node('claim','claim','Predicts elastic forces'),node('paper','article'),node('method')]
    snap=dict(nodes=nodes,hyperedges=[edge('c',['claim','paper']),edge('p',['paper','method'],relation='presents')])
    op=ThetaOperator(retrieval_graph(snap)); y=np.array([1.,0,0]); f,run=diffuse(op,y)
    answers=entity_groups([nodes[-1]])
    assert rank_entities(f,answers,op.ids)[0]['score']>0
    paths=connecting_paths(op,[dict(node_id='claim',y=1.)],['method'])
    assert [s['relation'] for s in paths[0]['path']]==['claims','presents']

def test_temporal_isolation_removes_future_seed_and_edge():
    snap=tiny(); snap['nodes'].append(node('future','claim','Future result',2026))
    snap['hyperedges'].append(edge('future_edge',['A','future'],year=2026))
    past=snapshot(snap,2025); op=ThetaOperator(retrieval_graph(past))
    assert 'future' not in op.ids and 'future_edge' not in [e.id for e in op.edges]
    assert np.allclose(op.apply_theta(np.ones(5)),dense_theta(tiny())@np.ones(5))

def test_mentions_safe_boundaries_no_answer_key_and_no_cascade():
    nodes=[node('a',text='AlphaNet'),node('ab',text='AB'),node('generic',text='model'),
        node('claim','claim','AlphaNet predicts forces'),node('version','claim','AlphaNet-v2 predicts forces'),node('substring','claim','SuperAlphaNet predicts forces')]
    groups=entity_groups(nodes[:3]); mentions=MentionIndex(nodes,groups,['claim'])
    alpha=next(g for g in groups if g['name']=='AlphaNet')
    assert mentions.mentions[alpha['id']]==['claim']
    assert not safe_mention_name('AB') and not safe_mention_name('model')
    y=np.array([0,0,0,1.,0,0]); assisted,meta=mentions.assist(y,groups,[n['id'] for n in nodes])
    assert assisted[0]==assisted[3]==.5 and assisted.sum()==1.
    assert 'ground_truth' not in inspect.getsource(MentionIndex)
    assert len(meta['transfers'])==1

def test_rrf_deterministic_rank_only_and_missing_entries():
    a=[dict(entity_id='a',score=9),dict(entity_id='b',score=4)]
    b=[dict(entity_id='b',score=.01),dict(entity_id='a',score=.001),dict(entity_id='c',score=0)]
    fused=rrf_fuse(a,b)
    assert [r['entity_id'] for r in fused]==['a','b','c']
    assert fused[0]['score']==pytest.approx(1/61+1/62)
    assert all(r['score']==sum(r['score_components'].values()) for r in fused)

def test_rrf_exact_score_ties_do_not_supply_hash_order_signal():
    a=[dict(entity_id='a',score=3.),dict(entity_id='b',score=2.),dict(entity_id='c',score=1.)]
    b=[dict(entity_id='c',score=0.),dict(entity_id='b',score=0.),dict(entity_id='a',score=0.)]
    result=rrf_fuse(a,b)
    assert [r['entity_id'] for r in result]==['a','b','c']
    assert {r['rank_diffusion'] for r in result}=={2.}

@pytest.mark.parametrize('name',['100','2020','THE','ALL','a','model'])
def test_mention_names_reject_numeric_and_generic_stopwords(name):
    assert not safe_mention_name(name)

def test_mention_assistance_does_not_cascade():
    nodes=[node('a',text='AlphaNet'),node('b',text='BetaNet'),node('c','claim')]
    groups=entity_groups(nodes[:2]); index=MentionIndex(nodes,groups,['claim'])
    aa=next(g for g in groups if g['name']=='AlphaNet'); bb=next(g for g in groups if g['name']=='BetaNet')
    index.mentions[aa['id']]=['c']; index.mentions[bb['id']]=['a']
    assisted,_=index.assist(np.array([0.,0.,1.]),groups,['a','b','c'])
    assert assisted[0]>.0 and assisted[1]==0

def test_shuffled_control_reproducible_and_preserves_edge_properties():
    graph=retrieval_graph(tiny()); a=shuffled_graph(graph,42); b=shuffled_graph(graph,42)
    assert [(e.id,e.original_members) for e in a.edges]==[(e.id,e.original_members) for e in b.edges]
    assert [(e.arity,e.weight,e.relation) for e in a.edges]==[(e.arity,e.weight,e.relation) for e in graph.edges]
    assert all(len(e.multiplicities)==e.arity for e in a.edges)
    assert graph.edges[0].original_members==('A','B','C')

def test_pairwise_projection_is_weight_conserving_and_same_seeds():
    native=retrieval_graph(tiny()); projected=clique_projection(native)
    assert sum(e.weight for e in projected.edges)==pytest.approx(sum(e.weight for e in native.edges))
    op=ThetaOperator(projected); assert set(op.arity)=={2}
    assert np.isfinite(diffuse(op,np.array([1.,0,0,0,0]))[0]).all()

def test_arity_two_operator_contains_identity_half_not_just_half_adjacency():
    snap=dict(nodes=[node('a'),node('b')],hyperedges=[edge('e',['a','b'])])
    op=ThetaOperator(retrieval_graph(snap))
    assert np.allclose(op.apply_theta(np.array([1.,0.])),[.5,.5])

def test_invalid_weights_and_empty_edges():
    snap=tiny(); snap['hyperedges'][0]['weight']=-1
    with pytest.raises(ValueError): retrieval_graph(snap)
    with pytest.raises(ValueError): retrieval_graph(snap,{'claims':-1.})
    snap['hyperedges']=[]; op=ThetaOperator(retrieval_graph(snap))
    assert np.allclose(op.apply_theta(np.ones(5)),0)

def test_diagnostics_export_mass_degree_and_convergence():
    snap=tiny(); op=ThetaOperator(retrieval_graph(snap)); f,run=diffuse(op,np.array([1.,0,0,0,0]))
    diagnostics=propagation_diagnostics(op,f,snap['nodes'],run)
    assert len(diagnostics['history'])==run['iterations']
    assert diagnostics['operator']['incidence_nnz']==5
    assert diagnostics['top_edges'] and diagnostics['degree_score_spearman'] is not None

def test_complete_synthetic_pipeline_preserves_history_and_type_b(tmp_path):
    import yaml
    from tkh_abstraction.config import read_config
    from tkh_abstraction.io import write_json,read_json
    from tkh_abstraction.diffusion_pipeline import run_diffusion
    cfg=read_config('configs/diffusion.yaml'); context=read_config('configs/contextual.yaml')
    context_file=tmp_path/'context.yaml'; context_file.write_text(yaml.safe_dump(context),encoding='utf-8')
    data_dir=tmp_path/'data'; data_dir.mkdir(); evaluation=tmp_path/'evaluation'; evaluation.mkdir()
    cfg.update(data_dir=str(data_dir),output=str(tmp_path/'full'),v2_output=str(evaluation),diffusion_output=str(evaluation/'diffusion'),context_config=str(context_file),
        diffusion_cutoffs=[2025],evaluation_model='synthetic-hash',diffusion_verify_reuse=False)
    nodes=[node('model',text='AlphaNet'),node('paper','article','Elastic force paper'),node('claim','claim','AlphaNet predicts elastic forces')]
    write_json(data_dir/'tkh_collection10.json',dict(nodes=nodes,hyperedges=[edge('p',['paper','model'],relation='presents'),edge('c',['paper','claim'])]))
    write_json(data_dir/'ground_truth.json',{'A_test':dict(type='A',expected_methods=['AlphaNet','Missing']), 'B_test':dict(type='B')})
    (data_dir/'questions.csv').write_text('question_id,type,question\nA_test,A,Which methods predict elastic forces?\nB_test,B,Why does it work?\n',encoding='utf-8')
    history=evaluation/'retrieval_ablation_results.json'; history.write_text('{"sentinel": true}',encoding='utf-8'); original=history.read_bytes()
    write_json(tmp_path/'full'/'metrics.json',dict(type_b_sentinel=dict(source_hit=.25)))
    run_diffusion(cfg,sensitivity=False)
    assert history.read_bytes()==original
    metrics=read_json(tmp_path/'full'/'metrics.json'); assert metrics['type_b_sentinel']==dict(source_hit=.25)
    assert {r['question_id'] for r in metrics['hypergraph_diffusion']['question_metrics']}=={'A_test'}
    results=read_json(evaluation/'hypergraph_diffusion_results.json')['results']
    assert [r['stage'] for r in results]==cfg['diffusion_stages']
    assert all(r['records'][0]['entity_metrics']['total_expected']==2 for r in results)
    assert all(r['summary']['rank_coverage']['canonical']==1 for r in results)
    for name in ['hypergraph_diffusion_diagnostics','propagation_traces','diffusion_per_target','diffusion_sensitivity']:
        assert (evaluation/(name+'.json')).exists()
    targets=read_json(evaluation/'diffusion_per_target.json'); traces=read_json(evaluation/'propagation_traces.json')
    assert all(t['rank_before']==targets[0]['ranks']['R3_answer_pool'] for t in traces)
    assert all('rank_atomic_similarity' in t and 'seed_is_target' in t for t in traces)

def test_production_operator_never_materializes_dense_theta(monkeypatch):
    def reject(*args,**kwargs): raise AssertionError('dense sparse conversion forbidden')
    monkeypatch.setattr(sparse.csr_matrix,'toarray',reject)
    monkeypatch.setattr(sparse.csc_matrix,'toarray',reject)
    op=ThetaOperator(retrieval_graph(tiny())); f,run=diffuse(op,np.array([1.,0,0,0,0]))
    assert np.isfinite(f).all() and not isinstance(op.L,np.ndarray)
