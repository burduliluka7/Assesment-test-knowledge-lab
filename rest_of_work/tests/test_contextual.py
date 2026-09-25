import numpy as np
import pytest
from tkh_abstraction.config import read_config
from tkh_abstraction.target_mapping import TargetResolver
from tkh_abstraction.contextual import build_representations,entity_groups,ContextIndex,analyze_question,type_prior,FIELDS
from tkh_abstraction.evaluation.retrieval_v2 import contextual_metrics,contextual_summary
from tkh_abstraction.semantics import Encoder
from tkh_abstraction.snapshots import snapshot

def n(i,name,typ='method',articles=(1,),year=2020):
    return dict(id=i,surface_form=name,type=typ,first_seen_year=year,provenance=dict(articles=list(articles)))

def graph(nodes,edges=()): return dict(snapshot=2025,nodes=nodes,hyperedges=list(edges))

def edge(i,rel,members): return dict(id=i,relation_type=rel,members=members,year=2020,provenance=dict(article_id=1,article_year=2020))

@pytest.fixture
def cc(): return read_config('configs/contextual.yaml')

@pytest.mark.parametrize('label,surface,state',[
    ('Alpha-v2','ALPHA–v2','EXACT_NODE'),('equivariant networks','equivariant network','ALIAS_NODE'),
    ('GTM','Graph Tensor Model (GTM)','ALIAS_NODE'),('AlphaNet','Universal AlphaNet predictor','ALIAS_NODE'),
    ('Alpha','Alpha-v2','UNRESOLVED'),('ab','Large ab model','UNRESOLVED')])
def test_generic_resolution(label,surface,state):
    assert TargetResolver(graph([n('x',surface)])).resolve(label)['resolution_type']==state

def test_ambiguous_and_evidence_only_are_not_identity():
    g=graph([n('x','AlphaNet potential'),n('y','AlphaNet graph model'),n('a','uses elastic tensor constraints','claim'),n('b','elastic tensor constraints improve stability','claim')])
    r=TargetResolver(g)
    assert r.resolve('AlphaNet')['resolution_type']=='AMBIGUOUS'
    e=r.resolve('elastic tensor constraints'); assert e['resolution_type']=='EVIDENCE_BACKED' and not e['canonical']
    # Semantic relatedness with no literal identity signal does not become an alias.
    assert r.resolve('strain-aware physics')['resolution_type']=='UNRESOLVED'

def test_composite_components_and_significant_plus():
    r=TargetResolver(graph([n('x','Alpha'),n('y','Beta'),n('z','Code++')]))
    c=r.resolve('Alpha+Beta'); assert c['resolution_type']=='COMPOSITE' and c['representable'] and not c['canonical']
    assert r.resolve('Alpha with Missing')['representable'] is False
    assert r.resolve('Code++')['resolution_type']=='EXACT_NODE'

def test_exact_grouping_does_not_fuzzy_merge():
    groups=entity_groups([n('a','Alpha-v2'),n('b','ALPHA–v2','cited_work'),n('c','Alpha-v3'),n('d','Alpha')])
    assert len(groups)==3 and any(g['node_ids']==['a','b'] for g in groups)

def fixture_graph():
    nodes=[n('m','AlphaNet'),n('p','Paper about elasticity','article'),n('c','AlphaNet predicts elastic forces','claim'),n('t','elastic force prediction','task'),n('d','elastic data','dataset'),n('comp','CompetitorNet')]
    edges=[edge('p','presents',['p','m']),edge('c','claims',['p','c']),edge('t','addresses',['m','t']),edge('e','extends',['m','comp'])]
    return graph(nodes,edges)

def test_context_relations_and_no_broad_neighbor_copy(cc):
    g=fixture_graph(); reps=build_representations(g,cc)
    assert reps['m']['fields']['task'][0]['node_id']=='t'
    assert reps['m']['fields']['claim'][0]['node_id']=='c'
    assert all('CompetitorNet' not in item['text'] for field in reps['m']['fields'].values() for item in field)
    cc['context_max_direct_arity']=1
    other=build_representations(graph(g['nodes'],[edge('large','addresses',['m','t','d'])]),cc)
    assert not other['m']['fields']['task']

def test_context_temporal_filter(cc):
    g=fixture_graph(); g['nodes'].append(n('future','Future scientific result','claim',year=2026))
    s=snapshot(g,2025); reps=build_representations(s,cc)
    assert 'future' not in reps and 'Future scientific result' not in str(reps)

def test_vectors_pooling_soft_priors_and_cost(cc,tmp_path):
    g=fixture_graph(); idx=ContextIndex(g,build_representations(g,cc),Encoder('synthetic-hash',cache=tmp_path),cc)
    assert np.allclose(np.linalg.norm(idx.X,axis=1),1.)
    q=idx.X[0]*4; intent=analyze_question('Which methods predict elastic forces?','A')
    a=idx.search([q],intent); b=idx.search([q],intent,type_enabled=True)
    assert a['cost']['vector_comparisons']==len(idx.texts)
    assert set(r['entity_id'] for r in a['ranked'])==set(r['entity_id'] for r in b['ranked'])
    assert all(r['score']==pytest.approx(sum(r['score_components'].values())) for r in b['ranked'])
    c=idx.search([q,q],intent); assert c['cost']['vector_comparisons']==2*a['cost']['vector_comparisons']
    assert [r['score'] for r in c['ranked']]==pytest.approx([r['score'] for r in a['ranked']])
    assert type_prior(['method'],intent)>type_prior(['article'],intent)

def test_extractive_views_and_unknown_intent():
    question='Which models predict forces with rotational constraints and scarce training data?'
    intent=analyze_question(question)
    assert all(view in question for view in intent['extractive_views'])
    assert analyze_question('Describe this topic')['kind']=='unknown'

def test_multianswer_metrics_and_raw_denominator():
    g=graph([n('a','Alpha'),n('b','Beta'),n('c','Gamma')]); resolver=TargetResolver(g)
    gold=[resolver.resolve(x) for x in ['Alpha','Beta','Missing']]
    rows=[dict(entity_id=x,node_ids=[x],name=name) for x,name in [('c','Gamma'),('a','Alpha'),('b','Beta')]]
    result=dict(ranked=rows,candidate_entity_ids=['c','a','b'])
    m=contextual_metrics(result,gold)
    assert m['at_k']['1']['recall']==0 and m['at_k']['5']['recall']==1
    assert m['at_k']['5']['raw_end_to_end_lower_bound']==2/3
    assert m['at_k']['5']['precision']==2/5 and m['mrr']==.5
    assert m['r_precision']==m['recall_at_r']==.5
    assert m['candidate_recall_50']==m['candidate_recall_100']==1
    assert m['target_outcomes'][-1]['outcome']=='UNREPRESENTABLE'

def test_grouped_topk_and_gold_denominator_stable(cc,tmp_path):
    g=graph([n('a','Alpha'),n('b','ALPHA','cited_work'),n('c','Beta')]); reps=build_representations(g,cc); enc=Encoder('synthetic-hash',cache=tmp_path)
    q=enc.encode(['Alpha'],'q'); intent=analyze_question('Which methods?'); gold=[TargetResolver(g).resolve('Alpha')]
    metrics=[]
    for grouped in [False,True]:
        result=ContextIndex(g,reps,enc,cc,grouped).search(q,intent,surface=True); metrics.append(contextual_metrics(result,gold))
        if grouped: assert len(result['ranked'])==2 and len({r['name'].casefold() for r in result['ranked']})==2
    assert metrics[0]['canonical_expected']==metrics[1]['canonical_expected']==1
    assert metrics[0]['at_k']['5']['precision']==metrics[1]['at_k']['5']['precision']==1/5

def test_no_self_text_double_counting(cc,tmp_path):
    g=graph([n('a','Same scientific name'),n('b','Same scientific name','dataset')])
    reps=build_representations(g,cc)
    assert all(not r['fields'][f] for r in reps.values() for f in FIELDS if f!='name')
    idx=ContextIndex(g,reps,Encoder('synthetic-hash',cache=tmp_path),cc)
    rows=idx.search([idx.X[0]],analyze_question('Find this'))['ranked']
    assert rows[0]['score']==pytest.approx(rows[1]['score'])
    assert rows[0]['score']==pytest.approx(cc['context_field_weights']['name'])

def test_pool_membership_and_candidate_cutoff_are_distinct():
    gold=TargetResolver(graph([n('a','Alpha')])).resolve('Alpha')
    rows=[dict(entity_id=str(i),node_ids=[str(i)],name=str(i)) for i in range(150)]
    rows.append(dict(entity_id='a',node_ids=['a'],name='Alpha'))
    result=dict(ranked=rows,candidate_entity_ids=[r['entity_id'] for r in rows])
    m=contextual_metrics(result,[gold],top_ks=[1,5])
    out=m['target_outcomes'][0]
    assert out['in_pool'] and not out['in_candidate_100'] and out['candidate_failure_reason']=='rank_below_top100'
    assert out['rank']==151 and m['candidate_recall_100']==0
    result=dict(ranked=rows[:-1],candidate_entity_ids=[r['entity_id'] for r in rows[:-1]])
    out=contextual_metrics(result,[gold])['target_outcomes'][0]
    assert not out['in_pool'] and out['candidate_failure_reason']=='not_in_scored_pool'

def test_duplicate_gold_aliases_keep_raw_instances_but_one_identity():
    g=graph([n('a','Universal AlphaNet predictor')]); rr=TargetResolver(g)
    gold=[rr.resolve('AlphaNet'),rr.resolve('Universal AlphaNet predictor'),rr.resolve('AlphaNet')]
    result=dict(ranked=[dict(entity_id='a',node_ids=['a'],name='Universal AlphaNet predictor')],candidate_entity_ids=['a'])
    m=contextual_metrics(result,gold)
    assert m['canonical_expected']==1 and m['total_expected']==3
    assert m['r_precision']==m['recall_at_r']==1
    assert m['at_k']['1']['precision']==m['at_k']['1']['raw_end_to_end_lower_bound']==1

def test_composites_and_evidence_do_not_inflate_canonical_coverage():
    rr=TargetResolver(graph([n('a','Alpha'),n('b','Beta'),n('c','elastic constraints improve speed','claim'),n('d','elastic constraints preserve accuracy','claim')]))
    gold=[rr.resolve('Alpha+Beta'),rr.resolve('elastic constraints')]
    result=dict(ranked=[dict(entity_id=x,node_ids=[x],name=x) for x in ['a','b','c']],candidate_entity_ids=['a','b','c'])
    m=contextual_metrics(result,gold)
    assert m['status']=='not_evaluable' and m['canonical_expected']==0
    assert m['at_k']['5']['representable_recall']==1 and m['at_k']['5']['recall'] is None
    assert all(x['outcome']=='RETRIEVED' for x in m['target_outcomes'])

@pytest.mark.parametrize('pooling,fusion',[('max','max'),('top_m','mean'),('top_m','max')])
def test_fusion_decomposition_and_empty_subset(cc,tmp_path,pooling,fusion):
    cc.update(context_pooling=pooling,context_fusion=fusion)
    g=fixture_graph(); idx=ContextIndex(g,build_representations(g,cc),Encoder('synthetic-hash',cache=tmp_path),cc)
    intent=analyze_question('Which methods predict elastic forces?')
    rows,cost=idx.score([idx.X[0],idx.X[-1]],intent,type_enabled=True)
    assert all(r['score']==pytest.approx(sum(r['score_components'].values())) for r in rows)
    empty,cost=idx.score([idx.X[0]],intent,ids=[])
    assert empty==[] and cost['vector_comparisons']==0

def test_article_expansion_ablation(cc):
    g=fixture_graph(); g['nodes'].append(n('unrelated','Other contextual assertion','claim'))
    a=build_representations(g,cc)
    b=build_representations(g,dict(cc,context_article_expansion=False))
    assert a['m']['fields']['claim_article'] and not b['m']['fields']['claim_article']
    assert a['m']['fields']['claim']==b['m']['fields']['claim']

def test_rerank_stopwords_and_candidate_boundary(cc,tmp_path):
    g=fixture_graph(); cc['context_candidate_k']=2
    idx=ContextIndex(g,build_representations(g,cc),Encoder('synthetic-hash',cache=tmp_path),cc)
    intent=analyze_question('Which methods predict elastic forces?'); q=[idx.X[0]]
    base=idx.search(q,intent); a=idx.search(q,intent,rerank=True)
    b=idx.search(q,dict(intent,task='the of for '+intent['task']),rerank=True)
    assert [r['entity_id'] for r in a['ranked'][2:]]==[r['entity_id'] for r in base['ranked'][2:]]
    assert {r['entity_id']:r['score'] for r in a['ranked']}=={r['entity_id']:r['score'] for r in b['ranked']}

def test_gold_scoring_cannot_change_retrieval(cc,tmp_path):
    g=fixture_graph(); idx=ContextIndex(g,build_representations(g,cc),Encoder('synthetic-hash',cache=tmp_path),cc)
    intent=analyze_question('Which methods predict forces?'); before=idx.search([idx.X[0]],intent)
    resolver=TargetResolver(g)
    contextual_metrics(before,[resolver.resolve('AlphaNet')])
    contextual_metrics(before,[resolver.resolve('Unseen model')])
    after=idx.search([idx.X[0]],intent)
    assert before['ranked']==after['ranked'] and before['cost']==after['cost']

def test_future_provenance_and_hyperedge_context_do_not_leak(cc):
    g=fixture_graph(); g['nodes'] += [n('future','Secret future claim','claim',articles=(2,),year=2026),n('paper2','Future publication','article',articles=(2,),year=2026)]
    g['nodes'][0]['provenance']['articles'].append(2)
    g['hyperedges'].append(dict(edge('future_edge','claims',['m','future']),year=2026))
    historical=snapshot(g,2025); reps=build_representations(historical,cc)
    assert 'Secret future claim' not in str(reps) and 'Future publication' not in str(reps)
    assert 2 not in historical['nodes'][0]['provenance']['articles']
