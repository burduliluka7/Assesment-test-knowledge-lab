import numpy as np
from tkh_abstraction.target_mapping import normalize,resolve_target,route_question,map_benchmark
from tkh_abstraction.search_v2 import SearchIndex,flat_search,retrieve_claims
from tkh_abstraction.evaluation.retrieval_v2 import entity_metrics
from tkh_abstraction.evaluation.claim_alignment import SourceIndex,decide_alignment
from tkh_abstraction.labels_v2 import evidence_split,generation_packet,generate_from_packet
from tkh_abstraction.snapshots import snapshot
from tkh_abstraction.config import read_config

def test_expected_recall_evaluability_distinguishes_unmapped_from_misses():
    from tkh_abstraction.evaluation.retrieval_v2 import question_evaluability,summarize
    questions=[dict(question_id=q,type=t) for q,t in [('available','A'),('unresolved','A'),('explanation','B')]]
    gt={'available':dict(type='A',expected_methods=['Known']),
        'unresolved':dict(type='A',expected_methods=['Unknown']), 'explanation':dict(type='B')}
    mappings=map_benchmark(gt,[node('n','Known')],2025)
    inventory=question_evaluability(questions,mappings)
    assert [r['expected_recall_evaluability']['status'] for r in inventory]==['evaluable','not_evaluable','not_applicable']
    records=[dict(variant='test',cutoff=2025,method='test',budget=10,question_id=q['question_id'],question_type=q['type'],
        **entity_metrics([], [m for m in mappings if m['question_id']==q['question_id']])) for q in questions]
    assert records[0]['expected_recall']==0 and records[1]['expected_recall'] is None
    summary=next(r for r in summarize(records) if r['question_subset']=='A')
    assert summary['questions']==2 and summary['expected_recall_evaluable']==1
    assert summary['expected_recall_excluded_questions'][0]['question_id']=='unresolved'
    assert summary['expected_recall_excluded_questions'][0]['excluded_targets'][0]['target']=='Unknown'

def test_real_type_a_denominator_keeps_all_csv_questions():
    from tkh_abstraction.io import read_csv,read_json
    from tkh_abstraction.evaluation.retrieval_v2 import question_evaluability
    questions=read_csv('data/data/questions.csv')
    inventory=question_evaluability(questions,read_json('artifacts/evaluation/target_mapping_2025.json'))
    type_a=[r for r in inventory if r['question_type']=='A']
    assert len(type_a)==14
    assert sum(r['expected_recall_evaluability']['status']=='evaluable' for r in type_a)==12
    excluded={r['question_id']:r['expected_recall_evaluability'] for r in type_a if r['expected_recall_evaluability']['status']=='not_evaluable'}
    assert set(excluded)=={'Q5','Q11'}
    assert all(r['total_expected_targets']==2 and r['mapped_expected_targets']==0 for r in excluded.values())

def node(i,text,typ='method',year=2020,articles=None):
    return dict(id=i,surface_form=text,type=typ,first_seen_year=year,provenance={'articles':articles or []})

def hierarchy(nodes):
    return dict(levels=[[dict(id='root',parent_id=None,member_ids=[n['id'] for n in nodes])],
        [dict(id='leaf-'+n['id'],parent_id='root',member_ids=[n['id']]) for n in nodes]])

def test_identifier_boundaries():
    nodes=[node('a','DeepH-E3'),node('b','DeepH'),node('c','Model++'),node('d','Model-v2')]
    assert resolve_target('DeepH',nodes)['node_ids']==['b']
    assert resolve_target('Model',nodes)['status']=='unmapped'
    assert normalize('E(3)')!=normalize('E3')
    assert normalize('X–v2++')=='x-v2++'

def test_ambiguous_alias_is_not_truth():
    assert resolve_target('ACE',[node('a','ACE with message passing'),node('b','ACE potential')])['status']=='ambiguous'
    assert resolve_target('ModelX',[node('a','Future study using ModelX','future_topic')])['status']=='unmapped'

def test_dataset_routing_and_methods():
    assert 'dataset' in route_question('What datasets and benchmarks evaluate potentials?')['types']
    assert 'method' in route_question('Which methods predict forces?')['types']
    nodes=[node('a','Test dataset','dataset'),node('b','Test model')]; X=np.eye(2)
    types=route_question('What datasets and benchmarks exist?')['types']
    assert flat_search(X[0],nodes,X,types)['returned_ids']==['a']
    assert SearchIndex(hierarchy(nodes),nodes,X,types).search(X[0],25)['returned_ids']==['a']

def test_exact_cited_entity_is_mappable_and_routable():
    nodes=[node('c','ScientificModel-v2','cited_work'),node('d','Paper mentioning ScientificModel-v2','cited_work')]
    assert resolve_target('ScientificModel-v2',nodes)['node_ids']==['c']
    assert resolve_target('ScientificModel',nodes)['status']=='unmapped'
    assert 'cited_work' in route_question('Which methods are used?')['types']
    assert 'cited_work' in route_question('What datasets are used?')['types']

def test_fp_categories_have_distinct_meaning():
    nodes=[node('a','Expected'),node('b','Valid'),node('c','Wrong'),node('d','Old')]
    gt={'q':dict(type='A',expected_methods=['Expected'],fp_categories=dict(valid_but_unlisted=['Valid'],wrong_domain=['Wrong'],outdated_as_main=['Old']))}
    m=entity_metrics(['b','c','d'],map_benchmark(gt,nodes,2025))
    assert m['expected_recall']==0 and m['strict_precision']==0
    assert m['extended_precision']==m['known_wrong_rate']==m['outdated_main_rate']==1/3
    absent=entity_metrics(['b'],map_benchmark(gt,nodes,2025),False)
    assert absent['extended_precision'] is absent['known_wrong_rate'] is absent['outdated_main_rate'] is None

def test_all_prototypes_charged_and_same_leaf_scorer():
    nodes=[node(str(i),str(i)) for i in range(5)]; X=np.eye(5); index=SearchIndex(hierarchy(nodes),nodes,X,{'method'},4)
    result=index.search(X[2],10); flat=flat_search(X[2],nodes,X,{'method'})
    assert result['score_comparisons']==10
    assert sum(t['comparisons'] for t in result['trace'])==10
    assert len({t['object_id'] for t in result['trace']})==len(result['trace'])
    assert result['trace'][0]['comparisons']==5
    assert result['leaf_scores']==flat['leaf_scores']
    assert result['returned_ids']==flat['returned_ids']
    assert index.search(X[2],4)['score_comparisons']==0

def test_prototypes_can_expose_diluted_branch_without_answer_key():
    nodes=[node(str(i),str(i)) for i in range(12)]
    X=np.array([[1.,0.],[1.,0.]]+[[-.9,np.sqrt(.19)]]*4+[[.1,np.sqrt(.99)]]*6)
    roots=[dict(id='a',parent_id=None,member_ids=[str(i) for i in range(6)]),dict(id='b',parent_id=None,member_ids=[str(i) for i in range(6,12)])]
    leaves=[dict(id='leaf-'+str(i),parent_id='a' if i<6 else 'b',member_ids=[str(i)]) for i in range(12)]
    index=SearchIndex(dict(levels=[roots,leaves]),nodes,X,{'method'},4)
    new=index.search(np.array([1.,0.]),16,beam=1)
    old=index.search(np.array([1.,0.]),16,method='legacy_typed_fixed_beam',beam=1)
    assert '0' in new['returned_ids'] and '0' not in old['returned_ids']
    assert new['score_comparisons']<=16 and new['root_layer_complete']
    assert new==index.search(np.array([1.,0.]),16,beam=1)

def test_optional_nli_cache_uses_exact_inputs(tmp_path):
    from tkh_abstraction.entailment import NLI
    n=NLI.__new__(NLI); n.cache={}; n.cache_path=tmp_path/'cache.json'; calls=[]
    def predict(pairs,batch_size):
        calls.extend(pairs); return [dict(entailment=.75)]*len(pairs)
    n._predict_uncached=predict
    assert n.predict([('premise','hypothesis')])==n.predict([('premise','hypothesis')])
    assert len(calls)==1
    n.predict([('other premise','hypothesis')]); assert len(calls)==2

def test_claim_paths_and_future_exclusion():
    nodes=[node('m','Method',articles=[1]),node('a','Paper','article',articles=[1]),node('c','Claim','claim',articles=[1]),node('future','Future claim','claim',2026,[1])]
    data=dict(nodes=nodes,hyperedges=[]); snap=snapshot(data,2025); X=np.eye(3)
    result=retrieve_claims(X[2],['m'],snap,X)
    assert result['returned_ids']==['c'] and result['score_comparisons']==1
    assert result['evidence'][0]['paths'][0]['path']=='entity_article_claim'
    reused=retrieve_claims(X[2],['c'],snap,X,known_scores={'c':1.})
    assert reused['score_comparisons']==0 and reused['reused_entity_scores']==['c']

def test_source_resolution_restricts_article():
    nodes=[node('a','Exact paper','article',articles=[1]),node('b','Other paper','article',articles=[2])]
    assert SourceIndex(dict(nodes=nodes,hyperedges=[])).resolve({'title':'Exact paper'})['article_ids']==[1]
    nodes[0]['provenance']['doi']=None
    assert SourceIndex(dict(nodes=nodes,hyperedges=[])).resolve({'title':'Exact paper','doi':None})['article_ids']==[1]

def test_prototypes_normalize_and_mixed_routing():
    nodes=[node('a','a'),node('b','b')]; X=np.array([[3.,0.],[0.,2.]])
    result=SearchIndex(hierarchy(nodes),nodes,X,{'method'},2).search(X[0],10)
    assert result['leaf_scores']==flat_search(X[0],nodes,X,{'method'})['leaf_scores']
    assert {'method','dataset'}<=set(route_question('Which methods and datasets are available?')['types'])
    assert result['root_layer_complete']
    assert not SearchIndex(hierarchy(nodes),nodes,X,{'method'},2).search(X[0],2)['root_layer_complete']

def test_alignment_requires_source_and_preserves_ambiguity():
    cfg=read_config('configs/evaluation_v2.yaml')
    c=dict(node_id='c',surface_form='Claim one',source_supported=True,cosine=.9,lexical_overlap=.8,numeric_support=True,nli=dict(entailment=.95,truncated=False))
    assert decide_alignment([c],cfg)==('confident_proxy',['c'])
    assert decide_alignment([dict(c,source_supported=False)],cfg)==('unmapped',[])
    assert decide_alignment([c,dict(c,node_id='d',surface_form='Distinct claim',cosine=.89)],cfg)==('ambiguous',[])

def test_generator_input_cannot_contain_heldout_features():
    nodes=[node(str(i),'Shared scientific term unique'+str(i)) for i in range(20)]
    gen,held=evidence_split(nodes); snap=dict(snapshot=2025,nodes=nodes,hyperedges=[])
    packet=generation_packet(gen,snap)
    assert {n['id'] for n in gen}.isdisjoint({n['id'] for n in held}) and len(held)==6
    assert not {n['surface_form'] for n in held}&{n['surface_form'] for n in packet['generation_members']}
    original=generate_from_packet(packet)
    for n in held: n['surface_form']='heldout adversarial contamination'
    assert generate_from_packet(generation_packet(gen,snap))==original

def test_alignment_never_accepts_better_unrelated_article():
    from tkh_abstraction.evaluation.claim_alignment import align
    cfg=read_config('configs/evaluation_v2.yaml')
    nodes=[node('a','Source paper','article',articles=[1]),node('b','Other paper','article',articles=[2]),node('c','Evidence','claim',articles=[1]),node('d','Perfect but wrong source','claim',articles=[2])]
    class E:
        metadata={}
        def encode(self,texts,split): return np.array([[1.,0.]]*len(texts))
    class N:
        metadata={}
        def predict(self,pairs): return [dict(entailment=.99,truncated=False)]*len(pairs)
    gt={'Q':dict(type='B',sources={'s':dict(title='Source paper')},required_claims=[dict(id='c1',text='Evidence',sources=['s'])])}
    result=align(gt,dict(snapshot=2025,nodes=nodes,hyperedges=[]),E(),N(),cfg)
    assert result['items'][0]['node_ids']==['c']
    assert [c['node_id'] for c in result['items'][0]['candidates']]==['c']

def test_label_overlay_temporal_and_holdout_integration(tmp_path):
    import pytest
    from tkh_abstraction.labels_v2 import label_overlay,evaluate_labels
    cfg=read_config('configs/evaluation_v2.yaml'); nodes=[node(str(i),'Common scientific evidence '+str(i)) for i in range(12)]
    h=hierarchy(nodes)
    for k,level in enumerate(h['levels']):
        for c in level: c.update(level=k,label='Common evidence',gloss='Common evidence.')
    snap=dict(snapshot=2025,nodes=nodes,hyperedges=[]); overlay,logs=label_overlay(h,snap,cfg)
    class N:
        metadata={}
        def predict(self,pairs): return [dict(entailment=.7,neutral=.2,contradiction=.1,prediction='entailment',truncated=False,input_tokens=10)]*len(pairs)
    result=evaluate_labels(logs,N(),cfg,tmp_path)
    assert result['summaries']['improved']['0']['evaluable']==1
    assert result['human_audit']['status']=='missing_data' and (tmp_path/'audit_key.json').exists()
    nodes[-1]['first_seen_year']=2026
    with pytest.raises(ValueError,match='Future label evidence'): label_overlay(h,snap,cfg)
