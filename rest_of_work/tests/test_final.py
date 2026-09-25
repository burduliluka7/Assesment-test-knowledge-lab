"""Final integration invariants and exact frozen-reference regressions."""
from pathlib import Path
from types import SimpleNamespace
import inspect
import numpy as np
import pytest
from tkh_abstraction.io import read_json, digest
from tkh_abstraction.final_search import HierarchySearch, flat_scan, final_decomposition, FineScorer
from tkh_abstraction.final_evaluation import claim_metrics, evaluate_result, evidence_export


class ToyScorer:
    def __init__(self):
        self.pools = {x:dict(name=x, node_ids=[x], types=['method']) for x in ['a','b','c']}
        self.calls = []
    def work(self, x): return {'a':2,'b':3,'c':1}[x]
    def score(self, x):
        self.calls.append(x)
        return dict(score={'a':.9,'b':.2,'c':.6}[x], score_components={})


@pytest.fixture
def toy():
    def unit(i, p, members): return dict(id=i, parent_id=p, member_ids=members)
    h = dict(levels=[[unit('root',None,['a','b','c'])],
                     [unit('ab','root',['a','b']),unit('single','root',['c'])],
                     [unit('aa','ab',['a']),unit('bb','ab',['b']),unit('cc','single',['c'])]])
    nodes = [dict(id=x,type='method') for x in ['a','b','c']]
    X = np.asarray([[1.,0.],[0.,1.],[.8,.2]])
    groups = [dict(id=x,node_ids=[x],types=['method']) for x in ['a','b','c']]
    return HierarchySearch(h,nodes,X,groups), X


def test_final_qualifiers_diagnostic_only():
    d = final_decomposition('Which neural methods are suitable for predicting signals?')
    assert d.source_spans['answer_qualifiers']
    assert not any(c['kind']=='answer_qualifier' for c in d.components)


def test_no_secondary_scorer_dependency():
    import tkh_abstraction.final_search as f
    source = inspect.getsource(f)
    for forbidden in ['RequirementNLIReranker','ControlledLocalNLI','rrf_fuse','expected_methods','method_claims','gold']:
        assert forbidden not in source
    assert 'allowed_groups' in inspect.signature(f.HierarchySearch.search).parameters


def test_root_budget_status_and_no_scoring(toy):
    ix,_=toy; s=ToyScorer(); r=ix.search(np.array([1.,0.]),ix.root_cost-1,s)
    assert r['status']=='INSUFFICIENT_ROOT_BUDGET' and not r['ranked'] and not s.calls
    assert r['cost']['total_comparisons']==0


def test_starts_all_roots_before_expanding(toy):
    ix,_=toy; r=ix.search(np.array([1.,0.]),None,ToyScorer())
    assert [t['object_id'] for t in r['trace'][:len(ix.roots)]]==ix.roots
    assert all(t['level']==0 for t in r['trace'][:len(ix.roots)])


def test_descendants_only_and_singleton_intermediate_expansion(toy):
    ix,_=toy; r=ix.search(np.array([1.,0.]),None,ToyScorer()); expanded=set()
    for t in r['trace']:
        if t['action']=='expand': expanded.add(t['object_id'])
        if t['action']=='score_leaf': assert t['parent_id'] in expanded
    assert 'single' in expanded and set(r['reached_node_ids'])=={'a','b','c'}


@pytest.mark.parametrize('budget',[4,6,10,20,None])
def test_determinism_and_exact_charges(toy,budget):
    ix,_=toy; s=ToyScorer(); a=ix.search(np.array([1.,0.]),budget,s); b=ix.search(np.array([1.,0.]),budget,ToyScorer())
    assert a==b
    assert a['cost']['total_comparisons']==sum(t['comparisons'] for t in a['trace'])
    assert budget is None or a['cost']['total_comparisons']<=budget
    assert set(s.calls)=={r['entity_id'] for r in a['ranked']}
    assert set(s.calls)<=set(a['reached_node_ids'])


def test_h2_intersection_restricts_fine_calls(toy):
    ix,_=toy;s=ToyScorer();r=ix.search(np.array([1.,0.]),None,s,{'b'})
    assert [x['entity_id'] for x in r['ranked']]==['b'] and set(s.calls)=={'b'}


def test_mixed_group_types_do_not_admit_unrelated_nodes():
    h=dict(levels=[[dict(id='r',parent_id=None,member_ids=['a','b','z'])],
                   [dict(id=x,parent_id='r',member_ids=[x]) for x in ['a','b','z']]])
    nodes=[dict(id='a',type='method'),dict(id='b',type='task'),dict(id='z',type='task')]
    groups=[dict(id='a',node_ids=['a','b'],types=['method','task'])]
    ix=HierarchySearch(h,nodes,np.asarray([[1.,0.],[1.,0.],[0.,1.]]),groups)
    assert ix.index.eligible=={'a','b'} and 'z' not in ix.index.units


def test_equal_work_flat_is_fixed_prefix(toy):
    _,X=toy;nv=dict(zip(['a','b','c'],X));s=ToyScorer()
    a=flat_scan(s,np.array([1.,0.]),nv,6);b=flat_scan(ToyScorer(),np.array([0.,1.]),nv,6)
    assert [t['entity_id'] for t in a['trace']]==[t['entity_id'] for t in b['trace']]
    assert a['cost']['total_comparisons']==sum(t['comparisons'] for t in a['trace'])<=6


def test_ground_truth_changes_evaluation_only(toy):
    ix,_=toy
    def run(expected):
        result=ix.search(np.array([1.,0.]),None,ToyScorer())
        # Resolution is evaluation-only and is not passed into the search.
        snap=dict(nodes=[dict(id=x,type='method',surface_form=x,provenance={}) for x in ['a','b','c']],hyperedges=[])
        from tkh_abstraction.target_mapping import TargetResolver
        from tkh_abstraction.diffusion_pipeline import evaluate_ranking
        resolver=TargetResolver(snap);res=[resolver.resolve(x) for x in expected]
        metric=evaluate_ranking(result['ranked'],res,dict(diffusion_top_ks=[1,5,10,20,50,100,200]))
        return result,metric
    a,ma=run(['a']);b,mb=run(['missing'])
    assert a==b and ma!=mb


def test_claim_selection_has_no_expected_claim_argument():
    assert list(inspect.signature(evidence_export).parameters)==['ranked','scorer','nodes','question_vector']


def test_unevaluable_claim_is_null_and_source_is_separate():
    snap=dict(snapshot=2025,nodes=[dict(id='a',type='method',surface_form='Alpha',provenance={})],hyperedges=[])
    gt={'Q':dict(type='A',expected_methods=['Alpha'],method_claims={'Alpha':['Alpha predicts X.']})}
    m=claim_metrics('Q',gt,snap,[],dict(items=[]))
    assert m['expected_claims_count']==1 and m['automatically_alignable_claims_count']==0
    assert m['strict_claim_recall'] is None and m['claim_evaluation_status']=='NOT_EVALUABLE_ALIGNMENT'
    assert m['source_hit'] is None


def test_final_fine_full_regression():
    root=Path('artifacts/evaluation/final')
    rows=read_json(root/'final_fine_rankings.json');frozen=read_json(root/'final_fine_regression.json')
    assert len(rows)==28
    by={(r['cutoff'],r['question_id']):r for r in frozen}
    history={(r['cutoff'],r['question_id']):r for r in read_json('artifacts/evaluation/requirement_evidence_selection.json')}
    for row in rows:
        expected=by[row['cutoff'],row['question_id']]
        assert digest([(r['entity_id'],r['score']) for r in row['returned']])==expected['ranking_hash']
        assert not any(c['kind']=='answer_qualifier' for c in row['decomposition']['components'])
        ranks={r['entity_id']:i+1 for i,r in enumerate(row['returned'])}
        scores={r['entity_id']:r['score'] for r in row['returned']}
        for candidate in history[row['cutoff'],row['question_id']]['candidates']:
            ident=candidate['entity_id']
            assert ranks[ident]==candidate['ranks']['without_qualifiers']
            if candidate['h2_rank']<=100: assert scores[ident]==candidate['without_qualifiers_score']
    for year,denom in [(2025,47),(2026,50)]:
        rr=[r for r in rows if r['cutoff']==year]
        assert sum(r['entity_metrics']['at_k']['10']['found'] for r in rr)==13
        assert sum(r['entity_metrics']['canonical_expected'] for r in rr)==denom


def test_frozen_hierarchy_bytes_laminarity_and_temporal_safety():
    from tkh_abstraction.hierarchy import validate_hierarchy
    from tkh_abstraction.snapshots import snapshot
    from tkh_abstraction.v2_pipeline import sha
    data=read_json('data/data/tkh_collection10.json')
    manifest=read_json('artifacts/evaluation/final/frozen_manifest.json')
    for path,h in manifest['files'].items():
        if '/hierarchy_' not in path: continue
        assert sha(path)==h
        hierarchy=read_json(path);snap=snapshot(data,hierarchy['snapshot'])
        validate_hierarchy(hierarchy,{n['id'] for n in snap['nodes']})
        assert [len(l) for l in hierarchy['levels'][:3]]==[12,40,120]


def test_shipped_search_counts_reachability_and_same_resolver():
    root=Path('artifacts/evaluation/final');run=read_json(root/'results.json')
    for group in run['results']:
        if not group.get('artifact'): continue
        for r in read_json(group['artifact']):
            assert r['cost']['total_comparisons']==sum(t['comparisons'] for t in r['trace'])
            if r['budget'] is not None: assert r['cost']['total_comparisons']<=r['budget']
            assert r['entity_metrics']['canonical_expected']>=0
            if group['system'].startswith('hierarchical'):
                reached={t['node_id'] for t in r['trace'] if t['action']=='score_leaf'}
                assert {t['entity_id'] for t in r['trace'] if t['action']=='score_candidate'}=={x['entity_id'] for x in r['returned']}
                for t in r['trace']:
                    if t['action']=='score_candidate': assert t['reached_via'] in reached
                assert r['cost']['fine_candidates_scored']==len(r['returned'])


def test_unrestricted_ceiling_matches_exhaustive():
    for y in [2025,2026]:
        flat=read_json(f'artifacts/evaluation/final/flat_exhaustive_none_{y}_all.json')
        hier=read_json(f'artifacts/evaluation/final/hierarchical_temporal_{y}_all.json')
        for a,b in zip(flat,hier): assert a['returned']==b['returned']


def test_frozen_old_sources_and_artifacts():
    from tkh_abstraction.final_pipeline import verify_frozen
    assert verify_frozen('artifacts/evaluation/final/frozen_manifest.json')>400
