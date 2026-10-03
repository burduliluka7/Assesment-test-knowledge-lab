"""Post-freeze artifact checks, including the isolated evaluation-only oracle."""
import hashlib
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def frozen():
    path=ROOT/'outputs/audit/freeze.json'
    if not path.exists():pytest.skip('Run prediction freeze first')
    return json.loads(path.read_text(encoding='utf8'))


def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf8'))


def test_new_prediction_integrity_and_phase_order(frozen):
    pred=ROOT/'outputs/retrieval/predictions.json'
    assert hashlib.sha256(pred.read_bytes()).hexdigest()==frozen['predictions_sha256']
    events=[json.loads(s) for s in (ROOT/'outputs/audit/events.jsonl').read_text().splitlines()]
    names=[e['event'] for e in events]
    if 'gold_loaded' in names:assert names.index('predictions_frozen')<names.index('evaluation_import')<names.index('gold_loaded')
    imports=read('outputs/audit/prediction_accesses.json')['local_imports']
    assert not any('evaluation' in name or 'strict_targets' in name for name in imports)


def test_new_rerank_and_verification_pools(frozen):
    predictions=read('outputs/retrieval/predictions.json')
    for q,row in predictions.items():
        ranks=row['rankings']; a=[r['entity_id'] for r in ranks['A5']]
        assert {r['entity_id'] for r in ranks['A4']}==set(a)
        for name,k in [('A6',500),('A6_100',100),('A6_200',200)]:
            if name not in ranks:continue
            b=[r['entity_id'] for r in ranks[name]]
            assert set(a[:k])==set(b[:k]) and a[k:]==b[k:]
        if 'A7' in ranks:
            b=[r['entity_id'] for r in ranks['A6']]; c=[r['entity_id'] for r in ranks['A7']]
            assert set(b[:20])==set(c[:20]) and b[20:]==c[20:]


def test_evidence_index_and_selected_paths_are_visible(frozen):
    for path in (ROOT/'outputs/entity_index').glob('*.json'):
        index=json.loads(path.read_text(encoding='utf8')); cutoff=index['snapshot']
        for entity in index['entities'].values():
            assert entity['first_seen_year']<=cutoff
            for rows in entity['evidence'].values():
                for row in rows:
                    assert row['evidence_id'] in index['items']
                    assert index['items'][row['evidence_id']]['first_seen_year']<=cutoff
                    for p in row['paths']:
                        assert all(e['year']<=cutoff and e['provenance'].get('article_year',cutoff)<=cutoff for e in p)
    for path in (ROOT/'outputs/retrieval/selected_evidence').glob('*.json'):
        for card in json.loads(path.read_text(encoding='utf8')).values():
            assert card['blocks'][0]['field']=='identity'
            assert len({r['text'].casefold() for r in card['selected']})==len(card['selected'])


def test_all_cross_encoder_identities_and_token_limits(frozen):
    for path in (ROOT/'outputs/retrieval/cross_encoder').glob('*.json'):
        for ident,row in json.loads(path.read_text(encoding='utf8')).items():
            assert row['included'][0]['field']=='identity'
            assert row['included'][0]['node_id']==ident
            assert row['token_count']<=512


def test_new_metric_denominators_and_positive_control(frozen):
    path=ROOT/'outputs/metrics.json'
    if not path.exists():pytest.skip('Run evaluation after freeze')
    metrics=read('outputs/metrics.json'); qs=read('outputs/per_question_evaluation.json')
    targets=[t for q in qs.values() for t in q['targets']]
    den=metrics['denominators']['target_occurrences']; mapped=metrics['denominators']['exact_alias_occurrences']
    assert den==len(targets)
    for system,m in metrics['systems'].items():
        for k in (1,5,10,25,50,100):
            hits=sum(t['direct_ranks'][system] is not None and t['direct_ranks'][system]<=k for t in targets)
            assert m['direct_hits'][str(k)]==hits and m['recall'][str(k)]==hits/den
        assert all(0<=h<=mapped for h in m['candidate_hits'].values())
    for t in targets:
        if t['status'] not in ('EXACT','ALIAS'):assert all(r is None for r in t['direct_ranks'].values())
    ctrl=read('outputs/evaluation_only/positive_controls.json')
    assert ctrl['passed'] and sum(r['positive_hits'] for r in ctrl['records'])==mapped
    assert all(r['empty_ranking_hits']==0 for r in ctrl['records'])


def test_verification_missing_evidence_never_contradicted(frozen):
    for path in (ROOT/'outputs/retrieval/requirement_verification').glob('*.json'):
        for row in json.loads(path.read_text(encoding='utf8')).values():
            for comp in row['components']:
                if comp['evidence_id'] is None:
                    assert comp['status']=='NOT_SUPPORTED_BY_AVAILABLE_EVIDENCE' and comp['probabilities'] is None


def test_full_cached_replay_is_identical_when_available(frozen):
    replay=ROOT/'outputs/reproduction/retrieval/predictions.json'
    if not replay.exists():pytest.skip('Reproducibility replay not run yet')
    assert hashlib.sha256(replay.read_bytes()).hexdigest()==frozen['predictions_sha256']
