import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from tkh_abstraction_v2.question_understanding import ExtractiveParser, StrictJSONParser
from tkh_abstraction_v2.candidate_cards import build_cards, eligible
from tkh_abstraction_v2.snapshots import snapshot
from tkh_abstraction_v2.models import pack
from tkh_abstraction_v2.ranking import rank, rrf, replace_prefix, requirement_score


@pytest.fixture
def cfg():
    return json.loads((ROOT / 'config.json').read_text())


def node(ident, typ, name, year=2020):
    return dict(id=ident, type=typ, surface_form=name, first_seen_year=year, provenance={'articles': [1]})


def graph():
    return dict(nodes=[node('m', 'method', 'ExampleEngine'), node('c', 'claim', 'ExampleEngine supports fast estimation.'),
        node('t', 'task', 'estimation'), node('f', 'claim', 'Future evidence', 2027)],
        hyperedges=[dict(id='e', relation_type='claims', members=['m', 'c'], year=2020, provenance={'article_year': 2020}),
                    dict(id='future', relation_type='claims', members=['m', 'f'], year=2027, provenance={'article_year': 2027})])


def test_parser_extracts_negation_and_date():
    q = 'Which methods by 2026 are suited for estimation with low memory and high accuracy without labels?'
    d = ExtractiveParser().parse(q)
    assert d['answer_type'] == 'method'
    assert d['requirements'] == ['low memory', 'high accuracy']
    assert d['exclusions'] == ['labels']
    assert d['temporal_condition'] == 'by 2026'
    for item in d['components']:
        assert q[item['start']:item['end']] == item['text']


def test_local_parser_never_fakes_or_accepts_facts():
    parser = StrictJSONParser(lambda _: '{"answer_type": "method"}')
    with pytest.raises(ValueError):
        parser.parse('Which methods estimate values?')


def test_snapshot_and_evidence_ids(cfg):
    snap = snapshot(graph(), 2026)
    cards = build_cards(snap, cfg)
    nodes = {n['id'] for n in snap['nodes']}
    edges = {e['id'] for e in snap['hyperedges']}
    assert 'f' not in cards
    for card in cards.values():
        for rows in card['fields'].values():
            for item in rows:
                assert item['node_id'] in nodes
                assert set(item['hyperedge_ids']) <= edges
                assert item['first_seen_year'] <= 2026
    assert cards['m']['fields']['direct_claims'][0]['node_id'] == 'c'


def test_high_arity_is_not_fact(cfg):
    g = graph()
    g['hyperedges'] = [dict(id='big', relation_type='addresses', members=['m', 'c', 't', 'f'], year=2028, provenance={})]
    cards = build_cards(snapshot(g, 2028), cfg)
    assert not cards['m']['fields']['task_problem']
    assert cards['m']['discarded'][0]['reason'] == 'high_arity_comembership'


def test_mapping_deterministic(cfg):
    assert eligible('method', 'method', cfg)
    assert eligible('component', 'method', cfg)
    assert not eligible('claim', 'method', cfg)
    assert not eligible('author', None, cfg)
    assert eligible('task', None, cfg)
    assert build_cards(snapshot(graph(), 2026), cfg) == build_cards(snapshot(graph(), 2026), cfg)


def test_frozen_prefix():
    rows = rank(['a', 'b', 'c'], [3, 2, 1])
    assert [r['node_id'] for r in replace_prefix(rows, {'a': 1, 'b': 4}, 2)] == ['b', 'a', 'c']
    with pytest.raises(ValueError):
        replace_prefix(rows, {'a': 1, 'c': 4}, 2)


class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()
    def decode(self, ids, skip_special_tokens=True):
        return ' '.join(ids)
    def num_special_tokens_to_add(self, pair=True):
        return 3
    def __call__(self, a, b, add_special_tokens=True):
        return {'input_ids': a.split() + b.split() + ['[X]'] * 3}


def test_identity_survives_token_packing():
    blocks = [dict(field='identity', text='NAME: ExampleEngine TYPE: method'), dict(field='direct_claims', text='long ' * 40)]
    q, doc, audit = pack(TinyTokenizer(), 'query ' * 30, blocks, 20)
    assert doc == blocks[0]['text']
    assert audit['included'][0] == blocks[0]
    assert audit['omitted'] and audit['token_count'] <= 20 and audit['query_truncated']


def test_requirement_and_tie_determinism():
    import numpy as np
    assert rank(['b', 'a'], [1, 1])[0]['node_id'] == 'a'
    assert rrf(rank(['a', 'b'], [2, 1]), rank(['a', 'b'], [1, 2]))[0]['node_id'] == 'a'
    score, _, _ = requirement_score(.9, [.8, .2], [.5])
    assert score == pytest.approx(.18)
    assert requirement_score(.9, np.array([.8, .2]), np.array([.5]))[0] == pytest.approx(.18)
    assert requirement_score(.9, np.array([]), np.array([]))[0] == .9
    json.dumps(requirement_score(.9, np.array([.8, .2], dtype=np.float32), np.array([.5], dtype=np.float32)))


def test_runtime_guard_blocks_protected_writes_and_gold_reads():
    code = '''
from pathlib import Path
from tkh_abstraction_v2.isolation import install
root=Path.cwd()
install(root)
for path,mode in [(root.parent/'clean_tshc'/'DO_NOT_CREATE','w'),(root/'data'/'ground_truth.json','r'),(root/'data'/'target_annotations.json','r')]:
    try:
        path.open(mode)
    except PermissionError:
        pass
    else:
        raise AssertionError('guard did not block')
try:
    __import__('tkh_abstraction_v2.evaluation')
except (PermissionError, ModuleNotFoundError):
    pass
else:
    raise AssertionError('evaluation import allowed')
'''
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not (ROOT.parent / 'clean_tshc' / 'DO_NOT_CREATE').exists()


def test_prediction_source_has_no_gold_or_evaluator_import():
    source = (ROOT / 'scripts/predict.py').read_text()
    for forbidden in ('ground_truth.json', 'target_annotations.json', 'claim_annotations.json', 'import evaluation', 'import benchmark'):
        assert forbidden not in source
    parser = (ROOT / 'src/tkh_abstraction_v2/question_understanding.py').read_text()
    assert 'open(' not in parser and 'read(' not in parser


def test_nli_is_not_used_as_relevance(cfg):
    assert cfg['relevance_model'] == 'cross-encoder/ms-marco-MiniLM-L6-v2'
    assert cfg['nli'] == 'not_run_optional'
    source = (ROOT / 'src/tkh_abstraction_v2/models.py').read_text()
    assert "self.model.config.num_labels != 1" in source


def test_frozen_artifacts_and_evaluation_order_when_present():
    freeze = ROOT / 'outputs/audit/freeze.json'
    if not freeze.exists():
        pytest.skip('Experiment has not frozen predictions yet')
    from tkh_abstraction_v2.io import read, sha
    for relative, digest in read(freeze)['files'].items():
        assert sha(ROOT / relative) == digest, relative
    events = [json.loads(line) for line in (ROOT / 'outputs/audit/events.jsonl').read_text().splitlines()]
    names = [e['event'] for e in events]
    if 'gold_loaded' in names:
        assert names.index('predictions_frozen') < names.index('evaluation_import') < names.index('gold_loaded')


def test_real_cards_visibility_and_provenance_when_present(cfg):
    path = ROOT / 'outputs/candidate_cards/cards.json'
    if not path.exists():
        pytest.skip('Cards not built yet')
    cards = json.loads(path.read_text(encoding='utf8'))
    snap = snapshot(json.loads((ROOT / 'data/tkh_collection10.json').read_text(encoding='utf8')), cfg['snapshot'])
    nodes = {n['id']: n for n in snap['nodes']}
    edges = {e['id']: e for e in snap['hyperedges']}
    for ident, card in cards.items():
        assert card['blocks'][0]['node_id'] == ident
        assert card['name'] == nodes[ident]['surface_form']
        for field, rows in card['fields'].items():
            assert len(rows) <= cfg['field_caps'][field]
            for row in rows:
                assert row['text'] == nodes[row['node_id']]['surface_form']
                assert row['first_seen_year'] <= cfg['snapshot']
                for eid in row['hyperedge_ids']:
                    assert {ident, row['node_id']} <= set(edges[eid]['members'])
                    assert len(edges[eid]['members']) <= cfg['direct_arity_cap']


def test_real_rerankings_only_permute_prefix_when_present(cfg):
    path = ROOT / 'outputs/retrieval/predictions.json'
    if not path.exists():
        pytest.skip('Rankings not frozen yet')
    predictions = json.loads(path.read_text(encoding='utf8'))
    for question in predictions.values():
        ranks = question['rankings']
        for source, target in [('A3', 'A4'), ('V2-B', 'V2-C'), ('V2-B', 'V2-D')]:
            if target not in ranks:
                continue
            a = [r['node_id'] for r in ranks[source]]
            b = [r['node_id'] for r in ranks[target]]
            k = cfg['rerank_k']
            assert len(b) == len(set(b))
            assert set(a[:k]) == set(b[:k])
            assert a[k:] == b[k:]


def test_actual_model_fixed_input_determinism_when_available():
    availability = ROOT / 'outputs/audit/model_availability.json'
    if not availability.exists():
        pytest.skip('Models not prepared yet')
    models = json.loads(availability.read_text())
    if any(models.get(role, {}).get('status') != 'available' for role in ('dense', 'relevance')):
        pytest.skip('Optional model unavailable')
    import numpy as np
    import torch
    from tkh_abstraction_v2.models import Encoder, RelevanceCrossEncoder
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    encoder = Encoder(ROOT / models['dense']['path'], ROOT / '.cache/test_embeddings')
    text = ['An example method estimates a value from measurements.']
    a = encoder.model.encode(text, normalize_embeddings=True)
    b = encoder.model.encode(text, normalize_embeddings=True)
    assert np.array_equal(a, b)
    ce = RelevanceCrossEncoder(ROOT / models['relevance']['path'])
    card = dict(node_id='synthetic', blocks=[dict(field='identity', text='NAME: ExampleEngine TYPE: method'),
        dict(field='direct_claims', text='The method estimates values from measurements.')])
    score_a = ce.score('Which methods estimate values?', [card])['synthetic']['relevance_score']
    score_b = ce.score('Which methods estimate values?', [card])['synthetic']['relevance_score']
    assert score_a == score_b


def test_bm25_independent_of_python_hash_seed():
    code = '''from tkh_abstraction_v2.ranking import BM25
import json
b=BM25(['alpha beta beta gamma delta','gamma delta epsilon alpha','beta epsilon delta'])
print(json.dumps(b.score('alpha beta gamma delta epsilon').tolist()))
'''
    outputs = []
    for seed in ('1', '42', '13579'):
        env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONHASHSEED=seed, PYTHONDONTWRITEBYTECODE='1')
        outputs.append(subprocess.check_output([sys.executable, '-B', '-c', code], env=env))
    assert outputs[0] == outputs[1] == outputs[2]


def test_composite_rank_requires_every_component():
    if not (ROOT / 'outputs/audit/freeze.json').exists():
        pytest.skip('Evaluation modules are imported only after prediction freeze')
    from tkh_abstraction_v2.evaluation import complete_rank
    assert complete_rank([['a', 'alias_a'], ['b']], {'a': 8, 'alias_a': 2, 'b': 7}) == 7
    assert complete_rank([['a'], ['b']], {'a': 1}) is None
    assert complete_rank([], {'a': 1}) is None


def test_evaluation_freeze_refuses_tampering():
    import importlib.util
    from tkh_abstraction_v2.io import write, sha, event
    spec = importlib.util.spec_from_file_location('freeze_driver', ROOT / 'scripts/evaluate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    folder = ROOT / '.cache/test_freeze'
    prediction = folder / 'outputs/retrieval/predictions.json'
    write(prediction, {'synthetic': []})
    digest = sha(prediction)
    write(folder / 'outputs/audit/freeze.json', dict(predictions_sha256=digest, files={'outputs/retrieval/predictions.json': digest}))
    event(folder / 'outputs/audit/events.jsonl', 'predictions_frozen', sha256=digest)
    assert module.verify_freeze(folder)['predictions_sha256'] == digest
    write(prediction, {'tampered': []})
    with pytest.raises(ValueError, match='Prediction hash'):
        module.verify_freeze(folder)


def test_final_metric_denominators_and_partial_no_credit_when_present():
    file = ROOT / 'outputs/metrics.json'
    if not file.exists():
        pytest.skip('Evaluation has not run yet')
    metrics = json.loads(file.read_text(encoding='utf8'))
    questions = json.loads((ROOT / 'outputs/per_question_evaluation.json').read_text(encoding='utf8'))
    targets = [t for q in questions.values() for t in q['targets']]
    denominator = metrics['denominators']['target_occurrences']
    assert denominator == len(targets)
    for target in targets:
        if target['status'] in ('PARTIAL', 'ABSENT', 'COMPOSITE'):
            assert all(r is None for r in target['direct_ranks'].values())
        if target['status'] in ('PARTIAL', 'ABSENT'):
            assert all(r is None for r in target['v1_compatible_bundle_ranks'].values())
    for system, values in metrics['systems'].items():
        for k in (1, 5, 10, 25, 50, 100):
            hits = sum(t['direct_ranks'][system] is not None and t['direct_ranks'][system] <= k for t in targets)
            assert hits == values['direct_hits'][str(k)]
            assert values['recall'][str(k)] == hits / denominator
    for before, after in [('V2-B', 'V2-C'), ('V2-B', 'V2-D'), ('A3', 'A4')]:
        if after in metrics['systems']:
            assert metrics['systems'][before]['recall']['100'] == metrics['systems'][after]['recall']['100']
