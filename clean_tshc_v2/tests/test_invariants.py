import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from tkh_abstraction_v2.question_understanding import ExtractiveParser, StrictJSONParser
from tkh_abstraction_v2.snapshots import snapshot
from tkh_abstraction_v2.models import pack


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
    assert cfg['nli'] == 'local_pinned_verification_proxy'
    source = (ROOT / 'src/tkh_abstraction_v2/models.py').read_text()
    assert "self.model.config.num_labels != 1" in source


def test_frozen_artifacts_and_evaluation_order_when_present():
    freeze = ROOT / 'artifacts/prediction/audit/freeze.json'
    if not freeze.exists():
        pytest.skip('Experiment has not frozen predictions yet')
    from tkh_abstraction_v2.io import read, sha
    for relative, digest in read(freeze)['files'].items():
        assert sha(ROOT / relative) == digest, relative
    events = [json.loads(line) for line in (ROOT / 'artifacts/prediction/audit/events.jsonl').read_text().splitlines()]
    names = [e['event'] for e in events]
    if 'gold_loaded' in names:
        assert names.index('predictions_frozen') < names.index('evaluation_import') < names.index('gold_loaded')






def test_actual_model_fixed_input_determinism_when_available():
    availability = ROOT / 'model_manifest.json'
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
    if not (ROOT / 'artifacts/prediction/audit/freeze.json').exists():
        pytest.skip('Evaluation modules are imported only after prediction freeze')
    from tkh_abstraction_v2.entity_evaluation import rank_components as complete_rank
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
    write(folder / 'outputs/audit/freeze.json', dict(status='PREDICTIONS_FROZEN_BEFORE_EVALUATION', predictions_sha256=digest, files={'outputs/retrieval/predictions.json': digest}))
    event(folder / 'outputs/audit/events.jsonl', 'predictions_frozen', sha256=digest)
    assert module.verify_freeze(folder,folder/'outputs')['predictions_sha256'] == digest
    write(prediction, {'tampered': []})
    with pytest.raises(ValueError, match='Prediction hash'):
        module.verify_freeze(folder,folder/'outputs')


