"""Gold-free coverage diagnostics. Does not change any predictions."""
import sys
from pathlib import Path
from collections import Counter
ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src'))
from tkh_abstraction_v2.io import read, write, sha
from tkh_abstraction_v2.snapshots import snapshot, anomalies
from tkh_abstraction_v2.candidate_cards import eligible
from tkh_abstraction_v2.ranking import BM25

cfg = read(ROOT / 'config.json')
graph = read(ROOT / 'data/tkh_collection10.json')
snap = snapshot(graph, cfg['snapshot'])
cards = read(ROOT / 'outputs/candidate_cards/cards.json')
lexical = BM25([c['text'] for c in cards.values()], cfg['bm25_k1'], cfg['bm25_b'])
queries = {}
for path in sorted((ROOT / 'outputs/query_understanding').glob('*.json')):
    d = read(path)
    queries[path.stem] = dict(answer_type=d['answer_type'], fallback=d.get('fallback'),
        requirements=len(d['requirements']), exclusions=len(d['exclusions']), time_recorded=bool(d['temporal_condition']),
        eligible_count=sum(eligible(c['node_type'], d['answer_type'], cfg) for c in cards.values()),
        lexical_positive_candidates=int((lexical.score(d['search_query']) > 0).sum()))
write(ROOT / 'outputs/audit/input_diagnostics.json', dict(snapshot=cfg['snapshot'], nodes=len(snap['nodes']), edges=len(snap['hyperedges']),
    candidate_count=len(cards), node_types=dict(Counter(n['type'] for n in snap['nodes'])),
    cards_without_non_title_evidence=sum(not any(c['fields'][f] for f in c['fields'] if f != 'article') for c in cards.values()),
    cards_without_any_evidence=sum(len(c['blocks']) == 1 for c in cards.values()),
    excluded_high_arity_incidences=sum(sum(x['reason'] == 'high_arity_comembership' for x in c['discarded']) for c in cards.values()),
    query_coverage=queries,
    copied_input_sha256={p.name: sha(p) for p in sorted((ROOT / 'data').iterdir()) if p.is_file()}))
write(ROOT / 'outputs/audit/snapshot_anomalies.json', anomalies(graph))
print('Gold-free input diagnostics written')
