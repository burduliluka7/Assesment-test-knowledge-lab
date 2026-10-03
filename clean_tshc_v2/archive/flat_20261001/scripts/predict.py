"""Gold-blind prediction entry point. Evaluation is a separate subsequent process."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src'))
for key in ('HF_HOME', 'TORCH_HOME', 'MPLCONFIGDIR', 'TMP', 'TEMP'):
    folder = ROOT / '.cache' / key.lower()
    folder.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(folder)
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from tkh_abstraction_v2.isolation import install
accesses = install(ROOT, prediction=True)
from tkh_abstraction_v2.io import read, write, sha, event
from tkh_abstraction_v2.snapshots import snapshot
from tkh_abstraction_v2.v1_documents import documents
from tkh_abstraction_v2.question_understanding import ExtractiveParser
from tkh_abstraction_v2.candidate_cards import build_cards, eligible
from tkh_abstraction_v2.ranking import BM25, rank, rrf, replace_prefix, requirement_score
from tkh_abstraction_v2.models import Encoder, RelevanceCrossEncoder
import csv
import time
import random
import importlib.metadata
import numpy as np


def main():
    out = ROOT / 'outputs'
    predictions_file = out / 'retrieval/predictions.json'
    if predictions_file.exists():
        raise FileExistsError('Frozen predictions already exist; use a fresh V2 copy/output archive for a new run')
    audit = out / 'audit/events.jsonl'
    cfg = read(ROOT / 'config.json')
    event(audit, 'prediction_started', config_sha256=sha(ROOT / 'config.json'), pid=os.getpid())
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    import torch
    torch.manual_seed(cfg['seed'])
    torch.set_num_threads(cfg['threads'])
    torch.use_deterministic_algorithms(True)
    versions = {n: importlib.metadata.version(n) for n in ['numpy', 'torch', 'transformers', 'sentence-transformers', 'huggingface-hub']}
    write(out / 'audit/configuration.json', cfg)
    write(out / 'audit/predeclaration.json', dict(primary='V2-C', first_stage='V2-B fixed before gold, not selected by measured performance',
        ablations={'A0': 'raw/V1/all', 'A1': 'raw/V1/hard-type', 'A2': 'raw/cards/all', 'A3': 'parsed/cards/all', 'A4': 'A3 top100 relevance'},
        systems={'V2-A': 'parsed/cards/hard-type/BGE', 'V2-B': 'V2-A + BM25 RRF60', 'V2-C': 'V2-B top100 relevance', 'V2-D': 'V2-B top100 requirement support'},
        development_only=True, no_hierarchy_routing=True))
    graph = read(ROOT / 'data/tkh_collection10.json')
    snap = snapshot(graph, cfg['snapshot'])
    with (ROOT / 'data/questions.csv').open(encoding='utf-8-sig', newline='') as stream:
        questions = list(csv.DictReader(stream, delimiter=';'))
    parsed = {q['question_id']: ExtractiveParser().parse(q['question']) for q in questions}
    for qid, d in parsed.items():
        write(out / 'query_understanding' / (qid + '.json'), d)
    print('Building snapshot-visible candidate cards', flush=True)
    cards = build_cards(snap, cfg)
    write(out / 'candidate_cards/cards.json', cards)
    docs = documents(snap, cfg)
    write(out / 'candidate_cards/v1_documents.json', docs)
    ids = sorted(cards)
    assert ids == [d['node_id'] for d in docs]
    texts = [cards[i]['text'] for i in ids]
    lexical = BM25(texts, cfg['bm25_k1'], cfg['bm25_b'])
    availability = read(out / 'audit/model_availability.json')
    encoder, cross = None, None
    failures = {}
    try:
        encoder = Encoder(ROOT / availability['dense']['path'], ROOT / '.cache/embeddings', cfg['batch_size'])
    except Exception as exc:
        failures['dense'] = repr(exc)
    try:
        cross = RelevanceCrossEncoder(ROOT / availability['relevance']['path'], cfg['cross_encoder_max_tokens'])
    except Exception as exc:
        failures['relevance'] = repr(exc)
    X = oldX = nodeX = None
    if encoder:
        print('Encoding V2 cards', flush=True)
        X = encoder.encode(texts)
        print('Encoding exact V1 document-style baseline', flush=True)
        oldX = encoder.encode([d['text'] for d in docs])
        evidence_texts = sorted({b['text'] for c in cards.values() for b in c['blocks']})
        nodeX = encoder.encode(evidence_texts)
        evidence_index = {t: j for j, t in enumerate(evidence_texts)}
        dense_token_counts = [len(encoder.model.tokenizer.encode(t, add_special_tokens=True, truncation=False)) for t in texts]
        write(out / 'audit/dense_token_packing.json', dict(limit=512, policy='identity-first prefix truncation',
            candidates_truncated=sum(n > 512 for n in dense_token_counts), candidate_counts=dict(zip(ids, dense_token_counts))))
    result = {}
    for question in questions:
        qid = question['question_id']
        print('Predicting', qid, flush=True)
        start = time.perf_counter()
        d = parsed[qid]
        allowed = [i for i in ids if eligible(cards[i]['node_type'], d['answer_type'], cfg)]
        allowed_set = set(allowed)
        def filtered(ranking):
            return [r for r in ranking if r['node_id'] in allowed_set]
        rankings = {'LexicalAll': rank(ids, lexical.score(d['search_query']))}
        rankings['Lexical'] = filtered(rankings['LexicalAll'])
        if encoder:
            queries = encoder.encode([d['original_question'], d['search_query'], d['structured_query']])
            rankings['A0'] = rank(ids, oldX @ queries[0])
            rankings['A1'] = filtered(rankings['A0'])
            rankings['A2'] = rank(ids, X @ queries[0])
            rankings['A3'] = rank(ids, X @ queries[1])
            rankings['DenseRaw'] = filtered(rankings['A2'])
            rankings['DenseStructured'] = filtered(rank(ids, X @ queries[2]))
            rankings['V2-A'] = filtered(rankings['A3'])
            rankings['SoftType'] = rank(ids, np.maximum(0., X @ queries[1]) * np.array([1. if i in allowed_set else cfg['soft_type_prior'] for i in ids]))
            rankings['V2-B'] = rrf(rankings['V2-A'], rankings['Lexical'], k=cfg['rrf_k'])
            prefix = rankings['V2-B'][:cfg['rerank_k']]
            components = [d['core_task'] or d['original_question'], *d['requirements'], *d['exclusions']]
            sims = np.maximum(0., nodeX @ encoder.encode(components).T)
            support = {}
            for row in prefix:
                ident = row['node_id']
                blocks = cards[ident]['blocks']
                ix = [evidence_index[b['text']] for b in blocks]
                best = np.argmax(sims[ix], axis=0)
                values = sims[ix].max(axis=0)
                nr = len(d['requirements'])
                score, gm, penalty = requirement_score(float(values[0]), values[1:1 + nr], values[1 + nr:])
                support[ident] = dict(score=score, core_support=float(values[0]), requirement_supports=[float(s) for s in values[1:1 + nr]],
                    exclusion_supports=[float(s) for s in values[1 + nr:]], geometric_mean=gm, exclusion_penalty=penalty, type_prior=1.,
                    supporting_blocks=[blocks[int(b)] for b in best], components=components)
            rankings['V2-D'] = replace_prefix(rankings['V2-B'], {i: s['score'] for i, s in support.items()}, cfg['rerank_k'])
            write(out / 'retrieval' / (qid + '_requirement_support.json'), support)
            if cross:
                for source, target in [('V2-B', 'V2-C'), ('A3', 'A4')]:
                    prefix_cards = [cards[r['node_id']] for r in rankings[source][:cfg['rerank_k']]]
                    details = cross.score(d['original_question'], prefix_cards)
                    rankings[target] = replace_prefix(rankings[source], {i: s['relevance_score'] for i, s in details.items()}, cfg['rerank_k'])
                    write(out / 'retrieval' / (qid + '_' + target + '_cross_encoder.json'), details)
        result[qid] = dict(question=question['question'], question_type=question['type'], eligible_count=len(allowed), rankings=rankings)
        write(out / 'audit' / (qid + '_runtime.json'), dict(seconds=time.perf_counter() - start))
    write(out / 'audit/model_version_manifest.json', dict(python=sys.version, packages=versions, models=availability, load_failures=failures,
        relevance_query='original question; A4/V2-C differ only from source by joint relevance scoring',
        parser='deterministic, optional local JSON interface not invoked', nli='optional diagnostic not run', device='cpu'))
    write(out / 'audit/prediction_accesses.json', dict(data_reads=sorted(set(accesses)),
        local_imports=sorted(n for n in sys.modules if n.startswith('tkh_abstraction')), forbidden_accesses=0,
        limitation='Python audit hooks enforce cooperative Python code, not hostile native code or OS-level isolation'))
    write(predictions_file, result)
    digest = sha(predictions_file)
    (out / 'retrieval/predictions.sha256').write_text(digest + '\n', encoding='ascii')
    # Freeze all interpretation-relevant artifacts; runtime logs need not be deterministic.
    frozen = [ROOT / 'config.json', *sorted((ROOT / 'src').rglob('*.py')), ROOT / 'scripts/predict.py',
        *sorted((ROOT / 'data').glob('*')), *sorted((out / 'query_understanding').glob('*.json')),
        *sorted((out / 'candidate_cards').glob('*.json')), *sorted((out / 'retrieval').glob('*.json'))]
    write(out / 'audit/freeze.json', dict(predictions_sha256=digest, files={str(p.relative_to(ROOT)): sha(p) for p in frozen},
        status='PREDICTIONS_FROZEN_BEFORE_EVALUATION'))
    event(audit, 'predictions_frozen', sha256=digest, questions=len(result))
    print('PREDICTIONS FROZEN SHA256', digest, flush=True)


if __name__ == '__main__':
    main()
