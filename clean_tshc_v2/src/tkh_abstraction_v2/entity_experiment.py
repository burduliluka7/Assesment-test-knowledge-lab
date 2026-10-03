"""Active V2: entities, independent high-recall channels, and pinned local models."""
import csv, hashlib, importlib.metadata, json, os, random, sys, time
import numpy as np
from .io import read, write, sha, event
from .snapshots import snapshot
from .intent import parse_intent
from .entity_index import build_index, select_evidence
from .entity_retrieval import EntityRetriever, rerank_prefix
from .models import Encoder, RelevanceCrossEncoder
from .verification import NliVerifier
from .requirement_ranking import atomic_components, score_requirements, order_candidates
from .v1_documents import documents


def content_key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def run(root, out, accesses):
    import torch
    prediction_path = out/'retrieval/predictions.json'
    if prediction_path.exists():
        raise FileExistsError('Predictions frozen; use --output for a replay')
    cfg = read(root/'config.json')
    availability = read(root/'model_manifest.json')
    for role in ('dense','relevance','nli'):
        model=availability[role]
        if model['status']=='available':
            assert model['revision']==cfg[role+'_revision']
            for relative,digest in model['files'].items():
                assert sha(root/model['path']/relative)==digest, 'Pinned model file changed: '+role+'/'+relative
    event(out/'audit/events.jsonl', 'prediction_started', config_sha256=sha(root/'config.json'), pid=os.getpid())
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    torch.set_num_threads(cfg['threads']); torch.use_deterministic_algorithms(True)
    write(out/'audit/configuration.json', cfg)
    write(out/'audit/predeclaration.json', dict(label='DEVELOPMENT', primary='A7', generator='A5',
        stages={'A0':'raw V1 document dense top500; canonical dedup', 'A1':'typed original+structured dense max top500',
        'A2':'dense/BM25/name round-robin union', 'A3':'+evidence dense', 'A4':'+bounded graph',
        'A5':'same A4 union ordered by RRF60', 'A6':'A5 top500 relevance, unchanged tail', 'A7':'RRF top500, core+requirements top3 evidence bundles, four NLI tiers then geometric/weakest relevance ranks, global/RRF/ID ties; unchanged tail'},
        controls=['TypedRaw','ParsedOnly','WithoutEvidenceRRF','WithoutGraphRRF','A6_100','A6_200','A7_NoNLI','A7_SingleEvidence'], no_training=True, no_fitted_weights=True, no_hierarchy=True, no_gold_based_selection=True))
    graph = read(root/'data/tkh_collection10.json')
    indices = {cfg['snapshot']: build_index(snapshot(graph, cfg['snapshot']), cfg)}
    print('Base index:', len(indices[cfg['snapshot']]['entities']), 'entities', flush=True)
    with (root/'data/questions.csv').open(encoding='utf-8-sig', newline='') as stream:
        questions = list(csv.DictReader(stream, delimiter=';'))
    intents = {q['question_id']: parse_intent(q['question'], cfg['snapshot']) for q in questions}
    for intent in intents.values():
        intent['ranking_components'] = atomic_components(intent)
        intent['answer_heads'] = intent['answer_types']
        intent['answer_families'] = sorted({t for a in intent['answer_types'] for t in cfg['answer_families'].get(a, [])})
    predictor_sources = [p for p in sorted((root/'src/tkh_abstraction_v2').glob('*.py')) if 'evaluation' not in p.name and p.name != 'strict_targets.py']
    source_key = content_key({p.name:sha(p) for p in predictor_sources})
    cache_root = root/'.cache/requirement_inference'/source_key
    encoder = Encoder(root/availability['dense']['path'], root/'.cache/embeddings', cfg['batch_size'])
    failures = {}
    try:
        cross = RelevanceCrossEncoder(root/availability['relevance']['path'], cfg['cross_encoder_max_tokens'])
    except Exception as exc:
        cross = None; failures['relevance'] = repr(exc)
    try:
        nli = NliVerifier(root/availability['nli']['path'])
    except Exception as exc:
        nli = None; failures['nli'] = repr(exc)
    retrievers = {}
    for cutoff in sorted({d['temporal_cutoff'] for d in intents.values()}):
        if cutoff not in indices:
            indices[cutoff] = build_index(snapshot(graph, cutoff), cfg)
        write(out/'entity_index'/(str(cutoff)+'.json'), indices[cutoff])
        print('Encoding cutoff', cutoff, flush=True)
        retrievers[cutoff] = EntityRetriever(indices[cutoff], documents(snapshot(graph, cutoff), cfg), encoder, cfg)
    if cross is None:
        raise RuntimeError('Pinned relevance model required; no silent replacement: '+str(failures))
    predictions, costs = {}, {}
    fingerprints = {k: content_key(v) for k,v in indices.items()}
    for question in questions:
        qid = question['question_id']; intent = intents[qid]; cutoff = intent['temporal_cutoff']
        idx, retriever = indices[cutoff], retrievers[cutoff]
        begin = time.perf_counter()
        print('Generating', qid, flush=True)
        generation = retriever.generate(intent)
        rankings = generation['systems']
        write(out/'query_understanding'/(qid+'.json'), intent)
        write(out/'retrieval/channels'/(qid+'.json'), dict(channels=generation['channels'], full_channels=generation['full_channels'], candidate_sources=generation['candidate_sources']))
        cards = {r['entity_id']: select_evidence(idx, r['entity_id'], generation['evidence_similarity'], cfg,
                 generation['graph_paths'].get(r['entity_id'], [])) for r in rankings['A5'][:cfg['rerank_k']]}
        write(out/'retrieval/selected_evidence'/(qid+'.json'), cards)
        cost = dict(generation['costs'], generation_wall_seconds=time.perf_counter()-begin)
        if cross:
            print('Reranking', qid, len(cards), 'entities', flush=True)
            key = content_key([cfg['relevance_revision'], intent['original_question'], cards])
            cache = cache_root/'global'/(key+'.json')
            start = time.perf_counter(); cached = cache.exists()
            details = read(cache) if cached else cross.score(intent['original_question'], list(cards.values()))
            if not cached:
                write(cache, details)
            cost.update(cross_encoder_pairs=len(details), cross_encoder_inference_pairs=0 if cached else len(details), cross_encoder_cache_hit=cached,
                cross_encoder_wall_seconds=time.perf_counter()-start,
                prefix_timing_estimates={str(k): sum(details[r['entity_id']]['runtime_seconds_amortized'] for r in rankings['A5'][:k]) for k in cfg['rerank_controls']},
                timing_note='100/200 reuse pairs from500; estimates, not independent production latency measurements')
            rankings['A6'] = rerank_prefix(rankings['A5'], details, cfg['rerank_k'])
            for k in cfg['rerank_controls']:
                rankings['A6_'+str(k)] = rerank_prefix(rankings['A5'], details, k)
            write(out/'retrieval/cross_encoder'/(qid+'.json'), details)
            requirement_results = {}
            for arm, top_k in [('A7',cfg['top_k_requirement_evidence']),('A7_SingleEvidence',1)]:
                print('Requirements', qid, arm, flush=True)
                key = content_key([cfg, fingerprints[cutoff], intent, list(cards), top_k, nli is not None])
                cache = cache_root/'requirements'/(key+'.json')
                start = time.perf_counter(); cached = cache.exists()
                if cached:
                    saved = read(cache); records, work = saved['records'], saved['costs']
                else:
                    records, work = score_requirements(intent, rankings['A5'], idx, retriever, cross, nli, cfg, top_k)
                    write(cache, dict(records=records,costs=work))
                rankings[arm], trace = order_candidates(rankings['A5'], details, intent['ranking_components'], records, cfg)
                if arm == 'A7':
                    rankings['A7_NoNLI'], no_nli_trace = order_candidates(rankings['A5'], details, intent['ranking_components'], records, cfg, use_nli=False)
                    write(out/'retrieval/requirement_ranking/A7_NoNLI'/(qid+'.json'),dict(nli_ignored_in_ranking=True, traces=no_nli_trace))
                for eid,row in trace.items():
                    row.update(name=idx['entities'][eid]['name'],candidate_generation=dict(
                        **generation['candidate_sources'][eid],rrf_rank=row['rrf_rank']))
                # Every final entity, including the unscored fusion tail, has a trace.
                for pos,row in enumerate(rankings[arm][cfg['rerank_k']:],cfg['rerank_k']+1):
                    eid=row['entity_id']
                    trace[eid]=dict(entity_id=eid,name=idx['entities'][eid]['name'],final_rank=pos,
                        rrf_rank=pos,stage='outside_rerank_pool',requirements=[],global_relevance=None,
                        supported_count=None,contradicted_count=None,unknown_count=None,
                        requirement_geometric_score=None,weakest_requirement_score=None,final_tier=None,
                        candidate_generation=generation['candidate_sources'][eid])
                write(out/'retrieval/requirement_ranking'/arm/(qid+'.json'),trace)
                requirement_results[arm]=dict(work,cache_hit=cached,actual_nli_pairs=0 if cached else work['nli_pairs'],
                    actual_requirement_cross_encoder_pairs=0 if cached else work['requirement_cross_encoder_pairs'],
                    actual_wall_seconds=time.perf_counter()-start)
            cost['requirements']=requirement_results
        cost['total_wall_seconds'] = time.perf_counter()-begin
        costs[qid] = cost
        predictions[qid] = dict(question=question['question'], question_type=question['type'], cutoff=cutoff,
            eligible_entities=generation['costs']['eligible_entities'], candidate_union_size=len(rankings['A5']), rankings=rankings)
        print('Completed', qid, 'union', len(rankings['A5']), 'seconds', round(cost['total_wall_seconds'],2), flush=True)
    write(out/'audit/efficiency.json', costs)
    write(out/'audit/model_version_manifest.json', dict(python=sys.version, models=availability, model_load_failures=failures,
        packages={n: importlib.metadata.version(n) for n in ('numpy','torch','transformers','sentence-transformers','huggingface-hub')},
        nli_role='requirement verification proxy, not relevance', seed=cfg['seed'], device='cpu'))
    write(prediction_path, predictions)
    digest = sha(prediction_path)
    (out/'retrieval/predictions.sha256').write_text(digest+'\n', encoding='ascii')
    files = [root/'config.json', root/'model_manifest.json', root/'scripts/predict.py', *sorted((root/'data').glob('*')),
        *[p for p in sorted((root/'src/tkh_abstraction_v2').glob('*.py')) if 'evaluation' not in p.name and p.name != 'strict_targets.py'],
        *[p for folder in ('query_understanding','entity_index','retrieval') for p in sorted((out/folder).rglob('*.json'))],
        out/'audit/configuration.json',out/'audit/predeclaration.json',out/'audit/model_version_manifest.json',out/'audit/efficiency.json',out/'retrieval/predictions.sha256']
    hashes={str(p.relative_to(root)):sha(p) for p in files}
    assert all(row['allowed'] for row in accesses), 'Denied project access during prediction'
    audit_path=out/'audit/prediction_accesses.json'
    write(audit_path,dict(project_reads=sorted({row['path'] for row in accesses}),denied_reads=[],
        local_imports=sorted(m for m in sys.modules if m.startswith('tkh_abstraction')),
        gold_reads=0,historical_evaluation_reads=0,enforcement='Strict input allowlist and cooperative Python audit hook; not an OS sandbox'))
    hashes[str(audit_path.relative_to(root))]=sha(audit_path)
    write(out/'audit/freeze.json', dict(predictions_sha256=digest, files=hashes,
        status='PREDICTIONS_FROZEN_BEFORE_EVALUATION', output_directory=str(out.relative_to(root))))
    event(out/'audit/events.jsonl','predictions_frozen',sha256=digest,questions=len(predictions))
    print('PREDICTIONS FROZEN', digest, flush=True)
