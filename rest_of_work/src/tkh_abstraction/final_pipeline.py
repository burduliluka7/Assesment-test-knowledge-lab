"""Final integration experiment; historical artifacts are inputs, never outputs."""
from pathlib import Path
from collections import defaultdict
import time
import numpy as np
from .io import read_json, read_csv, write_json, locate, digest
from .config import read_config
from .v2_pipeline import sha
from .snapshots import snapshot
from .semantics import node_texts
from .reranker_pipeline import FrozenH2Candidates, LocalEncoder
from .question_decomposition import decompose_question
from .target_mapping import TargetResolver
from .retrieval import normalize
from .final_search import FineScorer, HierarchySearch, flat_scan, rank_scored, final_decomposition
from .final_evaluation import TOP_KS, evidence_export, claim_metrics, evaluate_result, summarize
from .diffusion_pipeline import evaluate_ranking


def verify_frozen(path):
    manifest = read_json(path)
    for file, expected in manifest['files'].items():
        if file == manifest['allowed_additive_cli']: continue
        if sha(file) != expected: raise ValueError('Frozen file changed: '+file)
    return len(manifest['files'])-1


def prepare_scorer(question, encoder, source, pools, cfg, original, h2):
    """Preserve the already-frozen float32 dot-product shapes for exact replay.

    This prepares only the reference's historical top200 evidence similarities.
    Other entities are evaluated lazily when a search actually reaches them.
    """
    d0 = decompose_question(question); d = final_decomposition(question)
    texts0 = [c['text'] for c in d0.components]
    X0 = encoder.encode(texts0, 'constraint-components:'+digest(texts0))
    ids0 = sorted({i['node_id'] for b in original['candidates'] for xs in b['fields'].values() for i in xs})
    nv = {n['id']:source.context.X[source.context.text_idx[n['surface_form']]] for n in source.snap['nodes']}
    Z0 = X0 @ np.asarray([nv[n] for n in ids0]).T
    cache = {c['text']:dict(zip(ids0, map(float, Z0[i]))) for i, c in enumerate(d0.components)}
    # The historical CR6 run encoded all new components together. Keep that cache
    # key, even though qualifier vectors are never used by the production scorer.
    from .conditioned_decomposition import decompose_conditioned_question
    diagnostic = decompose_conditioned_question(question)
    texts = list(dict.fromkeys(texts0 + [c['text'] for c in diagnostic.components]))
    extra = [t for t in texts if t not in texts0]
    Xnew = encoder.encode(extra, 'conditioned-components:'+digest(extra)) if extra else []
    vectors = {t:X0[i] for i,t in enumerate(texts0)}
    vectors.update({t:Xnew[i] for i,t in enumerate(extra)})
    pool_ids = sorted({i['node_id'] for r in h2[:200] for xs in pools[r['entity_id']]['fields'].values() for i in xs})
    for c in d.components:
        t = c['text']; saved = cache.setdefault(t, {}); missing = [n for n in pool_ids if n not in saved]
        if missing: saved.update(zip(missing, map(float, vectors[t] @ np.asarray([nv[n] for n in missing]).T)))
    return FineScorer(question, cfg, pools, source.mentions.names, vectors, nv, cache)


def run_final(cfg, questions_path=None, truth_path=None):
    start = time.perf_counter(); root = Path(cfg['v2_output']); out = Path(cfg['final_output']); out.mkdir(parents=True, exist_ok=True)
    if cfg['final_verify_history']: verify_frozen(out/'frozen_manifest.json')
    ccfg = read_config(cfg['conditioned_config']); cr = read_config(ccfg['constraint_config'])
    rc = read_config(cr['reranking_config']); context_cfg = read_config(rc['context_config']); dc = read_config(rc['diffusion_config'])
    scoring = dict(cr, **{k:v for k,v in ccfg.items() if k.startswith('conditioned_')})
    data = read_json(locate(cfg['data_dir'], 'tkh_collection10.json'))
    questions = read_csv(questions_path or locate(cfg['data_dir'], 'questions.csv'))
    truth = read_json(truth_path or locate(cfg['data_dir'], 'ground_truth.json'))
    old = read_json(root/'conditioned_evidence_results.json')
    same_questions = digest(questions)==old['protocol']['questions_hash'] and digest(data)==old['protocol']['data_hash']
    if not same_questions: raise ValueError('Final integration replay requires the frozen graph and question text; no new benchmark construction')
    same_truth = digest(truth)==old['protocol']['ground_truth_hash']
    pools_all = read_json(root/'candidate_evidence_pools.json')
    old_selections = {(r['cutoff'],r['question_id']):r for r in read_json(root/'requirement_evidence_selection.json')}
    original = {(r['cutoff'],r['question_id']):r for r in read_json(root/'candidate_evidence.json')}
    rr = read_json(root/'reranker_results.json'); alignments = read_json(root/'claim_alignment.json')
    protocol = dict(config=cfg, fine_definition='Historical without_qualifiers: generic corrected decomposition, H2 top100, wider query-independent pools, corrected attribution, top3 component evidence/max, unchanged CR1 aggregate; no qualifiers/NLI/fusion.',
                    scoring_config=scoring, data_hash=digest(data), questions_hash=digest(questions), ground_truth_hash=digest(truth),
                    source_hashes={p.as_posix():sha(p) for p in Path('src').rglob('*.py')},
                    search='Existing centroid/prototype representations, .5/.5 best-two score, deterministic best-first; atomic member activates its frozen exact-name group once.',
                    universe='Main hierarchy and flat exhaustive search all H2-eligible entity groups. H2-intersection control restricts to FINAL_FINE top100; unreachable groups never scored.',
                    budget='Total coarse/prototype + leaf + component/evidence comparisons. Evidence charged per candidate/component/item including cached uses, an explicit standalone logical comparison count. Fine aggregation count separate.',
                    exclusions='Offline embeddings/hierarchy/pool construction, model load, graph enumeration, embedding inference, queue/sort/aggregation overhead and post-hoc evidence question-cosines excluded. H2 incidence visits reported separately, not equated with dot products.',
                    selection='No tuning; budgets25/50/100/200/500 plus unrestricted. Variant comparison500 chosen before outcomes.',
                    nli_calls=0, hierarchy_rebuilds=0)
    write_json(out/'protocol.json', protocol)
    encoder = LocalEncoder(cfg['evaluation_model'], cfg['evaluation_revision'])
    aa = [q for q in questions if q['type']=='A']; results = []; fine_archive = []; fingerprints = []; parity = []; pool_index_seconds = []
    # Phase 1: exact fine reference for both cutoffs, before ANY hierarchy traversal.
    sources = {}; scorers = {}; queries = {}; fine_results = {}
    for year in cfg['final_cutoffs']:
        snap = snapshot(data, year)
        h2rows = {r['question_id']:r for s in rr['results'] if s['cutoff']==year and s['stage']=='RR0' for r in s['records']}
        source = FrozenH2Candidates(snap, encoder, context_cfg, dc, h2rows, root/'diffusion'); sources[year] = source
        pools = {p['entity_id']:p for p in pools_all if p['cutoff']==year}
        year_rows = []
        for q in aa:
            qid = q['question_id']; h2 = source.get(q['question'], qid)
            scorer = prepare_scorer(q['question'], encoder, source, pools, scoring, original[year,qid], h2['ranked'])
            scorers[year,qid] = scorer
            queries[year,qid] = normalize(encoder.encode([q['question']], 'final-coarse-query:'+digest(q['question']))[0])
            prefix = rank_scored([r['entity_id'] for r in h2['ranked'][:100]], scorer)
            ranked = prefix + h2['ranked'][100:]
            expected = old_selections[year,qid]['candidates']
            actual_ranks = {r['entity_id']:i+1 for i,r in enumerate(ranked)}
            for c in expected:
                if actual_ranks[c['entity_id']] != c['ranks']['without_qualifiers']: raise ValueError('FINAL_FINE historical rank differs')
                if c['h2_rank']<=100 and scorer.details[c['entity_id']]['score'] != c['without_qualifiers_score']: raise ValueError('FINAL_FINE historical score differs')
            costs = dict(coarse_comparisons=0, leaf_comparisons=0,
                         evidence_comparisons=sum(scorer.work(r['entity_id']) for r in prefix), fine_candidates_scored=len(prefix),
                         h2_vector_comparisons=h2rows[qid]['cost']['vector_comparisons'],
                         h2_incidence_entry_visits=h2rows[qid]['cost'].get('incidence_entry_visits_upper',0),
                         cluster_expansions=0, reached_atomic_candidates=sum(len(r['node_ids']) for r in prefix))
            costs['total_comparisons'] = costs['evidence_comparisons']+costs['h2_vector_comparisons']
            result = dict(ranked=ranked, status='COMPUTED', cost=costs, trace=[], budget=None)
            resolver = TargetResolver(snap); resolved = [resolver.resolve(s) for s in truth.get(qid,{}).get('expected_methods',[])]
            fine_metric = evaluate_ranking(ranked, resolved, dict(diffusion_top_ks=TOP_KS))
            fine_results[year,qid] = (result, fine_metric, resolved, h2['ranked'])
            row = make_record('FINAL_FINE', 'none', year, q, result, fine_metric, resolved, scorer, source, queries[year,qid], truth, alignments[str(year)])
            year_rows.append(row); fine_archive.append(row)
            fingerprints.append(dict(cutoff=year, question_id=qid, ranking_hash=digest([(r['entity_id'],r['score']) for r in ranked]),
                                     top100_ids=[r['entity_id'] for r in prefix]))
        s = summarize(year_rows)
        if same_truth and s['at_k']['10']['found'] != 13: raise ValueError('FINAL_FINE does not reproduce 13 targets')
        results.append(dict(cutoff=year, system='FINAL_FINE', variant='none', budget=None, summary=s))
        parity.append(dict(cutoff=year, exact_scores_and_ranks=True, found=s['at_k']['10']['found'], canonical=s['at_k']['10']['canonical_target_instances']))
        print('FINAL_FINE VERIFIED', year, s['at_k']['10']['found'], flush=True)
    write_json(out/'final_fine_rankings.json', fine_archive)
    regression_path = out/'final_fine_regression.json'
    if regression_path.exists() and read_json(regression_path)!=fingerprints: raise ValueError('FINAL_FINE frozen regression changed')
    write_json(regression_path, fingerprints)
    write_json(out/'fine_parity.json', parity)
    # Phase 2: coarse-to-fine and flat runs with the already frozen scorer.
    for year in cfg['final_cutoffs']:
        source = sources[year]; snap = source.snap
        X = encoder.encode(node_texts(snap['nodes'], cfg['include_type']), f'v2-retrieval:{year}')
        indexes = {}; then = time.perf_counter()
        for variant in cfg['variants']:
            path = Path(cfg['output'])/'variants'/variant/f'hierarchy_{year}{"_benchmark" if year==2025 else ""}.json'
            indexes[variant] = HierarchySearch(read_json(path), snap['nodes'], X, source.groups, cfg['final_prototypes'])
        pool_index_seconds.append(dict(cutoff=year, seconds=time.perf_counter()-then))
        grouped = defaultdict(list)
        for q in aa:
            qid = q['question_id']; scorer = scorers[year,qid]; query = queries[year,qid]
            reference, fm, resolved, h2 = fine_results[year,qid]
            runs = [('flat_exhaustive','none',None), ('hierarchical','temporal',None), ('hierarchical_h2_intersection','temporal',None)]
            runs += [(s,'temporal' if s.startswith('hierarchical') else 'none',b) for b in cfg['final_work_budgets'] for s in ['hierarchical','hierarchical_h2_intersection','flat_equal_work']]
            runs += [('hierarchical',v,cfg['final_variant_budget']) for v in cfg['variants'] if v!='temporal']
            for system, variant, budget in runs:
                before = scorer.actual_new_dots
                if system.startswith('flat'):
                    result = flat_scan(scorer, query, scorer.node_vectors, budget, cfg['final_flat_seed'])
                else:
                    allowed = {r['entity_id'] for r in h2[:100]} if system.endswith('intersection') else None
                    result = indexes[variant].search(query, budget, scorer, allowed, cfg['final_centroid_weight'], cfg['final_prototype_top_r'])
                result['actual_incremental_evidence_dots'] = scorer.actual_new_dots-before
                row = make_record(system, variant, year, q, result, fm, resolved, scorer, source, query, truth, alignments[str(year)])
                if system.endswith('intersection') and budget is None:
                    if [r['entity_id'] for r in result['ranked']] != [r['entity_id'] for r in reference['ranked'][:100]]:
                        raise ValueError('Unrestricted hierarchy does not reproduce the H2-restricted fine ranking')
                grouped[system,variant,budget].append(row)
            exhaustive = grouped['flat_exhaustive','none',None][-1]['returned']
            ceiling = grouped['hierarchical','temporal',None][-1]['returned']
            if exhaustive != ceiling: raise ValueError('Hierarchy ceiling differs from exhaustive fine scoring')
            print('FINAL SEARCH', year, qid, 'elapsed', round(time.perf_counter()-start,1), flush=True)
        for (system,variant,budget), rows in grouped.items():
            path = out/f'{system}_{variant}_{year}_{budget if budget is not None else "all"}.json'
            write_json(path, rows)
            results.append(dict(cutoff=year, system=system, variant=variant, budget=budget, summary=summarize(rows), artifact=path.as_posix()))
    run = dict(protocol=protocol, results=results, fine_parity=parity, index_costs=pool_index_seconds,
               elapsed_seconds=time.perf_counter()-start, preserved_files=verify_frozen(out/'frozen_manifest.json') if cfg['final_verify_history'] else None)
    write_json(out/'results.json', run)
    return run


def make_record(system, variant, year, question, result, fine_metric, resolved, scorer, source, query, truth, alignment):
    metric, retention = evaluate_result(result, fine_metric, resolved)
    evidence = evidence_export(result['ranked'], scorer, {n['id']:n for n in source.snap['nodes']}, query)
    claims = claim_metrics(question['question_id'], truth, source.snap, evidence, alignment)
    costs = dict(result['cost'], vector_comparisons=result['cost']['total_comparisons'], entity_scores=result['cost']['fine_candidates_scored'])
    return dict(system=system, variant=variant, cutoff=year, question_id=question['question_id'], question_type='A', question=question['question'],
                budget=result['budget'], status=result['status'], entity_metrics=metric, retention=retention, claim_metrics=claims,
                evidence_metrics=dict(claim_comparisons_new=0), cost=costs, latency_seconds=0,
                returned=result['ranked'], trace=result['trace'], retrieved_evidence=evidence,
                root_layer_cost=result.get('root_layer_cost'), root_layer_complete=result.get('root_layer_complete'),
                actual_incremental_evidence_dots=result.get('actual_incremental_evidence_dots'),
                group_lift=result.get('group_lift',[]),
                decomposition=scorer.decomposition.to_dict())
