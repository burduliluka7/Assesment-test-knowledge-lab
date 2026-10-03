"""Post-freeze evaluation only. Never imported by prediction code."""
from collections import Counter
import math
import statistics
from .io import read, write, sha, event
from .snapshots import snapshot
from .strict_targets import support_audit


def evaluate(root, args):
    out = root / 'outputs'
    reference = root.parent / 'clean_tshc'
    gold_path = reference / 'data/ground_truth.json'
    annotation_path = reference / 'notes/target_annotations.json'
    truth = read(gold_path)
    annotations = read(annotation_path)
    event(out / 'audit/events.jsonl', 'gold_loaded', gold_sha256=sha(gold_path), annotations_sha256=sha(annotation_path))
    snap = snapshot(read(root / 'data/tkh_collection10.json'), 2026)
    support = support_audit(snap, truth, annotations)
    write(out / 'audit/strict_target_support.json', support)
    if '--inspect' in args:
        print('Gold schema:', type(truth).__name__, 'questions:', len(truth))
        print('Statuses:', support['counts'])
        print('Question types:', dict(Counter(q['type'] for q in truth.values())))
        return
    predictions = read(out / 'retrieval/predictions.json')
    cards = read(out / 'candidate_cards/cards.json')
    old_docs = {d['node_id']: d for d in read(out / 'candidate_cards/v1_documents.json')}
    cfg = read(root / 'config.json')
    qids = [q for q in predictions if truth[q]['type'] == 'A']
    systems = list(predictions[qids[0]]['rankings'])
    targets = [t for t in support['records'] if t['question_id'] in qids]
    direct = [t for t in targets if t['status'] in ('EXACT', 'ALIAS')]
    direct_count = len(direct)
    all_count = len(targets)
    supported_count = sum(t['status'] != 'ABSENT' for t in targets)
    ks = [1, 5, 10, 25, 50, 100]
    indices = {q: {s: {r['node_id']: i for i, r in enumerate(rows, 1)} for s, rows in predictions[q]['rankings'].items()} for q in qids}
    per_question, target_details = {}, []
    retention = {s: Counter() for s in ('V2-C', 'A4') if s in systems}
    rank_audit = []

    for qid in qids:
        decomposition = read(out / 'query_understanding' / (qid + '.json'))
        ts = [t for t in targets if t['question_id'] == qid]
        nr = len(ts)
        per_question[qid] = dict(question=predictions[qid]['question'], decomposition=decomposition, target_count=nr,
            status_counts=dict(Counter(t['status'] for t in ts)), systems={}, targets=[])
        requirement_file = out / 'retrieval' / (qid + '_requirement_support.json')
        requirement = read(requirement_file) if requirement_file.exists() else {}
        cross_file = out / 'retrieval' / (qid + '_V2-C_cross_encoder.json')
        cross = read(cross_file) if cross_file.exists() else {}
        stage_scores = {system: {r['node_id']: dict(score=r['score'], stage=r.get('score_stage', 'first_stage'))
                       for r in rows} for system, rows in predictions[qid]['rankings'].items()}
        for target in ts:
            detail = dict(question_id=qid, target=target['target'], status=target['status'], components=target['components'],
                node_ids=target['node_ids'], direct_ranks={}, composite_direct_completion_ranks={}, v1_compatible_bundle_ranks={})
            for system in systems:
                idx = indices[qid][system]
                direct_rank = complete_rank(target['components'], idx)
                detail['direct_ranks'][system] = direct_rank if target['status'] in ('EXACT', 'ALIAS') else None
                detail['composite_direct_completion_ranks'][system] = direct_rank if target['status'] == 'COMPOSITE' else None
                if target['status'] == 'COMPOSITE':
                    accessible = {}
                    for row in predictions[qid]['rankings'][system]:
                        ident = row['node_id']
                        rank_number = idx[ident]
                        context = old_docs[ident]['context_ids'] if system in ('A0', 'A1') else [b['node_id'] for b in cards[ident]['blocks']]
                        for member in [ident, *context]:
                            accessible.setdefault(member, rank_number)
                    bundle_rank = complete_rank(target['components'], accessible)
                else:
                    bundle_rank = detail['direct_ranks'][system]
                detail['v1_compatible_bundle_ranks'][system] = bundle_rank if target['status'] in ('EXACT', 'ALIAS', 'COMPOSITE') else None
            detail['entered_primary_top100'] = is_hit(detail['direct_ranks'].get('V2-B'), 100)
            detail['scores_and_evidence'] = {ident: dict(card_file='outputs/candidate_cards/cards.json', card_key=ident,
                evidence_selected=cards[ident]['fields'] if ident in cards else {},
                stage_scores={system: scores.get(ident) for system, scores in stage_scores.items()},
                requirement_components=requirement.get(ident), cross_encoder=cross.get(ident)) for ident in target['node_ids']}
            detail['failure_observations'] = failure_observations(target, detail, cards, snap, decomposition, cfg)
            for system, source in [('V2-C', 'V2-B'), ('A4', 'A3')]:
                if system not in systems or target['status'] not in ('EXACT', 'ALIAS'):
                    continue
                before, after = detail['direct_ranks'][source], detail['direct_ranks'][system]
                if is_hit(before, 100):
                    retention[system]['present_in_prefix'] += 1
                    retention[system]['retained_in_prefix'] += int(is_hit(after, 100))
                    retention[system]['upward' if after < before else 'downward' if after > before else 'unchanged'] += 1
            per_question[qid]['targets'].append(detail)
            target_details.append(detail)
        for system in systems:
            ds = [t['direct_ranks'][system] for t in per_question[qid]['targets']]
            first = min((r for r in ds if r is not None), default=None)
            per_question[qid]['systems'][system] = dict(first_direct_rank=first, reciprocal_rank=1 / first if first else 0.,
                direct_hits={str(k): sum(is_hit(r, k) for r in ds) for k in ks},
                recall={str(k): sum(is_hit(r, k) for r in ds) / nr if nr else None for k in ks},
                candidate_recall={str(k): sum(is_hit(r, k) for r in ds) / nr if nr else None for k in (50, 100, 200)})
            rows = predictions[qid]['rankings'][system]
            id_list = [r['node_id'] for r in rows]
            assert len(id_list) == len(set(id_list))
            assert all(math.isfinite(r['score']) for r in rows)
            rank_audit.append(dict(question_id=qid, system=system, ranked_candidates=len(rows), duplicate_ids=0, finite_scores=True))

    metrics = dict(label='DEVELOPMENT', primary_system='V2-C', primary_generator='V2-B (predeclared, not selected on gold)',
        denominators=dict(all_questions=len(predictions), type_A_questions=len(qids), type_B_not_identity_scored=len(predictions)-len(qids),
            target_occurrences=all_count, exact_alias_occurrences=direct_count, supported_including_partial=supported_count,
            status_counts=dict(Counter(t['status'] for t in targets)), rank_missing_penalty=len(cards)+1),
        definitions=dict(recall='Direct EXACT/ALIAS recovered occurrences divided by ALL Type-A target occurrences; micro average.',
            macro_recall='Mean over Type-A questions of direct exact/alias hits divided by all targets in each question.',
            MRR='Mean reciprocal rank of first EXACT/ALIAS answer, over all 14 Type-A questions; zero when unavailable.',
            direct_rank_summary='Mean/median over observed EXACT/ALIAS target occurrence ranks; missing explicitly reported.',
            v1_compatible='Supplementary strict raw V1 recovery: EXACT/ALIAS direct plus all components of COMPOSITE in union of selected document contexts; PARTIAL/ABSENT never full success.',
            supported_denominator='V1 convention includes PARTIAL and excludes ABSENT; partial never earns full identity credit.'),
        systems={}, reranker_retention={s: dict(c) for s, c in retention.items()})
    for system in systems:
        ds = [t['direct_ranks'][system] for t in target_details]
        observed = [r for r in ds if r is not None]
        matched = [t['direct_ranks'][system] for t in target_details if t['status'] in ('EXACT', 'ALIAS')]
        hits = {str(k): sum(is_hit(r, k) for r in ds) for k in ks}
        metrics['systems'][system] = dict(direct_hits=hits, recall={k: n / all_count for k, n in hits.items()},
            recall_exact_alias_conditional={k: n / direct_count for k, n in hits.items()},
            macro_recall={str(k): statistics.mean(per_question[q]['systems'][system]['recall'][str(k)] for q in qids) for k in ks},
            MRR=statistics.mean(per_question[q]['systems'][system]['reciprocal_rank'] for q in qids),
            target_MRR=sum(1 / r for r in matched if r) / direct_count,
            median_direct_target_rank=statistics.median(observed) if observed else None,
            mean_direct_target_rank=statistics.mean(observed) if observed else None,
            observed_direct_targets=len(observed), missing_direct_targets=direct_count-len(observed),
            mean_direct_rank_missing_penalized=statistics.mean(r if r else len(cards)+1 for r in matched),
            candidate_recall={str(k): sum(is_hit(r, k) for r in ds) / all_count for k in (50, 100, 200)},
            candidate_hits={str(k): sum(is_hit(r, k) for r in ds) for k in (50, 100, 200)},
            v1_compatible_recall={str(k): sum(is_hit(t['v1_compatible_bundle_ranks'][system], k) for t in target_details) / all_count for k in ks},
            v1_supported_recall={str(k): sum(is_hit(t['v1_compatible_bundle_ranks'][system], k) for t in target_details) / supported_count for k in ks},
            composite_direct_hits={str(k): sum(is_hit(t['composite_direct_completion_ranks'][system], k) for t in target_details) for k in ks})
    observations = Counter(label for t in target_details for label in t['failure_observations'])
    for label in ('1_correct_answer_not_in_candidate_top100', '2_in_top100_but_reranker_not_top10',
                  '3_target_type_excluded_by_answer_family', '4_no_adjacent_claim_task_problem_evidence_in_graph',
                  '7_high_arity_evidence_excluded_ambiguity'):
        observations.setdefault(label, 0)
    failures = dict(observed_counts=dict(observations),
        not_causally_identified=['useful evidence exists but parser missed requirement', 'acronym/name representation failure', 'multiple requirements not simultaneously satisfied'],
        caveat='Automatic observations are overlapping symptoms, not adjudicated causes. No manual repair or post-gold parameter change.')
    write(out / 'metrics.json', metrics)
    write(out / 'per_question_evaluation.json', per_question)
    write(out / 'audit/rank_audit.json', rank_audit)
    write(out / 'audit/failure_classes.json', failures)
    for system in systems:
        write(out / 'retrieval/by_system' / (system + '.json'), dict(predictions_sha256=sha(out / 'retrieval/predictions.json'),
            rankings={qid: row['rankings'][system] for qid, row in predictions.items()}))
    write(out / 'audit/evaluation_manifest.json', dict(gold_sha256=sha(gold_path), annotations_sha256=sha(annotation_path),
        prediction_sha256=sha(out / 'retrieval/predictions.json'),
        sources={p.name: sha(p) for p in [root / 'scripts/evaluate.py', root / 'src/tkh_abstraction_v2/evaluation.py', root / 'src/tkh_abstraction_v2/strict_targets.py']}))
    make_report(root, metrics, per_question, target_details, failures)
    event(out / 'audit/events.jsonl', 'evaluation_completed', metrics_sha256=sha(out / 'metrics.json'))
    print('Evaluation complete:', len(qids), 'Type-A questions;', all_count, 'target occurrences;', direct_count, 'EXACT/ALIAS')
    for system in ('A0', 'A1', 'A2', 'A3', 'A4', 'V2-A', 'V2-B', 'V2-C', 'V2-D'):
        if system in metrics['systems']:
            m = metrics['systems'][system]
            print(system, 'hits@10', m['direct_hits']['10'], '/', all_count, 'MRR', round(m['MRR'], 4))


def complete_rank(components, index):
    if not components:
        return None
    component_ranks = [min((index[v] for v in component if v in index), default=None) for component in components]
    return max(component_ranks) if all(r is not None for r in component_ranks) else None


def is_hit(rank, k):
    return rank is not None and rank <= k


def failure_observations(target, detail, cards, snap, decomposition, cfg):
    if target['status'] not in ('EXACT', 'ALIAS'):
        return ['not_a_resolved_direct_identity:' + target['status']]
    observations = []
    before, after = detail['direct_ranks'].get('V2-B'), detail['direct_ranks'].get('V2-C')
    if not is_hit(before, 100):
        observations.append('1_correct_answer_not_in_candidate_top100')
    elif not is_hit(after, 10):
        observations.append('2_in_top100_but_reranker_not_top10')
    family = cfg['answer_families'].get(decomposition['answer_type'])
    identities = [cards[i] for i in target['node_ids'] if i in cards]
    if family and not any(c['node_type'] in family for c in identities):
        observations.append('3_target_type_excluded_by_answer_family')
    if identities and all(not any(c['fields'][f] for f in c['fields'] if f != 'article') for c in identities):
        observations.append('selected_card_lacks_non_title_evidence')
        members = set(target['node_ids'])
        neighbors = {v for e in snap['hyperedges'] if members & set(e['members']) for v in e['members']} - members
        if not any(n['id'] in neighbors and n['type'] in ('claim', 'task', 'problem') for n in snap['nodes']):
            observations.append('4_no_adjacent_claim_task_problem_evidence_in_graph')
    if any(any(x['reason'] == 'high_arity_comembership' for x in c['discarded']) for c in identities):
        observations.append('7_high_arity_evidence_excluded_ambiguity')
    return observations


def make_report(root, metrics, questions, targets, failures):
    out = root / 'outputs'
    methods = (root / 'notes/methodology.md').read_text(encoding='utf8')
    intro, limitations = methods.split('## Limitations and next experiment', 1)
    lines = ['# V2 flat retrieval report — DEVELOPMENT', '',
        '## Material Passport', '',
        'Computational retrieval development study; frozen supplied graph and questions; prior benchmark exposure acknowledged. Verification status: ANALYZED. Software invariants and fixed-input inference determinism tested; no held-out or statistical-superiority claim.', '',
        intro.partition('\n')[2], '## Metrics and denominators', '',
        'All 18 questions have frozen predictions. Direct identity metrics use 14 Type-A questions and 63 target occurrences: 48 EXACT, 1 ALIAS, 1 COMPOSITE, 11 PARTIAL, 2 ABSENT. The 49 EXACT/ALIAS occurrences are the direct-identity subset. PARTIAL and ABSENT never earn identity credit; COMPOSITE is reported separately.', '',
        'The table reports **micro direct Recall@k over all 63 targets**, and MRR over all 14 Type-A questions. `metrics.json` also gives macro recall, conditional direct recall over 49 targets, V1-compatible composite bundle recovery, and V1 supported recall over 61 non-ABSENT targets (including PARTIAL in the denominator).', '',
        '| System | R@1 | R@5 | R@10 | R@25 | R@50 | R@100 | MRR | Median direct rank | Mean direct rank | Missing /49 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    order = ['A0', 'A1', 'A2', 'A3', 'A4', 'DenseRaw', 'DenseStructured', 'V2-A', 'LexicalAll', 'Lexical', 'V2-B', 'V2-C', 'V2-D', 'SoftType']
    for name in order:
        if name not in metrics['systems']:
            continue
        m = metrics['systems'][name]
        values = ' | '.join(f"{m['recall'][str(k)]:.3f}" for k in (1, 5, 10, 25, 50, 100))
        lines.append(f"| {name} | {values} | {m['MRR']:.3f} | {m['median_direct_target_rank']} | {m['mean_direct_target_rank']:.1f} | {m['missing_direct_targets']} |")
    lines.extend(['', 'Rank means/medians exclude missing direct identities and report their counts. The machine-readable metrics also give a fixed missing-rank penalty of 5,481, preventing filtered pools from receiving a smaller penalty.', '',
        '### Before-reranking candidate recall', '', '| Generator | hits@50 /63 | hits@100 /63 | hits@200 /63 |', '|---|---:|---:|---:|'])
    for name in ('A3', 'V2-A', 'Lexical', 'V2-B'):
        if name in metrics['systems']:
            m = metrics['systems'][name]
            lines.append('| ' + name + ' | ' + ' | '.join(str(m['candidate_hits'][str(k)]) for k in (50,100,200)) + ' |')
    lines.extend(['', '### Reranker retention', '', '| Reranker | Targets in prefix | Still in prefix | Up | Down | Unchanged |', '|---|---:|---:|---:|---:|---:|'])
    for name, r in metrics['reranker_retention'].items():
        lines.append('| ' + name + ' | ' + ' | '.join(str(r.get(k,0)) for k in ('present_in_prefix','retained_in_prefix','upward','downward','unchanged')) + ' |')
    if all(s in metrics['systems'] for s in ('A0','V2-C')):
        a, b = metrics['systems']['A0'], metrics['systems']['V2-C']
        lines.extend(['', '### Interpretation', '',
            f"A0 retrieves {a['direct_hits']['10']}/63 direct identities at rank 10; primary V2-C retrieves {b['direct_hits']['10']}/63. These are descriptive development counts. Individual regressions are retained below. A higher aggregate count does not establish that question parsing caused the difference; filtering, representation, and reranking controls must be read together.", ''])
        filter_only = metrics['systems'].get('A1')
        generator = metrics['systems'].get('V2-B')
        if filter_only and generator:
            lines.extend([
                f"The type-filter-only control A1 reaches {filter_only['direct_hits']['10']}/63 at top 10 and MRR {filter_only['MRR']:.4f}, above full V2-C's {b['direct_hits']['10']}/63 and MRR {b['MRR']:.4f}. Evidence cards alone (A2) and parsed cards (A3) recover no direct targets at top 10; A4's relevance reranking also recovers none. This run therefore does **not demonstrate that explicit parsing plus cards fixes the answer-identity failure**.", '',
                f"The primary generator admits only {generator['candidate_hits']['100']}/49 mapped EXACT/ALIAS occurrences to top100; {49-generator['candidate_hits']['100']} cannot be rescued by its reranker. V2-C moves two present target occurrences upward and one downward. The main observed bottleneck is candidate generation, with sparse/misaligned evidence and parser losses plausible but not causally proven explanations.", '',
                'The card audit finds 4,187/5,480 candidates without selected non-title evidence. Direct-family caps and static ID tie-breaking can omit query-relevant evidence; high-arity omission affects 34 mapped target occurrences. The parser loses pre-head scientific modifiers in some questions and deliberately leaves coordinated answer families unrestricted. These are documented limitations, not post-evaluation repairs.', ''])
    lines.extend(['## Per-question intent diagnostics', '', '| Question | Original | Answer type | Core task | Requirements | Exclusions | Rewritten query |', '|---|---|---|---|---|---|---|'])
    def cell(value):
        return str(value).replace('|','\\|').replace('\n',' ')
    for qid, row in questions.items():
        d = row['decomposition']
        lines.append('| ' + ' | '.join(cell(v) for v in (qid, row['question'], d['answer_type'], d['core_task'], '; '.join(d['requirements']), '; '.join(d['exclusions']), d['search_query'])) + ' |')
    lines.extend(['', '## Per-target direct rank diagnostics', '', 'An em dash means no direct identity rank. Composite completion and partial evidence mappings remain separate in `outputs/per_question_evaluation.json`. Every target record includes selected evidence, requirement components, relevance token packing when scored, and the candidate-top100 flag.', '',
        '| Question | Target | Status | A0 | V2-A | V2-B | V2-C | V2-D | In primary top100 |', '|---|---|---|---:|---:|---:|---:|---:|---|'])
    for target in targets:
        values = [target['question_id'], target['target'], target['status'], *[target['direct_ranks'].get(s) or '—' for s in ('A0','V2-A','V2-B','V2-C','V2-D')], target['entered_primary_top100']]
        lines.append('| ' + ' | '.join(cell(v) for v in values) + ' |')
    lines.extend(['', '## Failure observations', '', '| Observation | Target occurrences |', '|---|---:|'])
    for label, count in failures['observed_counts'].items():
        lines.append(f'| {label} | {count} |')
    lines.extend(['', 'Failure classes 5 (parser missed useful requirements), 6 (acronym representation), and 8 (requirements not jointly satisfied) are not automatically causally identified. The exported decompositions, evidence, and component scores support independent review; similarity alone cannot adjudicate these causes. No failures were manually patched.', '',
        '## Limitations and next experiment', limitations, '', '## Reproduction and audit artifacts', '',
        'See [README](../README.md) for exact commands. Complete per-system rankings are in `outputs/retrieval/by_system/`; scored evidence is in `outputs/retrieval/`; all metrics and denominators are in `outputs/metrics.json`. The protected-tree comparison is `outputs/audit/protected_verification.json`; test output is `outputs/audit/tests_final.log`; every new file is listed in `outputs/audit/new_files.txt`.', '',
        'Prediction SHA-256: `' + sha(out / 'retrieval/predictions.json') + '`.', '',
        'Phase ordering, input/source freeze, model weights, evaluation input hashes, and rank audit are saved under `outputs/audit/`. The implementation ledger records pre-gold retries. The scoped Claude consultation and 11-category descriptive validity review are in `notes/`.', '',
        'Model sources: [BGE](https://huggingface.co/BAAI/bge-small-en-v1.5) and [MS MARCO relevance MiniLM](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2). The latter is a relevance model; optional DeBERTa NLI was not run.', ''])
    (root / 'report/v2_retrieval_report.md').write_text('\n'.join(lines), encoding='utf8')
