"""Phase 0: audit actual existing rankings before contextual retrieval changes."""
from pathlib import Path
import unicodedata,re,hashlib
import numpy as np
from tkh_abstraction.io import read_json,read_csv,write_json,digest
from tkh_abstraction.snapshots import snapshot
from tkh_abstraction.target_mapping import normalize,resolve_target,ENTITY_TYPES,route_question,mapping_summary,map_benchmark
from tkh_abstraction.evaluation.extrinsic import map_methods

data=read_json('data/data/tkh_collection10.json'); gt=read_json('data/data/ground_truth.json'); qs=read_csv('data/data/questions.csv')
baseline=read_json('artifacts/baseline_v2/metrics.json'); current=read_json('artifacts/full/metrics.json')
assert baseline['extrinsic_v2']['records']==current['extrinsic_v2']['records'],'Recomputed v2 benchmark changed before modifications'
names=sorted({s for g in gt.values() for s in g.get('expected_methods',[])})
basic=lambda t:re.sub(r'\s+',' ',unicodedata.normalize('NFKC',t).casefold()).strip()
reconciliation=dict(question_counts={t:sum(q['type']==t for q in qs) for t in ['A','B']},unique_targets=len(names),
    target_instances=sum(len(g.get('expected_methods',[])) for g in gt.values()),recomputed_v2_records_identical=True,
    full_export_basic_exact=sum(any(basic(n['surface_form'])==basic(s) for n in data['nodes']) for s in names),
    full_export_v2_normalized_exact=sum(any(normalize(n['surface_form'])==normalize(s) for n in data['nodes']) for s in names),
    duplicate_handling='Mapping retains every matching node ID but counts one named target; old retrieval ranks nodes without entity deduplication.',
    normalization=dict(basic='NFKC, casefold, whitespace only',v1='NFKC, casefold, Unicode dashes; punctuation except plus/hyphen becomes whitespace',v2='v1 plus preservation of parentheses; no singularization or composite decomposition'),cutoffs={})
audits=[]
for year in [2025,2026]:
    snap=snapshot(data,year); nodes={n['id']:n for n in snap['nodes']}; old=map_methods(gt,list(nodes.values())); mapping=map_benchmark(gt,list(nodes.values()),year)
    records=read_json(f'artifacts/evaluation/retrieval/temporal_{year}.json'); flat={r['metrics']['question_id']:r for r in records if r['metrics']['method']=='flat_exhaustive'}
    reconciliation['cutoffs'][str(year)]=dict(v1_method_only=sum(bool(r['node_ids']) for r in old.values()),
        v2_typed=mapping_summary(mapping),basic_exact_all_types=sum(any(basic(n['surface_form'])==basic(s) for n in nodes.values()) for s in names),
        v2_exact_all_types=sum(any(normalize(n['surface_form'])==normalize(s) for n in nodes.values()) for s in names),
        resolver_types=sorted(ENTITY_TYPES|{'claim'}),retrieval_types='Question-dependent route; recorded per question',
        question_evaluability=current['extrinsic_v2']['question_evaluability'][str(year)])
    for q in qs:
        if q['type']!='A': continue
        result=flat[q['question_id']]['entities']; rank={x:i+1 for i,x in enumerate(result['ranked_ids'])}; route=route_question(q['question'],q['type'])
        for label in gt[q['question_id']]['expected_methods']:
            resolved=resolve_target(label,list(nodes.values())); full=resolve_target(label,data['nodes']); ids=resolved['node_ids']; detail=[]
            for x in sorted(set(ids)|set(full['node_ids'])):
                n=next(n for n in data['nodes'] if n['id']==x); visible=x in nodes; eligible=visible and n['type'] in route['types']
                detail.append(dict(node_id=x,node_type=n['type'],surface_form=n['surface_form'],resolved_at_cutoff=x in ids,
                    present_in_snapshot=visible,eligible_after_type_routing=eligible,present_in_flat_candidates=x in rank,
                    surface_score=result['leaf_scores'].get(x),surface_rank=rank.get(x),
                    exclusion_reason='not_visible_at_cutoff' if not visible else 'node_type_excluded_by_route' if not eligible else None))
            available=[r for r in detail if r['resolved_at_cutoff'] and r['present_in_flat_candidates']]
            audits.append(dict(question_id=q['question_id'],question=q['question'],cutoff=year,gold_label=label,resolution_status=resolved['status'],
                resolved_node_ids=ids,route=route,node_diagnostics=detail,present_in_snapshot=bool(ids),eligible_after_type_routing=bool(available),
                present_in_flat_candidates=bool(available),best_surface_score=max((r['surface_score'] for r in available),default=None),
                best_surface_rank=min((r['surface_rank'] for r in available),default=None),
                exclusion_reason='no_accepted_target_mapping' if not ids else 'all_resolved_nodes_excluded_by_route' if not available else None))
    rows=[r for r in audits if r['cutoff']==year]; resolved=[r for r in rows if r['resolved_node_ids']]; candidates=[r for r in resolved if r['present_in_flat_candidates']]
    ranks=[r['best_surface_rank'] for r in candidates]
    reconciliation['cutoffs'][str(year)]['candidate_audit']=dict(target_instances=len(rows),resolved_instances=len(resolved),
        candidate_instances=len(candidates),route_excluded_instances=len(resolved)-len(candidates),
        eligible_gold_nodes=sum(r['resolved_at_cutoff'] and r['present_in_flat_candidates'] for x in rows for r in x['node_diagnostics']),
        excluded_gold_nodes=sum(r['resolved_at_cutoff'] and not r['present_in_flat_candidates'] for x in rows for r in x['node_diagnostics']),
        best_rank_quantiles=dict(zip(['min','q25','median','q75','max'],np.quantile(ranks,[0,.25,.5,.75,1]).tolist())),
        at_10=sum(r<=10 for r in ranks),at_50=sum(r<=50 for r in ranks),at_100=sum(r<=100 for r in ranks))
reconciliation['historical_numbers']=dict(v1_21='Method-only resolver at 2025; historical 21/50 is reproducible.',
    v2_33='Typed resolver at conservative 2025 after unsafe future-topic containment was removed.',
    preliminary_37='Earlier annual-2026 result before removal of one unsafe future-topic containment; retained in notes/improvement/initial_v2_run.log and consultation history, not final current coverage.',
    literal_34='Compare full_export_basic_exact with the stricter temporal and normalization-specific counts; this audit recomputes rather than assumes 34.',
    twelve_vs_fourteen='14 Type A questions; Q5/Q11 have no accepted v2 mappings and null conditional recall. All remain in records and raw denominators.')
write_json('artifacts/evaluation/baseline_reconciliation.json',reconciliation)
write_json('artifacts/evaluation/gold_candidate_audit.json',dict(protocol='Existing v2 BGE cosine of type-prefixed surface text, exhaustive rank, no context. Gold used only for diagnostics.',records=audits))
write_json('artifacts/baseline_v2/recomputed_verification.json',dict(status='passed',record_count=len(current['extrinsic_v2']['records']),records_hash=digest(current['extrinsic_v2']['records'])))
print({y:x['candidate_audit'] for y,x in reconciliation['cutoffs'].items()})
print('Literal full-export matches:',reconciliation['full_export_basic_exact'],reconciliation['full_export_v2_normalized_exact'])
