"""Post-evaluation tables, plots and minimal verified handoff for Claude."""
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'src'))
os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
from tkh_abstraction_v2.io import read,write,sha
from tkh_abstraction_v2.ranking import tokens


def main():
    out=ROOT/'outputs'; report=ROOT/'report'; report.mkdir(exist_ok=True)
    m=read(out/'metrics.json'); efficiency=read(out/'audit/efficiency.json')
    predictions=read(out/'retrieval/predictions.json')
    questions=read(out/'per_question_evaluation.json')
    indices={int(p.stem):read(p) for p in (out/'entity_index').glob('*.json')}
    statuses=Counter()
    verification_candidates=0
    for p in (out/'retrieval/requirement_verification').glob('*.json'):
        records=read(p);verification_candidates+=len(records)
        statuses.update(c['status'] for r in records.values() for c in r['components'])
    events=[json.loads(s) for s in (out/'audit/events.jsonl').read_text().splitlines()]
    start=next(e['time'] for e in events if e['event']=='prediction_started')
    end=next(e['time'] for e in events if e['event']=='predictions_frozen')
    phase_seconds=(datetime.fromisoformat(end)-datetime.fromisoformat(start)).total_seconds()
    exact_bm25=0
    for q,pred in predictions.items():
        intent=read(out/'query_understanding'/(q+'.json'))
        exact_bm25+=len(indices[pred['cutoff']]['entities'])*len(set(tokens(intent['original_question']+'\n'+intent['structured_query'])))
    totals=dict(prediction_phase_wall_seconds=phase_seconds,query_wall_seconds=sum(r['total_wall_seconds'] for r in efficiency.values()),
        dense_comparisons=sum(r['dense_comparisons'] for r in efficiency.values()),
        evidence_generation_comparisons=sum(r['evidence_comparisons'] for r in efficiency.values()),
        bm25_document_term_lookups_exact=exact_bm25,
        cross_encoder_pairs=sum(r.get('cross_encoder_pairs',0) for r in efficiency.values()),
        cross_encoder_actual_inference_pairs=sum(r.get('cross_encoder_inference_pairs',0) for r in efficiency.values()),
        verification_candidates=verification_candidates,
        verification_components=sum(r.get('verification',{}).get('requirement_components',0) for r in efficiency.values()),
        nli_pairs=sum(r.get('verification',{}).get('actual_nli_pairs',0) for r in efficiency.values()),
        requirement_evidence_comparisons=sum(r.get('verification',{}).get('evidence_comparisons',0) for r in efficiency.values()),
        verification_status_counts=dict(statuses),
        prefix_timing_estimates={str(k):sum(r.get('prefix_timing_estimates',{}).get(str(k),0) for r in efficiency.values()) for k in (100,200,500)},
        timing_note='CPU descriptive wall times; prefix controls share scores from500 and are not separately timed; counters are logical work, not production latency.')
    write(out/'efficiency_report.json',totals)
    overview={str(year):dict(entities=len(index['entities']),evidence_items=len(index['items']),
        entity_evidence_links=sum(len(rs) for e in index['entities'].values() for rs in e['evidence'].values()),
        inferred_role_arcs=sum(len(v) for v in index['arcs'].values()),rejected_edges=len(index['rejected_edges']),
        entities_without_candidate_specific_evidence=sum(not any(r['candidate_specific'] for rs in e['evidence'].values() for r in rs) for e in index['entities'].values())) for year,index in indices.items()}
    old=read(ROOT/'archive/flat_20261001/outputs/metrics.json')
    old_summary={s:old['systems'][s] for s in ('A0','A1','V2-B','V2-C')}
    handoff=dict(label='DEVELOPMENT; frozen evaluation; no held-out claim',date='2026-10-02',metrics=m,efficiency=totals,index=overview,
        previous_V2=old_summary,failure_classes=read(out/'audit/failure_classes.json'),prediction_sha256=sha(out/'retrieval/predictions.json'),
        per_question={q:dict(question=r['question'],first_direct_ranks={s:r['systems'][s]['first_direct_rank'] for s in ('A0','A1','A5','A6','A7') if s in r['systems']},
            target_ranks=[dict(target=t['target'],status=t['status'],ranks={s:t['direct_ranks'][s] for s in ('A0','A1','A5','A6','A7') if s in t['direct_ranks']},failure=t['failure']['primary']) for t in r['targets']]) for q,r in questions.items()})
    write(ROOT/'notes/entity_redesign/report_handoff.json',handoff)
    lines=['# V2 entity-centric retrieval — DEVELOPMENT results','',
        'The detailed LaTeX report is authored by Claude through the user-requested Ultralight workflow. This summary is generated directly from frozen evaluation artifacts.','',
        'Direct recall counts EXACT/ALIAS target occurrences only. All-target denominator: '+str(m['denominators']['target_occurrences'])+'; mapped denominator: '+str(m['denominators']['exact_alias_occurrences'])+'. All results use14 Type-A questions;18 questions have predictions.','',
        '| Stage | Candidate@100 /mapped | Candidate@500 /mapped | Union /mapped | Avg pool | R@1 | R@5 | R@10 | R@25 | R@100 | MRR |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,ms in m['systems'].items():
        values=[name,ms['candidate_hits']['100'],ms['candidate_hits']['500'],ms['candidate_union_hits'],f"{ms['average_candidate_set_size']:.1f}",
            *[f"{ms['recall'][str(k)]:.3f}" for k in (1,5,10,25,100)],f"{ms['MRR']:.3f}"]
        lines.append('| '+' | '.join(map(str,values))+' |')
    lines+=['','Candidate metrics for A6/A7 use the actual pre-rerank input pool, not the reranked output. Larger union recovery is reported separately from its top500 ceiling. Conditional rank averages exclude misses; the JSON reports missing counts and a fixed missing-rank penalty.','',
        '## Conditional reranking','',json.dumps(m['conditional_reranking'],indent=2,ensure_ascii=False),'',
        '## Efficiency','', '```json',json.dumps(totals,indent=2),'```','',
        '## Failures','', '```json',json.dumps(handoff['failure_classes'],indent=2),'```','',
        '## Reproducibility and interpretation','',
        'See README.md for commands; config.json for every fixed parameter; outputs/failure_traces.json for all target traces; outputs/evaluation_only/positive_controls.json for isolated oracle tests. The previous experiment is retained under archive/flat_20261001. The experiment does not route through or modify the hierarchy.','',
        'Prediction SHA-256: `'+handoff['prediction_sha256']+'`.','',
        'V1 was already dirty at task start. The user approved preserving it; the final audit must show zero additional V1 changes, including an unchanged starting git diff. No GitHub operations are performed.','']
    (report/'v2_retrieval_report.md').write_text('\n'.join(lines),encoding='utf8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    stages=[s for s in ('A0','A1','A2','A3','A4','A5','A6','A7') if s in m['systems']]
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    axs[0].bar(stages,[100*m['systems'][s]['candidate_recall']['500'] for s in stages],color='#0072B2')
    axs[1].bar(stages,[100*m['systems'][s]['recall']['10'] for s in stages],color='#D55E00')
    axs[0].set(title='Candidate recovery at500',ylabel='Percent of mapped EXACT/ALIAS occurrences')
    axs[1].set(title='Final direct Recall@10',ylabel='Percent of all target occurrences')
    for ax in axs:ax.set_ylim(0,100);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.savefig(report/'stage_metrics.pdf');fig.savefig(report/'stage_metrics.png',dpi=160);plt.close(fig)
    print('Wrote report summary, figure, efficiency totals, and Claude handoff')


if __name__=='__main__':main()
