"""Build current V2 report materials from verified, post-freeze evaluation only."""
import json
import sys
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'src'))
from tkh_abstraction_v2.io import read,write,sha
from tkh_abstraction_v2.freeze import verify_freeze


def main():
    pred=ROOT/'artifacts/prediction';ev=ROOT/'artifacts/evaluation';report=ROOT/'report'
    freeze=verify_freeze(ROOT,pred);m=read(ev/'metrics.json');costs=read(pred/'audit/efficiency.json')
    totals={k:sum(c.get(k,0) for c in costs.values()) for k in ('dense_comparisons','evidence_comparisons','cross_encoder_pairs','cross_encoder_inference_pairs','cross_encoder_wall_seconds','total_wall_seconds')}
    totals['bm25_document_term_lookups_exact']=0
    from tkh_abstraction_v2.ranking import tokens
    indices={int(p.stem):read(p) for p in (pred/'entity_index').glob('*.json')}
    for q,cost in costs.items():
        intent=read(pred/'query_understanding'/(q+'.json'))
        totals['bm25_document_term_lookups_exact']+=len(indices[intent['temporal_cutoff']]['entities'])*len(set(tokens(intent['original_question']+'\n'+intent['structured_query'])))
    totals['requirement_arms']={arm:{k:sum(c['requirements'][arm].get(k,0) for c in costs.values()) for k in (
        'requirement_dense_comparisons','requirement_cross_encoder_pairs','actual_requirement_cross_encoder_pairs','nli_pairs','actual_nli_pairs','actual_wall_seconds')} for arm in ('A7','A7_SingleEvidence')}
    totals['wall_note']='Completed predictor invocation with mixed cold/cached inference; interrupted pre-evaluation invocations are not included. Logical totals describe all pairs, actual totals only cache misses. Single CPU run descriptive time; logical comparisons are not production latency. NoNLI reuses primary relevance scores;100/200 controls reuse500-pool global scores.'
    totals['prediction_process']=read(ROOT/'artifacts/run_logs/prediction_exit.json')
    write(ev/'efficiency_report.json',totals)
    per=read(ev/'per_question_evaluation.json')
    compact={q:dict(question=d['question'],targets=[dict(target=t['target'],status=t['status'],ranks={a:t['direct_ranks'][a] for a in ('A5','A6','A7','A7_NoNLI','A7_SingleEvidence')},failure=t['failure']) for t in d['targets']]) for q,d in per.items()}
    handoff=dict(label='DEVELOPMENT; no training or gold fitting',metrics=m,efficiency=totals,
        failure_classes=read(ev/'audit/failure_classes.json'),index={str(c):dict(entities=len(i['entities']),evidence_items=len(i['items']),entity_evidence_links=sum(len(rs) for e in i['entities'].values() for rs in e['evidence'].values()),role_arcs=sum(len(v) for v in i['arcs'].values()),rejected_edges=len(i['rejected_edges']),entities_without_candidate_specific_evidence=sum(not any(r['candidate_specific'] for rs in e['evidence'].values() for r in rs) for e in i['entities'].values())) for c,i in indices.items()},
        per_question=compact,prediction_sha256=freeze['predictions_sha256'],
        access_audit=read(pred/'audit/prediction_accesses.json'))
    samples=[]
    for path in sorted((pred/'retrieval/requirement_ranking/A7').glob('*.json')):
        multi=read(path);single=read(pred/'retrieval/requirement_ranking/A7_SingleEvidence'/path.name)
        for eid,row in multi.items():
            for c,one in zip(row['requirements'],single[eid]['requirements']):
                if c['status']!=one['status'] and len(samples)<6:
                    samples.append(dict(question_id=path.stem,entity_id=eid,name=row['name'],component=c['text'],kind=c['kind'],
                        top3_status=c['status'],top1_status=one['status'],top3_probabilities=c['probabilities'],top1_probabilities=one['probabilities'],
                        selected_evidence=[dict(id=e['evidence_id'],text=e['text']) for e in c['evidence']],
                        actual_nli_included=(c.get('nli_input') or {}).get('included',[])))
    handoff['requirement_status_examples']=samples
    # Large dependency read list remains in the audit; the author only needs counts.
    audit=handoff['access_audit'];audit['project_read_count']=len(audit.pop('project_reads'))
    write(ROOT/'notes/requirement_design/report_handoff.json',handoff)
    lines=['# V2: entity-centric, requirement-aware retrieval','',
        'Frozen pretrained models; no benchmark training or weight/threshold fitting. Questions are prediction inputs; target annotations are evaluation-only. These are development diagnostics.','',
        '[Detailed method/results PDF](v2_entity_retrieval_report.pdf) | [LaTeX](v2_entity_retrieval_report.tex)','',
        '| Arm | Candidate@100 /49 | Candidate@500 /49 | Union /49 | R@1 | R@5 | R@10 | R@25 | R@50 | R@100 | MRR |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,a in m['systems'].items():
        vals=[name,a['candidate_hits']['100'],a['candidate_hits']['500'],a['candidate_union_hits'],*[f"{a['recall'][str(k)]:.4f}" for k in (1,5,10,25,50,100)],f"{a['MRR']:.4f}"]
        lines.append('| '+' | '.join(map(str,vals))+' |')
    lines+=['','Final recall denominator is all target occurrences; candidate denominator is directly mapped occurrences. A6/A7 candidate union refers to the actual500-prefix, while A5 reports the full retrieval union. Tails remain in final rankings.','',
        '## Conditional movement','', '```json',json.dumps(m['A6_to_A7'],indent=2),'```','',
        '## Requirement diagnostics','', '```json',json.dumps(m['requirement_diagnostics'],indent=2),'```','',
        '## Failure categories','', '```json',json.dumps(handoff['failure_classes'],indent=2),'```','',
        '## Efficiency','', '```json',json.dumps(totals,indent=2),'```','',
        'Prediction SHA-256: `'+freeze['predictions_sha256']+'`.','',
        'Full candidate traces are in `artifacts/prediction/retrieval/requirement_ranking/`. Scored target traces are evaluation-only in `artifacts/evaluation/failure_traces.json`. No model or ranking policy was changed after evaluation.','']
    report.mkdir(exist_ok=True);(report/'v2_retrieval_report.md').write_text('\n'.join(lines),encoding='utf8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    stages=['A0','A1','A2','A3','A4','A5','A6','A7','A7_NoNLI','A7_SingleEvidence']
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    axes[0].bar(stages,[100*m['systems'][s]['candidate_recall']['500'] for s in stages],color='#0072B2')
    axes[1].bar(stages,[100*m['systems'][s]['recall']['10'] for s in stages],color='#D55E00')
    axes[0].set(title='Candidate Recall@500',ylabel='Percent of mapped target occurrences')
    axes[1].set(title='Final Recall@10',ylabel='Percent of all target occurrences')
    for ax in axes:ax.set_ylim(0,100);ax.tick_params(axis='x',rotation=55);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.savefig(report/'stage_metrics.pdf');fig.savefig(report/'stage_metrics.png',dpi=160);plt.close(fig)
    print('Current V2 materials generated; no historical evaluation used')


if __name__=='__main__':main()
