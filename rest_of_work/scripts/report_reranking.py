"""Audit and report the frozen-candidate experiment; never runs or selects a ranker."""
from pathlib import Path
from collections import Counter
import json
import xml.etree.ElementTree as ET
import numpy as np
from tkh_abstraction.io import read_json,write_json,digest,read_csv
from tkh_abstraction.v2_pipeline import sha,verify_reuse
from tkh_abstraction.snapshots import snapshot

root=Path('artifacts/evaluation'); out=root/'reranking'
bundle=read_json(root/'reranker_results.json'); protocol=bundle['protocol']; results=bundle['results']
historical=read_json(root/'hypergraph_diffusion_results.json')
targets=read_json(root/'reranker_per_target.json'); evidence=read_json(root/'candidate_evidence.json')
failure=read_json(root/'reranker_failure_analysis.json'); oracle=read_json(root/'reranker_oracle.json')
metrics=read_json('artifacts/full/metrics.json'); cfg=protocol['config']
data=read_json(Path(cfg['data_dir'])/'tkh_collection10.json'); qs=read_csv(Path(cfg['data_dir'])/'questions.csv')
gt=read_json(Path(cfg['data_dir'])/'ground_truth.json')
assert digest(data)==protocol['data_hash'] and digest(qs)==protocol['questions_hash'] and digest(gt)==protocol['ground_truth_hash']
for p,h in protocol['historical_hashes'].items(): assert sha(p)==h,('Historical artifact changed',p)
for p,h in protocol['source_hashes'].items(): assert sha(p)==h,('Source mismatch; rerun experiment',p)
for key,h in protocol['historical_metrics_hashes'].items(): assert digest(metrics[key])==h,key
verify_reuse(protocol['context_config'],data)
assert sha('report/hypergraph_diffusion.md')==sha('artifacts/baseline_diffusion/hypergraph_diffusion.md')
suite=ET.parse(root/'reranker_tests.xml').getroot().find('testsuite')
assert int(suite.attrib['failures'])==int(suite.attrib['errors'])==int(suite.attrib['skipped'])==0
lookup={(r['cutoff'],r['stage']):r for r in results}
old={(r['cutoff'],r['stage']):r for r in historical['results']}
qids={q['question_id'] for q in qs if q['type']=='A'}; assert len(qids)==14
evidence_lookup={(q['cutoff'],q['question_id'],b['entity_id']):b for q in evidence for b in q['candidates']}
for year in cfg['reranker_cutoffs']:
    baseline={r['question_id']:r for r in lookup[year,'RR0']['records']}
    original={r['question_id']:r for r in old[year,'H2']['records']}
    for qid,row in baseline.items():
        assert row['entity_metrics']==original[qid]['entity_metrics'] and row['returned']==original[qid]['returned']
    node_years={n['id']:n['first_seen_year'] for n in snapshot(data,year)['nodes']}
    for result in [r for r in results if r['cutoff']==year]:
        assert {r['question_id'] for r in result['records']}==qids
        assert result['summary']['questions']==14 and result['summary']['evaluable']==12
        assert result['summary']['at_k']['10']['total_target_instances']==63
        for row in result['records']:
            qid=row['question_id']; metric=row['entity_metrics']; prior=baseline[qid]
            assert metric['status']==('not_evaluable' if qid in {'Q5','Q11'} else 'evaluable')
            if qid in {'Q5','Q11'}: assert metric['at_k']['10']['recall'] is None and metric['evaluability_reason']
            assert metric['candidate_recall_100']==prior['entity_metrics']['candidate_recall_100']
            assert metric['candidate_at_k']==prior['entity_metrics']['candidate_at_k']
            k=row['candidate_k']
            if k is None: continue
            assert row['candidate_preserved'] and {r['entity_id'] for r in row['returned'][:k]}=={r['entity_id'] for r in prior['returned'][:k]}
            assert row['returned'][k:]==prior['returned'][k:]
            for detail in row['score_details'].values(): assert abs(detail['score']-sum(detail['score_components'].values()))<1e-6
    for q in [q for q in evidence if q['cutoff']==year]:
        for b in q['candidates']:
            assert set(b['node_ids'])<=node_years.keys()
            for f,items in b['fields'].items():
                assert len(items)<=cfg['reranker_field_caps'][f]
                for item in items: assert item['node_id'] in node_years and item['first_seen_year']<=year

def mean(xs):
    xs=[x for x in xs if x is not None]; return float(np.mean(xs)) if xs else None

def fmt(v): return '—' if v is None else f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v)
def cell(v): return fmt(v).replace('|','/').replace('\n',' ')
def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(cell(v) for v in row)+' |\n' for row in rows)+'\n'

def result_row(year,stage): return lookup.get((year,stage),old.get((year,stage)))
def records(year,stage): return {r['question_id']:r for r in result_row(year,stage)['records']}
def rank(row,node_ids): return next((i+1 for i,r in enumerate(row['returned']) if set(r['node_ids'])&set(node_ids)),None)

movements=[]; counts=[]; availability=[]; oracle_summary=[]
for year in cfg['reranker_cutoffs']:
    tt=[t for t in targets if t['cutoff']==year]
    for stage in ['RR1-100','RR1-200','RR3-100','RR3-200']:
        counts.append(dict(cutoff=year,stage=stage,counts=dict(Counter(t['rerankings'][stage]['category'] for t in tt))))
        bins=[('R3 top10','R3_rank',lambda t:t['R3_rank'] is not None and t['R3_rank']<=10),
              ('H2 top10','H2_rank',lambda t:t['H2_rank'] is not None and t['H2_rank']<=10)]
        for lo,hi in [(11,50),(51,100),(101,200)]: bins.append((f'H2 {lo}-{hi}','H2_rank',lambda t,lo=lo,hi=hi:t['H2_rank'] is not None and lo<=t['H2_rank']<=hi))
        for label,key,predicate in bins:
            xs=[t for t in tt if predicate(t)]; diff=[t[key]-t['rerankings'][stage]['rank'] for t in xs]
            movements.append(dict(cutoff=year,stage=stage,bin=label,reference=key,count=len(xs),improved=sum(v>0 for v in diff),unchanged=sum(v==0 for v in diff),worsened=sum(v<0 for v in diff),top10=sum(t['rerankings'][stage]['rank']<=10 for t in xs)))
    for gold in [True,False]:
        rows=[d for d in failure['evidence_presence'] if d['cutoff']==year and d['gold']==gold]
        availability.append(dict(cutoff=year,gold=gold,candidate_question_pairs=len(rows),specific_present=sum(r['specific_evidence_count']>0 for r in rows),
            specific_fraction=mean([r['specific_evidence_count']>0 for r in rows]),fields={f:dict(selected_mean=mean([r['counts'][f] for r in rows]),
                available_mean=mean([r['available_counts'][f] for r in rows]),presence_fraction=mean([r['counts'][f]>0 for r in rows]),
                best_similarity_mean_when_present=mean([r['best_similarity_by_field'][f] for r in rows]),
                selected_quantiles=np.quantile([r['counts'][f] for r in rows],[0,.25,.5,.75,1]).tolist()) for f in cfg['reranker_field_caps']}))
    for k in [100,200]:
        rows=[r for r in oracle['records'] if r['cutoff']==year and r['candidate_k']==k]
        total=sum(r['canonical_targets'] for r in rows); found=sum(r['oracle_found_at10'] for r in rows)
        oracle_summary.append(dict(cutoff=year,candidate_k=k,found=found,total=total,micro_recall=found/total,
            macro_recall=mean([r['oracle_recall_at10'] for r in rows]),raw_found=sum(r['oracle_raw_found'] for r in rows)))
failure.update(category_counts=counts,rank_movement_bins=movements,evidence_distribution=availability)
attribution=[]
for year in cfg['reranker_cutoffs']:
    for gold in [True,False]:
        selected=[evidence_lookup[year,d['question_id'],d['entity_id']] for d in failure['evidence_presence'] if d['cutoff']==year and d['gold']==gold]
        kinds=Counter(item.get('support_kind',item['kind']) for b in selected for items in b['fields'].values() for item in items)
        attribution.append(dict(cutoff=year,gold=gold,selected_support_kinds=dict(kinds),
            path_sources_via_generic_bridge=sum(item.get('via_generic_bridge',False) for b in selected for item in b['fields']['path_evidence']),
            diagnostic_paths_via_generic_bridge=sum(p['via_generic_bridge'] for b in selected for p in b['selected_paths'])))
failure['attribution_diagnostics']=attribution
write_json(root/'reranker_failure_analysis.json',failure); oracle['summary']=oracle_summary; write_json(root/'reranker_oracle.json',oracle)

text='# Evidence-aware candidate reranking\n\n'
text+='The fixed evidence scorer improves conservative 2025 top-ten recovery from H2’s 3/47 to 6/47 canonical target instances (R3: 4/47). It brings three H2 rank 11–100 targets into the top ten and preserves all three existing H2 hits. This is a modest gain: 41/47 still fail at ten, and median gold rank worsens from 66 to 89 at K100 and 104 at K200. Enlarging the candidate pool provides no additional top-ten hits. The evidence hypothesis is only partially supported; final answer quality does not yet justify hierarchy approximation as the next priority.\n\n'
text+='These are development diagnostics on a repeatedly used benchmark, not held-out estimates. No answer-trained weights, parameter sweep, new aliases, remote relevance API, hierarchy change, or change to Type B retrieval was made. The previous [diffusion report](hypergraph_diffusion.md) and every historical metric section remain unchanged.\n\n'
text+='## Frozen protocol and evidence\n\n'
text+='`FrozenH2Candidates` composes the existing `ContextIndex`, original extractive query view, normalized BGE embeddings, native `ThetaOperator`, unchanged `MentionIndex`, entity grouping and RRF. H2 remains alpha=.85, M=100, uniform native relation weights, original convergence settings and RRF60. Existing questions replay archived full node vectors and reconstruct the complete ranking; all 28 top-200 dictionaries and per-question metric dictionaries match historical H2 exactly before scoring. New questions execute the same H2 functions. Only each frozen top-K prefix is reordered; the entire tail retains its original order. Candidate recall always uses the original H2 prefix. H3 remains an archived comparison.\n\n'
text+='`CandidateEvidenceIndex.build_candidate_evidence` uses graph-only pools, then ranks evidence by tier, specificity and original-question cosine, with stable ID tie breaks. Caps are direct claims 3, safe explicit-name claims 3, task/problem 3, technical 2, dataset/metric 2, titles 2 and paths 2. Identical normalized text cannot occupy multiple selected fields. Direct relations are limited to claims/addresses/solves/uses_technique/uses_component/evaluated_on/presents with arity ≤16. Direct items are labelled low-arity (≤3) or co-membership only; undirected incidence is not a subject–predicate assertion. Explicit names use the frozen safe-boundary MentionIndex. Only titles from presenting or provenance publications are included; whole-article claim expansion is excluded.\n\n'
text+='Genericness combines the positive-degree 90th percentile, short labels (≤3 tokens), broad node type or ≥3 source articles, and an identifier exemption. It filters context nodes, never removes candidate entities. Up to two approximate high-product paths from the twenty highest positive semantic seeds are kept within three hops. The traversal reuses native Theta transition factors, bounds retained paths per vertex, and is not exact diffusion attribution. Generic bridges remain in diagnostic paths; only non-generic claim/task/problem source text can enter semantic fields. Final path destinations are labelled target entities. Named but semantically broad nodes can escape this heuristic. `specific_evidence_count` means selected direct/name-linked non-title evidence is present, not that it entails an answer.\n\n'
text+='RR1 reuses BGE-small-en-v1.5 revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` and H2’s cached atomic question/text cosines. Negative cosines are clamped to zero; each field uses its maximum. Fixed weights are name .10, direct claims .20, explicit mentions .20, task/problem .20, technical .10, dataset .10, article .05, path source .05. These are new predeclared reranker defaults, not inherited R3 weights or fitted values.\n\n'
text+='`score = (sum_available(weight × field_cosine) / sum_available(weight)) × (0.5 + 0.5 × sqrt(sum_available(weight) / sum_all(weight)))`. Missing fields are excluded from the semantic denominator, while coverage mildly discounts sparse evidence. The discount can still disadvantage sparse graph extractions; it does not repair them. Metadata H2/R3 rank and score never enter semantic text. RR3 fuses H2 with RR1 using the existing RRF60 and average ranks for exact-score ties. “Best available” means RR1 because no eligible RR2 is local, not retrospective score-based selection.\n\n'
text+='RR2 is **unavailable**. The local cache contains BGE, MiniLM and a three-way NLI DeBERTa classifier; none is a relevance cross-encoder. `CandidateReranker` and `LocalCrossEncoderReranker` supply the optional interface, explicit local-only loaders and whole-block token packing at a 480-token pair budget, with included/omitted blocks and token counts. No RR2 metric is fabricated and no model is downloaded. The CE path is covered by loader/packing safeguards, not a real-model relevance experiment.\n\n'
text+='All 14 Type A questions appear in every stage and cutoff; Q5 (Bead-mapping, GNN+GPP) and Q11 (CE→MC→NN→PF, hybrid frameworks) have no canonical named-entity evaluation units, so canonical recall is explicitly null while their supplied labels remain in the raw /63 denominator. Canonical counts are 47 instances/34 unique labels at 2025 and 50/37 at 2026. The unchanged resolver records individual composite, evidence-backed, ambiguous or unresolved reasons; noncanonical targets are not silently dropped.\n\n'

headers=['Variant','K','Hit1','Hit5','Hit10','Hit20','MacroR5','MacroR10','MacroR20','MicroR5','MicroR10','MicroR20','UniqueR10','Raw /63','MRR','R-Prec','R@R','Median rank','Mean rank','RR calls','RR seconds','Total seconds*']
for year in cfg['reranker_cutoffs']:
    text+=f'## Quality and cost: {year}\n\n'; rows=[]
    for stage in ['R3','H3','RR0','RR1-100','RR1-200','RR3-100','RR3-200']:
        s=result_row(year,stage)['summary']; a=s['at_k']; cost=s.get('reranking_cost',{})
        raw=sum(r['entity_metrics']['at_k']['10']['raw_found'] for r in result_row(year,stage)['records'])
        rows.append([stage,stage.split('-')[-1] if '-' in stage else '—']+[a[str(k)]['macro_hit'] for k in [1,5,10,20]]+
            [a[str(k)]['macro_recall'] for k in [5,10,20]]+[a[str(k)]['micro_recall'] for k in [5,10,20]]+
            [a['10']['unique_target_recall'],f'{raw}/63',s['mrr'],s['r_precision'],s['recall_at_r'],s['rank_quantiles']['median'],s['mean_target_rank'],
             cost.get('reranker_comparisons',0),cost.get('reranker_seconds',0),cost.get('total_query_seconds',s['cost']['runtime_seconds'])])
    text+=table(headers,rows)
    text+='RR2-100 and RR2-200: unavailable; no quality/cost values. Hit and macro recall average the 12 evaluable questions; costs average all 14. Unique recall is the union of canonical target names recovered across associated questions. R-Precision and Recall@R coincide for these distinct grouped answer units.\n\n'
    text+=table(['Variant','Frozen C100 (macro)','Frozen C200 (macro)','Candidate preservation','Evidence selection seconds','Unique cached scores consulted','New evidence dots'],[
        [stage,lookup[year,stage]['summary']['candidate_recall']['100'],lookup[year,stage]['summary']['candidate_recall']['200'],lookup[year,stage]['summary']['candidate_preservation_rate'],
         lookup[year,stage]['summary']['reranking_cost']['evidence_selection_seconds'],lookup[year,stage]['summary']['reranking_cost']['evidence_selection_scores_reused'],0]
        for stage in ['RR0','RR1-100','RR1-200','RR3-100','RR3-200']])
text+='*Total seconds is a component-sum estimate: measured historical H2 search time + current path/evidence construction + current reranking, not a fresh end-to-end H2 execution. Cold model loading, snapshot/index building, embedding encoding and evaluation/export are excluded. Path traversal is shared per query; K100 is charged shared traversal plus its first 100 bundle builds, K200 all 200. Scalar reranker calls double, but cached dot products do not. Existing H2 search comparisons remain in RR0 cost; new reranker comparisons are separate. Query encoding/replay wall time is not substituted for historical search timing. RR3 cost includes scoring and fusion. Timings are indicative single-run measurements; part of the final run overlapped test verification, so these are not controlled latency benchmarks.\n\n'

text+='## Candidate-pool oracle\n\n'
text+='The evaluation-only oracle maximizes distinct canonical coverage in ten candidate slots using exact finite subset coverage. Alias labels merge before optimization; raw label-instance recovery is recorded separately. It never feeds expected labels or gold membership to evidence/scoring. For distinct answer groups and >10 golds, ten slots cannot recover all targets. Missing candidates and top-ten capacity both limit this diagnostic.\n\n'
text+=table(['Cutoff','K','Oracle found','Micro oracle R10','Macro oracle R10','RR1 R10','Gap (micro)','RR3 R10'],[
    [o['cutoff'],o['candidate_k'],f"{o['found']}/{o['total']}",o['micro_recall'],o['macro_recall'],
     lookup[o['cutoff'],f"RR1-{o['candidate_k']}"]['summary']['at_k']['10']['micro_recall'],
     o['micro_recall']-lookup[o['cutoff'],f"RR1-{o['candidate_k']}"]['summary']['at_k']['10']['micro_recall'],
     lookup[o['cutoff'],f"RR3-{o['candidate_k']}"]['summary']['at_k']['10']['micro_recall']] for o in oracle_summary])
text+=table(['Cutoff','Question','Canonical targets','Present100','Oracle100','Present200','Oracle200'],[
    [year,qid,next(o for o in oracle['records'] if o['cutoff']==year and o['question_id']==qid)['canonical_targets']]+
    [v for k in [100,200] for o in oracle['records'] if o['cutoff']==year and o['question_id']==qid and o['candidate_k']==k for v in [o['canonical_candidates'],o['oracle_recall_at10']]]
    for year in cfg['reranker_cutoffs'] for qid in sorted(qids,key=lambda q:int(q[1:]))])

text+='## Evidence availability and failure categories\n\n'
text+=table(['Cutoff','Group','Candidate/question pairs','Specific evidence present','Fraction'],[
    [a['cutoff'],'gold' if a['gold'] else 'not in answer key',a['candidate_question_pairs'],a['specific_present'],a['specific_fraction']] for a in availability])
text+='Counts above are candidate-group/question pairs, not raw labels or globally unique entities. At 2025, 41/43 in-pool canonical target instances have selected specific evidence; at 2026, 43/45 do. Presence is a structural proxy. Non-gold means absent from the benchmark answer list, not scientifically incorrect.\n\n'
text+=table(['Cutoff','Group','Field','Mean selected','Mean available before cap','Presence fraction','Mean best cosine if available'],[
    [a['cutoff'],'gold' if a['gold'] else 'not gold',f,d['selected_mean'],d['available_mean'],d['presence_fraction'],d['best_similarity_mean_when_present']]
    for a in availability for f,d in a['fields'].items()])
text+='Full min/quartile/median/max count distributions and per-candidate similarities are in `reranker_failure_analysis.json`. Counts after selection reflect genericness filtering and cross-field deduplication; they do not prove the original graph lacks useful evidence.\n\n'
text+=table(['Cutoff','Group','Low-arity selected','Co-membership only selected','Explicit-name selected','Indirect source via generic bridge','Diagnostic paths via generic bridge'],[
    [a['cutoff'],'gold' if a['gold'] else 'not gold',a['selected_support_kinds'].get('low_arity_direct',0),a['selected_support_kinds'].get('comembership_only',0),a['selected_support_kinds'].get('name_match',0),a['path_sources_via_generic_bridge'],a['diagnostic_paths_via_generic_bridge']] for a in attribution])
text+='Paths through generic bridges are retained and flagged, not recast as direct linkage. Their non-generic source text can still contribute the fixed .05 path-field weight. This is weaker attribution than direct or explicit-name evidence, and high source similarity is partly induced by semantic seed selection. Co-membership-only text is also explicitly marked in the compact document. Neither diagnostic is used to tune weights after seeing outcomes.\n\n'
categories=['NOT_IN_H2_100','NOT_IN_H2_200','IN_CANDIDATES_NO_SPECIFIC_EVIDENCE','IN_CANDIDATES_EVIDENCE_PRESENT_RERANK_FAILED','RERANK_IMPROVED_BUT_BELOW_10','RETRIEVED_TOP10','RERANK_DEGRADED_EXISTING_GOOD_RESULT']
text+=table(['Cutoff','Stage']+categories,[[c['cutoff'],c['stage']]+[c['counts'].get(cat,0) for cat in categories] for c in counts])
text+='Categories are mutually exclusive per stage/pool. Absence from that H2 pool takes precedence; top-ten recovery then takes precedence over degradation. A good R3/H2 result is a top-ten target. Remaining in-pool targets are classified by selected specific evidence and rank direction. “Evidence present/rerank failed” is operational, not a judgment that the evidence entails the answer.\n\n'
text+='## Gains and degradation\n\n'
text+=table(['Cutoff','Stage','Original bin','Targets','Improved','Unchanged','Worsened','Now top10'],[
    [m['cutoff'],m['stage'],m['bin'],m['count'],m['improved'],m['unchanged'],m['worsened'],m['top10']] for m in movements])
text+='R3-bin movement is relative to R3; H2 bins are relative to H2. Bins overlap across reference systems. In 2025 all four original R3 top-ten targets and all three H2 top-ten targets are recovered by both RR1 and RR3; substantial degradation nevertheless occurs among H2’s rank 11–100 targets. RR3 moderates overall displacement but also keeps SchNet at rank 11 instead of RR1’s 8.\n\n'

def explain(qid,label,kind):
    t=next(t for t in targets if t['cutoff']==2025 and t['question_id']==qid and t['label']==label)
    base=records(2025,'RR0')[qid]; rr=records(2025,'RR1-200')[qid]
    candidate=next((r for r in base['returned'] if set(r['node_ids'])&set(t['node_ids'])),None)
    if candidate is None: return ''
    b=evidence_lookup[2025,qid,candidate['entity_id']]; detail=rr['score_details'][candidate['entity_id']]
    s=f'### {kind}: {qid}, {label}\n\nQuestion: {rr["question"]}\n\nR3 → H2 → RR1-100/200 → RR3-100/200: {t["R3_rank"]} → {t["H2_rank"]} → {t["rerankings"]["RR1-100"]["rank"]}/{t["rerankings"]["RR1-200"]["rank"]} → {t["rerankings"]["RR3-100"]["rank"]}/{t["rerankings"]["RR3-200"]["rank"]}. RR1 score {detail["score"]:.4f}; available semantic mean {detail["available_semantic_score"]:.4f}; coverage {detail["coverage"]:.2f}.\n\n'
    rows=[]
    for f,items in b['fields'].items():
        if not items: continue
        item=max(items,key=lambda x:x['similarity'])
        rows.append([f,item['node_id'],item['kind'],item['similarity'],detail['score_components'].get(f),item['text']])
    s+=table(['Field','Node','Support','Cosine','Contribution','Selected maximum-scoring text'],rows)
    s+='These are actual score inputs and additive contributions, not causal proof from an evidence-removal ablation. Full selected texts, source years, provenance, discarded items and approximate paths are in `candidate_evidence.json`.\n\n'
    return s

text+='## Actual evidence traces\n\n'
for qid,label in [('Q2','D4FT'),('Q12','M3GNet'),('Q13','SchNet')]: text+=explain(qid,label,'Improvement')
text+='D4FT benefits from task/claim evidence about self-consistency, scaling and electronic structure. M3GNet and SchNet gain from relevant claims across selected fields. However, their highest explicit-mention text sometimes states that a *different* model outperforms them. Max-cosine pooling cannot identify the favored subject or comparison polarity; a recovered benchmark answer is therefore not proof of constraint-level reasoning.\n\n'
for qid,label in [('Q2','HamGNN'),('Q1','MACE')]: text+=explain(qid,label,'Remaining failure/degradation')
text+='HamGNN is an example of an available gold candidate demoted by scoring, while MACE has selected explicit mentions but sparse direct/task fields. The availability denominator avoids zero-filled penalties, yet the fixed coverage confidence still discounts sparse bundles. High-similarity path-source text may describe a neighbor’s comparative result. These observations identify extraction/attribution ambiguity and cosine weakness; the experiment cannot causally apportion them.\n\n'
for qid in ['Q1','Q12']:
    base=records(2025,'RR0')[qid]; rr=records(2025,'RR1-200')[qid]
    gold_nodes={n for t in targets if t['cutoff']==2025 and t['question_id']==qid for n in t['node_ids']}
    candidates=[r for r in base['returned'][:10] if not set(r['node_ids'])&gold_nodes and rank(rr,r['node_ids'])>30]
    cand=max(candidates,key=lambda c:rank(rr,c['node_ids']))
    b=evidence_lookup[2025,qid,cand['entity_id']]; d=rr['score_details'][cand['entity_id']]
    text+=f'### Suppressed unjudged candidate: {qid}, {cand["name"]}\n\nH2 rank {b["h2_rank"]} → RR1-200 rank {rank(rr,cand["node_ids"])}; score {d["score"]:.4f}, coverage {d["coverage"]:.2f}. This entry is not a canonical gold answer for this question. It remains **unjudged**, not a proven false positive. The question requests scientific methods/models; article/framework nodes can be topically related without identifying the requested method. Its complete selected semantic input is:\n\n```text\n{b["document"]}\n```\n\n'
text+='Sparse or less query-specific selected fields explain these demotions numerically; no external scientific adjudication or counterfactual evidence test was conducted.\n\n'

text+='## Every canonical target\n\n'
text+=table(['Cutoff','Q','Target','R3','H2','RR1-100','RR1-200','RR3-100','RR3-200'],[
    [t['cutoff'],t['question_id'],t['label'],t['R3_rank'],t['H2_rank']]+[t['rerankings'][s]['rank'] for s in ['RR1-100','RR1-200','RR3-100','RR3-200']] for t in targets])
text+='## Decisions after this pass\n\n'
text+='1. At 2025, RR1 improves micro Recall@10 by 3/47 = .0638 over H2 and 2/47 = .0426 over R3, reaching .1277. RR3 reaches .1064. At 2026, RR1/RR3 reach 5/50 = .10 versus H2’s 3/50 = .06.\n'
text+='2. K200 does not improve RR1/RR3 top-ten recovery over K100 at either cutoff. It doubles scoring calls and adds evidence construction; RR3-200 gains one 2025 hit at twenty. That does not establish a sufficient quality benefit for K200, though it remains essential as a diagnostic of discrimination.\n'
text+='3. Oracle micro Recall@10 is 31/47 = .6596 and 43/47 = .9149 at 2025 for K100/K200; 32/50 = .64 and 44/50 = .88 at 2026. Macro oracle values are separately tabulated and must not be confused with these micro denominators.\n'
text+='4. RR1’s oracle gaps are 25/47 = .5319 and 37/47 = .7872 at 2025; .54 and .78 at 2026. High candidate coverage does not translate to final discrimination.\n'
text+='5. Selected specific evidence is present for 41/43 primary in-pool gold instances and 43/45 at 2026. Only two in-pool instances at each cutoff have none. This does not establish that their evidence is discriminative or sufficient.\n'
text+='6. Failures predominantly have selected evidence present, rather than total evidence absence. Comparative-subject ambiguity, sparse extraction, and max-cosine’s inability to evaluate conjunctions, direction, scale and scientific constraints are plausible mechanisms supported by traces. Their relative causal contributions remain unmeasured.\n'
text+='7. Deterministic evidence scoring yields three new 2025 top-ten hits but worsens median rank. It helps a few answers and does not solve the ranking bottleneck.\n'
text+='8. A relevance cross-encoder comparison is unavailable; the local NLI classifier is not substituted. No conclusion about cross-encoder superiority is supported.\n'
text+='9. RRF preserves all H2 primary top-ten strengths and moderates rank degradation, but sacrifices SchNet’s top-ten gain. It does not beat RR1 at ten.\n'
text+='10. Hierarchy approximation is not justified as the next priority. The quality prerequisite remains unmet. A future pass should test stronger relevance/constraint scoring and better evidence attribution on unseen questions before optimizing the same weak ranking for speed. No hierarchy implementation was added.\n\n'
text+='Semantic retrieval locates query-relevant text; diffusion transfers that relevance to structurally related entities; reranking must test which entities satisfy the question. Neither graph proximity nor semantic cosine is entailment, and this pass does not fix graph extraction.\n\n'

text+='## Reproduction and verification\n\n'
text+='```powershell\n$env:HF_HUB_OFFLINE=\'1\'\n.\\.venv\\Scripts\\python.exe -m tkh_abstraction.cli evaluate-reranking --config configs/reranking.yaml\n.\\.venv\\Scripts\\python.exe -m pytest -q --junitxml=artifacts/evaluation/reranker_tests.xml\n.\\.venv\\Scripts\\python.exe scripts/report_reranking.py\n.\\.venv\\Scripts\\python.exe scripts/package_submission.py\n```\n\n'
text+='For unseen questions on the same graph, pass `--questions questions_new.csv --ground-truth ground_truth_new.json`; use separate configured output directories to preserve assessment exports. New data additionally requires `reranker_verify_history: false` and explicit data/output paths. `--cutoff` selects a partial experiment; the assessment report audit expects both cutoffs. Current question IDs and target names occur in evaluation/report fixtures, not production evidence/scoring.\n\n'
text+=f'The complete suite passed {suite.attrib["tests"]} tests with no skips/errors/failures. New tests cover deterministic capped evidence, explicit mentions/dedup/temporal exclusion, generic bridges, normalized missing fields, coverage and pooling, K100/K200 preservation, unchanged groups, RRF determinism, multi-answer/alias oracle coverage, frozen candidate metrics, no oracle/gold production inputs, offline/NLI rejection, all-null unseen evaluation, Type B/history preservation, ground-truth perturbation invariance, and exact RR0 historical metrics. The report audit also checks source hashes, every frozen prefix, all 14 question statuses, old artifact/metric hashes, native hierarchy manifest, temporal evidence containment and score decomposition.\n\n'
text+='Implementation: `candidate_evidence.py`, `reranking.py`, `reranker_pipeline.py`, `evaluation/reranker_oracle.py`, CLI extension, `configs/reranking.yaml`, `tests/test_reranking.py`, and this reporting script. Required exports are `reranker_results.json`, `reranker_per_target.json`, `candidate_evidence.json`, `reranker_failure_analysis.json`, `reranker_oracle.json`, and `reranker_tests.xml`; detailed protocols/stage exports/audit are under `artifacts/evaluation/reranking`. Pre-pass metrics and diffusion report are preserved under `artifacts/baseline_diffusion`.\n\n'
text+='Ultralight used serial read-only Claude Opus planning and Claude Sonnet review through the official local subscription helper. Codex implemented, integrated and executed the work. Prompts, results and decisions are under `notes/reranking`; no human scientific validation is claimed.\n'
Path('report/evidence_reranking.md').write_text(text,encoding='utf-8')
section='\n## Evidence-aware candidate reranking\n\nFixed candidate-specific RR1 scoring raises conservative top-ten recovery from H2’s 3/47 to 6/47 (R3: 4/47), preserving H2’s three hits and adding three. Both 100- and 200-candidate pools yield the same top-ten hits; RR3 fusion yields 5/47. Oracle ceilings are 31/47 and 43/47, so most available answers remain missed; median gold rank worsens. No local relevance cross-encoder was available. All 14 Type A questions remain explicit; Q5/Q11 have null canonical evaluability and remain in the raw /63 denominator. Type B, H2, H3 and historical metrics are unchanged. Final quality still does not justify hierarchy approximation as the next priority. See [evidence_reranking.md](evidence_reranking.md) for complete metrics, actual evidence traces, failures, costs and audit.\n'
main=Path('report/report.md'); main.write_text(main.read_text(encoding='utf-8').split('\n## Evidence-aware candidate reranking')[0]+section,encoding='utf-8')
audit=dict(status='passed_with_research_limitations',tests=suite.attrib,question_records=sum(len(r['records']) for r in results),
    RR0_exact_historical_metric_and_prefix_parity=True,all14_type_a_retained=True,candidate_preservation=True,temporal_evidence_verified=True,
    historical_artifacts_and_metrics_unchanged=True,diffusion_report_unchanged=True,type_b_and_hierarchy_unchanged=True,source_hashes_verified=True,
    RR2_status=bundle['RR2_status']['status'],hierarchy_next=False,conclusion='Modest top10 gain; large oracle gap and worse median rank; stronger evidence discrimination still needed')
write_json(out/'final_audit.json',audit)
check=Path('artifacts/submission_checklist.md'); check.write_text(check.read_text(encoding='utf-8').split('\n## Evidence reranking follow-up')[0]+'\n## Evidence reranking follow-up\n\nPASS: frozen H2 parity, both candidate pools/cutoffs, all 14 Type A statuses, evidence/score/path exports, exact evaluation-only oracle, no label training, historical/Type B/hierarchy preservation, full tests and source audit. See `evaluation/reranking/final_audit.json` and `../report/evidence_reranking.md`.\n\nLIMITATIONS: RR2 unavailable locally; modest top-ten improvement, substantial rank degradation and large oracle gap; graph co-membership and explicit mention do not imply entailment. Hierarchy approximation remains deferred.\n',encoding='utf-8')
print(json.dumps(audit,indent=2))
