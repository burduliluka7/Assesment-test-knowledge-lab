"""Evaluation-only CR6 report, diagnostics and historical preservation audit."""
from pathlib import Path
from collections import Counter
import json
import xml.etree.ElementTree as ET
import numpy as np
from scipy.stats import spearmanr
from tkh_abstraction.io import read_json, write_json, digest
from tkh_abstraction.v2_pipeline import sha
from tkh_abstraction.snapshots import snapshot
from tkh_abstraction.constraint_scoring import aggregate_support

ROOT=Path('artifacts/evaluation')
run=read_json(ROOT/'conditioned_evidence_results.json'); p=run['protocol']
qs=read_json(ROOT/'requirement_evidence_selection.json'); deltas=read_json(ROOT/'evidence_selection_delta.json')
pools=read_json(ROOT/'candidate_evidence_pools.json'); targets=read_json(ROOT/'conditioned_ranks_per_target.json')
old=read_json(ROOT/'constraint_reranking_results.json'); rr=read_json(ROOT/'reranker_results.json')
metrics=read_json('artifacts/full/metrics.json'); cfg=p['scoring_config']; years=p['config']['conditioned_cutoffs']
lookup={(r['cutoff'],r['stage']):r for r in run['results']}
previous={(r['cutoff'],r['stage']):r for r in old['results']}
candidate_lookup={(q['cutoff'],q['question_id'],c['entity_id']):c for q in qs for c in q['candidates']}
delta_lookup={(r['cutoff'],r['question_id'],r['entity_id']):r for r in deltas}
pool_lookup={(r['cutoff'],r['entity_id']):r for r in pools}

def mean(values): return float(np.mean(values)) if len(values) else None
def quantiles(values): return dict(zip(['min','q25','median','q75','max'],map(float,np.quantile(values,[0,.25,.5,.75,1])))) if len(values) else None
def fmt(x): return '—' if x is None else f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x)
def table(headers,rows):
    def cell(v): return fmt(v).replace('|','/').replace('\n',' ')
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(cell(v) for v in row)+' |\n' for row in rows)+'\n'
def rho(x,y):
    if len(x)<3 or len(set(x))<2 or len(set(y))<2: return None
    return float(spearmanr(x,y).statistic)
def rank(t,s): return t['ranks'][s]

# Audit before writing a report that could conceal inconsistent exports.
for path,h in p['source_hashes'].items(): assert sha(path)==h,('source',path)
for path,h in p['historical_hashes'].items(): assert sha(path)==h,('history',path)
for key,h in p['historical_metric_hashes'].items(): assert digest(metrics[key])==h,('metric',key)
suite=ET.parse(ROOT/'conditioned_evidence_tests.xml').getroot().find('testsuite')
assert all(int(suite.attrib[k])==0 for k in ['failures','errors','skipped'])
assert set(years)=={2025,2026}, 'Assessment report requires both cutoffs'
assert len(run['inventory'])==28
old_inv={(i['cutoff'],i['question_id']):i for i in old['inventory']}
for i in run['inventory']:
    assert i['reference_parity'] and i['reference_metric_parity']
    for key in ['prefix100_hash','prefix200_hash']: assert i[key]==old_inv[i['cutoff'],i['question_id']][key]
for o in run['oracle']: assert o==next(x for x in old['oracle'] if all(x[k]==o[k] for k in ['cutoff','question_id','candidate_k']))
for result in run['results']:
    year,stage=result['cutoff'],result['stage']; assert result['summary']['questions']==14 and result['summary']['evaluable']==12
    assert {r['question_id'] for r in result['records']}=={f'Q{i}' for i in range(1,15)}
    h2={r['question_id']:r for r in lookup[year,'H2']['records']}
    assert result['summary']['at_k']['10']['canonical_target_instances']==(47 if year==2025 else 50)
    assert result['summary']['at_k']['10']['total_target_instances']==63
    for r in result['records']:
        if stage!='H2':
            k=r.get('candidate_k',100); a=r['returned']; b=h2[r['question_id']]['returned']
            assert {x['entity_id'] for x in a[:k]}=={x['entity_id'] for x in b[:k]} and a[k:]==b[k:]
        if stage in ['CR0-100','CR1-100','CR3-100']:
            prior=next(x for x in previous[year,stage]['records'] if x['question_id']==r['question_id'])
            assert r['returned']==prior['returned'] and r['entity_metrics']==prior['entity_metrics']
        if r['question_id'] in ['Q5','Q11']: assert r['entity_metrics']['status']=='not_evaluable' and r['entity_metrics']['at_k']['10']['recall'] is None
data=read_json(Path(p['config']['data_dir'])/'tkh_collection10.json')
visible={y:snapshot(data,y) for y in years}
for pool in pools:
    nodes={n['id'] for n in visible[pool['cutoff']]['nodes']}; edges={e['id'] for e in visible[pool['cutoff']]['hyperedges']}
    texts=[]
    for f,items in pool['fields'].items():
        assert len(items)<=p['config']['conditioned_pool_caps'][f]
        for item in items:
            assert item['node_id'] in nodes and item['first_seen_year']<=pool['cutoff']
            assert all(e in edges for e in item['path'])
            texts.append(item['text'].casefold())
    assert len(texts)==len(set(texts))
for q in qs:
    for component in q['decomposition']['components']:
        assert q['question'][component['start']:component['end']]==component['text']
        assert all(component['end']<=t['start'] or component['start']>=t['end'] for t in q['decomposition']['source_spans']['temporal_condition'])
    for c in q['candidates']:
        d=c['CR6']; pool=pool_lookup[q['cutoff'],c['entity_id']]; ids={i['node_id'] for xs in pool['fields'].values() for i in xs}
        for req in d['requirements']:
            sel=req['selected_evidence']; assert len(sel)<=3 and all(e['node_id'] in ids and e['weight']>0 for e in sel)
            assert sel==sorted(sel,key=lambda r:(-r['support'],-r['cosine'],r['node_id']))
            assert all(abs(e['support']-max(0,e['cosine'])*e['weight'])<1e-12 for e in sel)
        comp=[r['component'] for r in d['requirements']]; supports=[r['cosine_support'] for r in d['requirements']]
        agg=aggregate_support(comp,supports,[s for x,s in zip(comp,supports) if x['kind']=='exclusion'],pool['types'],q['decomposition']['answer_type'],cfg)
        assert agg['score']==d['score']

support_stats=[]; noise=[]; movements=[]; preservation=[]; failures=[]; control_stats=[]
for year in years:
    ts=[t for t in targets if t['cutoff']==year]
    for k in [100,200]:
        for gold in [True,False]:
            candidates=[c for q in qs if q['cutoff']==year for c in q['candidates'] if c['h2_rank']<=k and c['gold']==gold]
            ds=[r for row in deltas if row['cutoff']==year and row['h2_rank']<=k and row['gold']==gold for r in row['components'] if r['comparison_status']=='matched' and r['component']['kind']!='exclusion']
            gains=[r['support_delta'] for r in ds]
            support_stats.append(dict(cutoff=year,k=k,gold=gold,candidates=len(candidates),matched=len(ds),mean_delta=mean(gains),median_delta=float(np.median(gains)) if gains else None,
                improved=sum(v>1e-7 for v in gains),same=sum(abs(v)<=1e-7 for v in gains),worsened=sum(v< -1e-7 for v in gains),
                changed_node=sum(r['changed'] for r in ds),previously_unselected=sum(r['previously_unselected'] for r in ds),
                newly_supported=sum(r['old_support']<.5<=r['new_support'] for r in ds),lost_support=sum(r['new_support']<.5<=r['old_support'] for r in ds),
                pool_larger=sum(c['CR6']['pool_size']>c['old_pool_size'] for c in candidates),
                mean_pool_only_delta=mean([r['pool_only_support']-r['old_support'] for r in ds]),
                new_components=sum(r['comparison_status']!='matched' for row in deltas if row['cutoff']==year and row['h2_rank']<=k and row['gold']==gold for r in row['components'])))
            sizes=[c['CR6']['pool_size'] for c in candidates]; scores=[c['CR6']['score'] for c in candidates]
            component_sizes=[c['CR6']['pool_size'] for c in candidates for r in c['CR6']['requirements'] if r['maximum_raw_cosine'] is not None]
            maxima=[r['maximum_raw_cosine'] for c in candidates for r in c['CR6']['requirements'] if r['maximum_raw_cosine'] is not None]
            noise.append(dict(cutoff=year,k=k,gold=gold,candidates=len(candidates),pool_distribution=quantiles(sizes),
                rho_pool_max_cosine=rho(component_sizes,maxima),rho_degree_score=rho([c['weighted_degree'] for c in candidates],scores),
                rho_items_score=rho(sizes,scores),rho_attributable_items_score=rho([c['CR6']['attributable_items'] for c in candidates],scores)))
    for stage in p['stages']:
        failures.append(dict(cutoff=year,stage=stage,counts=dict(Counter(t['failures'][stage] for t in ts))))
        for name,lo,hi in [('1-10',1,10),('11-50',11,50),('51-100',51,100),('>100',101,float('inf'))]:
            group=[t for t in ts if lo<=(rank(t,'H2') or float('inf'))<=hi]
            movements.append(dict(cutoff=year,stage=stage,bin=name,n=len(group),improved=sum(rank(t,stage)<rank(t,'H2') for t in group),
                same=sum(rank(t,stage)==rank(t,'H2') for t in group),worsened=sum(rank(t,stage)>rank(t,'H2') for t in group),
                entered10=sum(rank(t,stage)<=10<rank(t,'H2') for t in group),left10=sum(rank(t,'H2')<=10<rank(t,stage) for t in group)))
        for ref in ['H2','CR0-100','CR1-100']:
            group=[t for t in ts if rank(t,ref)<=10]
            preservation.append(dict(cutoff=year,stage=stage,reference=ref,prior=len(group),preserved=sum(rank(t,stage)<=10 for t in group),lost=sum(rank(t,stage)>10 for t in group)))
    for stage in ['pool_only','without_qualifiers']:
        control_stats.append(dict(cutoff=year,diagnostic=stage,found=sum(rank(t,stage)<=10 for t in ts),total=len(ts),median=float(np.median([rank(t,stage) for t in ts]))))
pool_losses=[]; attribution_changes=[]; question_noise=[]; exclusion_stats=[]
for year in years:
    for gold in [True,False]:
        rs=[r for row in deltas if row['cutoff']==year and row['h2_rank']<=100 and row['gold']==gold for r in row['components'] if r['comparison_status']=='matched' and r['component']['kind']=='exclusion']
        exclusion_stats.append(dict(cutoff=year,gold=gold,n=len(rs),mean_violation_delta=mean([r['support_delta'] for r in rs]),
            larger_violation=sum(r['support_delta']>1e-7 for r in rs),smaller_violation=sum(r['support_delta']< -1e-7 for r in rs)))
for q in qs:
    for c in q['candidates']:
        pool_losses.extend(dict(cutoff=q['cutoff'],question_id=q['question_id'],candidate=c['candidate'],gold=c['gold'],h2_rank=c['h2_rank'],**x) for x in c['old_positive_items_missing_from_pool'])
        attribution_changes.extend(dict(cutoff=q['cutoff'],question_id=q['question_id'],candidate=c['candidate'],**x,
            selected=any(r['best_evidence'] and r['best_evidence']['node_id']==x['node_id'] for r in c['CR6']['requirements'])) for x in c['CR6']['attribution_changes'])
    for k in [100,200]:
        cs=q['candidates'][:k]
        question_noise.append(dict(cutoff=q['cutoff'],question_id=q['question_id'],k=k,
            rho_items_score=rho([c['CR6']['pool_size'] for c in cs],[c['CR6']['score'] for c in cs]),
            rho_degree_score=rho([c['weighted_degree'] for c in cs],[c['CR6']['score'] for c in cs])))
diagnostics=dict(support_changes=support_stats,noise=noise,per_question_noise=question_noise,movement=movements,preservation=preservation,
    failures=failures,controls=control_stats,old_positive_items_missing=pool_losses,attribution_changes=attribution_changes,exclusions=exclusion_stats)
write_json(ROOT/'conditioned_failure_analysis.json',dict(targets=targets,**diagnostics,qualitative_failure_categories=['DECOMPOSITION_FAILURE','ATTRIBUTION_FAILURE'],
    policy='Operational support is cosine >=.5, not entailment. Non-gold means unjudged. Qualitative failure categories are not assigned automatically.'))

lines=['# Requirement-conditioned evidence retrieval\n']
def put(text): lines.append(text+'\n')
for year in years:
    s=lookup[year,'CR6-100']['summary']; old_s=lookup[year,'CR1-100']['summary']; fusion=lookup[year,'CR7-100']['summary']
    put(f"At {year}, CR6 recovers {s['at_k']['10']['found']}/{s['at_k']['10']['canonical_target_instances']} canonical target instances at ten, compared with CR1’s {old_s['at_k']['10']['found']}; CR7 recovers {fusion['at_k']['10']['found']}. CR6 median gold rank is {fmt(s['rank_quantiles']['median'])}, versus CR1’s {fmt(old_s['rank_quantiles']['median'])}.")
put('These are development diagnostics on the repeatedly used benchmark, not held-out validation or human scientific adjudication. H2, CR0, CR1 and CR3 reproduce exactly. CR2/CR4 remain archived; the new path makes zero NLI calls and downloads no models. All previous reports and metric sections are preserved.')
put('## Frozen protocol and implementation')
put('`CandidateEvidencePool` reuses `CandidateEvidenceIndex.pools` before whole-question semantic selection. The same safe mentions, eligible low-arity relations, genericness filter, snapshot, entity groups and provenance-title policy apply. Pools are built once per candidate × snapshot, with fixed field caps 20 direct claims, 20 explicit-name claims, 15 tasks/problems, 15 technical items, 10 datasets/metrics and 10 titles. Stable structural priority and node IDs decide caps; normalized identical text appears once, with merged source links. The pool is bounded eligible graph evidence, not every possible statement about an entity. Some larger structurally capped pools may still omit useful text.')
put('No new path expansion is performed: safe named claims are already discovered by the unchanged MentionIndex. Historical query-seeded paths remain in each selection export as diagnostics; they are not inputs to the query-independent pool. This deliberately avoids importing a neighboring unnamed claim as candidate support. The new pool need not be a strict superset of the old bundle because old query-seeded paths and structural-cap choices can differ. Whole-article claim expansion is excluded.')
put('`ConditionedEvidenceReranker` extends the existing cosine-reranker interface. Each original component substring is embedded using pinned normalized BGE-small-en-v1.5; evidence vectors come from the existing ContextIndex cache. Top three items are selected by nonnegative cosine × attribution weight, with the historical cosine/ID tie rule. Maximum support enters the unchanged `aggregate_support`: core × geometric mean of positive components × exclusion factor × soft type prior. Top-three export does not mean three items are averaged. Weights remain 1 direct/winner, .8 explicit name, .15 mention, .25 uncertain and 0 comparison baseline/unnamed path. The .5 threshold is diagnostic only. Exclusion cosine cannot establish negation or compliance.')
put('`ConditionedQuestionDecomposition` subclasses the frozen `QuestionDecomposition`. It recovers answer-head modifiers before a recognized generic noun, splits a leading hyphenated adjective from a remaining compound noun, and retains coordinated requirements with a shared modal predicate as one exact source span. It adds qualifiers as positive components without filtering candidates. All old parser/scorer modules are byte-preserved. The additive attribution adapter recognizes sentence-initial participial “Including …” followed by a limited generic finite-predicate pattern as safe explicit-name support (.8), while preserving comparison/negation/list safeguards. It does not upgrade this to direct candidate subject support. The heuristic remains incomplete.')
put('The main comparison changes evidence access and these explicitly requested generic grammar fixes together. Two fixed evaluation diagnostics separate them: “pool_only” applies old decomposition and old attribution to the new pool; “without_qualifiers” removes only qualifier components from otherwise identical CR6. Neither control chooses a configuration or trains weights. A within-candidate permutation of the same component scores is largely invariant under geometric pooling and would be an uninformative null; no shuffled variant is added.')
put(table(['Q','Answer type','Qualifiers','Core','Requirements','Exclusions'],[[q['question_id'],q['decomposition']['answer_type'],'; '.join(q['decomposition']['answer_qualifiers']),q['decomposition']['core_task'],'; '.join(q['decomposition']['requirements']),'; '.join(q['decomposition']['exclusions'])] for q in qs if q['cutoff']==2025]))
put('## Quality, oracle and cost')
for year in years:
    put(f'### {year}')
    rows=[]
    for stage in ['H2','CR0-100','CR1-100','CR3-100',*p['stages']]:
        s=lookup[year,stage]['summary']; a=s['at_k']; rows.append([stage,*[a[str(k)]['macro_hit'] for k in [1,5,10,20]],*[a[str(k)]['macro_recall'] for k in [5,10,20]],*[a[str(k)]['micro_recall'] for k in [5,10,20]],a['10']['unique_target_recall'],f"{a['10']['found']}/63",s['mrr'],s['r_precision'],s['recall_at_r'],s['rank_quantiles']['median'],s['mean_target_rank']])
    put(table(['Stage','Hit1','Hit5','Hit10','Hit20','MacroR5','MacroR10','MacroR20','MicroR5','MicroR10','MicroR20','UniqueR10','Raw','MRR','R-Prec','R@R','Median','Mean rank'],rows))
put('All 14 Type A questions appear in every stage, with explicit evaluability. Q5 and Q11 have no canonical named-entity units: canonical recall is null, their raw labels remain in /63. Macro/Hit averages use 12 evaluable questions; micro denominators are 47 (2025) and 50 (2026), with 34/37 unique canonical labels. Costs average all 14. Type B retrieval and historical outputs are unchanged. Complete per-question metrics are in both the results export and the additive metrics.json section.')
oracle_rows=[]
for year in years:
    for stage in ['CR1-100',*p['stages']]:
        k=int(stage.split('-')[1]); os=[o for o in run['oracle'] if o['cutoff']==year and o['candidate_k']==k]; n=sum(o['canonical_targets'] for o in os); found=sum(o['oracle_found_at10'] for o in os)
        actual=lookup[year,stage]['summary']['at_k']['10']['micro_recall']
        oracle_rows.append([year,stage,f'{found}/{n}',found/n,actual,found/n-actual,mean([o['oracle_recall_at10'] for o in os if o['oracle_recall_at10'] is not None])])
put(table(['Year','Stage','Frozen oracle','Oracle microR10','Actual microR10','Gap','Oracle macroR10'],oracle_rows))
put('The exact prior subset-coverage oracle remains evaluation-only; every per-question oracle dictionary is identical. Candidate prefixes and the entire unreranked tail are preserved. No oracle or expected label reaches pool construction, decomposition, evidence selection or scoring.')
put(table(['Year','Pool build seconds','Candidate pools','Evidence items'],[[r['cutoff'],r['seconds'],r['candidates'],r['items']] for r in run['pool_build_costs']]))
cost_rows=[]
for year in years:
    for stage in p['stages']:
        rows=lookup[year,stage]['records']; fields=['evidence_selection_dots_new','evidence_selection_scores_reused','evidence_embeddings_reused','new_component_strings','semantic_support_comparisons','reranker_seconds','total_query_seconds']
        cost_rows.append([year,stage,*[mean([r['cost'][k] for r in rows]) for k in fields]])
put(table(['Year','Stage','New component dots','Reused CR1 scalar scores','Reused evidence vectors','New component strings','Support comparisons','New seconds','Total seconds*'],cost_rows))
put('*Component-sum estimates use historical H2 search plus current vector preparation, evidence scoring and fusion. Pool construction is offline and separately tabulated. Model/index loading, historical reference recomputation, evaluation exports and diagnostic controls are excluded. Primary runs conservatively pay shared top-200 vector preparation, so K100 may be overcharged. Component vectors use the existing persistent embedding cache; reported new strings are relative to CR1, not necessarily neural cache misses on a rerun. Evidence encodings and NLI calls are zero. These single-machine timings are not controlled serving benchmarks.')
put('## Evidence selection and noise')
put(f"The pool audit finds {len(pool_losses)} repeated candidate/question/item instances with positive old attribution missing from the new pool: {dict(Counter(x['reason'] for x in pool_losses))}. Full texts and reasons are exported. The additive participial rule changes {len(attribution_changes)} repeated item roles, of which {sum(x['selected'] for x in attribution_changes)} become a selected component maximum. These are not globally unique claims.")
put(table(['Year','K','Group','Matched components','Mean delta','Median delta','Improved','Same','Worsened','Changed node','Newly exposed','New support','Lost support','Pool-only mean delta','New/restructured'],[[r[k] for k in ['cutoff','k','gold','matched','mean_delta','median_delta','improved','same','worsened','changed_node','previously_unselected','newly_supported','lost_support','mean_pool_only_delta','new_components']] for r in support_stats]))
put('True/false groups mean canonical gold / absent from the answer key; the latter remain unjudged. The preceding table includes matched positive components only. Components are matched by kind and exact original span/text. New qualifiers and restructured conjunctions have null old support and delta, rather than invented zero baselines; removed old components are separately exported. Improved/same/worsened uses a 1e-7 numerical tolerance. “Newly exposed” means the selected normalized text was absent from the previous small bundle. Newly supported uses the fixed .5 threshold, not adjudicated entailment. Exclusions are tabulated separately: a larger exclusion value is a stronger violation proxy, not a support gain.')
put(table(['Year','Gold','Matched exclusions (K100)','Mean violation delta','Larger violation','Smaller violation'],[[r[k] for k in ['cutoff','gold','n','mean_violation_delta','larger_violation','smaller_violation']] for r in exclusion_stats]))
put(table(['Year','K','Gold','Pool min/Q1/median/Q3/max','rho(pool,max cosine)','rho(degree,score)','rho(items,score)','rho(attributable,score)'],[[r['cutoff'],r['k'],r['gold'],r['pool_distribution'],r['rho_pool_max_cosine'],r['rho_degree_score'],r['rho_items_score'],r['rho_attributable_items_score']] for r in noise]))
put('Correlations pool candidate/question observations within a cutoff and K; max-cosine correlation uses component observations. Repeated entities/components are dependent, so these are descriptive Spearman coefficients without significance claims. Correlation does not identify a causal size effect. Full distributions are retained. No pool-size penalty was added after viewing outcomes.')
put(table(['Year','Diagnostic (K100)','Top10','Canonical','Median'],[[r['cutoff'],r['diagnostic'],r['found'],r['total'],r['median']] for r in control_stats]))
put('## Movement and retention')
put(table(['Year','Stage','H2 bin','N','Improved','Same','Worsened','Entered10','Left10'],[[r[k] for k in ['cutoff','stage','bin','n','improved','same','worsened','entered10','left10']] for r in movements]))
put(table(['Year','Stage','Reference','Prior hits','Preserved','Lost'],[[r[k] for k in ['cutoff','stage','reference','prior','preserved','lost']] for r in preservation]))
put('Reference hit sets overlap and must not be added. CR7 uses the unchanged RRF60 and average ranks for tied scores. It can preserve prior hits while also suppressing standalone gains.')
put(table(['Year','Stage','Failure counts'],[[r['cutoff'],r['stage'],r['counts']] for r in failures]))
put('Failure precedence is pool absence, top-ten recovery, empty pool, no direct/name/winner attribution, weak qualifier, zero positive coverage, partial coverage, then supported-but-below-ten. NO_ATTRIBUTABLE_EVIDENCE means no strong direct/name/winner item; uncertain or mention-only evidence may still score. REQUIREMENT_EVIDENCE_NOT_FOUND and ANSWER_QUALIFIER_FAILURE mean low selected cosine support, not proven graph absence or scientific invalidity. DECOMPOSITION_FAILURE and ATTRIBUTION_FAILURE require qualitative inspection; no automated counts are fabricated from answer membership.')

def trace(q,c,title):
    put('### '+title)
    put(q['question']); put('Ranks: '+', '.join(f'{s}={c["ranks"][s]}' for s in ['H2','CR0-100','CR1-100','CR3-100',*p['stages']])+'.')
    put(f"Pool size {c['CR6']['pool_size']} versus old bundle {c['old_pool_size']}; CR6 coverage {fmt(c['CR6']['requirement_coverage'])}; score {fmt(c['CR6']['score'])}.")
    ds=delta_lookup[q['cutoff'],q['question_id'],c['entity_id']]['components']
    rows=[]
    for r in ds:
        a,b=r['old_evidence'],r['new_evidence']
        rows.append([r['component']['kind'],r['component']['text'],a['node_id'] if a else None,r['old_support'],b['node_id'] if b else None,r['new_support'],b['attribution'] if b else None,r['previously_unselected'],b['text'] if b else None])
    put(table(['Kind','Component','Old node','Old support','New node','New support','Attribution','New text','Selected evidence'],rows))
    put('Top three matches and source-link provenance are available in requirement_evidence_selection.json and candidate_evidence_pools.json. These are actual score inputs, not proof that the text satisfies the scientific condition.')

put('## Actual graph evidence and qualifier diagnostics')
q1=next(q for q in qs if q['cutoff']==2025 and q['question_id']=='Q1')
mace=next(c for c in q1['candidates'] if c['candidate']=='MACE')
trace(q1,mace,'Required Q1/MACE diagnostic')
put('Compare the efficiency and scale rows directly with their selected texts. A high similarity to an accuracy statement does not establish efficiency or million-atom scalability; only explicitly stated conditions count as textual evidence in a qualitative reading. Absence from the bounded eligible pool does not prove absence from the literature.')
all_primary=[(q,c) for q in qs if q['cutoff']==2025 for c in q['candidates'] if c['h2_rank']<=100]
hidden=[(q,c) for q,c in all_primary if c['gold'] and c['ranks']['CR6-100']<c['ranks']['CR1-100'] and any(r['previously_unselected'] and (r['support_delta'] or 0)>1e-7 for r in delta_lookup[q['cutoff'],q['question_id'],c['entity_id']]['components'])]
if hidden:
    q,c=max(hidden,key=lambda qc:qc[1]['ranks']['CR1-100']-qc[1]['ranks']['CR6-100']); trace(q,c,'Previously hidden evidence with improved gold rank: '+q['question_id']+'/'+c['candidate'])
else: put('No primary gold candidate simultaneously meets the newly exposed evidence and improved-rank criteria; no example is invented.')
different=[(q,c) for q,c in all_primary if len({r['best_evidence']['node_id'] for r in c['CR6']['requirements'] if r['best_evidence']})>=3]
if different:
    q,c=next(((q,c) for q,c in different if c['gold']),different[0]); trace(q,c,'Different components select different evidence: '+q['question_id']+'/'+c['candidate'])
boost=[(q,c) for q,c in all_primary if not c['gold'] and c['CR6']['pool_size']>c['old_pool_size'] and c['ranks']['CR6-100']<c['ranks']['CR1-100']]
if boost:
    q,c=max(boost,key=lambda qc:qc[1]['ranks']['CR1-100']-qc[1]['ranks']['CR6-100']); trace(q,c,'Expanded-pool boost for an unjudged candidate: '+q['question_id']+'/'+c['candidate'])
    put('This candidate is absent from the supplied answer key, so the boost is a noise warning to inspect, not a certified false positive.')
missing=[(q,c) for q,c in all_primary if c['gold'] and c['CR6']['requirement_coverage']<1 and c['ranks']['CR6-100']>10]
if missing:
    q,c=missing[0]; trace(q,c,'Remaining weak component support: '+q['question_id']+'/'+c['candidate'])
qualifier_rows=[]
for q in qs:
    if q['question_id']!='Q12': continue
    for gold in [True,False]:
        cs=[c for c in q['candidates'] if c['h2_rank']<=100 and c['gold']==gold]
        qualifier_rows.append([q['cutoff'],gold,len(cs),mean([c['without_qualifiers_score'] for c in cs]),mean([c['CR6']['score'] for c in cs]),sum(c['ranks']['without_qualifiers']<=10 for c in cs),sum(c['ranks']['CR6-100']<=10 for c in cs)])
put('### Q12 answer-head qualifier control')
put(table(['Year','Gold','Candidates','Mean without qualifiers','Mean CR6','Top10 without qualifiers','Top10 CR6'],qualifier_rows))
q12=next(q for q in qs if q['cutoff']==2025 and q['question_id']=='Q12')
changed=sorted(q12['candidates'][:100],key=lambda c:abs(c['ranks']['CR6-100']-c['ranks']['without_qualifiers']),reverse=True)[:6]
put(table(['Candidate','Gold','Without-qualifier rank','CR6 rank','Qualifier supports'],[[c['candidate'],c['gold'],c['ranks']['without_qualifiers'],c['ranks']['CR6-100'],[(r['component']['text'],round(r['cosine_support'],4)) for r in c['CR6']['requirements'] if r['component']['kind']=='answer_qualifier']] for c in changed]))
put('This paired control isolates the effect of including the extracted modifiers on this fixed pool/scorer. Benchmark membership is not a scientific label for semantic class; count/rank changes alone do not certify better class discrimination. Adding a component to a geometric mean can either raise or lower a score depending on its support, so qualifiers are not guaranteed penalties.')
rescued=[t for t in targets if t['cutoff']==2025 and rank(t,'H2')<=10<rank(t,'CR6-100') and rank(t,'CR7-100')<=10]
put(table(['H2 hit rescued by CR7','Q','H2','CR6','CR7'],[[t['label'],t['question_id'],rank(t,'H2'),rank(t,'CR6-100'),rank(t,'CR7-100')] for t in rescued]))
if not rescued: put('No 2025 H2 top-ten loss is rescued by CR7 in this run; no positive example is invented.')

put('## Every canonical target')
put(table(['Year','Q','Target','R3','H2','CR0','CR1','CR3','CR6-100','CR7-100','CR6-200'],[[t['cutoff'],t['question_id'],t['label'],t['R3_rank'],*[rank(t,s) for s in ['H2','CR0-100','CR1-100','CR3-100',*p['stages']]]] for t in targets]))
put('## Decisions')
for year in years:
    a=lookup[year,'CR1-100']['summary']; b=lookup[year,'CR6-100']['summary']; c=lookup[year,'CR7-100']['summary']
    put(f"{year}: CR6 versus CR1 changes micro Recall@10 by {fmt(b['at_k']['10']['micro_recall']-a['at_k']['10']['micro_recall'])} and median gold rank by {fmt(b['rank_quantiles']['median']-a['rank_quantiles']['median'])}. CR7 recovers {c['at_k']['10']['found']} canonical instances at ten. The oracle and retention tables quantify the remaining gap and losses; neither fusion nor K200 is presumed better.")
a=lookup[2025,'CR1-100']['summary']; b=lookup[2025,'CR6-100']['summary']; f=lookup[2025,'CR7-100']['summary']
g=next(r for r in support_stats if r['cutoff']==2025 and r['k']==100 and r['gold'])
u=next(r for r in support_stats if r['cutoff']==2025 and r['k']==100 and not r['gold'])
ng=next(r for r in noise if r['cutoff']==2025 and r['k']==100 and r['gold'])
nu=next(r for r in noise if r['cutoff']==2025 and r['k']==100 and not r['gold'])
entered=sum(r['entered10'] for r in movements if r['cutoff']==2025 and r['stage']=='CR6-100' and r['bin'] in ['11-50','51-100'])
loss={r['reference']:r['lost'] for r in preservation if r['cutoff']==2025 and r['stage']=='CR6-100'}
q12gold=next(r for r in qualifier_rows if r[0]==2025 and r[1])
pool_control=next(r for r in control_stats if r['cutoff']==2025 and r['diagnostic']=='pool_only')
answers=[
 f"Recall@10: CR1 {a['at_k']['10']['found']}/47 → CR6 {b['at_k']['10']['found']}/47. The pure pool-access control recovers {pool_control['found']}/47; it must be considered alongside the combined change.",
 f"Median gold rank: CR1 {fmt(a['rank_quantiles']['median'])} → CR6 {fmt(b['rank_quantiles']['median'])} (lower is better).",
 f"Matched positive components selecting a different node: {g['changed_node']}/{g['matched']} gold, {u['changed_node']}/{u['matched']} unjudged. Newly exposed selected texts: {g['previously_unselected']} and {u['previously_unselected']} respectively.",
 f"Support increases: gold {g['improved']}/{g['matched']} = {fmt(g['improved']/g['matched'])}; unjudged {u['improved']}/{u['matched']} = {fmt(u['improved']/u['matched'])}. Mean deltas are {fmt(g['mean_delta'])} and {fmt(u['mean_delta'])}; these are model-support changes, not correctness annotations.",
 f"Q12 qualifiers: the paired control recovers {q12gold[5]} gold candidates at ten without qualifiers and {q12gold[6]} with qualifiers. Semantic class discrimination remains qualitatively unvalidated even if benchmark ranks improve.",
 f"Pool-size/score Spearman: gold {fmt(ng['rho_items_score'])}, unjudged {fmt(nu['rho_items_score'])}; degree/score {fmt(ng['rho_degree_score'])} and {fmt(nu['rho_degree_score'])}. Full max-cosine and per-question diagnostics are exported; no corrective penalty was fitted.",
 f"CR6 moves {entered} canonical targets from H2 ranks 11–100 into the top ten.",
 f"CR6 loses prior top-ten targets: {loss}. These reference sets overlap.",
 f"CR7 recovers {f['at_k']['10']['found']}/47 versus CR6 {b['at_k']['10']['found']}/47; consult the separate preservation table for the retention tradeoff.",
 f"The primary oracle remains 31/47. Gap before: {(31-a['at_k']['10']['found'])}/47 = {fmt((31-a['at_k']['10']['found'])/47)}; after CR6: {(31-b['at_k']['10']['found'])}/47 = {fmt((31-b['at_k']['10']['found'])/47)}.",
 'Failure categories describe operational missing candidates, attribution and support; they cannot determine which scientific mechanism is the dominant cause without independent evidence annotations.',
 'Do not proceed to hierarchy approximation on this evidence alone; validate quality and attribution on unseen questions first.'
]
put('\n'.join(f'{i}. {answer}' for i,answer in enumerate(answers,1)))
put('Selection counts establish whether hidden candidate evidence was accessed; matched-component support statistics compare gold with unjudged candidates. Pool-only controls distinguish access from grammar changes, and the Q12 paired control isolates qualifier participation. Pool-size correlations are descriptive; where positive, they warn of max-pooling opportunity bias without establishing that size caused gains. Grammar changes were motivated by development-set failure traces and the user’s specification, then corrected using unrelated synthetic cases; they were not tuned against benchmark scores. No weights, caps or thresholds were selected by benchmark score. No informative misalignment null was run, which limits causal interpretation.')
put('Remaining failures mix unavailable candidates, weak selected attribution/support and supported-but-low ranks. Operational categories cannot causally apportion missing graph facts, decomposition errors, attribution errors and semantic ranking. Cosine still cannot verify numerical scale, comparative direction beyond shallow patterns, conjunction truth or scientific applicability. Fuller evidence does not make those limitations disappear.')
put('Author judgement: hierarchy approximation remains premature. Recovery and retention are assessed on reused development questions, with an oracle gap and no independent validation of evidence sufficiency. Small changes in recovered targets do not establish generalization or statistical significance. Keep this pass as a quality experiment and test the mechanisms on unseen questions before optimizing retrieval speed. No new neural reranker, H2 discovery change, hierarchy, parameter sweep or post-hoc size correction was added.')
put('## Reproduction and verification')
put("```powershell\n$env:HF_HUB_OFFLINE='1'\n.\\.venv\\Scripts\\python.exe -m tkh_abstraction.cli evaluate-conditioned --config configs/conditioned.yaml\n.\\.venv\\Scripts\\python.exe -m pytest -q --junitxml=artifacts/evaluation/conditioned_evidence_tests.xml\n.\\.venv\\Scripts\\python.exe scripts/report_conditioned.py\n.\\.venv\\Scripts\\python.exe scripts/package_submission.py\n```")
put(f"Full suite: {suite.attrib['tests']} passed, no failures/errors/skips. Tests cover generic qualifiers and conjunctions, participial versus list attribution, wider deterministic deduplicated pools/provenance, component-specific selection, temporal safety, frozen aggregation, exclusions, zero baseline/path weights, exact prefixes and tails, deterministic fusion, no answer-specific rules, and ground-truth independence. The report audit verifies all old source/artifact/metric hashes, reference rankings/metrics, oracle dictionaries, all 14 statuses, source spans, temporal pool containment and score decomposition.")
put('For unseen questions, use `--questions` and `--ground-truth` with separate output paths. An isolated run without copied baseline artifacts requires `conditioned_verify_history: false`; the full assessment report writer expects the supplied two-cutoff benchmark. Historical CR2/CR4 are never executed. Do not run old report writers against the extended workspace. Pools are cached in memory per snapshot and exported; existing embedding caches persist between runs.')
put('Implementation: conditioned_decomposition.py, conditioned_evidence.py, conditioned_pipeline.py, additive CLI command, configs/conditioned.yaml, tests/test_conditioned.py and scripts/report_conditioned.py. Required machine-readable artifacts and test XML are alongside this report; protocol/checkpoints/audit are under conditioned/. Historical constraints metrics/report are archived under artifacts/baseline_constraints.')
put('Ultralight local adaptation completed serial read-only Claude Opus 5 planning and Claude Sonnet 5 review through the official first-party subscription helper. The review prompted conservative generic attribution/conjunction safeguards and stronger pipeline tests; advice was verified rather than adopted blindly. Prompts, results and decisions are in notes/conditioned. Codex implemented, integrated and executed the work. No human scientific adjudication is claimed.')
(ROOT/'conditioned_evidence.md').write_text('\n'.join(lines),encoding='utf8')
write_json(ROOT/'conditioned/final_audit.json',dict(status='passed',tests=int(suite.attrib['tests']),historical_files=len(p['historical_hashes']),frozen_queries=len(run['inventory']),oracle_records=len(run['oracle']),candidate_pools=len(pools),candidate_question_pairs=len(candidate_lookup)))
print('Conditioned report and audit complete')
