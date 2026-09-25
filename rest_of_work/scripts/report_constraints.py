"""Evaluation-only constraint report and preservation audit. Never selects a model or parameter."""
from pathlib import Path
from collections import Counter
import json,re
import xml.etree.ElementTree as ET
import numpy as np
from tkh_abstraction.io import read_json,read_csv,write_json,digest
from tkh_abstraction.v2_pipeline import sha,verify_reuse
from tkh_abstraction.config import read_config
from tkh_abstraction.snapshots import snapshot

root=Path('artifacts/evaluation'); out=root/'constraints'
run=read_json(root/'constraint_reranking_results.json'); p=run['protocol']; cfg=p['config']; rc=p['reranking_config']
support=read_json(root/'requirement_support.json'); attrs=read_json(root/'evidence_attribution.json')
targets=read_json(root/'constraint_reranking_per_target.json'); failures=read_json(root/'constraint_failure_analysis.json')
decomps=read_json(root/'question_decompositions.json'); previous=read_json(root/'reranker_results.json')
diffusion=read_json(root/'hypergraph_diffusion_results.json'); metrics=read_json('artifacts/full/metrics.json')
data=read_json(Path(cfg['data_dir'])/'tkh_collection10.json'); questions=read_csv(Path(cfg['data_dir'])/'questions.csv'); gt=read_json(Path(cfg['data_dir'])/'ground_truth.json')
for path,h in p['historical_hashes'].items(): assert sha(path)==h,('historical',path)
for path,h in p['source_hashes'].items(): assert sha(path)==h,('source',path)
for key,h in p['historical_metrics_hashes'].items(): assert digest(metrics[key])==h,('metrics',key)
assert digest(data)==p['data_hash'] and digest(questions)==p['questions_hash'] and digest(gt)==p['ground_truth_hash']
verify_reuse(read_config(rc['context_config']),data)
suite=ET.parse(root/'constraint_reranking_tests.xml').getroot().find('testsuite')
assert all(int(suite.attrib[k])==0 for k in ['errors','failures','skipped'])
lookup={(r['cutoff'],r['stage']):r for r in run['results']}
old={(r['cutoff'],r['stage']):r for r in previous['results']}
history={(r['cutoff'],r['stage']):r for r in diffusion['results']}
qs={q['question_id'] for q in questions if q['type']=='A'}; assert len(qs)==14
def mean(xs):
    xs=[x for x in xs if x is not None]; return float(np.mean(xs)) if xs else None
def fmt(x): return '—' if x is None else f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x)
def table(headers,rows):
    def cell(x): return fmt(x).replace('|','/').replace('\n',' ')
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(cell(x) for x in r)+' |\n' for r in rows)+'\n'
def target_rank(t,stage): return t['rerankings'][stage]['rank']
def rank_of(rows,ident): return next((i+1 for i,r in enumerate(rows) if r['entity_id']==ident),None)

for r in run['results']:
    year,stage=r['cutoff'],r['stage']; h2={q['question_id']:q for q in old[year,'RR0']['records']}; prior={q['question_id']:q for q in old[year,'RR1-100']['records']}
    assert {q['question_id'] for q in r['records']}==qs
    assert r['summary']['questions']==14 and r['summary']['evaluable']==12 and r['summary']['at_k']['10']['total_target_instances']==63
    for q in r['records']:
        k=q['candidate_k']; base=h2[q['question_id']]; m=q['entity_metrics']
        assert m['candidate_at_k']==base['entity_metrics']['candidate_at_k']
        assert q['candidate_preserved'] and {x['entity_id'] for x in q['returned'][:k]}=={x['entity_id'] for x in base['returned'][:k]}
        assert q['returned'][k:]==base['returned'][k:]
        if stage=='CR0-100': assert q['returned']==prior[q['question_id']]['returned'] and m==prior[q['question_id']]['entity_metrics']
        if q['question_id'] in {'Q5','Q11'}: assert m['status']=='not_evaluable' and m['at_k']['10']['recall'] is None
old_oracle={(o['cutoff'],o['question_id'],o['candidate_k']):o for o in read_json(root/'reranker_oracle.json')['records']}
for o in run['oracle']: assert o==old_oracle[o['cutoff'],o['question_id'],o['candidate_k']]
prior_inv={(i['cutoff'],i['question_id']):i for i in previous['inventory']}
for i in run['inventory']:
    for key in ['prefix100_hash','prefix200_hash']: assert i[key]==prior_inv[i['cutoff'],i['question_id']][key]
for d in decomps:
    for c in d['components']:
        assert d['original_question'][c['start']:c['end']]==c['text']==c['source_span']
        for temporal in d['source_spans']['temporal_condition']: assert c['end']<=temporal['start'] or c['start']>=temporal['end']
visible_by_year={year:{n['id'] for n in snapshot(data,year)['nodes']} for year in cfg['constraint_cutoffs']}
for a in attrs:
    visible=visible_by_year[a['cutoff']]
    for item in a['items']:
        assert item['node_id'] in visible and item['first_seen_year']<=a['cutoff']
        if item['attribution']['category'] in {'COMPARISON_BASELINE','INDIRECT_GRAPH_NEIGHBOR'}: assert item['attribution_weight']==0

support_lookup={(q['cutoff'],q['question_id'],c['entity_id']):c for q in support for c in q['candidates']}
coverage=[]; attribution_counts=[]; movement=[]; category_counts=[]; nli_rows=[]
for q in support:
    for c in q['candidates']:
        for req in c['CR2']['requirements']:
            for e in req.get('nli_evidence',[]):
                prob=e.get('nli')
                if prob and prob['status']=='computed':
                    assert abs(sum(prob[k] for k in ['entailment','neutral','contradiction'])-1)<1e-5
                    nli_rows.append(dict(cutoff=q['cutoff'],question_id=q['question_id'],candidate=c['candidate'],entity_id=c['entity_id'],h2_rank=c['h2_rank'],
                        component=req['component'],hypothesis=req['hypothesis'],hypothesis_role=req['hypothesis_role'],**e))
for year in cfg['constraint_cutoffs']:
    for k in [100,200]:
        for gold in [True,False]:
            rows=[a for a in attrs if a['cutoff']==year and a['h2_rank']<=k and a['gold']==gold]
            for model in ['CR1','CR2']:
                ds=[support_lookup[year,a['question_id'],a['entity_id']][model] for a in rows]
                coverage.append(dict(cutoff=year,k=k,gold=gold,scorer=model,n=len(ds),mean_coverage=mean([d['requirement_coverage'] for d in ds]),
                    mean_soft_coverage=mean([d['soft_coverage'] for d in ds]),all_supported=sum(d['requirement_coverage']==1 for d in ds),
                    partial_supported=sum(0<d['requirement_coverage']<1 for d in ds),none_supported=sum(d['requirement_coverage']==0 for d in ds),
                    contradicted=sum(bool(d['contradicted_requirements']) for d in ds),no_direct_attributable=sum(d['no_direct_attributable_evidence'] for d in ds)))
            counts=Counter(i['attribution']['category'] for a in rows for i in a['items'])
            attribution_counts.append(dict(cutoff=year,k=k,gold=gold,candidate_pairs=len(rows),counts=dict(counts),
                co_membership_only=sum(i['attribution']['co_membership_only'] for a in rows for i in a['items']),
                path_only=sum(i['attribution']['path_only'] for a in rows for i in a['items'])))
    ts=[t for t in targets if t['cutoff']==year]
    for stage in p['stages'][1:]:
        category_counts.append(dict(cutoff=year,stage=stage,counts=dict(Counter(t['rerankings'][stage]['category'] for t in ts))))
        bins=[('H2 1-10',lambda t:t['H2_rank'] is not None and t['H2_rank']<=10),('H2 11-50',lambda t:t['H2_rank'] is not None and 11<=t['H2_rank']<=50),
            ('H2 51-100',lambda t:t['H2_rank'] is not None and 51<=t['H2_rank']<=100),('H2 >100',lambda t:t['H2_rank'] is None or t['H2_rank']>100)]
        for name,predicate in bins:
            xs=[t for t in ts if predicate(t)]
            movement.append(dict(cutoff=year,stage=stage,bin=name,count=len(xs),improved=sum(target_rank(t,stage)<t['H2_rank'] for t in xs),
                same=sum(target_rank(t,stage)==t['H2_rank'] for t in xs),worsened=sum(target_rank(t,stage)>t['H2_rank'] for t in xs),
                entered10=sum(t['H2_rank']>10 and target_rank(t,stage)<=10 for t in xs),left10=sum(t['H2_rank']<=10 and target_rank(t,stage)>10 for t in xs)))
        for reference,key in [('H2','H2_rank'),('CR0','CR0-100')]:
            xs=[t for t in ts if (t['H2_rank'] if reference=='H2' else target_rank(t,key))<=10]
            movement.append(dict(cutoff=year,stage=stage,bin=reference+' prior top10',count=len(xs),
                improved=sum(target_rank(t,stage)<(t['H2_rank'] if reference=='H2' else target_rank(t,key)) for t in xs),
                same=sum(target_rank(t,stage)==(t['H2_rank'] if reference=='H2' else target_rank(t,key)) for t in xs),
                worsened=sum(target_rank(t,stage)>(t['H2_rank'] if reference=='H2' else target_rank(t,key)) for t in xs),entered10=0,left10=sum(target_rank(t,stage)>10 for t in xs)))

diagnostics=dict(coverage=coverage,attribution_counts=attribution_counts,rank_movement=movement,category_counts=category_counts,
    decomposition=dict(mean_explicit_requirements=mean([len(d['requirements']) for d in decomps if d['question_type']=='A']),
        mean_positive_components=mean([sum(c['kind']!='exclusion' for c in d['components']) for d in decomps if d['question_type']=='A']),
        fallback_questions=[d['question_id'] for d in decomps if d['fallback']]),
    nli=dict(pairs=len(nli_rows),label_counts=dict(Counter(e['nli']['prediction'] for e in nli_rows)),truncated=sum(e['nli']['truncated'] for e in nli_rows),
        mean_entailment=mean([e['nli']['entailment'] for e in nli_rows]),mean_neutral=mean([e['nli']['neutral'] for e in nli_rows]),
        interpretation='Selected-pair scientific NLI behavior, not human scientific entailment labels'),
    unassigned_categories=dict(DECOMPOSITION_INCORRECT='Requires manual adjudication; no false claim of exhaustive ground truth',
        DECOMPOSITION_MISSING_REQUIREMENT='No independently annotated decomposition target; zero explicit requirements is not automatically a failure',
        COMPARISON_ATTRIBUTION_FAILURE='Requires trace adjudication; rule-role distributions and controlled probes reported separately'))
tie_rows=[]
for q in support:
    for k in [100,200]:
        for scorer in ['CR1','CR2']:
            ds=[c[scorer] for c in q['candidates'][:k]]; values=[d['score'] for d in ds]; counts=Counter(values)
            tie_rows.append(dict(cutoff=q['cutoff'],question_id=q['question_id'],k=k,scorer=scorer,exact_zeros=sum(v==0 for v in values),
                distinct_scores=len(counts),largest_tie=max(counts.values(),default=0),score_std=float(np.std(values)),fallback_candidates=sum(d['status']!='computed' for d in ds)))
diagnostics['score_ties']=tie_rows
failures.update(diagnostics); write_json(root/'constraint_failure_analysis.json',failures)

stages=p['stages']; primary=2025
def s(stage,year=2025): return lookup[year,stage]['summary']
def r10(stage,year=2025): return s(stage,year)['at_k']['10']['micro_recall']
def hits(stage,year=2025): return s(stage,year)['at_k']['10']['found']
def gap(stage,year=2025):
    k=int(stage.split('-')[1]); os=[o for o in run['oracle'] if o['cutoff']==year and o['candidate_k']==k]
    return sum(o['oracle_found_at10'] for o in os)/sum(o['canonical_targets'] for o in os)-r10(stage,year)

text='# Requirement-aware candidate reranking\n\n'
text+=f'At the conservative 2025 cutoff, CR0 (unchanged RR1-100) recovers {hits("CR0-100")}/47 canonical target instances at ten. Requirement cosine CR1 recovers {hits("CR1-100")}/47; controlled NLI CR2 recovers {hits("CR2-100")}/47; H2 fusion CR3/CR4 recover {hits("CR3-100")}/47 and {hits("CR4-100")}/47. The frozen H2-100 oracle remains 31/47. These are fixed-default development diagnostics on the repeatedly used benchmark; no held-out generalization, human scientific validation or calibrated entailment probability is claimed.\n\n'
text+='The experiment reuses `FrozenH2Candidates`, the exact prior `CandidateEvidenceIndex` bundles, `CandidateReranker`, `ThetaOperator`, `MentionIndex`, grouped entities, resolver, BGE cache, local `NLI`, temporal snapshots, reranking metrics and the unchanged oracle. H2 alpha=.85/M100/weights/eligibility/RRF and every previous source module except the CLI remain unchanged. Every historical H2 prefix, per-question metric, CR0 ranking/metric, previous report and metric section is audited. Primary K is 100; CR1/CR2 K200 are diagnostics. CR5 was not added. No hierarchy or new candidate-discovery mechanism was implemented.\n\n'
text+='## Generic decomposition and fixed scoring\n\n'
text+='`decompose_question(question, declared_type)` uses generic interrogative framing, task/condition connectors, conjunctions, negative markers and dates. Fields are original contiguous substrings with exact start/end offsets; requirements are free-form strings, with no materials-science taxonomy or expected-answer logic. No suitable local generative model exists; no model was used for decomposition, no model downloaded and no remote decomposition API called. Original question text is always retained. Temporal phrases are diagnostic and excluded from support vectors; the existing selected snapshot remains authoritative, including the original 2025 conservative / 2026 annual sensitivity interpretation. Dates such as “after” do not introduce a new lower-bound filter.\n\n'
text+='The parser is deliberately shallow: it can fragment conjunctions, mistake descriptive answer-type modifiers for framing, or leave a compound task intact. It does not infer missing properties. The report exports every decomposition rather than claiming correctness from exact source spans. A source span verifies extraction fidelity, not that the phrase is a logically complete condition.\n\n'
text+='A concrete limitation is Q12: the extracted task is “large-scale atomistic simulation”, while the answer-head qualifiers “ML-based interatomic potential” are not retained as separate scored requirements. Q5 splits “crystalline symmetries” away from the later shared predicate “must be preserved”. These are qualitative decomposition concerns, not automatically inferred from retrieval failure. They limit a causal claim that the experiment understands every condition. No answer-specific paraphrase was inserted to repair them.\n\n'
text+=table(['Q','Answer type','Core task','Requirements','Exclusions','Temporal','Warnings'],[
    [d['question_id'],d['answer_type'],d['core_task'],'; '.join(d['requirements']),'; '.join(d['exclusions']),d['temporal_condition'],'; '.join(d['warnings'])] for d in decomps])
text+=f'Type A mean explicit requirements/question: {fmt(diagnostics["decomposition"]["mean_explicit_requirements"])}; mean positive components including the core: {fmt(diagnostics["decomposition"]["mean_positive_components"])}. Type B decompositions are diagnostics only; Type B retrieval and all its prior outputs remain unchanged.\n\n'
text+='CR1 compares each task/requirement string with each previously selected evidence text using normalized BGE-small-en-v1.5 vectors. Evidence embeddings are reused from `ContextIndex.X`; only component strings need encoding. Each component takes the maximum of nonnegative cosine × attribution weight. Different requirements can select different items.\n\n'
text+='Attribution weights, fixed before outcomes: direct subject 1, safe explicit-name support .8, comparison winner 1, comparison baseline 0, mention-only .15, uncertain .25, indirect graph neighbor 0. Safe names use unchanged boundaries and graph-derived aliases. Generic active/passive comparison direction is checked before subject/list heuristics. An unnegated comparison baseline cannot unconditionally boost the named candidate. A direct low-arity edge without an explicit subject remains UNCERTAIN; co-membership is not a predicate. Unnamed path/provenance text is diagnostic-only; attributable path text may qualify by its textual role. Original graph labels and paths are preserved. These heuristics are not a complete relation extractor.\n\n'
text+='For positive component supports `s`, the ranking score is `core_support × geometric_mean(s) × product(1 − exclusion_violation) × soft_type_prior`. A missing core gates the score to zero; one strong component cannot compensate freely for weak support elsewhere. Answer type uses only a .95 mismatch factor for recognized generic entity families and never filters candidates. “Requirement coverage” is separately defined as the fraction of positive components ≥.5; this fixed descriptive threshold is not used in ranking. Geometric support is a smooth strength/coverage proxy, not a probability of satisfying every condition.\n\n'
text+='CR2 uses `cross-encoder/nli-deberta-v3-small`, pinned revision `fa2804872c3b4bd748f38c0185cc85775361e735`, strictly local. It is a three-way NLI classifier, **not a relevance cross-encoder**. For each candidate/component, the single highest attributed-cosine eligible item is preselected. The hypothesis explicitly includes the candidate: task “<candidate> is suitable for <task>”; condition “<candidate> supports <task> with <condition>” or “is suitable … when …”. This literal templating avoids invented paraphrases but sometimes produces awkward or stronger-than-premise statements.\n\n'
text+='Positive NLI support is `weight × max(0, entailment − contradiction)`. Exclusions use a positive violation hypothesis, “<candidate> involves <excluded phrase> for <task>”; entailment reduces the final score. Neutral neither supplies positive task support nor proves exclusion compliance. CR1 uses similarity to the excluded phrase as a coarse violation proxy and cannot distinguish its negation. CR2 preserves all three probabilities, predicted labels, input token counts and premise-only truncation flags. Any NLI execution failure/unavailability makes the entire question use explicit CR1 fallback, avoiding mixed cosine/NLI scales; failed probabilities remain null. Fallback is never silently presented as NLI success. Exact premise/hypothesis cache reuse and forward work are separately counted.\n\n'
text+='CR3 and CR4 fuse H2 with CR1 and CR2 respectively using unchanged RRF60 and deterministic average ranks for exact score ties. No mixing weights, support thresholds, parser rules, attribution weights or NLI parameters were selected by benchmark score. An attribution or decomposition change motivated by an actual generic bug is logged separately from modeling limitations.\n\n'

headers=['Stage','K','Hit1','Hit5','Hit10','Hit20','MacroR5','MacroR10','MacroR20','MicroR5','MicroR10','MicroR20','UniqueR10','Raw /63','MRR','R-Prec','R@R','Median','Mean rank','Preserved','Support comparisons','NLI requests','NLI new pairs','NLI sec','New stage sec','Total sec*']
for year in cfg['constraint_cutoffs']:
    text+=f'## Metrics: {year}\n\n'; rows=[]
    for stage in stages:
        result=lookup[year,stage]; sm=result['summary']; a=sm['at_k']; cost=sm['constraint_cost']; raw=sum(q['entity_metrics']['at_k']['10']['raw_found'] for q in result['records'])
        rows.append([stage,stage.split('-')[1]]+[a[str(k)]['macro_hit'] for k in [1,5,10,20]]+[a[str(k)]['macro_recall'] for k in [5,10,20]]+
            [a[str(k)]['micro_recall'] for k in [5,10,20]]+[a['10']['unique_target_recall'],f'{raw}/63',sm['mrr'],sm['r_precision'],sm['recall_at_r'],sm['rank_quantiles']['median'],sm['mean_target_rank'],
            sm['candidate_preservation_rate'],cost['semantic_support_comparisons'],cost['nli_pair_requests'],cost['nli_forward_pairs'],cost['nli_seconds'],cost['runtime_seconds'],cost['total_query_seconds']])
    text+=table(headers,rows)
    text+=table(['Reference','MicroR10','Median gold rank','Frozen C100 macro','Frozen C200 macro'],[
        [stage,history[year,stage]['summary']['at_k']['10']['micro_recall'],history[year,stage]['summary']['rank_quantiles']['median'],history[year,stage]['summary']['candidate_recall']['100'],history[year,stage]['summary']['candidate_recall']['200']] for stage in ['R3','H2','H3']])
text+='All 14 Type A questions remain in every stage. Q5/Q11 have no canonical entity units and explicit null recall; their labels remain in the raw /63 denominator. Macro metrics average 12 evaluable questions; micro denominators are 47 canonical instances at 2025 and 50 at 2026. Unique recall retains the prior union-of-target-names definition. Costs average all 14. Candidate preservation is exact at both pools; candidates outside the reranked prefix retain their H2 order.\n\n'
text+='*Total seconds is historical H2 search + prior evidence construction + measured new preparation/scoring/fusion, not a cold end-to-end latency benchmark. Shared top200 component/evidence-vector preparation and NLI aggregation are conservatively charged to primary runs; this can overcount K100. CR4 reuses CR2’s batch results but is reported with standalone component cost. Cached NLI calls do not imply new neural forward work. Model loading, evaluation/export and most existing graph indexing are outside the per-query component sum. Inspect exact cache hits/forward pairs and run logs before comparing machines or reruns.\n\n'
text+='## Oracle gap\n\n'
orows=[]
for year in cfg['constraint_cutoffs']:
    for stage in stages:
        k=int(stage.split('-')[1]); os=[o for o in run['oracle'] if o['cutoff']==year and o['candidate_k']==k]; found=sum(o['oracle_found_at10'] for o in os); total=sum(o['canonical_targets'] for o in os)
        orows.append([year,stage,f'{found}/{total}',found/total,r10(stage,year),gap(stage,year)])
text+=table(['Year','Stage','Unchanged oracle','Oracle microR10','Actual microR10','Gap'],orows)
text+='The oracle remains evaluation-only and byte-equivalent per question to the preceding experiment. It maximizes distinct canonical coverage in ten slots; no oracle membership enters decomposition, attribution, cosine, NLI, or fusion.\n\n'

text+='## Requirement and attribution diagnostics\n\n'
text+=table(['Year','K','Group','Scorer','Candidates','Mean coverage','Soft coverage','All supported','Partial','None','Contradiction','No direct attributable evidence'],[
    [a['cutoff'],a['k'],'gold' if a['gold'] else 'not gold',a['scorer'],a['n'],a['mean_coverage'],a['mean_soft_coverage'],a['all_supported'],a['partial_supported'],a['none_supported'],a['contradicted'],a['no_direct_attributable']] for a in coverage])
text+='Counts are candidate/question pairs, with fraction = count/candidates. “Supported” means model score ≥.5 under the fixed attribution policy, not human-verified entailment. No direct attributable evidence means no direct-subject, explicit-name, or comparison-winner item among the capped prior bundle; uncertain graph-linked items can still contribute .25. Neither this count nor a zero NLI score proves the underlying graph lacks sufficient information. Gold membership is applied only in evaluation.\n\n'
roles=list(cfg['constraint_attribution_weights'])
text+=table(['Year','K','Group']+roles+['Co-membership flag','Path-only flag'],[[a['cutoff'],a['k'],'gold' if a['gold'] else 'not gold']+[a['counts'].get(role,0) for role in roles]+[a['co_membership_only'],a['path_only']] for a in attribution_counts])
text+='Attribution categories are mutually exclusive per item; graph co-membership/path flags overlap them. Comparison winner/baseline labels refer to the detected text relation, not whether a method is globally better or meets the question.\n\n'
text+='## Rank movement and failures\n\n'
text+=table(['Year','Stage','Original bin','N','Improved','Same','Worsened','Entered10','Left10'],[[a[k] for k in ['cutoff','stage','bin','count','improved','same','worsened','entered10','left10']] for a in movement])
cats=sorted({cat for c in category_counts for cat in c['counts']})
text+=table(['Year','Stage']+cats,[[c['cutoff'],c['stage']]+[c['counts'].get(cat,0) for cat in cats] for c in category_counts])
text+='Primary categories are mutually exclusive: pool absence, top-ten recovery, no direct attributable item, predicted contradiction, zero/partial diagnostic coverage, then supported-but-below-ten. A zero-coverage NLI failure is an operational scoring failure relative to a gold target, not proof the classifier misunderstood the premise. Decomposition errors and comparison-attribution errors require qualitative adjudication; they are not fabricated automatically from answer membership. Additional flags are exported.\n\n'

def candidate_for(year,qid,label):
    t=next(t for t in targets if t['cutoff']==year and t['question_id']==qid and t['label']==label)
    q=next(q for q in support if q['cutoff']==year and q['question_id']==qid)
    c=next((c for c in q['candidates'] if set(c['node_ids'])&set(t['node_ids'])),None)
    return t,q,c
def trace(year,qid,label,title):
    t,q,c=candidate_for(year,qid,label)
    if c is None: return f'### {title}: {qid}, {label}\n\nOutside H2 top200; no support score was computed.\n\n'
    rows=[]
    for i,r in enumerate(c['CR1']['requirements']):
        n=c['CR2']['requirements'][i]; ce=r['best_evidence']; ne=n.get('best_evidence'); pr=ne.get('nli') if ne else None
        rows.append([r['component']['kind'],r['component']['text'],ce['node_id'] if ce else None,ce['attribution'] if ce else None,
            ce['text'] if ce else None,r['cosine_support'],n.get('nli_support'),pr['entailment'] if pr else None,pr['neutral'] if pr else None,pr['contradiction'] if pr else None,r['hypothesis']])
    result=f'### {title}: {qid}, {label} ({year})\n\nQuestion: {q["original_question"]}\n\nR3 {t["R3_rank"]} → H2 {t["H2_rank"]}; '+', '.join(f'{stage} {target_rank(t,stage)}' for stage in stages)+'.\n\n'
    result+=f'CR1 coverage {c["CR1"]["requirement_coverage"]:.3f}, score {c["CR1"]["score"]:.6f}; CR2 coverage {c["CR2"]["requirement_coverage"]:.3f}, score {c["CR2"]["score"]:.6f}.\n\n'
    return result+table(['Kind','Extracted condition','Evidence node','Attribution','Evidence text','Cosine support','NLI support','Entail','Neutral','Contradict','Candidate-specific hypothesis'],rows)

text+='## Actual traces and known failure patterns\n\n'
for qid,label in [('Q12','M3GNet'),('Q13','SchNet'),('Q2','HamGNN'),('Q1','MACE')]: text+=trace(2025,qid,label,'Required diagnostic')
text+='M3GNet Q12 is a clear technical NLI failure in this trace: the premise reports successful molecular-dynamics simulations and compatible conductivity/activation energies, yet the model assigns about .995 contradiction to suitability for large-scale atomistic simulation. The premise does not establish the requested scale, so neutrality would be defensible; it does not assert that large-scale use is impossible. Contradiction minus entailment clips support to zero and the zero-score tie policy further affects rank. This is a qualitative logical assessment of the actual trace, not an external scientific performance certification.\n\n'
text+='A remaining attribution error affects SchNet: “Including force information in the training loss causally improves SchNet’s generalization …” is classified as MENTION_ONLY because the generic “including” heuristic mistakes a participial subject for an illustrative list. Its graph provenance is retained, but the .15 weight can make a less specific task label win evidence selection. This is reported as an attribution limitation; the rule was not patched specifically for this benchmark sentence.\n\n'
comparison_examples=[]
for a in attrs:
    if a['cutoff']!=2025 or a['candidate'].casefold() not in {'m3gnet','schnet'}: continue
    for item in a['items']:
        if item['attribution']['comparison'] is not None: comparison_examples.append([a['question_id'],a['candidate'],item['node_id'],item['attribution']['category'],item['text'],item['attribution_weight']])
text+=table(['Q','Candidate','Node','Detected role','Comparative premise','New positive support weight'],comparison_examples[:8])
text+='Detected baselines now have zero positive support. The multi-comparison MEGNet/SchNet statement is UNCERTAIN and receives .25 rather than unconditional full support. The former SevenNet/M3GNet baseline statement receives zero. This tests attribution behavior; it does not establish that the resulting rank improves. A comparison can still be relevant background even when it is not positive evidence for the requested condition.\n\n'
primary_targets=[t for t in targets if t['cutoff']==2025 and t['H2_rank']<=100]
used=set()
for title,predicate,key in [
    ('All scored requirements supported and rank improved',lambda t,c:c['CR1']['requirement_coverage']==1 and target_rank(t,'CR1-100')<t['H2_rank'],lambda t:t['H2_rank']-target_rank(t,'CR1-100')),
    ('Partial support and demotion',lambda t,c:0<c['CR1']['requirement_coverage']<1 and target_rank(t,'CR1-100')>t['H2_rank'],lambda t:target_rank(t,'CR1-100')-t['H2_rank']),
    ('NLI improves a gold rank relative to cosine',lambda t,c:target_rank(t,'CR2-100')<target_rank(t,'CR1-100'),lambda t:target_rank(t,'CR1-100')-target_rank(t,'CR2-100')),
    ('NLI worsens a gold rank relative to cosine',lambda t,c:target_rank(t,'CR2-100')>target_rank(t,'CR1-100'),lambda t:target_rank(t,'CR2-100')-target_rank(t,'CR1-100')),
    ('Good H2 rank but no direct attributable evidence',lambda t,c:t['H2_rank']<=50 and c['CR1']['no_direct_attributable_evidence'],lambda t:100-t['H2_rank'])]:
    choices=[t for t in primary_targets if (c:=candidate_for(2025,t['question_id'],t['label'])[2]) is not None and predicate(t,c)]
    if choices:
        t=max(choices,key=key); text+=trace(2025,t['question_id'],t['label'],title)
    else: text+=f'### {title}\n\nNo matching gold example occurred under the fixed operational definition; none is invented.\n\n'

text+='## Local NLI validity diagnostics\n\n'
text+='Standalone exact-score ties retain the existing entity-ID tie break; fusion uses average tied ranks before combining with H2. A zero component makes the geometric score zero, so large tied groups can lose H2 ordering. No epsilon or new tie policy was fitted after observing results. The core is deliberately counted both as a gate and inside the geometric mean: its exponent is 1 + 1/n for n positive components.\n\n'
text+=table(['Year','Q','K','Scorer','Exact zeros','Distinct scores','Largest tie','Score std','Fallback candidates'],[
    [a[k] for k in ['cutoff','question_id','k','scorer','exact_zeros','distinct_scores','largest_tie','score_std','fallback_candidates']] for a in tie_rows])
text+=f'Selected evidence/hypothesis pairs: {len(nli_rows)} (includes repeated uses across candidates/cutoffs, unlike unique forward-pair cost). Predicted labels: {json.dumps(diagnostics["nli"]["label_counts"])}; truncated premises: {diagnostics["nli"]["truncated"]}. Mean entailment {fmt(diagnostics["nli"]["mean_entailment"])}, mean neutral {fmt(diagnostics["nli"]["mean_neutral"])}.\n\n'
diagnostic_cases={
    'Highest entailment':sorted(nli_rows,key=lambda x:-x['nli']['entailment'])[:3],
    'Highest contradiction':sorted(nli_rows,key=lambda x:-x['nli']['contradiction'])[:3],
    'Highest neutral':sorted(nli_rows,key=lambda x:-x['nli']['neutral'])[:3],
    'Numerical/scaling wording':sorted([x for x in nli_rows if re.search(r'\d|\b(?:thousands|millions|scale|scaling)\b',x['component']['text'],re.I)],key=lambda x:-x['nli']['entailment'])[:3],
    'Negation/exclusion':sorted([x for x in nli_rows if x['hypothesis_role']=='exclusion_violation' or re.search(r'\b(?:not|without|never)\b',x['text'],re.I)],key=lambda x:-x['nli']['entailment'])[:3],
    'Conjunction':sorted([x for x in nli_rows if re.search(r'\b(?:and|or)\b',x['hypothesis'],re.I)],key=lambda x:-x['nli']['entailment'])[:3]}
write_json(out/'nli_trace_diagnostics.json',diagnostic_cases)
for name,rows in diagnostic_cases.items():
    text+=f'### {name}\n\n'+table(['Year/Q','Candidate','Evidence','Hypothesis','Entail','Neutral','Contradict'],[
        [str(e['cutoff'])+'/'+e['question_id'],e['candidate'],e['text'],e['hypothesis'],e['nli']['entailment'],e['nli']['neutral'],e['nli']['contradiction']] for e in rows])
probe_path=out/'nli_control_probes.json'
if probe_path.exists():
    probes=read_json(probe_path); text+='### Controlled comparison and technical probes\n\n'
    text+=table(['Kind','Premise','Hypothesis','Entail','Neutral','Contradict'],[[r['kind'],r['premise'],r['hypothesis'],r['nli']['entailment'],r['nli']['neutral'],r['nli']['contradiction']] for r in probes['records']])
    text+=probes['interpretation']+'\n\n'
text+='The actual rank disagreements above separate cosine/NLI behavior, but not their causal scientific correctness. High neutral probabilities can be appropriate when a premise states a property but the template asserts general suitability; a high-entailment numerical extrapolation or wrong comparison subject is stronger evidence of failure. The supplied answer key does not label these premise/hypothesis pairs, so classifier errors are not equated automatically with gold rank losses.\n\n'

text+='## Every canonical target\n\n'
text+=table(['Year','Q','Target','R3','H2']+stages,[[t['cutoff'],t['question_id'],t['label'],t['R3_rank'],t['H2_rank']]+[target_rank(t,stage) for stage in stages] for t in targets])
text+='## Decision questions\n\n'
text+=f'1. CR1 versus CR0 at 2025: {hits("CR1-100")}/47 versus {hits("CR0-100")}/47; micro Recall@10 change {r10("CR1-100")-r10("CR0-100"):+.4f}.\n'
text+=f'2. CR2 versus CR1: {hits("CR2-100")}/47 versus {hits("CR1-100")}/47; change {r10("CR2-100")-r10("CR1-100"):+.4f}.\n'
text+='3. Candidate-specific NLI is evaluated as a controlled entailment component only. Its technical suitability is assessed by the quality table, rank losses, neutral/contradiction distributions and actual probes; existence of a local classifier is not evidence it should become default.\n'
text+=f'   Here it does not improve the cosine result: CR2 retrieves {hits("CR2-100")}/47 against CR1’s {hits("CR1-100")}/47, with the technical contradiction failure illustrated above.\n'
text+=f'4. Oracle gap: CR0 {gap("CR0-100"):.4f}, CR1 {gap("CR1-100"):.4f}, CR2 {gap("CR2-100"):.4f}. The combined decomposition/attribution/aggregation ablation cannot isolate the causal effect of decomposition alone.\n'
text+='5. New top-ten entries from H2 ranks 11–100: '+', '.join(f'{stage}: {sum(11<=t["H2_rank"]<=100 and target_rank(t,stage)<=10 for t in primary_targets)}' for stage in stages[1:5])+'.\n'
text+='6. Existing top-ten losses are tabulated separately against H2 and CR0; the two references overlap and are not added together.\n'
text+='   Primary losses (H2 / CR0): '+', '.join(f'{stage}: {sum(t["H2_rank"]<=10 and target_rank(t,stage)>10 for t in primary_targets)} / {sum(target_rank(t,"CR0-100")<=10 and target_rank(t,stage)>10 for t in primary_targets)}' for stage in stages[1:5])+'.\n'
text+='7. The SevenNet/M3GNet statement receives zero positive support. The multi-comparison SchNet statement is UNCERTAIN (.25). Unconditional full support is removed, but actual ranks need not improve.\n'
text+='8. Primary failures and diagnostic flags distinguish absent candidates, weak textual attribution, zero/partial support, model contradiction and supported-but-below-ten. The experiment cannot reliably quantify decomposition correctness or scientific attribution accuracy without independent annotations.\n'
text+='9. CR3/CR4 retention and entry counts are in the movement table; RRF can retain H2 relevance but also suppress a standalone scorer’s gains. Fixed fusion is not an automatic improvement.\n'
text+='   At 2025, CR3 preserves all three H2 hits and five of six CR0 hits, while CR1 preserves two and three respectively. CR4 preserves two H2 hits and two CR0 hits. CR3 is the better retention tradeoff in this run.\n'
text+='10. The next-step decision is based on absolute recovery, protection of prior hits, oracle gap and trace validity, not the best single benchmark score. No hierarchical approximation, extra model, CR5 or parameter sweep was added in response to poor results.\n\n'
text+='**Decision:** retain generic requirement decomposition and attributed cosine as a promising experimental direction. Reject this pinned local NLI model/template/top-one configuration as the default for this retrieval use case: it underperforms CR1, adds inference work and loses prior good answers. This is not a rejection of every possible NLI model. CR3 offers a retention tradeoff, not a reason to replace H2 discovery. With roughly three quarters of canonical targets still absent from the top ten, imperfect decomposition/attribution, and no unseen-question validation, hierarchy approximation remains premature. The next quality study should validate these mechanisms on unseen questions before optimizing fine-level retrieval for speed.\n\n'

text+='## Reproduction, tests and provenance\n\n'
text+='```powershell\n$env:HF_HUB_OFFLINE=\'1\'\n.\\.venv\\Scripts\\python.exe -m tkh_abstraction.cli evaluate-constraints --config configs/constraints.yaml\n.\\.venv\\Scripts\\python.exe -m pytest -q --junitxml=artifacts/evaluation/constraint_reranking_tests.xml\n.\\.venv\\Scripts\\python.exe scripts/constraint_nli_probes.py\n.\\.venv\\Scripts\\python.exe scripts/report_constraints.py\n.\\.venv\\Scripts\\python.exe scripts/package_submission.py\n```\n\n'
text+='Unseen questions use the existing `--questions` and `--ground-truth` CLI arguments and separate configured output paths. For another graph, or an isolated output directory without copied frozen baseline artifacts, set `constraint_verify_history: false`; no new question IDs or answer names need code changes. Changing the answer key changes evaluation only. The assessment audit expects the supplied full two-cutoff run. Do not run old report writers to overwrite historical reports.\n\n'
text+=f'Full suite: {suite.attrib["tests"]} passed, no errors/failures/skips. Added tests cover frozen old sources/oracle, CR0 parity, exact extractive spans, unrelated question wording, independent requirements, exclusions, temporal isolation, identity-bearing hypotheses, different evidence per condition, comparison direction/negation, mention/path/co-membership discount, core gating, NLI probabilities and fallback, offline loaders, no benchmark-rule strings, candidate preservation, ground-truth perturbation invariance and Type B/history preservation. Existing H2 recomputation, RRF and hierarchy regression tests remain in the suite.\n\n'
text+='Files: `question_decomposition.py` (`QuestionDecomposition`, `decompose_question`, `candidate_hypothesis`), `evidence_attribution.py` (`classify_attribution`), `constraint_scoring.py` (`ControlledLocalNLI`, `RequirementCosineReranker`, `RequirementNLIReranker`, `PreparedReranker`, `aggregate_support`), `constraint_pipeline.py` (`run_constraints`, diagnostic classification), CLI extension, `configs/constraints.yaml`, tests and two report/probe scripts. All seven requested machine-readable/test artifacts and this report are under `artifacts/evaluation`; checkpoints/protocol/audit/probes are under `constraints/`. Previous metrics/report are also archived under `artifacts/baseline_reranking`.\n\n'
text+='Ultralight local adaptation provided serial read-only Claude Opus planning and Claude Sonnet review via the official first-party subscription helper. Codex implemented, integrated and executed. Advice was checked rather than adopted blindly: graph co-membership is not subject support, and unavailable NLI probabilities are null rather than invented neutral results. Prompts/results, generic bug fixes and decisions are retained in `notes/constraints`. No human scientific adjudication is claimed.\n'
(root/'constraint_reranking.md').write_text(text,encoding='utf-8')
audit=dict(status='passed_with_research_limitations',tests=suite.attrib,records=sum(len(r['records']) for r in run['results']),
    H2_prefix_and_metric_parity=True,CR0_exact_RR1_parity=True,oracle_unchanged=True,candidate_preservation=True,all14_questions_retained=True,
    source_hashes_verified=True,historical_reports_artifacts_metrics_unchanged=True,type_b_and_hierarchy_unchanged=True,
    source_spans_verified=True,temporal_evidence_verified=True,nli_probabilities_verified=True,nli_status=run['nli_model']['status'])
write_json(out/'final_audit.json',audit); print(json.dumps(audit,indent=2))
