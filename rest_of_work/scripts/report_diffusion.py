"""Audit frozen history, inspect diffusion mechanisms, and render measured results."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,xml.etree.ElementTree as ET
import numpy as np
from tkh_abstraction.io import read_json,read_csv,write_json,digest
from tkh_abstraction.snapshots import snapshot
from tkh_abstraction.config import read_config
from tkh_abstraction.v2_pipeline import verify_reuse,sha

root=Path('artifacts/evaluation'); out=root/'diffusion'; cfg=read_config('configs/diffusion.yaml')
bundle=read_json(root/'hypergraph_diffusion_results.json'); assert bundle['status']=='computed'
protocol=bundle['protocol']; results=bundle['results']; summary={(r['cutoff'],r['stage']):r['summary'] for r in results}
diagnostics=read_json(root/'hypergraph_diffusion_diagnostics.json'); targets=read_json(root/'diffusion_per_target.json')
traces=read_json(root/'propagation_traces.json'); sensitivities=read_json(root/'diffusion_sensitivity.json')['records']
metrics=read_json('artifacts/full/metrics.json'); baseline=read_json('artifacts/baseline_contextual/metrics.json')
assert all(metrics[k]==v for k,v in baseline.items()),'Historical metrics changed'
assert all(sha(p)==h for p,h in protocol['historical_artifact_hashes'].items()),'Historical contextual files changed'
assert all(sha(p)==h for p,h in protocol['source_hashes'].items()),'Diffusion source changed after run'
verify_reuse(read_config('configs/contextual.yaml'),read_json('data/data/tkh_collection10.json'))
questions=read_csv('data/data/questions.csv'); qids={q['question_id'] for q in questions if q['type']=='A'}
data=read_json('data/data/tkh_collection10.json'); node_lookup={n['id']:n for n in data['nodes']}
visible_ids={y:{n['id'] for n in snapshot(data,y)['nodes']} for y in cfg['diffusion_cutoffs']}
expected=set(cfg['diffusion_stages']); counts=Counter()
for result in results:
    year=result['cutoff']; stage=result['stage']; counts[year]+=1
    assert {r['question_id'] for r in result['records']}==qids and len(result['records'])==len(qids)
    assert result['summary']['at_k']['10']['total_target_instances']==63
    assert result['summary']['rank_coverage']['ranked']==result['summary']['rank_coverage']['canonical']
    for row in result['records']:
        # Historical contextual field aggregation uses float32; retain its established tolerance.
        tolerance=1e-6 if stage.startswith('R') else 1e-10
        assert all(abs(r['score']-sum(r['score_components'].values()))<tolerance for r in row['returned'])
    if stage in {'R0','R1','R3'}:
        old=read_json(root/'contextual'/f'{stage}_{year}.json')
        historical={r['question_id']:r for r in old['records'] if r['question_type']=='A'}
        for row in result['records']:
            prior=historical[row['question_id']]['entity_metrics']; current=row['entity_metrics']
            assert all(current['at_k'][k]==v for k,v in prior['at_k'].items()),'Baseline scoring changed'
            for key in ['mrr','r_precision','recall_at_r','candidate_recall_50','candidate_recall_100']:
                assert current[key]==prior[key]
for year in cfg['diffusion_cutoffs']:
    assert {r['stage'] for r in results if r['cutoff']==year}==expected
    assert {s['stage'] for s in sensitivities if s['cutoff']==year}==set(cfg['diffusion_sensitivity_stages'])
for d in diagnostics:
    assert d['theta_applications']==d['iterations']==len(d['history'])
    assert d['iterations']<=cfg['diffusion_max_iterations']
    with np.load(d['node_scores_artifact']) as a:
        assert np.isfinite(a['f']).all() and np.min(a['f'])>=0
        assert np.isclose(a['y'].sum(),1.)
        assert set(a['node_ids'])==visible_ids[d['cutoff']]

# Enrich diagnostic categories from exported seed paths; these are proxies, not human relevance labels.
bydiag={(d['cutoff'],d['question_id'],d['stage']):d for d in diagnostics}
for target in targets:
    support=target['support'].get('H0',{}); paths=support.get('paths',[])
    top20={s['node_id'] for s in bydiag[(target['cutoff'],target['question_id'],'H0')]['semantic_seeds']['top_seeds'][:20]}
    target['strong_claim_article_seed_proxy']=any(p['seed']['type'] in {'claim','article','task','problem'} and p['seed']['node_id'] in top20 and 1<=p['hops']<=2 for p in paths)
    if target['strong_claim_article_seed_proxy'] and 'top20_evidence_seed_within_two_hops' not in target['categories']:
        target['categories'].append('top20_evidence_seed_within_two_hops')
    before=target['ranks']['R3_answer_pool']; after=target['ranks'].get('H3')
    target['H3_vs_matched_R3_rank_change']=before-after if before is not None and after is not None else None
write_json(root/'diffusion_per_target.json',targets)
primary=[t for t in targets if t['cutoff']==2025]; p={k[1]:v for k,v in summary.items() if k[0]==2025}
pools={y:read_json(out/f'pool_controls_{y}.json') for y in cfg['diffusion_cutoffs']}
matched=pools[2025]['R3_answer_pool']['summary']
category_rows=[]
for category in sorted({c for t in primary for c in t['categories']}):
    group=[t for t in primary if category in t['categories']]; delta=[t['H3_vs_matched_R3_rank_change'] for t in group]
    category_rows.append(dict(category=category,instances=len(group),improved=sum(x>0 for x in delta),unchanged=sum(x==0 for x in delta),worsened=sum(x<0 for x in delta),median_rank_gain=float(np.median(delta))))
seed_counts=Counter(); seed_mass=defaultdict(float); path_types=Counter(); path_relations=Counter()
for d in diagnostics:
    if d['cutoff']==2025 and d['stage']=='H0':
        for s in d['semantic_seeds']['top_seeds']: seed_counts[s['type']]+=1; seed_mass[s['type']]+=s['y']
for target in primary:
    if target['ranks']['H2']<=100:
        for path in target['support']['H2']['paths']:
            path_types[path['seed']['type']]+=1
            for step in path['path']: path_relations[step['relation']]+=1
mechanisms=dict(categories=category_rows,seed_node_counts=dict(seed_counts),mean_seed_mass_by_type={t:v/len(qids) for t,v in seed_mass.items()},
    seed_types_in_H2_top100_paths=dict(path_types),relations_in_H2_top100_paths=dict(path_relations),
    interpretation='Path counts overlap and describe selected short positive-seed paths, not causal utility, flow fractions or scientific entailment')
write_json(out/'mechanism_summary.json',mechanisms)

def f(x): return 'null' if x is None else f'{x:.4f}'
def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(str(x).replace('|','/') for x in row)+' |\n' for row in rows)+'\n'
def c(stage,k=100): return p[stage]['candidate_recall'][str(k)]
def rec(stage): return p[stage]['at_k']['10']['found']
def med(stage): return p[stage]['rank_coverage']['censored_median']

text='# Query-seeded native hypergraph diffusion\n\n'
text+='This is a Type A candidate-generation experiment. Graph construction, hierarchy memberships, resolver, grouping rules, historical R0–R5 artifacts and Type B evidence outputs are preserved. Defaults were declared before diffusion outcomes: alpha=0.85, top-M=100, tolerance=1e-8, maximum 100 iterations, RRF k=60. Sensitivities do not select a new default.\n\n'
text+='## Mathematical mechanism and attribution\n\n'
text+='[Zhou, Huang and Schölkopf (NIPS 2006), sections 5 and 7](https://proceedings.neurips.cc/paper_files/paper/2006/file/dff8e9c2ac33381546d96deea9922999-Paper.pdf) define the symmetric normalized hypergraph operator and transductive inference from initial **labels**. They did not evaluate semantic-query retrieval. Our adaptation supplies atomic query–node cosine relevance as the initial signal. The existing spectral code returns L=I−Theta, so `ThetaOperator.apply_theta(x)` reuses `x−L@x`. Dense allocation and inversion occur only in tiny correctness tests.\n\n'
text+='```text\nTheta = Dv^(-1/2) H W De^(-1) H.T Dv^(-1/2)\ny = L1_normalize(positive top-M atomic semantic scores)\nf_next = (1-alpha)*y + alpha*Theta*f\nf_limit = (1-alpha)*(I-alpha*Theta)^(-1)*y\n```\n\n'
text+='The scalar (1−alpha) does not change rankings relative to the unscaled linear solve. The reused incidence path keeps edge-cardinality normalization and weighted vertex degrees. Original export edges have no numeric weight field, so native weight=1; optional explicit future weights are supported by the retrieval adapter. No confidence field is silently promoted to a weight. Singleton/internal edges follow the existing spectral exclusion policy; the supplied fine graph has valid arity≥2. Isolated seeded vertices retain only their restart term.\n\n'
text+='The symmetric operator does not conserve L1 probability mass: exported per-type “mass” is a sum of relevance scores. We do not renormalize between iterations. Stopping uses the specified L1 change; exported L2 error bounds divide the last change by (1−alpha). A maximum-iteration stop is explicitly marked unconverged. Test solves verify scaled equality at alpha=.50/.70/.85/.95 with stricter tolerance and sufficient test iterations.\n\n'
text+='The pairwise control uses the existing weight-conserving clique projection (w/choose(arity,2) per pair), followed by the same symmetric arity-2 operator. On nonisolated vertices this is 0.5*(I+normalized adjacency), including self retention. It is a lazy symmetric graph diffusion, not ordinary non-lazy PPR. Native Theta is itself a linear operator with an equivalent weighted pair expansion; an empirical difference here tests this particular projection/normalization, not irreducible higher-order expressivity.\n\n'
text+='## Frozen retrieval protocol\n\n'
text+='Atomic BGE vectors and original-question vectors reuse the existing exact-text/model cache and are explicitly normalized. Authors get no initial semantic mass, but remain structural intermediates in H0. Top-M ties break by node ID; negatives are zeroed. If all selected scores are nonpositive, the deterministic fallback is uniform over the selected eligible nodes and is flagged. A genuinely empty eligible pool retains zero signal.\n\n'
text+='Final answer groups admit method, technique, component, dataset, cited_work, article and metric nodes. Claims, tasks, problems, future topics and authors remain in the propagation graph. A group qualifies when any member has an admitted type; score aggregation is maximum, never duplicate-counting sum. H0 uses no added numeric type prior; fused variants inherit R3 priors through ranks. R3_answer_pool is the required matched-pool control; every canonical target remains eligible at both cutoffs. Historical R3 alone is not a fair causal contrast for filtering gains.\n\n'
text+='H1 combines matched-pool R3 and H0 by equal-weight RRF with fixed k=60 and average ranks for exact score ties; an arbitrary zero-score ID ordering contributes no distinct fusion ranks. A flagged nonpositive/empty-seed fallback makes fused variants retain R3 rather than treating an uninformative diffusion order as semantic evidence. H2 builds a graph-only explicit-mention index using bounded whole-name matching: distinctive identifiers or sufficiently informative multiword names, no numeric-only names, stopword acronyms, fuzzy aliases or generic short strings. It raises a group’s maximum own seed to its maximum mentioning-node seed, writing only one deterministic member to avoid replication across duplicate nodes, then globally normalizes. The evidence source keeps its seed, so total pre-normalization mass can grow and every unmodified seed is rescaled downward. All updates read original y, so mention links cannot cascade. H2 consequently can increase the number of nonzero seed nodes; it is not a fixed-support M control. H2 then diffuses and fuses with R3. Atomic top-M itself is node-level and can include repeated text at different nodes; max answer aggregation does not undo that seed-level multiplicity.\n\n'
text+='H3 keeps H2 and uses predeclared relation multipliers: claims/presents/solves/addresses=1; evaluated_on/uses_technique/uses_component=.75; extends/cites=.25; proposes_future_work=.10; authored_by=0. Weighted vertex degrees are recomputed. These are design priors, not fitted weights; changing a relation also changes normalization, so H3 is not a single-relation causal ablation. H3 was executed only after the uniform/mention checkpoints.\n\n'
text+='The shuffled control independently randomizes each edge’s distinct memberships, retaining node count, arity, relation labels and edge weights with seed42. It does not preserve node degrees, type mixes or connectivity. One shuffle gives a limited structural sanity check, not a significance test or clean semantic-only intervention. Pairwise and shuffled controls use exactly H0’s seeds. Type B is not scored or replaced by diffusion.\n\n'
for year in cfg['diffusion_cutoffs']:
    text+=f'## {year} '+('conservative primary' if year==2025 else 'annual sensitivity; may include evidence after February')+'\n\n'
    ss=[r for r in results if r['cutoff']==year]
    text+=table(['Variant','Candidate R@20 / 50 / 100 / 200 (macro)','R@10 micro','Raw recovered /63','Median / mean gold rank','MRR','R-Prec = R@R'],[
        [r['stage'],' / '.join(f(r['summary']['candidate_recall'][str(k)]) for k in [20,50,100,200]),f(r['summary']['at_k']['10']['micro_recall']),str(r['summary']['at_k']['10']['found'])+'/63',
         f(r['summary']['rank_coverage']['censored_median'])+' / '+f(r['summary']['rank_coverage']['censored_mean']),f(r['summary']['mrr']),f(r['summary']['r_precision'])] for r in ss])
    text+=table(['Variant','Hit@1 / 5 / 10 / 20','Recall@5 / 10 / 20 / 50 / 100 (macro)','Precision@5 / 10','Unique-target R@10','Ranked / canonical'],[
        [r['stage'],' / '.join(f(r['summary']['at_k'][str(k)]['macro_hit']) for k in [1,5,10,20]),' / '.join(f(r['summary']['at_k'][str(k)]['macro_recall']) for k in [5,10,20,50,100]),
         ' / '.join(f(r['summary']['at_k'][str(k)]['macro_precision']) for k in [5,10]),f(r['summary']['at_k']['10']['unique_target_recall']),str(r['summary']['rank_coverage']['ranked'])+'/'+str(r['summary']['rank_coverage']['canonical'])] for r in ss])
    text+=table(['Pool-only diagnostic','Candidate R@50 / 100','R@10 micro','Median rank'],[[name,f(v['summary']['candidate_recall']['50'])+' / '+f(v['summary']['candidate_recall']['100']),f(v['summary']['at_k']['10']['micro_recall']),f(v['summary']['rank_coverage']['censored_median'])] for name,v in pools[year].items()])
    text+=table(['Variant','Seed / R3 dot products','Theta applications (=iterations)','Incidence entry visits upper bound','Component runtime seconds','Unconverged queries'],[
        [r['stage'],f(r['summary']['cost']['seed_comparisons'])+' / '+f(r['summary']['cost']['r3_comparisons']),f(r['summary']['cost']['theta_applications']),f(r['summary']['cost']['incidence_entry_visits_upper']),f(r['summary']['cost']['runtime_seconds']),r['summary']['unconverged_queries']] for r in ss])
text+='All 14 Type A records remain explicit. Canonical recall uses 47 target instances at 2025 and 50 at 2026; Q5 and Q11 remain non-evaluable for canonical recall, with their expected labels retained in the raw /63 denominator. Macro candidate recall averages 12 evaluable questions; micro recall counts canonical instances; unique recall counts targets recovered in any associated question. Missing ranks would be censored at pool-size+1, with coverage shown; none are omitted here. Candidate prefixes and output ranks are identical because this pass has no reranker. Precision is fixed-K judged-target precision under incomplete labels. All per-K macro/micro/unique/raw values are in JSON.\n\n'
text+='Cost columns are per-query means. A sparse Theta application performs two sparse incidence multiplies; incidence entry visits are an operation-count proxy, not FLOPs or seconds. R3 dots and atomic-seed dots are separate. Fusion rows charge the sum of standalone component work even when the experiment reuses shared results; actual-new application counts are also exported. Runtime is a sum of measured query components including seed preparation, mention assistance, diffusion, answer aggregation and fusion; it excludes offline model/index/operator construction and diagnostic path searches. H3 weighting and pairwise projection are built once per snapshot; operator build time, nnz and edge counts are exported. This is local timing, not a controlled serving benchmark.\n\n'
text+='## Mechanistic evidence and limitations\n\n'
text+=table(['Seed type','Selected node occurrences across 14 queries','Mean initial score mass'],[[typ,seed_counts[typ],f(mechanisms['mean_seed_mass_by_type'][typ])] for typ in sorted(seed_counts,key=lambda x:-seed_mass[x])])
text+=table(['Selected H2 top-100 path relation','Occurrences'],path_relations.most_common())
text+=table(['Diagnostic category (overlapping)','Instances','H3 better / same / worse than matched R3','Median rank gain'],[[r['category'],r['instances'],f"{r['improved']} / {r['unchanged']} / {r['worsened']}",f(r['median_rank_gain'])] for r in category_rows])
text+='All 47 primary canonical instances have at least one selected positive-seed structural path within four hops, so the no-path category has zero instances. This does not imply that every path carries relevant scientific evidence: several failures connect only through generic materials-science or machine-learning nodes. Across the 35 targets originally below rank 100 in full R3, H3 improves 31 relative to matched-pool R3; among the four original top-ten targets, two worsen.\n\n'
text+='“Strong evidence” here means a top-20 semantic claim/article/task/problem seed with a selected path of one or two hyperedges; it is not a human relevance judgment. Zero-hop direct seeds and lexical mention transfers are reported separately from structural propagation. The no-path category means no positive top-M structural seed path found within four hops, not proof that the full graph is disconnected. Paths are undirected incidence paths, ordered toward large seeds, short connections and weighted small edges. They include actual node texts, relation IDs, arity and local Theta transition factors, but do not uniquely decompose the full diffusion score. Structural paths do not prove that a claim entails an answer.\n\n'
drift=[]
for stage in ['H0','H0_shuffled','H0_pairwise','H2','H3']:
    dd=[d for d in diagnostics if d['cutoff']==2025 and d['stage']==stage]
    types=defaultdict(float)
    for d in dd:
        for typ,mass in d['final_type_fractions'].items(): types[typ]+=mass/len(dd)
    drift.append([stage,', '.join(t+': '+f(v) for t,v in sorted(types.items(),key=lambda x:-x[1])[:4]),f(np.mean([d['high_degree_score_fraction'] for d in dd])),f(np.mean([d['high_arity_edge_mass_fraction'] for d in dd])),f(np.mean([d['degree_score_spearman'] for d in dd if d['degree_score_spearman'] is not None]))])
text+=table(['Variant','Mean final type fractions (largest four)','Top-degree-decile score share','Arity>16 edge-output share','Degree-score Spearman'],drift)
text+='These measures expose generic-node/hub drift without treating correlation as causation. Each query exports full type-score history, top nodes/edges, relation contributions and compact NPZ vectors of semantic scores, seeds, propagated scores and weighted degrees. Edge mass decomposes one final Theta application, not every walk contributing to the inverse. High-degree share uses each graph’s own 90th-percentile threshold; degree ties may include more than 10% of vertices.\n\n'

def explain(target,stage):
    support=target['support'][stage]; lines=[]
    for path in support['paths'][:2]:
        seed=path['seed']; chain=seed['type']+' '+seed['node_id']+' ('+seed['surface_form'][:130]+')'
        for step in path['path']:
            dest=node_lookup[step['to_node']]
            chain+=' → '+step['relation']+'['+step['edge_id']+'] → '+dest['type']+' '+step['to_node']+' ('+dest['surface_form'][:75]+')'
        lines.append(chain if path['path'] else chain+' [direct seed at answer]')
    mention='; '.join(t['evidence_node_id']+' → explicit mention → '+t['destination'] for t in support['mention_transfers'])
    return '<br>'.join(lines+([mention] if mention else [])) or 'No selected positive-seed path within four hops'
success=sorted([t for t in primary if t['ranks']['H2']<=100 and t['ranks']['R3_answer_pool']>100],key=lambda t:t['ranks']['H2'])[:3]
failures=sorted(primary,key=lambda t:t['H3_vs_matched_R3_rank_change'])[:3]
text+='### Examples of gains and failures\n\n'
text+=table(['Question / target','Matched R3 → H0 → H2 → H3 rank','Approximate support'],[[t['question_id']+' / '+t['gold_label'],' → '.join(str(t['ranks'][s]) for s in ['R3_answer_pool','H0','H2','H3']),explain(t,'H2')] for t in success])
text+=table(['Question / target','Matched R3 → H3 rank','Failure evidence'],[[t['question_id']+' / '+t['gold_label'],str(t['ranks']['R3_answer_pool'])+' → '+str(t['ranks']['H3']),explain(t,'H3')] for t in failures])
text+='H2 gains cannot be attributed to copying all article context: only safe literal name mentions transfer seed support. Nevertheless a lexical mention may describe a comparison, limitation or negation, and the symmetric graph has no entailment semantics. Fusion can elevate many moderately supported entities while lowering an already-correct top result. H3 removes author-edge transmission and changes weighted normalization, so it can also suppress a useful incidental connection. Inspect the before/after rows and trace text rather than interpreting every short path as scientifically valid.\n\n'
text+='## Alpha and seed-count sensitivity\n\n'
text+='The default remains alpha=.85, M=100. These are one-at-a-time checks, not a Cartesian search or model selection exercise.\n\n'
text+=table(['Cutoff / variant','alpha','M','Macro candidate R@100','Micro R@10','Median gold rank','Mean iterations','Unconverged'],[
 [str(s['cutoff'])+' / '+s['stage'],s['alpha'],s['seed_top_m'],f(s['summary']['candidate_recall']['100']),f(s['summary']['at_k']['10']['micro_recall']),f(s['summary']['rank_coverage']['censored_median']),f(s['summary']['cost']['iterations']),s['summary']['unconverged_queries']] for s in sensitivities])
text+='If any 100-iteration run is unconverged, its row is a bounded-iteration sensitivity result, not a certified fixed point; inspect the exported residual/error bound. No alpha or seed count was changed to maximize these results.\n\n'
text+='In the executed runs all default rows converged and no seed fallback was used. Every alpha=.95 sensitivity reached the 100-iteration cap. At 2025, H3 C@100 is .7667 at alpha=.50/.70, .7230 at .85 and .6368 in the truncated .95 run; M=50/100/200 gives .7444/.7230/.6277 at alpha=.85. These variations show sensitivity to propagation depth and seed breadth; they do not license selecting the best development-benchmark setting. The seed/claim/graph text remains the same across these controls.\n\n'
text+='## Thirteen explicit answers\n\n'
text+=f'1. **Does pure native diffusion improve candidates?** H0 macro C@50/C@100 = {f(c("H0",50))}/{f(c("H0"))}, versus R3 {f(c("R3",50))}/{f(c("R3"))} and matched-pool R3 {f(matched["candidate_recall"]["50"])}/{f(matched["candidate_recall"]["100"])}. This separates diffusion from output filtering.\n'
text+=f'2. **Does pure diffusion improve median rank?** H0 median {med("H0")} versus full R3 {med("R3")} and matched R3 {matched["rank_coverage"]["censored_median"]}. Candidate recall and median rank need not move together. H2/H3 fusion medians are {med("H2")}/{med("H3")}.\n'
text+='3. **Which seed types are useful?** The mass and path tables identify which types actually supplied seeds and connected to recovered targets. These are observed support proxies; there was no type-removal causal ablation, so they cannot establish a unique best seed type.\n'
text+='4. **Does evidence transfer reach correct entities?** Yes where the exported positive-weight claim/task/article paths or explicit-mention transfers terminate at mapped target nodes. Selected examples above demonstrate the mechanism; they do not certify claim entailment.\n'
text+=f'5. **Real versus shuffled?** H0 C@100 {f(c("H0"))}, shuffled {f(c("H0_shuffled"))}; medians {med("H0")}/{med("H0_shuffled")}. This favors the real graph over this null, but degree/connectivity changes and one shuffle prevent a strong semantic-only causal claim.\n'
text+=f'6. **Native versus pairwise?** C@100 is {f(c("H0"))} native and {f(c("H0_pairwise"))} pairwise; medians {med("H0")}/{med("H0_pairwise")}. The full K tables expose differences; no general native-higher-order advantage follows from one projection comparison.\n'
text+=f'7. **Does mention support help?** H1→H2 macro C@100 {f(c("H1"))}→{f(c("H2"))}, median {med("H1")}→{med("H2")}; sparse cited-work rows and mention-source IDs are exported. This is an association with the prescribed mention mechanism, including its changed seed support and normalization.\n'
text+=f'8. **Does relation weighting help?** H2→H3 C@100 {f(c("H2"))}→{f(c("H3"))}, C@200 {f(c("H2",200))}→{f(c("H3",200))}, median {med("H2")}→{med("H3")}. Treat mixed changes as sensitivity to a design prior, not learned scientific relation quality.\n'
text+='9. **Which relations carry useful relevance?** The selected recovered-target path counts and edge-mass diagnostics show their frequencies. Counts include shared paths and generic hubs; useful scientific support requires checking the actual statements.\n'
text+='10. **What causes failures?** No seed path within four hops is a connectivity/seed-support warning; a path through high-degree generic nodes indicates possible drift; direct semantic/mention support can still be weak or irrelevant. These are diagnostic proxies, not exhaustive causal labels. Candidate gains alongside weaker top-ten retrieval show that remaining answer ranking is unsolved.\n'
text+=f'11. **Does top-ten recovery exceed 4/47?** H0/H1/H2/H3 recover {rec("H0")}/{rec("H1")}/{rec("H2")}/{rec("H3")} instances, respectively, versus R3’s {rec("R3")}/47. Raw denominator remains 63.\n'
text+=f'12. **Does candidate recall improve beyond ~.37?** H2/H3 macro C@100 reaches {f(c("H2"))}/{f(c("H3"))}, against full R3 {f(c("R3"))} and matched-pool R3 {f(matched["candidate_recall"]["100"])}. The exact per-question and micro values are exported; twelve evaluable questions are too few to claim broad generalization.\n'
text+='13. **Investigate hierarchy next?** The candidate improvements justify a later controlled approximation study of the fine-graph relevance target, especially H2, with candidate-quality loss and actual sparse work measured. They do not justify deploying hierarchical retrieval or claiming solved QA. No hierarchy diffusion, cross-encoder, LLM reranking or tuned relation-weight search was implemented in this pass. Independent questions, stronger structural nulls and source/claim adjudication remain necessary.\n\n'
text+='## Every primary canonical target: ranks\n\n'
text+=table(['Q / target','R3','R3 matched','H0','H1','H2','H3','H3 gain vs matched R3'],[[t['question_id']+' / '+t['gold_label']]+[t['ranks'][s] for s in ['R3','R3_answer_pool','H0','H1','H2','H3']]+[t['H3_vs_matched_R3_rank_change']] for t in primary])
text+='Positive gain means a lower/better rank. Both cutoffs, score values, groups, mention sources, selected paths, and categories are in `diffusion_per_target.json` and `propagation_traces.json`.\n\n'
text+='## Implementation, verification and reproduction\n\n'
suite=ET.parse(out/'tests.xml').getroot().find('testsuite'); assert int(suite.attrib['failures'])==int(suite.attrib['errors'])==0
text+=f'The full suite passed **{suite.attrib["tests"]} tests**, including new dense/sparse and dense-solve equivalence, De/Dv normalization, zero/weighted/isolated edges, deterministic seeds and fallback, temporal separation, group max, evidence intermediates, conservative mentions, RRF, reproducible shuffle, projection weights, no dense sparse conversion, and a complete synthetic pipeline preserving historical files and Type B data. Automated artifact audit verifies all 14 questions per variant/cutoff, raw denominators, canonical rank coverage, score decompositions, current source hashes, frozen historical artifacts/metrics, original hierarchy hashes and exact R0/R1/R3 metric parity.\n\n'
text+='New implementation: `src/tkh_abstraction/diffusion.py` (`ThetaOperator`, `diffuse`, `semantic_seeds`, `MentionIndex`, `rank_entities`, `rrf_fuse`, diagnostics/paths) and `diffusion_pipeline.py` (`run_diffusion`, metric adapter). They reuse spectral incidence/operator, native graph classes, projection, snapshots, normalized cached BGE vectors, entity groups, question analysis, unchanged resolver, metrics and artifact writers. CLI adds `evaluate-diffusion`; defaults live in `configs/diffusion.yaml`; tests in `tests/test_diffusion.py`; reporting/audit in `scripts/report_diffusion.py`. Documentation and archive packaging were updated. Construction and historical evaluators were not rewritten.\n\n'
text+='Executed primary commands (the initial checkpoint stopped before H3; full run includes both cutoffs and sensitivities):\n\n```powershell\n$env:HF_HUB_OFFLINE=\'1\'\n.venv\\Scripts\\python.exe -m tkh_abstraction.cli evaluate-diffusion --config configs/diffusion.yaml --cutoff 2025 --diffusion-stage R0 --diffusion-stage R1 --diffusion-stage R3 --diffusion-stage H0 --diffusion-stage H0_shuffled --diffusion-stage H0_pairwise --diffusion-stage H1 --diffusion-stage H2 --no-sensitivity\n.venv\\Scripts\\python.exe -m pytest -q --junitxml=artifacts/evaluation/diffusion/tests.xml\n.venv\\Scripts\\python.exe -m tkh_abstraction.cli evaluate-diffusion --config configs/diffusion.yaml\n.venv\\Scripts\\python.exe scripts/report_diffusion.py\n.venv\\Scripts\\python.exe scripts/package_submission.py\n```\n\n'
text+='`--diffusion-stage` and `--cutoff` select subsets; `--no-sensitivity` disables only the sensitivity pass. Partial invocations replace only the diffusion artifact set, never R0–R5 historical outputs. Run the complete command before the full report audit. For another dataset, configure input/output paths and disable assessment-specific `diffusion_verify_reuse`; no current Q ID or expected answer is used by production retrieval. The report’s benchmark assertions intentionally validate this supplied assessment.\n\n'
text+='Ultralight Opus planning and Sonnet review were read-only consultations through the official local subscription helper; Codex implemented and executed all experiments. Advice was checked against the code/math: suggestions to double the pairwise operator or assume a stochastic L1 error bound were rejected. The first benchmark attempt exposed a duplicate diagnostics-key error, which was fixed with regression coverage before successful checkpoints. Full prompts, decisions and run logs are preserved in `notes/diffusion`. No expert scientific validation is claimed.\n'
Path('report/hypergraph_diffusion.md').write_text(text,encoding='utf-8')
section='\n## Query-seeded native hypergraph diffusion\n\n'
section+=f'The new Type A candidate experiment reuses the native sparse Zhou operator with atomic semantic seeds. At conservative 2025, macro Candidate Recall@100 is {f(c("R3"))} for historical R3, {f(matched["candidate_recall"]["100"])} for R3 with the same answer pool, {f(c("H0"))} for pure native H0, {f(c("H2"))} for mention-assisted H2 fusion, and {f(c("H3"))} for relation-weighted H3. H2/H3 median gold ranks are {med("H2")}/{med("H3")}, but both recover only {rec("H2")}/47 canonical instances in the top ten versus R3’s 4/47.\n\n'
section+=f'Real H0 exceeds its shuffled control ({f(c("H0_shuffled"))} C@100), while the weight-conserving pairwise control matches native C@100 ({f(c("H0_pairwise"))}). This supports further candidate-mechanism research, not a native-hypergraph superiority or solved-QA claim. Type B and historical R0–R5 numbers remain unchanged. See [hypergraph_diffusion.md](hypergraph_diffusion.md) for all variants, costs, sensitivities, every target’s rank changes, approximate paths and thirteen explicit conclusions.\n'
for path in [Path('report/contextual_retrieval.md'),Path('report/report.md')]:
    original=path.read_text(encoding='utf-8').split('\n## Query-seeded native hypergraph diffusion')[0]
    path.write_text(original+section,encoding='utf-8')
audit=dict(status='passed_with_research_limitations',tests=suite.attrib,question_records=sum(len(r['records']) for r in results),canonical_target_records=len(targets),
    historical_metrics_unchanged=True,historical_artifacts_unchanged=True,source_hashes_verified=True,baseline_R0_R1_R3_exact_metric_parity=True,all_canonical_targets_ranked=True,
    type_b_unchanged=True,hierarchy_unchanged=True,sensitivity_runs=len(sensitivities),main_unconverged=sum(not d['converged'] for d in diagnostics),
    final_conclusion='Candidate generation improves with mention-assisted fusion; top-ten answer retrieval remains weak; no native superiority over pairwise at C100 established')
write_json(out/'final_audit.json',audit)
checklist=Path('artifacts/submission_checklist.md'); original=checklist.read_text(encoding='utf-8').split('\n## Diffusion follow-up')[0]
checklist.write_text(original+'\n## Diffusion follow-up\n\nPASS: mathematical equivalence tests, nine required stages at both cutoffs, real/shuffled/pairwise controls, fixed alpha/M sensitivities, all-question denominator retention, source/cost/path diagnostics, and preservation of historical/contextual/Type B/hierarchy outputs. See `evaluation/diffusion/final_audit.json` and `../report/hypergraph_diffusion.md`.\n\nLIMITATIONS: weak top-ten QA, single degree-changing shuffle, heuristic mention identity and undirected path explanations, no proof of entailment or native higher-order superiority. Hierarchical diffusion remains a proposed next study, not implemented.\n',encoding='utf-8')
print(json.dumps(audit,indent=2))
