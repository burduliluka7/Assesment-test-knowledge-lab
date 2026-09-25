"""Consolidate frozen intrinsic evidence and the executed final retrieval study."""
from pathlib import Path
from collections import Counter
from copy import deepcopy
import json
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tkh_abstraction.io import read_json, write_json, digest
from tkh_abstraction.final_pipeline import verify_frozen
from tkh_abstraction.v2_pipeline import sha
from tkh_abstraction.hierarchy import validate_hierarchy
from tkh_abstraction.snapshots import snapshot

ROOT = Path('artifacts/evaluation/final')
FIG = Path('report/figures')


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])


def fmt(x, digits=3): return 'N/A' if x is None else f'{x:.{digits}f}'


def record(run, year, system, budget=None, variant=None):
    return next(r for r in run['results'] if r['cutoff']==year and r['system']==system and r['budget']==budget and (variant is None or r['variant']==variant))


def result_row(r):
    s=r['summary']; m=s['at_k']['10']; w=s['work']; c=s['claims']; t=s['retention']
    label=r['system']+('/'+r['variant'] if r['variant']!='none' else '')+(' all' if r['budget'] is None else ' '+str(r['budget']))
    return [label, f"{m['found']}/{m['canonical_target_instances']} ({fmt(m['micro_recall'])})", fmt(m['unique_target_recall']),fmt(s['mrr']),
            f"{t['retained_fine_hits']}/{t['fine_hits']}",fmt(w['fine_candidates_scored'],1),fmt(w['total_comparisons'],1),fmt(w['cluster_expansions'],1),
            fmt(c['candidate_source_hit']),fmt(c['strict_claim_recall'])]


def main():
    preserved=verify_frozen(ROOT/'frozen_manifest.json'); run=read_json(ROOT/'results.json')
    for path,h in run['protocol']['source_hashes'].items():
        if sha(path)!=h: raise ValueError('Final source differs from executed run: '+path)
    legacy=read_json('artifacts/full/metrics.json'); descriptive=read_json('artifacts/full/descriptive.json')
    data=read_json('data/data/tkh_collection10.json'); variants=legacy['variants']
    labels={v:read_json(f'artifacts/evaluation/labels/{v}/faithfulness.json')['summaries']['improved'] for v in variants}
    events=read_json('artifacts/full/temporal_events.json'); event_counts=Counter(e['event_type'] for e in events)
    if set(event_counts)!={'birth','growth','continuation','merge','split','death'}: raise ValueError('Missing event type')
    testroot=ET.parse(ROOT/'tests.xml').getroot(); suites=list(testroot.iter('testsuite'))
    newest=max(p.stat().st_mtime for folder in ['src','tests'] for p in Path(folder).rglob('*.py'))
    if (ROOT/'tests.xml').stat().st_mtime < newest: raise ValueError('JUnit predates current source/tests; rerun the final test command')
    tests=sum(int(s.attrib.get('tests',0)) for s in suites)
    if sum(int(s.attrib.get(k,0)) for s in suites for k in ['failures','errors','skipped']): raise ValueError('Tests not all passing')
    # Prose uses rounded frozen numbers. Reject drift rather than silently narrating
    # old constants against different measurements.
    quality=Counter(x['code'] for x in read_json('artifacts/full/data_quality.json')['issues'])
    assert quality['edge_before_members_seen']==170 and quality['edge_before_asserting_paper']==134 and quality['origin_after_first_seen']==3
    assert round(100*legacy['projection_loss']['fraction_arity_gt2'],2)==80.48
    assert legacy['projection_loss']['pair_instances']==72701 and legacy['projection_loss']['unique_pairs']==70022
    assert [(x['snapshot'],x['nodes'],x['hyperedges']) for x in descriptive['snapshots']]==[(2020,1505,374),(2022,2164,526),(2024,4164,983),(2026,5798,1429)]
    l0=labels['temporal']['0'];l1=labels['temporal']['1']
    assert [round(l0['prediction_rates'][k],3) for k in ['entailment','neutral','contradiction']]==[.458,.479,.062]
    assert [round(l1['prediction_rates'][k],3) for k in ['entailment','neutral','contradiction']]==[.521,.354,.125]
    ari=variants['temporal']['perturbation']['summary']['0']['ari']
    assert round(ari['mean'],3)==.550 and [round(x,3) for x in ari['ci95']]==[.497,.603]
    for year in ['2025','2026']:
        alignment=read_json('artifacts/evaluation/claim_alignment.json')[year]
        for typ,count in [('A',188),('B',36)]:
            claims=[x for x in alignment['items'] if x['question_type']==typ and x['kind']=='claim']
            assert len(claims)==count and not any(x['alignment_status']=='confident_proxy' for x in claims)
    for r in run['results']:
        s=r['summary'];assert s['questions']==14 and s['evaluable']==12 and s['at_k']['10']['total_target_instances']==63
    # Submission-facing label overlay exports; immutable hierarchy inputs remain intact.
    hierarchy_files=[]
    for year in [2020,2022,2024,2026]:
        h=read_json(f'artifacts/full/hierarchy_{year}.json'); original=deepcopy(h)
        nodes={n['id'] for n in snapshot(data,year)['nodes']}; validate_hierarchy(h,nodes)
        overlay={r['id']:r for r in read_json(f'artifacts/evaluation/labels/temporal/labels_{year}.json')}
        for level in h['levels']:
            for c in level:
                for key in ['id','persistent_id','level','parent_id','member_ids','label','gloss']:
                    if key not in c: raise ValueError('Missing hierarchy export field '+key)
                if c['id'] in overlay:
                    c['label']=overlay[c['id']]['label'];c['gloss']=overlay[c['id']]['gloss']
        assert [[(c['id'],c['parent_id'],c['member_ids']) for c in l] for l in h['levels']]==[[(c['id'],c['parent_id'],c['member_ids']) for c in l] for l in original['levels']]
        h['label_export_note']='Frozen v2 generation-only overlays at L0/L1; original labels at deeper levels. Memberships and persistent identities unchanged.'
        path=ROOT/f'hierarchy_{year}.json';write_json(path,h);hierarchy_files.append(path.as_posix())
    write_json(ROOT/'temporal_events.json',[dict(e,from_snapshot=e['source_snapshot'],to_snapshot=e['target_snapshot'],overlap_statistics=e['overlap_scores']) for e in events])
    fine={str(y):record(run,y,'FINAL_FINE')['summary'] for y in [2025,2026]}
    fine_rows=read_json(ROOT/'final_fine_rankings.json')
    search_coverage=[]
    intersection_costs=[]
    for r in run['results']:
        rows=read_json(r['artifact']) if 'artifact' in r else [x for x in fine_rows if x['cutoff']==r['cutoff']]
        search_coverage.append({k:r[k] for k in ['cutoff','system','variant','budget']} | dict(
            zero_candidate_questions=sum(not x['returned'] for x in rows),
            mean_returned_at10=float(np.mean([min(10,len(x['returned'])) for x in rows])),
            root_layer_cost=rows[0]['root_layer_cost'],status_counts=r['summary']['status_counts']))
        if r['system']=='hierarchical_h2_intersection':
            h2=fine[str(r['cutoff'])]['work']
            intersection_costs.append({k:r[k] for k in ['cutoff','variant','budget']} | dict(
                traversal_comparisons=r['summary']['work']['total_comparisons'],
                prerequisite_h2_vector_comparisons=h2['h2_vector_comparisons'],
                total_including_h2=r['summary']['work']['total_comparisons']+h2['h2_vector_comparisons'],
                h2_incidence_entry_visits=h2['h2_incidence_entry_visits']))
    claim_inventory=[]
    for year in [2025,2026]:
        items=[x for x in read_json('artifacts/evaluation/claim_alignment.json')[str(year)]['items'] if x['kind']=='claim']
        for qid in sorted({x['question_id'] for x in items},key=lambda q:int(q[1:])):
            claims=[x for x in items if x['question_id']==qid]
            claim_inventory.append(dict(cutoff=year,question_id=qid,question_type=claims[0]['question_type'],
                expected_claims_count=len(claims),automatically_alignable_claims_count=sum(x['alignment_status']=='confident_proxy' for x in claims),
                retrieved_aligned_claims_count=0,strict_claim_recall=None,status='NOT_EVALUABLE_ALIGNMENT',
                evaluation_scope='Final Type A experiment' if claims[0]['question_type']=='A' else 'Frozen Type B alignment; retrieval not rerun'))
    write_json(ROOT/'claim_evaluability.json',claim_inventory)
    hier=[r for r in run['results'] if r['system'].startswith('hierarchical')]
    compact=[]
    for v, detail in variants.items():
        coh=detail['coherence']['2026']['0'];pert=detail['perturbation']['summary']['0']['ari']
        retrieval=record(run,2025,'hierarchical',500,v)['summary']
        compact.append(dict(variant=v,coherence=coh['node_weighted'],null=coh['null']['mean'],null_p=coh['null']['p_upper'],
                            perturbation_ari=pert,temporal_ari=detail['cross_snapshot']['2024->2026']['levels']['0']['ari'],
                            label_overclaim_proxy=labels[v]['0']['unsupported_rate'],retrieval=retrieval))
    consolidated=dict(descriptive=descriptive,hierarchy=dict(budgets=[12,40,120],primary_variant='temporal',exports=hierarchy_files),
                      coherence={v:d['coherence'] for v,d in variants.items()},
                      coherence_null={v:{y:{l:{k:c[k] for k in ['null','type_stratified_null'] if k in c} for l,c in cs.items()} for y,cs in d['coherence'].items()} for v,d in variants.items()},
                      perturbation_stability={v:d['perturbation'] for v,d in variants.items()},temporal_stability={v:d['cross_snapshot'] for v,d in variants.items()},
                      temporal_events=dict(artifact=(ROOT/'temporal_events.json').as_posix(),counts=dict(event_counts)),label_faithfulness=labels,
                      fine_retrieval=fine,hierarchical_retrieval=hier,
                      claim_evidence=[{k:r[k] for k in ['cutoff','system','variant','budget']}|{'metrics':r['summary']['claims']} for r in run['results']],
                      variant_comparison=compact,final_retrieval=run['results'],historical_metrics='artifacts/full/metrics.json',
                      search_coverage=search_coverage,intersection_prerequisite_costs=intersection_costs,claim_evaluability=claim_inventory,
                      provenance=dict(frozen_manifest=(ROOT/'frozen_manifest.json').as_posix(),preserved_files=preserved,tests=tests))
    write_json('metrics.json',consolidated)
    assert {'descriptive','hierarchy','coherence','coherence_null','perturbation_stability','temporal_stability','temporal_events','label_faithfulness','fine_retrieval','hierarchical_retrieval','claim_evidence','variant_comparison'}<=set(consolidated)

    # Primary quality/cost plot: invalid root budgets are omitted and labeled.
    fig,axes=plt.subplots(1,2,figsize=(10,3.1),layout='constrained')
    for ax,year in zip(axes,[2025,2026]):
        for system,label in [('hierarchical','Temporal hierarchy'),('flat_equal_work','Flat equal work')]:
            rows=[r for r in run['results'] if r['cutoff']==year and r['system']==system and (r['variant']=='temporal' or system.startswith('flat')) and r['budget'] is not None and 'INSUFFICIENT_ROOT_BUDGET' not in r['summary']['status_counts']]
            ax.plot([r['summary']['work']['total_comparisons'] for r in rows],[r['summary']['at_k']['10']['micro_recall'] for r in rows],'-o',label=label,markersize=4)
        ref=record(run,year,'FINAL_FINE')['summary'];ax.axhline(ref['at_k']['10']['micro_recall'],color='black',linestyle='--',label='FINAL_FINE quality')
        ax.scatter([ref['work']['total_comparisons']],[ref['at_k']['10']['micro_recall']],color='black',marker='*',s=80,label='FINAL_FINE charged work')
        for system,marker in [('flat_exhaustive','s'),('hierarchical','^')]:
            r=record(run,year,system,None)['summary'];ax.scatter([r['work']['total_comparisons']],[r['at_k']['10']['micro_recall']],marker=marker,s=35,label=system+' unrestricted')
        ax.set_xscale('symlog',linthresh=50);ax.set_xlabel('Charged comparisons (log above 50)');ax.set_ylabel('Micro Recall@10');ax.set_title(str(year));ax.set_ylim(-.01,.40);ax.grid(alpha=.2)
    axes[0].legend(fontsize=6);fig.savefig(FIG/'hierarchy_retrieval_tradeoff.png',dpi=180);fig.savefig(FIG/'hierarchy_retrieval_tradeoff.svg');plt.close(fig)
    # Optional static display of all three abstraction levels at four snapshots.
    fig,axes=plt.subplots(1,4,figsize=(11,2.5),layout='constrained')
    for ax,year in zip(axes,[2020,2022,2024,2026]):
        h=read_json(ROOT/f'hierarchy_{year}.json');positions={};counter=0
        children={c['id']:[x for x in h['levels'][2] if x['parent_id']==c['id']] for c in h['levels'][1]}
        for root in h['levels'][0]:
            mids=[c for c in h['levels'][1] if c['parent_id']==root['id']];xx=[]
            for mid in mids:
                xs=[]
                for c in children[mid['id']]: positions[c['id']]=(counter,0);xs.append(counter);counter+=1
                positions[mid['id']]=(float(np.mean(xs)),1);xx.append(positions[mid['id']][0])
            positions[root['id']]=(float(np.mean(xx)),2)
        for level in h['levels'][:3]:
            for c in level:
                x,y=positions[c['id']];ax.scatter(x,y,s=3+15*len(c['member_ids'])/len(h['levels'][-1]),color='steelblue')
                if c['parent_id'] in positions:
                    px,py=positions[c['parent_id']];ax.plot([px,x],[py,y],color='gray',linewidth=.25)
        ax.set_title(str(year));ax.set_yticks([0,1,2],['L2','L1','L0']);ax.set_xticks([])
    fig.savefig(FIG/'hierarchy_levels.png',dpi=180);plt.close(fig)

    headers=['System / budget','R@10','Unique R@10','MRR*','Retained fine hits','Fine scores','Comparisons','Expansions','Candidate source hit','Strict claim recall']
    primary={y:[record(run,y,'FINAL_FINE'),record(run,y,'flat_exhaustive'),record(run,y,'flat_equal_work',500),record(run,y,'hierarchical',500,'temporal'),record(run,y,'hierarchical',None,'temporal'),record(run,y,'hierarchical_h2_intersection',None,'temporal')] for y in [2025,2026]}
    f=record(run,2025,'FINAL_FINE')['summary'];h=record(run,2025,'hierarchical',500,'temporal')['summary'];flat=record(run,2025,'flat_equal_work',500)['summary']
    reduction=1-h['work']['total_comparisons']/f['work']['total_comparisons']
    fine_reduction=1-h['work']['fine_candidates_scored']/f['work']['fine_candidates_scored']
    assert all(record(run,y,'hierarchical',500,'temporal')['summary']['retention']['retained_fine_hits']==0 for y in [2025,2026])
    conclusion='Retain temporal hybrid as the scientific browsing hierarchy, with faithfulness still only partially validated. The current search policy is not justified as a retrieval accelerator: reduced work comes with loss of all FINAL_FINE top-ten hits at budget500 in both cutoffs.'
    metric_note='*MRR uses each system\'s complete available ranking: FINAL_FINE retains its historical H2 tail, while budgeted searches rank only scored groups. MRR and uncapped gold-rank statistics therefore have different candidate coverage; compare top-ten recovery and fine-hit retention for the primary approximation question. H2-intersection comparison cells report traversal work only; the prerequisite H2 retrieval is charged separately in the detailed cost table.'
    variant_table=table(['Variant','BGE / null','Perturbation ARI [95% CI]','Temporal ARI','L0 non-entailment proxy','Final hierarchy R@10 (500)','Comparisons'],
                        [[r['variant'],f"{fmt(r['coherence'])} / {fmt(r['null'])}",f"{fmt(r['perturbation_ari']['mean'])} [{fmt(r['perturbation_ari']['ci95'][0])}, {fmt(r['perturbation_ari']['ci95'][1])}]",fmt(r['temporal_ari']),fmt(r['label_overclaim_proxy']),fmt(r['retrieval']['at_k']['10']['micro_recall']),fmt(r['retrieval']['work']['total_comparisons'],1)] for r in compact])
    fine_table=table(['Cutoff','Top10 / canonical','Macro R10','Unique R10','Raw /63','MRR','R-Prec / R@R','Median / mean gold rank'],
                     [[y,f"{s['at_k']['10']['found']}/{s['at_k']['10']['canonical_target_instances']}",fmt(s['at_k']['10']['macro_recall'],4),fmt(s['at_k']['10']['unique_target_recall'],4),f"{s['at_k']['10']['found']}/63",fmt(s['mrr'],4),f"{fmt(s['r_precision'],4)} / {fmt(s['recall_at_r'],4)}",f"{fmt(s['rank_quantiles']['median'],1)} / {fmt(s['mean_target_rank'],2)}"] for y,s in fine.items()])
    text=f'''# Temporal native-hypergraph abstraction: final assessment

## Problem and formal statement (T1, T2; Deliverable 0)

The supplied TKH contains **5,798 nodes, 1,429 native hyperedges, 12 node types and 11 relation types**; 80.48% of edges have arity greater than two. Consistently visible snapshots contain 1,505/374 nodes/edges (2020), 2,164/526 (2022), 4,164/983 (2024) and 5,798/1,429 (2026). `artifacts/full/descriptive.json` gives distributions and growth. Data-quality checks find 170 edges preceding a member's visibility, 134 preceding the asserting article, and three origin-year/first-seen inconsistencies; counts overlap. Visibility requires first-seen nodes and an edge whose date, asserting paper and all members are available. This is retrospective publication visibility, not ingestion history.

For H_t=(V_t,E_t), seek exclusive covering partitions P_0,…,P_3 with hard budgets **12 / 40 / 120 / singleton leaves**, and each child contained in exactly one parent. Greedy restricted-candidate merges minimize increments of

**J(P_t)=0.30 J_hyper + 0.20 J_spec + 0.35 J_sem + 0.15 J_temp.**

J_hyper is native edge-weighted fragmentation: (1−Σ_C(m_eC/n_e)²)/(1−1/n_e), averaged over original edge weights. J_spec is normalized within-cluster dispersion in global native-hypergraph spectral coordinates; J_sem is the analogous MiniLM semantic dispersion. J_temp is predecessor-partition VI on common base nodes divided by log(common-node count), zero with at most one common node. Active weights are renormalized when a term is disabled, including time at the first snapshot. Semantics and structure explicitly share the objective; neither acts as a post-hoc veto. No global optimum is claimed.

**P1 laminarity and P2 budgets are guaranteed by construction; P4 fidelity is procedurally guaranteed by native collapse. P3 coherence, P5 stability and P6 label faithfulness are empirical.** Display budgets do not prove natural communities. Sparse incidence products cost O(I+n) for I incidences; storage is O(I+n) plus embeddings. Neighbor proposals can cost O(n²(d+q)); restricted greedy merges can approach quadratic work. Matching coarse identities costs O(N³). No large-corpus scaling guarantee is established.

## Related work

[Zhou, Huang & Schölkopf (2006)](https://proceedings.neurips.cc/paper_files/paper/2006/hash/dff8e9c2ac33381546d96deea9922999-Abstract.html) supplies normalized native incidence geometry; this method adds fragmentation, semantics and temporal VI. [Chen, Saad & Zhang (2022)](https://doi.org/10.1007/s40324-021-00282-x) motivates recursive coarsening; here each next level consumes the actual collapsed hypergraph, without an approximation-error guarantee. [Schaub, Li & Peel (2023)](https://doi.org/10.1103/PhysRevE.107.054305) distinguishes producing a hierarchy from demonstrating meaningful hierarchy; our null and downstream tests address that distinction. [Chi et al. (2007)](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/01/evospe.pdf) motivates the current-fit/temporal-smoothness tradeoff; our VI regularizer differs from their evolutionary spectral objective. Existing source verification and fuller synthesis are in `report/literature_notes.md`.

## Method: temporal coupling, collapse and labels (T3–T5)

Construction-time **VI regularization** supplies stability. Post-construction **Hungarian overlap matching** tracks persistent identities; matching itself does not regularize construction. Birth, growth, continuation, merge, split and death are exported with source/target snapshots, IDs and overlap statistics in `artifacts/evaluation/final/temporal_events.json`. On common nodes in 2024→2026, temporal L0 ARI is .636 versus static .351, at a small coherence change from .777 to .775.

Collapse handles all three cases. For **m=n**, retain the fully internal edge as provenance/evidence and remove it from inter-supernode spectral incidence. For **1<m<n**, merge repeated endpoints and retain multiplicity m_eC: {{a,b,c,d}}→{{S1:3,S2:1}}. If endpoints occupy **three or more supernodes**, retain one coarse hyperedge over every distinct supernode with multiplicities. Original members, relation identity and provenance remain traceable; fine endpoint distinctions disappear from working coarse geometry. Implemented collapse is the next level's structural input. Binary spectral incidence omits multiplicity, while the fragmentation term uses it.

The rejected pairwise alternative expands 1,429 native edges into **72,701 pair instances / 70,022 unique pairs (50.88×)**. Different higher-order configurations can induce identical weighted adjacency. Native edge identity is therefore retained for TKH fidelity, not a claim that every objective term is irreducibly higher-order or wins every metric; pairwise perturbation ARI is higher.

Labels are extractive noun/keyphrases plus a short non-comparative one-sentence gloss. At L0/L1 the repaired labeler sees only `cutoff`, `generation_members`, `generation_type_counts`, and `generation_only_relation_counts`; generation relations are fully contained in the generation split. Snapshot-valid generation/evaluation members are separated where feasible. The final exports overlay these frozen labels without changing hierarchy membership. No post-cutoff member/evidence enters the packets, but unversioned text and retrospective model pretraining limit stronger temporal claims.

## Evaluation validity (T6)

**MiniLM drives clustering; BGE independently evaluates coherence**, excluding singletons. Fifty exact-size random-label permutations and a within-type null accompany observed scores. Temporal L0 has observed .7748 versus size-null .7602 (difference .0146, permutation p=.0196); the type-stratified difference is .0054. These are small effects and uncorrected descriptive p-values across several levels/variants.

Perturbation removes round(10% of hyperedges), then rebuilds with five fixed seeds **11,23,37,53,71** and fixed predecessor. Temporal L0 ARI is .550, approximate Student-t 95% CI **[.497,.603]**; normalized VI is .158 [.133,.182]. These are perturbation intervals, separate from exhaustive real cross-snapshot ARI. Native construction, nulls, perturbation, labeling and all historical metrics were preserved, not retuned.

Held-out gloss proxies over four snapshots give L0 entailment/neutral/contradiction **.458/.479/.062**, non-entailment **.542**; L1 **.521/.354/.125**, non-entailment **.479** (48 evaluable samples each). NLI is not ground truth, neutral is not demonstrated falsehood, and non-entailment is only an overclaim proxy. Shorter/weaker glosses can improve this proxy. Blind expert adjudication remains absent: **P6 is only partially validated**.

### Fixed fine reference and hierarchy experiment

**FINAL_FINE** is the historical `without_qualifiers` configuration: generic CR1-derived decomposition with the already implemented generic grammar/attribution fixes → frozen H2 top100 → query-independent wider evidence pools → component-only BGE cosine, attribution weights and top-three evidence/max support → unchanged CR1 core-gated geometric aggregation → top10. Qualifiers remain diagnostics; there is **no qualifier score, NLI, fusion, new alias or primary K200**. Both cutoffs reproduced exact archived candidate scores/ranks before hierarchy traversal; full rankings and hash regressions are archived.

{fine_table}

All 14 Type A questions remain; Q5/Q11 have no canonical units and null recall. Macro metrics average 12 evaluable questions; micro denominators are 47/50, raw denominator 63. Unique recall retains the union-of-target-names definition. Four Type B questions and all historical outputs remain archived; the new entity experiment is Type A only. FINAL_FINE was selected on this reused benchmark, so these are development diagnostics, not held-out generalization.

The frozen temporal hierarchy starts at all eligible L0 roots, scores **.5 centroid cosine + .5 mean(best two of four farthest-point prototype cosines)**, and expands deterministically best-first through L1/L2 to atomic leaves. One reached member activates its existing exact-name entity group once; full group evidence is then accessible, explicitly extending access beyond that one atomic occurrence. Only reached groups receive fine scores. Main hierarchy/flat exhaustive search all fixed H2-eligible groups, while FINAL_FINE uses its H2 prefix; the H2-intersection ceiling checks approximation separately from changed candidate discovery. No expected names or claims guide search.

Predeclared total-work budgets are **25,50,100,200,500**, plus unrestricted traversal. Budgets below the full root-layer cost are `INSUFFICIENT_ROOT_BUDGET`, not evidence of hierarchy failure. Work charges every centroid/prototype/leaf and component–evidence comparison, including logically reused comparisons; aggregation counts are separate. Offline embeddings, hierarchy/pools, model loading, query encoding, graph enumeration and export diagnostics are excluded. FINAL_FINE additionally charges archived H2 vector work; its sparse incidence visits remain separate. Cross-system caches affect execution time, not standalone charges. Thus counts are a declared work proxy, never a production latency speedup.

Flat equal-work uses the same budget and scorer over a fixed seed-42 random group prefix; a candidate is included only if its complete cost fits. It stops at the first unaffordable group; hierarchy continues through its frontier and can score later affordable groups. This policy asymmetry is fixed and disclosed, not adjusted after outcomes. Flat exhaustive scores every eligible group with the identical final scorer. Both are grouped entity baselines; the historical all-node surface-only baseline remains supplemental.

## Results

{variant_table}

Intrinsic columns use 2026 L0 and temporal ARI uses 2024→2026; label proxies aggregate four snapshots. Retrieval columns are the **new 2025 final experiment at budget500**, not obsolete leaf-ranking results. Semantic-only has highest coherence and ignores edges (ARI=1 under edge removal is therefore expected); structural has strong temporal continuity but weaker semantics; static has good semantics but weaker continuity; temporal pays a small coherence cost for persistence; pairwise is perturbation-stable but discards native relation identity. Ship temporal for the required combination, not as a scalar winner.

Conservative 2025 extrinsic results (all costs mean per 14 questions):

{table(headers,[result_row(r) for r in primary[2025]])}

{metric_note}

At budget500, hierarchy retains **{h['retention']['retained_fine_hits']}/13** fine hits, with **{h['retention']['additional_hits']} hierarchy-only recoveries**; the latter are distinct from retention. It scores {fmt(h['work']['fine_candidates_scored'],1)} rather than100 fine candidates ({fmt(fine_reduction*100,1)}% fewer), and charges {fmt(h['work']['total_comparisons'],1)} rather than {fmt(f['work']['total_comparisons'],1)} comparisons ({fmt(reduction*100,1)}% fewer). Equal-work flat recovers {flat['at_k']['10']['found']}/47. Counts exclude H2 incidence work and other operations above; lower work alone is not success.

Unrestricted hierarchy exactly matches exhaustive flat rankings, recovering 9/47 and 10/50 and retaining 9/13 fine hits at both cutoffs, with additional traversal cost. Expanding the candidate universe changes competition even with the same fine scorer. Unrestricted H2 intersection recovers the reference 13/47 and 13/50, but already requires H2 membership: including H2 vector work costs {fmt(next(r['total_including_h2'] for r in intersection_costs if r['cutoff']==2025 and r['budget'] is None),1)} and {fmt(next(r['total_including_h2'] for r in intersection_costs if r['cutoff']==2026 and r['budget'] is None),1)} comparisons, plus H2 incidence visits. It is an approximation control, not an independent cheap discovery mechanism.

![Quality versus charged work](report/figures/hierarchy_retrieval_tradeoff.png)

{conclusion}

Supporting claims have three separate layers: entity recovery; selected evidence/provenance; and strict alignment where evaluable. Every top-ten entity exports selected node IDs/text, attribution, article IDs, graph paths, component cosine and question similarity. Expected claims enter evaluation only. Archived strict alignment resolves **0/188 required Type A claims and 0/36 Type B claims**; all required-claim recalls remain **null / NOT_EVALUABLE_ALIGNMENT**. Two supplemental Type A evidence excerpts previously aligned; they are not substitutes for required claims. Candidate source hit measures the fraction of source-mappable expected claim items with source overlap in evidence of the matching returned entity, then averages over 12 source-evaluable questions. It is claim-item-weighted within each question, not entity recall or entailment. The weaker any-entity source hit is separately exported. All per-question counts, unmapped sources and evidence traces are retained; `final/claim_evaluability.json` includes all 18 questions at both cutoffs.

Annual-2026 sensitivity, which can include papers after February, retains FINAL_FINE's 13/50. Hierarchy500 recovers 1/50, retains 0/13 and scores 33.9 fine groups for 500 comparisons, versus FINAL_FINE's 100 groups and 14,978.6 comparisons. Equal-work flat recovers 3/50 and retains 1/13. Full 2026 tables, zero-candidate question counts and every budget are in `artifacts/evaluation/final/integration_details.md`. CR2/CR4 remain negative local-NLI ablations; CR7 preserves some H2 hits but has lower top10 recovery than standalone conditioned scoring. They are archived and never invoked by FINAL_FINE.

## Verified versus assumed, limitations and four more weeks (T7)

**Verified:** snapshot/arity counts; budget, exclusivity and laminarity checks; native collapse use and event exports; executed coherence/null/ARI values; final scores/ranks/cost traces; **{tests} passing tests**; **{preserved} frozen files** preserved (additive CLI dispatch excluded). Verification concerns implementation and measured proxies, not scientific truth.

**Assumed or proxy:** publication-date semantics; unversioned text's historical validity; MiniLM/BGE scientific meaning; NLI gloss support; lexical entity identity; reasonable fixed weights; incomplete benchmark labels. Source overlap is not claim entailment. Evidence access can improve rank while comparative baselines, numerical extrapolation or wrong entity types remain unsupported. FINAL_FINE still misses most canonical targets. P6 and supporting-claim validity are incomplete, and no expert or unseen-query validation is claimed.

With four more weeks: obtain versioned source text, independently adjudicate labels and claims, improve claim alignment, reserve an independent query set, and study split-aware refinement and larger-corpus scaling. No further reranker, threshold sweep or hierarchy redesign is part of this final pass.

### Final decision checklist

**T1:** heterogeneous, genuinely higher-arity growing TKH; counts and date anomalies above. **T2:** explicit weighted native/spectral/semantic/VI objective, fixed budgets, greedy local optimization; P1/P2/P4 guaranteed procedurally, P3/P5/P6 empirical. **T3:** VI enforces smoothness, overlap matching tracks identities/events; .636 versus .351 temporal ARI with small coherence cost. **T4:** internal evidence retained, partial endpoints keep multiplicities, ≥3 regions keep one hyperedge; coarse geometry loses endpoint detail. **T5:** snapshot-valid split packets generate extractive labels/glosses; held-out non-entailment .542/.479, expert faithfulness unsupported. **T6:** independent BGE/nulls, five edge-removal seeds/CIs and separate temporal ARI; FINAL_FINE13/47 and13/50; hierarchy retention/work above; source recovery measurable, strict required-claim recall unevaluable. **T7:** verified facts and proxy assumptions separated; concrete four-week study above. Per-item PASS/PARTIAL/NOT_EVALUABLE and all deliverables are in `artifacts/evaluation/final_requirement_audit.json`.
'''
    required=['Problem and formal statement','Related work','Method: temporal','Evaluation validity','Results','Verified versus assumed','Final decision checklist']
    if any(s not in text for s in required): raise ValueError('Required report section missing')
    Path('report.md').write_text(text,encoding='utf-8')
    details=['# Final integration details','', 'All rows are development diagnostics. Root-insufficient rows are not valid hierarchy quality trials.',metric_note]
    for year in [2025,2026]:
        details += ['',f'## {year}',table(headers,[result_row(r) for r in run['results'] if r['cutoff']==year])]
    details += ['','## Fine metric completeness',fine_table,'','## Costs and statuses',
                table(['Year','System','Variant','Budget','Coarse','Leaf','Evidence','Fine scores','Total','Status'],
                      [[r['cutoff'],r['system'],r['variant'],r['budget'],*[fmt(r['summary']['work'][k],2) for k in ['coarse_comparisons','leaf_comparisons','evidence_comparisons','fine_candidates_scored','total_comparisons']],json.dumps(r['summary']['status_counts'])] for r in run['results']]),
                '', 'Standalone logical evidence comparisons deliberately charge repeated/cached candidate-component-item evaluations. Actual new dot products are exported separately. H2 sparse incidence visits are an additional operation class, not silently converted to vector comparisons.']
    details += ['## Search coverage',table(['Year','System','Variant','Budget','Zero-candidate questions /14','Mean returned at10','Root cost','Status'],
        [[r['cutoff'],r['system'],r['variant'],r['budget'],r['zero_candidate_questions'],fmt(r['mean_returned_at10']),r['root_layer_cost'],json.dumps(r['status_counts'])] for r in search_coverage]),
        '## H2-intersection prerequisite costs',
        'These controls require the existing H2 top100 membership. The budget limits traversal only. Adding the prerequisite H2 vector comparisons produces the standalone total below; incidence visits remain a separate operation class.',
        table(['Year','Budget','Traversal','H2 vector prerequisite','Total including H2','H2 incidence visits'],
              [[r['cutoff'],r['budget'],*[fmt(r[k],1) for k in ['traversal_comparisons','prerequisite_h2_vector_comparisons','total_including_h2','h2_incidence_entry_visits']]] for r in intersection_costs]),
        '## Required-claim evaluability, every question',
        'Type B counts reuse the frozen alignment audit; its retrieval was not rerun. Zero retrieved alignments with zero alignable claims gives null recall, not zero recall.',
        table(['Year','Q','Type','Expected','Alignable','Retrieved aligned','Strict recall','Status'],
              [[r['cutoff'],r['question_id'],r['question_type'],r['expected_claims_count'],r['automatically_alignable_claims_count'],r['retrieved_aligned_claims_count'],fmt(r['strict_claim_recall']),r['status']] for r in claim_inventory])]
    (ROOT/'integration_details.md').write_text('\n\n'.join(details),encoding='utf-8')
    audit=[]
    def add(req,status,artifact,evidence,limitation=''):
        if not Path(artifact).exists(): raise ValueError('Audit artifact missing: '+artifact)
        audit.append(dict(requirement=req,status=status,artifact=artifact,metric_or_evidence=evidence,limitation=limitation))
    for req in ['P1','P2','P4','T2','T3','T4']:
        add(req,'PASS','report.md','Frozen hierarchy construction, all snapshot invariants, explicit formal/collapse/temporal mechanisms; see tests and hierarchy exports.','No global optimum or naturally occurring hierarchy claim.')
    add('P3','PARTIAL','metrics.json','Independent BGE coherence .7748 vs null .7602; small positive effect.','Embedding proxy, no expert semantic adjudication.')
    add('P5','PASS','metrics.json','Real temporal L0 ARI .636 vs static .351; persistent IDs and all six event types exported.','Empirical continuity, not guaranteed locality of every change.')
    for req in ['P6','T5']:
        add(req,'PARTIAL','metrics.json','Held-out L0/L1 non-entailment .542/.479 and temporally filtered packets.','NLI proxy; no blind expert audit or versioned historical text.')
    add('T1','PASS','artifacts/full/descriptive.json','5798 nodes,1429 hyperedges; four snapshots; type/arity distributions and growth.')
    add('T6','PARTIAL','metrics.json','Intrinsic controls and final hierarchy-vs-fine/flat study executed.','Required claims unalignable; proxy label/evidence validity incomplete.')
    add('T7','PASS','report.md','Explicit verified/assumed limitations and four-more-weeks section.')
    for req,key in [('T6.independent_coherence','coherence'),('T6.null_model','coherence_null'),('T6.perturbation_five_seeds','perturbation_stability'),('T6.confidence_intervals','perturbation_stability'),('T6.real_temporal_stability','temporal_stability'),('T6.extrinsic_hierarchy','hierarchical_retrieval'),('T6.hierarchy_vs_flat','final_retrieval')]:
        add(req,'PASS','metrics.json',key,'Negative downstream results do not fail experiment completeness; repeated benchmark is not held out.')
    add('T6.label_overclaim','PARTIAL','metrics.json','label_faithfulness','Measured NLI non-entailment, not true overclaim ground truth.')
    add('T6.strict_supporting_claims','NOT_EVALUABLE',str(ROOT/'results.json'),'0/188 required Type A claims align; null strict recall.','No convenient evidence substituted for claims.')
    add('T6.source_provenance','PASS','metrics.json','Per-question source_hit/provenance_overlap and candidate-associated diagnostics.','Weaker than claim entailment.')
    for req,artifact in [('Deliverable0','report.md'),('Deliverable1','README.md'),('Deliverable2',str(ROOT/'hierarchy_2026.json')),('Deliverable3','metrics.json'),('Deliverable4','report.md'),('Deliverable5','AI_USAGE.md'),('Deliverable6_optional','report/figures/hierarchy_levels.png')]:
        add(req,'PASS',artifact,'Runnable pinned code; four snapshot exports/events; consolidated metrics; compact report; AI disclosure; static hierarchy figure as applicable.','Markdown supplied; rendered page count depends on conversion settings.' if req=='Deliverable4' else '')
    required_ids={f'P{i}' for i in range(1,7)}|{f'T{i}' for i in range(1,8)}|{f'Deliverable{i}' for i in range(6)}|{'Deliverable6_optional'}
    if not required_ids <= {r['requirement'] for r in audit}: raise ValueError('Requirement omitted')
    write_json('artifacts/evaluation/final_requirement_audit.json',dict(status_counts=dict(Counter(r['status'] for r in audit)),records=audit,
                                                                   preserved_files=preserved,tests=tests,report_words=len(text.split())))
    print('FINAL REPORT COMPLETE',tests,'tests;',preserved,'preserved files;',len(text.split()),'report words')


if __name__=='__main__': main()
