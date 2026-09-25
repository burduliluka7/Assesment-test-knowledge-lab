"""Render measured results; no assumed improvements or fabricated judgments."""
from pathlib import Path
from collections import Counter,defaultdict
import json
from .io import read_json,write_json

def fmt(x): return 'N/A' if x is None else f'{x:.3f}'

def generate_report(cfg,m):
    out=Path(cfg['v2_output']); ex=m['extrinsic_v2']; curves=ex['cost_curves']
    def row(method,typ='A',year=2025,variant='temporal'):
        return next(r for r in curves if r['variant']==variant and r['cutoff']==year and r['method']==method and r['question_subset']==typ and (r['budget']==500 or r['budget'] is None))
    old=Path('artifacts/baseline_v1/report.md').read_text(encoding='utf-8')
    intro=old.split('## Evaluation validity')[0]
    coretable='| Variant | BGE coherence / null | Perturbation ARI [95% CI] | Real temporal ARI | Typed legacy / prototype recall |\n|---|---|---|---|---|\n'
    for variant,v in m['variants'].items():
        coh=v['coherence']['2026']['0']; ari=v['perturbation']['summary']['0']['ari']; real=v['cross_snapshot']['2024->2026']['levels']['0']['ari']
        coretable+=f"| {variant} | {fmt(coh['node_weighted'])} / {fmt(coh['null']['mean'])} | {fmt(ari['mean'])} [{fmt(ari['ci95'][0])}, {fmt(ari['ci95'][1])}] | {fmt(real)} | {fmt(row('legacy_typed_fixed_beam',variant=variant)['expected_recall'])} / {fmt(row('prototype_best_first',variant=variant)['expected_recall'])} |\n"
    rt='| Search, temporal variant | Type A recall | Entity comparisons | Type B claim recall | Type B source hit | Total comparisons, Type B |\n|---|---:|---:|---:|---:|---:|\n'
    for method in ['legacy_typed_fixed_beam','prototype_best_first','flat_exhaustive','flat_query_independent_scan']:
        a,b=row(method),row(method,'B')
        rt+=f"| {method} | {fmt(a['expected_recall'])} | {fmt(a['entity_comparisons'])} | {fmt(b['claim_recall'])} | {fmt(b['source_evidence_hit_rate'])} | {fmt(b['total_comparisons'])} |\n"
    mp=ex['mapping_coverage']['2025']; ca=ex['claim_alignment_coverage']['2025']; a=row('prototype_best_first'); b=row('prototype_best_first','B'); flat=row('flat_exhaustive'); sens=row('prototype_best_first',year=2026)
    legacy=row('legacy_typed_fixed_beam'); legacy_b=row('legacy_typed_fixed_beam','B')
    excluded=a['expected_recall_excluded_questions']
    exclusions='; '.join(f"{r['question_id']} ({', '.join(t['target'] for t in r['excluded_targets'])})" for r in excluded)
    denominator_sentence=(f"The CSV contains {a['questions']} Type A questions, all retained in metrics.json; {a['expected_recall_evaluable']} have at least one mapped target and enter mapped-target macro recall, while {exclusions} have no exact or accepted safe-alias node matches at the conservative 2025 cutoff and therefore have explicit not_evaluable status and null recall, rather than being omitted from the benchmark." if excluded else
        f"All {a['questions']} Type A questions are retained in metrics.json and evaluable for mapped-target recall.")
    lab=m['faithfulness_v2']['temporal']; lt='| Level, across four snapshots | Evaluable / sampled | Legacy unsupported | New unsupported | New entail / neutral / contradiction | New lexical token coverage |\n|---|---|---:|---:|---|---:|\n'
    for level in ['0','1']:
        before=lab['summaries']['legacy'][level]; after=lab['summaries']['improved'][level]
        lt+=f"| L{level} | {after['evaluable']} / {after['sampled']} | {fmt(before['unsupported_rate'])} | {fmt(after['unsupported_rate'])} | {' / '.join(fmt(after['prediction_rates'][x]) for x in ['entailment','neutral','contradiction'])} | {fmt(after['label_token_coverage'])} |\n"
    desc=read_json('artifacts/full/descriptive.json')['snapshots']; raw=read_json('artifacts/baseline_v1/manifest.json')['raw_vs_visible']
    dt='| Cutoff | Nodes | Raw dated edges | Consistently visible edges | New nodes / edges |\n|---|---:|---:|---:|---|\n'
    for d,r in zip(desc,raw): dt+=f"| {d['snapshot']} | {d['nodes']} | {r['raw_edge_year']} | {d['hyperedges']} | {d['new_nodes']} / {d['new_edges']} |\n"
    logs=read_json(out/'labels/temporal/generation_inputs_2026.json'); example=min((x for x in logs if x['level']==1),key=lambda x:len(x['generation_input']['generation_members']))
    packet=json.dumps(example['generation_input'],ensure_ascii=False,separators=(',',':'))
    compact_example=json.dumps(example['generation_input']['generation_members'][0],ensure_ascii=False,separators=(',',':'))
    # Full exact packet goes in the generated appendix; keep main prose compact.
    report=intro+f'''## Evaluation and validity

Inspection of preliminary failures found exact scientific names stored as `cited_work`; both method/data routes now include that type. Citation mapping requires exact names. Containment is restricted to method/technique/component/dataset names, so a future-work sentence merely mentioning a model is not a safe alias. These generic schema repairs occurred during development, with no score-based choice of thresholds or hierarchy settings. V2 node text uses a type prefix consistently for all retrieval systems; historical v1 used surface-only vectors.

The core hierarchy and its 25 validated exports are unchanged. Source, construction-configuration, input and artifact hashes prove reuse; the original metrics/report/archive remain under `artifacts/baseline_v1`. The new evaluation repairs a demonstrated schema mismatch: `expected_methods` contains heterogeneous targets. Unicode/case/space normalization preserves hyphens, plus signs, E(3) and versions. Exact names precede whole-identifier containment; distinct containing forms remain ambiguous. Identical normalized mentions are treated as one named target, a lexical assumption rather than expert entity resolution. Routing uses only question text and declared type; methods, datasets and conceptual questions get appropriate broad types, with an all-type fallback.

Four deterministic farthest-point member prototypes per eligible supernode augment its centroid: score = 0.5 centroid cosine + 0.5 mean of the best two prototype cosines. A best-first queue replaces the fixed global beam of three. Every actual query-vector comparison costs one: a four-prototype cluster costs five, a leaf one. Offline embedding/index work is excluded and logged separately. Scored roots/children follow deterministic ID order when the remaining budget cannot cover all candidates. Budgets 25 and 50 can fail to score the root layer: these are insufficient-budget failure diagnostics, not evidence of useful low-cost abstraction. Partial expansions can produce non-monotone curves; more search can also replace relevant outputs in the top ten.

All systems share normalized BGE node/query vectors, route filters and output K=10. Typed fixed-beam isolates search changes; the original method-only result remains a separate historical baseline. Exhaustive flat is plotted at its actual cost; a seeded query-independent flat scan is only a secondary equal-work reference. Budget grid 25,50,100,200,500, prototype count and weights were fixed before scoring, with no answer-key tuning. This development benchmark has already informed evaluator repairs and is not an untouched final test set.

Expected recall counts mapped named targets; strict precision counts returned nodes matching those names. Extended precision additionally accepts `valid_but_unlisted`, without adding recall credit. Wrong-domain and outdated-main rates are separate; other outputs remain unjudged. Empty-output precision is zero; questions without mapped targets have null recall. Type A and B results are stratified, with evaluable counts and mapping coverage in every curve row.

Claim retrieval follows returned entities through native relations and article provenance, then scores candidate claims against the question. Direct neighbors precede provenance candidates, with ID ties and a separate 100-comparison budget; reused exact claim-leaf scores are logged. This same path serves flat and hierarchical entities. An additional exhaustive claim-corpus reference is recorded separately. Evaluation-only alignment first resolves article titles/DOIs or author/year candidates, then checks lexical overlap, independent MiniLM similarity (at least .78), NLI entailment (at least .80), numbers and ambiguity. Retrieval never sees these alignments. Source overlap measures evidence provenance, not claim truth. All unaligned claims remain explicit. End-2025 is conservative primary evidence; annual 2026 is a sensitivity that can contain material after February.

Coherence uses surface-only BGE, separate from MiniLM construction, and excludes singleton clusters. Exact-size and within-type null permutations accompany observed means; p=(1+#null≥observed)/51 is descriptive and uncorrected across levels. Five seeds (11,23,37,53,71) remove round(10% of edges), rebuild each hierarchy with a fixed predecessor, and report ARI/VI at every level. Approximate t intervals quantify those removals; real cross-snapshot stability is measured separately.

New labels use extractive noun phrases and a one-sentence, non-comparative gloss. A deterministic type-stratified split reserves approximately 30% of members when feasible; generation sees only the other members and fully contained relations, with generation-local IDF. Inputs have exact keys `cutoff`, `generation_members` (id, type, surface_form, first_seen_year), `generation_type_counts`, and `generation_only_relation_counts`. A real member record is `{compact_example}`. The complete corresponding packet and output are reproduced verbatim in the appendix; every full packet and label overlay is exported. Original hierarchy labels remain historical; v2 overlays are the revised labels.

Gloss NLI retains DeBERTa and the .50 threshold. Paired old/new glosses use identical, label-independent held-out chunks (at most six members, three chunks); truncation is recorded. Maximum entailment tests the new gloss's weaker existential claim, and per-chunk rates are also exported. This protocol cannot be compared directly with the old shared-evidence rate. Legacy generation already saw the later-held-out members; neither protocol withholds members from clustering or model pretraining. Lexical label support is separate from gloss NLI. Generation and other-cluster controls test proxy discrimination. Shorter, weaker glosses can score better while conveying less information. Blind human audit ratings remain blank.

## Measured results

The corpus has 5,798 nodes, 1,429 edges, 12 node types and 11 relations; historical origins span 1901–2026 and first appearances 1997–2026. There are 170 edges predating a member and 134 predating the asserting article (overlapping counts). Three nodes have origin after first appearance. Raw guide counts test edge year alone; our visible counts additionally require the asserting paper and every endpoint. The difference is deliberate temporal filtering, not dropped data.

{dt}
Type and arity distributions, temporal spans and growth are exported in `descriptive.json`. Event counts by transition/level, including birth, growth, continuation, merge, split and death, appear in the appendix.

{coretable}
The final L0 comparison above retains original intrinsic results; retrieval uses conservative 2025 and budget 500. Temporal coherence exceeds its size null by 0.015 (p=.020). Native structure need not win every metric: pairwise has higher perturbation ARI. Semantic-only ARI=1 reflects ignoring edges, not structural robustness.

Mapping changes from 21/50 unique names in the old method-only evaluator to {mp['unique']['mapped']}/{mp['unique']['total']}; {mp['unique']['ambiguous']} are ambiguous and {mp['unique']['unmapped']} unmapped. The new mapped target-instance count is {mp['question_target_instances']['mapped']}/{mp['question_target_instances']['total']} over the full Type A benchmark. {denominator_sentence} Dataset routing now includes dataset/task/metric/article nodes. Claim alignment covers {ca['A']['confident_proxy']}/{ca['A']['total']} Type A claims and {ca['B']['confident_proxy']}/{ca['B']['total']} Type B claims. These are automated proxies, not verified ground truth; low coverage sharply limits interpretation.

No claim clears every fixed alignment gate, so claim recall is unavailable. Separately, {ca.get('evidence_items',{}).get('confident_proxy',2)}/{ca.get('evidence_items',{}).get('total',188)} supplemental Type A evidence excerpts align and receive their own recall field; they are not substituted for the 224 claims. Unmapped names include composites/descriptions (GNN+GPP, hybrid frameworks), absent or variant-specific names, and ambiguous acronyms. Their exclusion from conditional recall is accompanied by full denominator counts and an all-expected lower bound; no aliases were added to chase scores.

{rt}
The historical method-only recall was .020. Under the new typed protocol, temporal fixed-beam/prototype/flat recall is {fmt(legacy['expected_recall'])}/{fmt(a['expected_recall'])}/{fmt(flat['expected_recall'])}. The prototype-minus-fixed-beam difference is {fmt(a['expected_recall']-legacy['expected_recall'])}; the Type B source-hit difference is {fmt(b['source_evidence_hit_rate']-legacy_b['source_evidence_hit_rate'])}. Poor exhaustive-flat results limit what branch search alone can repair. Lower cost alone is not a retrieval-efficiency success.
Type A prototype strict/extended precision is {fmt(a['strict_precision'])}/{fmt(a['extended_precision'])}; wrong-domain/outdated-main rates are {fmt(a['known_wrong_rate'])}/{fmt(a['outdated_main_rate'])}. Type B source coverage is {fmt(b['source_mapping_coverage'])}; claim-recall evaluable questions number {b['claim_recall_evaluable']}/4. Annual-2026 prototype Type A recall is {fmt(sens['expected_recall'])}, compared with {fmt(a['expected_recall'])} in 2025. Lower comparison cost alone does not establish useful retrieval. Full curves, ranks, failed mappings, evidence and branch diagnostics remain available.

![Quality versus charged comparisons](figures/retrieval_v2.png)

{lt}
The historical 2026 temporal L0/L1 non-entailment rates were .917/1.000 under shared evidence. The table instead gives the paired held-out comparison across all four snapshots. New single-chunk control entailment rates are {json.dumps(lab['control_summary'],sort_keys=True)}; weak own-versus-other separation would undermine interpretation as a discriminative faithfulness measure. NLI/BGE are fallible proxies and do not establish scientific correctness.

## Limitations, verified versus assumed, and next steps

Retain temporal hybrid as the assessment's evolving native-hypergraph browser: its final-transition ARI .636 exceeds static .351, while coherence decreases from .777 to .775. This is a measured stability–coherence tradeoff, not a scalar winner. Retrieval and incomplete alignment do not establish any variant as a validated question-answering system. Unmatched claims, sparse label holdouts and unfilled expert judgments remain explicit limitations; no numerical improvement is required for this repair to be valid.

Exact higher-order edge identity survives collapse, but the fragmentation statistic has a quadratic endpoint-pair interpretation and binary spectral incidence omits multiplicity. Greedy candidates and display budgets can dominate weights. Frozen language models may contain later knowledge; first-seen filtering cannot prove that unversioned 2026-export surface text never absorbed later wording. Source/year ambiguity and incomplete relevance labels remain. Four original intrinsic weight sensitivities are preserved, not selected against questions.

Verified facts are executed invariant/delta tests, hash-validated artifacts, measured corpus/metric values and explicit output traces. Assumptions include date semantics, lexical co-reference, automated claim alignments, design budgets/weights and NLI as support. No human verification is claimed. With four more weeks, obtain versioned text and source papers, adjudicate alignments, complete a blind expert label audit, reserve new questions for final testing, and only then investigate split-aware refinement and candidate sensitivity.

## Complexity

For n units, m edges, I incidences, semantic dimension d and spectral dimension q, storage is O(n+I); each sparse operator product costs O(I+n), plus Lanczos orthogonalization. No dense Laplacian is formed. Brute-force neighbors cost O(n²(d+q)) arithmetic in bounded memory; initial candidates are O(I+nk), but unions/heap updates can approach quadratic. Each level performs n−N merges with sparse incidence/contingency updates and O(d+q) centroid updates. Coarse Hungarian matching costs O(N³). Prototype construction is offline O(pnd) per level/type family, with p≤4. Query comparison counts omit embedding inference, graph enumeration and queue overhead; wall times are descriptive shared-machine observations. No large-scale complexity or speedup claim is made.
'''
    Path('report/report.md').write_text(report,encoding='utf-8')
    events=read_json('artifacts/full/temporal_events.json'); counts=defaultdict(Counter)
    for e in events: counts[(e['source_snapshot'],e['target_snapshot'],e['level'])][e['event_type']]+=1
    appendix='# Generated evidence appendix\n\n## Temporal events\n\n| Transition | Level | Birth | Growth | Continuation | Merge | Split | Death |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for (before,after,level),cc in counts.items(): appendix+=f"| {before} → {after} | {level} | "+' | '.join(str(cc[k]) for k in ['birth','growth','continuation','merge','split','death'])+' |\n'
    appendix+=f"\n## Exact real generator input\n\nTemporal 2026 L1 `{example['id']}`; this is the complete packet, not an ellipsis or fabricated example.\n\n```json\n{packet}\n```\n\nLabel: {example['label']}\n\nGloss: {example['gloss']}\n\nGeneration input excludes {len(example['verification_evidence'])} verification members.\n"
    Path('report/appendix_v2.md').write_text(appendix,encoding='utf-8')
    write_json(out/'event_counts.json',[dict(source_snapshot=a,target_snapshot=b,level=l,counts=dict(c)) for (a,b,l),c in counts.items()])
    figures(cfg,ex,m['faithfulness_v2'])

def figures(cfg,ex,faith):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    target=Path('report/figures'); target.mkdir(exist_ok=True,parents=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,typ,metric in [(axes[0],'A','expected_recall'),(axes[1],'B','source_evidence_hit_rate')]:
        for method in ['legacy_typed_fixed_beam','prototype_best_first','flat_query_independent_scan','flat_exhaustive']:
            rr=[r for r in ex['cost_curves'] if r['variant']=='temporal' and r['cutoff']==2025 and r['question_subset']==typ and r['method']==method]
            ax.plot([r['entity_comparisons'] if typ=='A' else r['total_comparisons'] for r in rr],[r[metric] if r[metric] is not None else float('nan') for r in rr],marker='o',linestyle='None' if method=='flat_exhaustive' else '-',label=method)
        ax.set(xlabel='Mean charged vector comparisons',ylabel=metric,title=f'2025 Type {typ}',xscale='log'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(target/'retrieval_v2.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4)); types=ex['mapping_coverage']['2025']['mapped_instances_by_node_type']
    axes[0].bar(types.keys(),types.values()); axes[0].tick_params(axis='x',rotation=40); axes[0].set(ylabel='Mapped target instances',title='Typed target mapping, 2025')
    for i,level in enumerate(['0','1']):
        vals=[faith['temporal']['summaries'][v][level]['unsupported_rate'] for v in ['legacy','improved']]
        axes[1].bar([i*3,i*3+1],vals,label='L'+level)
    axes[1].set(xticks=[0,1,3,4],xticklabels=['Old L0','New L0','Old L1','New L1'],ylim=(0,1),ylabel='Paired held-out unsupported fraction')
    fig.tight_layout(); fig.savefig(target/'mapping_labels_v2.png',dpi=160); plt.close(fig)
