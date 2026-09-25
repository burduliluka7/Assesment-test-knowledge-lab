import re
import unicodedata
import numpy as np
from ..retrieval import flat,hierarchical,supporting_claims
from ..io import read_csv,read_json,locate,write_json
from .statistics import paired_bootstrap

def normalize_name(text):
    text=unicodedata.normalize('NFKC',text).casefold()
    text=''.join('-' if unicodedata.category(c)=='Pd' else c for c in text)
    # '+' and '-' distinguish scientific model variants (DeepH vs DeepH-E3).
    return re.sub(r'[^\w+\-]+',' ',text).strip()

def map_methods(gt,nodes):
    all_methods=[n for n in nodes if n['type']=='method']; out={}
    for name in sorted({m for g in gt.values() for m in g.get('expected_methods',[])}):
        norm=normalize_name(name)
        exact=[n['id'] for n in all_methods if normalize_name(n['surface_form'])==norm]
        boundary=[n['id'] for n in all_methods if re.search(r'(?<![\w+\-])'+re.escape(norm)+r'(?![\w+\-])',normalize_name(n['surface_form']))]
        ids=exact or boundary
        out[name]=dict(node_ids=ids,tier='exact' if exact else 'whole_token' if boundary else 'unmapped',ambiguous=len(ids)>1)
    return out

def metrics(result,expected,mapping):
    targets={m:set(mapping[m]['node_ids']) for m in expected}; available={m:s for m,s in targets.items() if s}
    returned=set(result['returned_ids']); hits={m for m,s in available.items() if s&returned}
    correct={x for x in returned if any(x in s for s in targets.values())}
    recall=len(hits)/len(available) if available else None
    precision=len(correct)/len(returned) if returned else 0.
    ranks=[i+1 for i,x in enumerate(result['ranked_ids']) if any(x in s for s in available.values())]
    all_ranks=[next((i+1 for i,x in enumerate(result['ranked_ids']) if x in s),None) for s in available.values()]
    return dict(status='computed' if available else 'missing_data',recall_mapped=recall,
        recall_all_expected_lower_bound=len(hits)/len(expected) if expected else None,
        lower_bound_caveat='Only rule-based lexical recoveries counted; alignments are unadjudicated and unmapped targets may be present under other names',
        strict_node_precision=precision if available else None,
        f1=(2*precision*recall/(precision+recall) if precision+recall else 0.) if recall is not None else None,
        hit=bool(hits) if available else None,mapped_targets=len(available),expected_targets=len(expected),recovered_methods=sorted(hits),
        first_target_rank=min(ranks) if ranks else None,
        score_inspections_to_first=result['inspections'] if ranks else None,
        rank_to_recover_all=max(all_ranks) if all_ranks and all(x is not None for x in all_ranks) else None,
        censored_no_target=not bool(ranks),inspections=result['inspections'],returned_ids=result['returned_ids'])

def evaluate(h,snap,X,encoder,cfg,directory,output):
    questions=read_csv(locate(directory,'questions.csv')); gt=read_json(locate(directory,'ground_truth.json'))
    articles=read_csv(locate(directory,'collection10_articles.csv'))
    if not all('question_id' in q and 'question' in q for q in questions): raise ValueError('unknown questions schema')
    if not all('id' in a and 'year' in a and 'title' in a for a in articles): raise ValueError('unknown articles schema')
    mapping=map_methods(gt,snap['nodes']); write_json(output,mapping)
    supported=[q for q in questions if gt.get(q['question_id'],{}).get('expected_methods')]
    Q=encoder.encode([q['question'] for q in supported],f"retrieval-queries-{snap['snapshot']}")
    records=[]
    for q,v in zip(supported,Q):
        expected=gt[q['question_id']]['expected_methods']; f=flat(v,snap['nodes'],X)
        rows={}
        for b in cfg['retrieval_budgets']:
            hr=hierarchical(v,h,snap['nodes'],X,b,cfg['retrieval_beam'])
            rows[str(b)]=metrics(hr,expected,mapping)
            rows[str(b)]['supporting_evidence']=supporting_claims(v,hr['returned_ids'],snap,X)
        fm=metrics(f,expected,mapping); fm['supporting_evidence']=supporting_claims(v,f['returned_ids'],snap,X)
        records.append(dict(question_id=q['question_id'],flat=fm,hierarchical=rows))
    summary={}
    for b in cfg['retrieval_budgets']:
        rr=[r for r in records if r['flat']['recall_mapped'] is not None]
        if not rr:
            summary[str(b)]=dict(status='not_applicable',reason='No expected methods map at this cutoff',questions_with_mapped_targets=0)
            continue
        hv=[r['hierarchical'][str(b)]['recall_mapped'] for r in rr]; fv=[r['flat']['recall_mapped'] for r in rr]
        summary[str(b)]=dict(status='computed',hierarchical_recall=float(np.mean(hv)),flat_recall=float(np.mean(fv)),
            questions_with_mapped_targets=len(rr),
            hierarchical_hit_rate=float(np.mean([r['hierarchical'][str(b)]['hit'] for r in rr])),flat_hit_rate=float(np.mean([r['flat']['hit'] for r in rr])),
            hierarchical_inspections=float(np.mean([r['hierarchical'][str(b)]['inspections'] for r in rr])),
            flat_inspections=float(np.mean([r['flat']['inspections'] for r in rr])),paired_recall_difference=paired_bootstrap(hv,fv,cfg['seed']))
    return dict(status='computed',snapshot=snap['snapshot'],question_count=len(questions),evaluated_questions=len(supported),
        questions_without_mappable_targets=[r['question_id'] for r in records if r['flat']['mapped_targets']==0],
        excluded_questions=[q['question_id'] for q in questions if q not in supported],
        excluded_reason='claim-only questions have no expected_methods',top_k=10,scorer=encoder.metadata,
        protocol='Fixed output K=10; exhaustive flat with every comparison charged versus budget-limited hierarchy. Costs differ and are reported; no free flat ranking. Whole child batches; global beam fixed before results.',
        precision_caveat='Strict listed-target node precision; unlisted valid methods can be penalized, so this is not relevance precision.',
        supporting_claim_recall=dict(status='missing_data',reason='GT claim paraphrases have no verified claim-node alignment; no fuzzy matches treated as truth'),
        mapping_coverage=dict(mapped=sum(bool(x['node_ids']) for x in mapping.values()),total=len(mapping),ambiguous=sum(x['ambiguous'] for x in mapping.values())),
        summary=summary,per_question=records)
