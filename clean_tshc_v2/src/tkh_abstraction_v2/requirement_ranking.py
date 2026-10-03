"""Frozen-model, gold-blind multi-evidence ranking with explicit conjunctions."""
import math
import re
from .entity_index import normalize_name
from .models import pack
from .verification import status_from_probabilities


def atomic_components(intent):
    """Supplement retrieval intent without changing any first-stage query strings."""
    question = intent['original_question']
    result = []
    for component in intent['components']:
        start, end = component['start'], component['end']
        text = question[start:end]
        # Keep disjunctions and the core task intact. Split positive/excluded lists.
        separators = [] if component['kind'] == 'core_task' or re.search(r'\bor\b', text, re.I) else list(
            re.finditer(r',\s+(?:and\s+)?|\s+(?:and|nor)\s+', text, re.I))
        bounds = [start] + [start + m.end() for m in separators]
        ends = [start + m.start() for m in separators] + [end]
        for left, right in zip(bounds, ends):
            while left < right and question[left].isspace(): left += 1
            while right > left and question[right-1].isspace(): right -= 1
            if not question[left:right].strip(): continue
            result.append(dict(id='r'+str(len(result)), kind=component['kind'],
                text=question[left:right], source_span=dict(start=left, end=right, text=question[left:right]),
                connector=component.get('connector', ''),
                polarity='excluded_property' if component['kind']=='exclusion' else 'positive'))
    if not any(c['kind']=='core_task' for c in result):
        result.insert(0, dict(id='core_fallback', kind='core_task', text=question,
            source_span=dict(start=0,end=len(question),text=question),connector='',polarity='positive'))
    return result


def select_requirement_evidence(index, entity_id, similarities, k):
    eligible = {}
    for field, rows in index['entities'][entity_id]['evidence'].items():
        if field in ('articles', 'graph_evidence'): continue
        for row in rows:
            if not row['candidate_specific']: continue
            eid = row['evidence_id']
            priority = 0 if field == 'claims' else 1 if field == 'explicit_mentions' else 2
            item = dict(row, text=index['items'][eid]['text'], provenance=index['items'][eid]['provenance'],
                priority=priority, similarity=float(similarities[eid]))
            if eid not in eligible or priority < eligible[eid]['priority']: eligible[eid] = item
    ordered = sorted(eligible.values(), key=lambda r:(-r['similarity'],r['priority'],r['evidence_id']))
    selected, seen = [], set()
    for row in ordered:
        key = normalize_name(row['text'])
        if key in seen: continue
        seen.add(key); selected.append(row)
        if len(selected) == k: break
    return selected


def bundle(entity, component, evidence):
    return dict(node_id=entity['entity_id']+'::'+component['id'],
        query='Candidate '+entity['name']+': '+component['text'],
        blocks=[dict(field='identity',entity_id=entity['entity_id'],
            text='Candidate: '+entity['name']+'\nType: '+', '.join(entity['types']))] +
            [dict(field='requirement_evidence',evidence_id=r['evidence_id'],text=r['text']) for r in evidence])


def hypothesis(entity, component):
    # For exclusions this asserts the forbidden property; swap polarity later.
    return entity['name'] + (' involves ' if component['kind']=='exclusion' else
        ' is suitable for ' if component['kind']=='core_task' else ' satisfies the condition: ') + component['text'] + '.'


def competition_ranks(scores):
    """Equal logits share a rank, so identity is not a hidden relevance feature."""
    values=sorted(set(scores.values()),reverse=True)
    ranks={v:1+sum(s>v for s in scores.values()) for v in values}
    return {i:ranks[s] for i,s in scores.items()}


def aggregation(records, epsilon=1e-12, use_nli=True):
    positives=[r for r in records if r['kind']!='exclusion']
    supported=sum(r['status']=='SUPPORTED' for r in records) if use_nli else 0
    contradicted=sum(r['status']=='CONTRADICTED' for r in records) if use_nli else 0
    positive_supported=sum(r['status']=='SUPPORTED' for r in positives) if use_nli else 0
    tier=3 if contradicted else 0 if positives and positive_supported==len(positives) else 1 if positive_supported else 2
    reciprocals=[1/r['requirement_rank'] for r in positives]
    return dict(supported_count=supported,positive_supported_count=positive_supported,
        contradicted_count=contradicted,unknown_count=len(records)-supported-contradicted,
        requirement_geometric_score=math.exp(sum(math.log(x+epsilon) for x in reciprocals)/len(reciprocals)) if reciprocals else 0.,
        weakest_requirement_score=min(reciprocals,default=0.), final_tier=tier)


def order_candidates(rrf_rows, global_details, components, records, cfg, use_nli=True):
    pool=rrf_rows[:cfg['rerank_k']]
    ids=[r['entity_id'] for r in pool]
    global_ranks=competition_ranks({i:global_details[i]['relevance_score'] for i in ids})
    for position, component in enumerate(components):
        scores={i:records[i][position]['requirement_relevance_logit'] for i in ids
                if records[i][position]['requirement_relevance_logit'] is not None}
        ranks=competition_ranks(scores)
        neutral=(len(scores)+1)/2 if scores else 1.
        for i in ids:
            records[i][position]['requirement_rank']=ranks.get(i,neutral)
            records[i][position]['rank_imputed']=i not in ranks
            records[i][position]['scored_candidates']=len(scores)
    traces={}
    for rrf_rank,i in enumerate(ids,1):
        traces[i]=dict(entity_id=i,rrf_rank=rrf_rank,global_relevance=dict(
            cross_encoder_logit=global_details[i]['relevance_score'],rank=global_ranks[i]),
            requirements=records[i],**aggregation(records[i],cfg['requirement_epsilon'],use_nli))
    def key(i):
        t=traces[i]
        return (t['final_tier'],-t['requirement_geometric_score'],-t['weakest_requirement_score'],
            global_ranks[i],t['rrf_rank'],i)
    ordered=sorted(ids,key=key)
    for rank,i in enumerate(ordered,1): traces[i]['final_rank']=rank
    rows=[dict(entity_id=i,score=traces[i]['requirement_geometric_score'],stage='requirement_conjunction') for i in ordered]
    return rows+[dict(r,stage='unreranked_fusion_tail') for r in rrf_rows[cfg['rerank_k']:]],traces


def score_requirements(intent, rrf_rows, index, retriever, cross, nli, cfg, top_k):
    components=intent['ranking_components']
    ids=[r['entity_id'] for r in rrf_rows[:cfg['rerank_k']]]
    records={i:[] for i in ids}; cards=[]; pending=[]; nli_pairs=[]; nli_pending=[]
    costs=dict(requirement_dense_comparisons=0,requirement_cross_encoder_pairs=0,nli_pairs=0)
    # Candidate name is in model inputs; dense selection uses literal component text
    # and is restricted to the candidate's attached evidence.
    queries=retriever.encoder.encode([c['text'] for c in components])
    matrix=retriever.evidence_vectors @ queries.T
    costs['requirement_dense_comparisons']=int(matrix.size)
    similarities_by_component=[dict(zip(retriever.evidence_ids,map(float,matrix[:,j]))) for j in range(len(components))]
    for i in ids:
        entity=index['entities'][i]
        for j,component in enumerate(components):
            similarities=similarities_by_component[j]
            evidence=select_requirement_evidence(index,i,similarities,top_k)
            card=bundle(entity,component,evidence)
            record=dict(component, selected_evidence_ids=[e['evidence_id'] for e in evidence],
                evidence=evidence,requirement_relevance_logit=None,probabilities=None,
                status='NOT_SUPPORTED_BY_AVAILABLE_EVIDENCE',hypothesis=hypothesis(entity,component),
                relevance_input=None,nli_input=None)
            records[i].append(record)
            if not evidence: continue
            if cross:
                cards.append(card);pending.append(record)
            if nli:
                hyp,premise,detail=pack(nli.tokenizer,record['hypothesis'],card['blocks'],cfg['cross_encoder_max_tokens'])
                record['nli_input']=dict(hypothesis=hyp,premise=premise,**detail)
                if len(detail['included'])>1:
                    nli_pairs.append((premise,hyp));nli_pending.append(record)
            else:
                record['status']='NOT_EVALUABLE'
    if cards:
        details=cross.score_pairs(cards,batch=cfg['batch_size'])
        for card,record in zip(cards,pending):
            detail=details[card['node_id']]
            record['relevance_input']=detail
            if len(detail['included'])>1:
                record['requirement_relevance_logit']=detail['relevance_score']
        costs['requirement_cross_encoder_pairs']=len(cards)
    if nli_pairs:
        for record,detail in zip(nli_pending,nli.score(nli_pairs)):
            record['probabilities']=detail['probabilities']
            record['status']=status_from_probabilities(detail['probabilities'],True,cfg['nli_threshold'],record['kind']=='exclusion')
        costs['nli_pairs']=len(nli_pairs)
    return records,costs
