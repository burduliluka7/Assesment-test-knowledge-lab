"""Same cosine scorer in both systems. Every scored object costs one inspection."""
import heapq
import numpy as np

def normalize(x): return x/max(float(np.linalg.norm(x)),1e-12)

def flat(query,nodes,X,topk=10,budget=None):
    eligible=[i for i,n in enumerate(nodes) if n['type']=='method']
    # Full exhaustive ranking is the primary baseline, with all scoring charged.
    if budget is not None: eligible=eligible[:budget]
    query=normalize(query)
    ranking=sorted(eligible,key=lambda i:(-float(normalize(X[i])@query),nodes[i]['id']))
    return dict(inspections=len(eligible),scored_leaf_ids=[nodes[i]['id'] for i in eligible],
        returned_ids=[nodes[i]['id'] for i in ranking[:topk]],ranked_ids=[nodes[i]['id'] for i in ranking])

def hierarchical(query,h,nodes,X,budget=250,beam=3,topk=10):
    query=normalize(query)
    idx={n['id']:i for i,n in enumerate(nodes)}; methods={n['id'] for n in nodes if n['type']=='method'}
    clusters={c['id']:c for level in h['levels'] for c in level}
    children={c['id']:[] for level in h['levels'] for c in level}
    for level in h['levels'][1:]:
        for c in level: children[c['parent_id']].append(c['id'])
    inspected=[]; leaves=[]; frontier=[c['id'] for c in h['levels'][0]]
    for level in range(len(h['levels'])):
        candidates=[uid for uid in sorted(frontier) if methods.intersection(clusters[uid]['member_ids'])]
        if len(inspected)+len(candidates)>budget:
            # Whole sibling batches prevent free ranking or arbitrary partial scanning.
            break
        scored=[]
        for uid in candidates:
            c=clusters[uid]; ii=[idx[x] for x in c['member_ids'] if x in methods]
            score=float(normalize(X[ii].mean(axis=0))@query)
            inspected.append(uid); scored.append((score,uid))
            if level==len(h['levels'])-1: leaves.append((score,c['member_ids'][0]))
        selected=sorted(scored,key=lambda p:(-p[0],p[1]))[:beam]
        frontier=[child for _,uid in selected for child in children[uid]]
    ranked=[uid for _,uid in sorted(leaves,key=lambda p:(-p[0],p[1]))]
    return dict(inspections=len(inspected),inspected_objects=inspected,scored_leaf_ids=ranked,
        returned_ids=ranked[:topk],ranked_ids=ranked,budget=budget,budget_remaining=budget-len(inspected))

def supporting_claims(query,method_ids,snap,X,budget=50,topk=5):
    """Article-linked evidence drill-down, without pretending it verifies GT paraphrases.

    Same procedure and extra scoring budget for either retrieval system. Candidate
    ordering is by id before scoring; no unpaid query ranking selects the budget.
    """
    query=normalize(query)
    nodes=snap['nodes']; lookup={n['id']:n for n in nodes}; idx={n['id']:i for i,n in enumerate(nodes)}
    method_articles={x:set(lookup[x].get('provenance',{}).get('articles',[])) for x in method_ids}
    for e in snap['hyperedges']:
        article=e.get('provenance',{}).get('article_id')
        if article is not None:
            for x in method_ids:
                if x in e['members']: method_articles[x].add(article)
    links={}
    for n in nodes:
        if n['type']!='claim': continue
        articles=set(n.get('provenance',{}).get('articles',[]))
        relevant={m:sorted(aa&articles) for m,aa in method_articles.items() if aa&articles}
        if relevant: links[n['id']]=relevant
    chosen=sorted(links)[:budget]
    ranked=sorted(chosen,key=lambda x:(-float(normalize(X[idx[x]])@query),x))
    return dict(inspections=len(chosen),budget=budget,candidate_count=len(links),
        claims=[dict(id=x,text=lookup[x]['surface_form'],method_article_links=links[x]) for x in ranked[:topk]],
        limitation='Shared source article is contextual evidence, not verified support for a particular method or benchmark claim')
