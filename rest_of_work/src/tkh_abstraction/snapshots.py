from copy import deepcopy
from collections import Counter
import numpy as np

def snapshot(data, cutoff):
    """Corpus-publication visibility, never historical origin-year visibility."""
    nodes=[deepcopy(n) for n in data['nodes'] if n['first_seen_year']<=cutoff]
    visible={n['id'] for n in nodes}
    article_years={a: n['first_seen_year'] for n in data['nodes'] if n['type']=='article'
                   for a in n.get('provenance',{}).get('articles',[])}
    for n in nodes:
        n.pop('last_seen_year',None)
        for field in ['year','origin_year']:
            if n.get(field) is not None and n[field]>cutoff: n.pop(field)
        p=n.get('provenance',{})
        p['articles']=[a for a in p.get('articles',[]) if article_years.get(a,9999)<=cutoff]
        for k in ['ingested_at','extracted_at']: p.pop(k,None)
    edges=[]; deferred=[]
    for e in data['hyperedges']:
        if e['year']>cutoff: continue
        py=e.get('provenance',{}).get('article_year',e['year'])
        if py>cutoff or not set(e['members'])<=visible:
            deferred.append(e['id']); continue
        edges.append(deepcopy(e))
    return dict(snapshot=cutoff,nodes=nodes,hyperedges=edges,deferred_edge_ids=deferred)

def describe(snap, previous=None):
    ns=snap['nodes']; es=snap['hyperedges']; ar=np.array([len(e['members']) for e in es])
    prevn={n['id'] for n in previous['nodes']} if previous else set()
    preve={e['id'] for e in previous['hyperedges']} if previous else set()
    incident={x for e in es for x in e['members']}
    years=[n['first_seen_year'] for n in ns]+[e['year'] for e in es]
    return dict(snapshot=snap['snapshot'],nodes=len(ns),hyperedges=len(es),
        node_types=dict(Counter(n['type'] for n in ns)),relation_types=dict(Counter(e['relation_type'] for e in es)),
        arity_distribution=dict(Counter(map(int,ar))),arity_quantiles=dict(zip(['min','q25','median','q75','max'],np.quantile(ar,[0,.25,.5,.75,1]).tolist())) if len(ar) else None,
        fraction_arity_gt2=float(np.mean(ar>2)) if len(ar) else None,
        time_span=[min(years),max(years)] if years else None,
        new_nodes=len({n['id'] for n in ns}-prevn),new_edges=len({e['id'] for e in es}-preve),
        isolated_nodes=len({n['id'] for n in ns}-incident),deferred_edges=len(snap['deferred_edge_ids']))
