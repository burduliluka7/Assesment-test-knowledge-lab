from collections import Counter
from .hypergraph import Edge, Hypergraph, Unit

def collapse(graph, groups, ids=None):
    """Keep one evidence record per original edge, including internal records."""
    lookup={u.id:u for u in graph.units}; mapping={}; units=[]
    for j,group in enumerate(groups):
        uid=ids[j] if ids else f'c{j}'
        children=[lookup[x] for x in group]
        members=tuple(sorted(x for u in children for x in u.members))
        types=Counter()
        for u in children:
            if u.id in mapping: raise ValueError('overlapping collapse groups')
            mapping[u.id]=uid; types.update(u.types)
        units.append(Unit(uid,members,sum(u.size*u.centroid for u in children)/len(members),dict(types)))
    if set(mapping)!=set(lookup): raise ValueError('collapse must cover current units')
    edges=[]
    for e in graph.edges:
        mult=Counter()
        for u,m in e.multiplicities.items(): mult[mapping[u]]+=m
        if sum(mult.values())!=e.arity: raise ValueError('multiplicity conservation violated')
        edges.append(Edge(e.id,e.relation,dict(mult),e.original_members,e.provenance,e.weight))
    return Hypergraph(units,edges),mapping
