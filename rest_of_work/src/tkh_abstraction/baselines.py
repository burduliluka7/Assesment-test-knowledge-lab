from collections import Counter,defaultdict
from itertools import combinations
from .hypergraph import Edge,Hypergraph

def clique_projection(graph):
    """Total edge weight conserved: each pair receives w / choose(n,2)."""
    weights=Counter(); contributors=defaultdict(list)
    for e in graph.edges:
        support=sorted(e.multiplicities)
        if len(support)<2: continue
        for pair in combinations(support,2):
            weights[pair]+=e.weight/(len(support)*(len(support)-1)/2)
            contributors[pair].append(e.id)
    edges=[Edge(f'pair_{i}','projected',dict.fromkeys(pair,1),pair,{},w) for i,(pair,w) in enumerate(sorted(weights.items()))]
    return Hypergraph(graph.units,edges)

def projection_loss(graph):
    c=defaultdict(list)
    for e in graph.edges:
        for p in combinations(sorted(e.original_members),2): c[p].append(e.id)
    instances=sum(len(x) for x in c.values()); m=len(graph.edges)
    return dict(fraction_arity_gt2=sum(e.arity>2 for e in graph.edges)/m if m else None,
        pair_instances=instances,unique_pairs=len(c),expansion_factor=instances/m if m else None,
        pairs_with_multiple_sources=sum(len(x)>1 for x in c.values()),max_sources_per_pair=max(map(len,c.values()),default=0),
        examples=[dict(pair=list(p),original_edges=ids) for p,ids in c.items() if len(ids)>1][:5],
        nonidentifiability_example='One weighted triple {a,b,c} and three pair edges of weight 1/3 produce the same projected adjacency; relation identity and arity cannot be recovered.')
