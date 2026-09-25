from dataclasses import dataclass, field
from collections import Counter
import numpy as np

@dataclass
class Unit:
    id: str
    members: tuple
    centroid: np.ndarray
    types: dict = field(default_factory=dict)
    @property
    def size(self): return len(self.members)

@dataclass
class Edge:
    id: str
    relation: str
    multiplicities: dict
    original_members: tuple
    provenance: dict
    weight: float = 1.
    @property
    def arity(self): return len(self.original_members)
    @property
    def internal(self): return len(self.multiplicities)==1

@dataclass
class Hypergraph:
    units: list
    edges: list

def from_snapshot(snap, embeddings, relation_weights=None):
    rw=relation_weights or {}
    if any(w<=0 for w in rw.values()): raise ValueError('relation weights must be positive')
    units=[Unit(n['id'],(n['id'],),embeddings[i],{n['type']:1}) for i,n in enumerate(snap['nodes'])]
    edges=[Edge(e['id'],e['relation_type'],dict.fromkeys(e['members'],1),tuple(e['members']),e.get('provenance',{}),rw.get(e['relation_type'],1.)) for e in snap['hyperedges']]
    return Hypergraph(units,edges)

def fragmentation(edge, mapping=None):
    c=Counter()
    for u,m in edge.multiplicities.items(): c[mapping[u] if mapping else u]+=m
    n=edge.arity
    return (1-sum((m/n)**2 for m in c.values()))/(1-1/n) if n>1 else 0.

def objective_hyper(graph,mapping=None):
    den=sum(e.weight for e in graph.edges)
    return sum(e.weight*fragmentation(e,mapping) for e in graph.edges)/den if den else 0.
