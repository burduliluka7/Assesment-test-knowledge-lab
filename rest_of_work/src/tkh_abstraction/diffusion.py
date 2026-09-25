"""Query-seeded Zhou propagation using the existing matrix-free spectral operator.

Retrieval inputs are graph, query vectors and configuration only. No evaluation data.
"""
from collections import defaultdict, deque
from copy import deepcopy
import re
import time
import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from .hypergraph import from_snapshot
from .spectral import laplacian
from .baselines import clique_projection
from .contextual import entity_groups
from .target_mapping import normalize, contains_name
from .retrieval import normalize as unit


def retrieval_graph(snap, relation_weights=None):
    """Local adapter: preserve export weights without changing construction code."""
    graph=from_snapshot(snap,np.zeros((len(snap['nodes']),1)))
    rw=relation_weights or {}
    if any(not np.isfinite(w) or w<0 for w in rw.values()): raise ValueError('Invalid relation multiplier')
    for edge,raw in zip(graph.edges,snap['hyperedges']):
        native=float(raw.get('weight',1.))
        if not np.isfinite(native) or native<0: raise ValueError('Invalid native edge weight')
        if len(raw['members'])!=len(set(raw['members'])): raise ValueError('Duplicate hyperedge membership')
        edge.weight=native*float(rw.get(edge.relation,1.))
        if not np.isfinite(edge.weight) or edge.weight<0: raise ValueError('Nonnegative finite retrieval weights required')
    return graph


def shuffled_graph(graph,seed=42):
    """Arity/weight/relation-preserving null; node degrees are NOT preserved."""
    result=deepcopy(graph); rng=np.random.default_rng(seed); ids=sorted(u.id for u in graph.units)
    for edge in sorted(result.edges,key=lambda e:e.id):
        members=tuple(sorted(rng.choice(ids,size=len(edge.multiplicities),replace=False).tolist()))
        edge.original_members=members; edge.multiplicities=dict.fromkeys(members,1)
    return result


class ThetaOperator:
    def __init__(self,graph):
        self.graph=graph
        # Reuse exactly the construction operator: L = I - F F.T, no n*n array.
        self.L,self.B,self.degree=laplacian(graph)
        self.edges=[e for e in graph.edges if not e.internal]
        self.ids=[u.id for u in graph.units]; self.idx={x:i for i,x in enumerate(self.ids)}
        self.arity=np.asarray(self.B.sum(axis=0)).ravel()
        self.weights=np.asarray([e.weight for e in self.edges])
        self.inv=np.zeros_like(self.degree); positive=self.degree>0
        self.inv[positive]=1/np.sqrt(self.degree[positive])

    def apply_theta(self,x):
        x=np.asarray(x,dtype=float)
        return x-self.L@x

    def edge_contributions(self,x):
        """Sum of Theta*x contribution over destination vertices per edge."""
        incoming=np.asarray(self.B.T@(self.inv*x)).ravel()
        outgoing=np.asarray(self.B.T@self.inv).ravel()
        return incoming*(self.weights/self.arity)*outgoing

    @property
    def metadata(self):
        return dict(nodes=len(self.ids),edges=len(self.edges),incidence_nnz=self.B.nnz,
            isolated_nodes=int(sum(self.degree==0)),theta_implementation='x - existing spectral.laplacian(graph) @ x',
            singleton_policy='Existing spectral incidence omits singleton/internal edges',
            pairwise_policy='Existing weight-conserving clique projection, then the same arity-2 Zhou normalization (lazy symmetric graph diffusion)')


def apply_theta(x,graph,relation_weights=None):
    """Convenience API; retain a ThetaOperator instance for repeated iterations."""
    if isinstance(graph,dict): graph=retrieval_graph(graph,relation_weights)
    return ThetaOperator(graph).apply_theta(x)


def diffuse(operator,y,alpha=.85,tolerance=1e-8,max_iterations=100,node_types=None):
    if not 0<alpha<1 or tolerance<=0 or max_iterations<1: raise ValueError('Invalid diffusion parameters')
    y=np.asarray(y,dtype=float)
    if y.shape!=(len(operator.ids),) or not np.isfinite(y).all() or np.any(y<0): raise ValueError('Invalid nonnegative seed vector')
    f=y.copy(); history=[]; started=time.perf_counter()
    types=node_types or ['node']*len(y); indices={t:np.where(np.asarray(types)==t)[0] for t in sorted(set(types))}
    for iteration in range(1,max_iterations+1):
        next_f=(1-alpha)*y+alpha*operator.apply_theta(f)
        # x-(I-Theta)x can incur roundoff at an exact zero; no score pruning.
        if np.min(next_f,initial=0)<-1e-12: raise ValueError('Negative propagation beyond floating-point roundoff')
        next_f=np.maximum(next_f,0.); delta=float(np.linalg.norm(next_f-f,ord=1)); f=next_f
        mass={t:float(f[ii].sum()) for t,ii in indices.items()}
        history.append(dict(iteration=iteration,delta_l1=delta,total_score_mass=float(f.sum()),score_mass_by_type=mass))
        if delta<tolerance: break
    return f,dict(iterations=iteration,theta_applications=iteration,converged=delta<tolerance,last_delta_l1=delta,
        error_l2_upper_bound=delta/(1-alpha),sparse_incidence_multiply_operations=2*iteration,
        incidence_entry_visits_upper=2*operator.B.nnz*iteration,runtime_seconds=time.perf_counter()-started,history=history,
        interpretation='Symmetric-normalized relevance scores are not probability mass; no per-iteration L1 renormalization')


def semantic_seeds(scores,nodes,top_m=100,seed_types=None):
    if top_m<1: raise ValueError('seed_top_m must be positive')
    allowed=set(seed_types) if seed_types is not None else {n['type'] for n in nodes}-{'author'}
    scores=np.asarray(scores,dtype=float)
    if scores.shape!=(len(nodes),) or not np.isfinite(scores).all(): raise ValueError('Invalid semantic scores')
    order=sorted((i for i,n in enumerate(nodes) if n['type'] in allowed),key=lambda i:(-scores[i],nodes[i]['id']))[:top_m]
    y=np.zeros(len(nodes)); y[order]=np.maximum(scores[order],0.)
    fallback=None
    if y.sum()>0: y/=y.sum()
    elif order: y[order]=1/len(order); fallback='uniform_over_deterministic_top_M_non_author_nodes; no_positive_semantic_signal'
    else: fallback='empty_eligible_seed_pool; zero_signal'
    return y,dict(fallback=fallback,selected_nodes=len(order),positive_seed_nodes=int(sum(y>0)),seed_l1=float(y.sum()),
        top_seeds=[dict(node_id=nodes[i]['id'],type=nodes[i]['type'],surface_form=nodes[i]['surface_form'],semantic_score=float(scores[i]),y=float(y[i])) for i in order])


def safe_mention_name(name):
    """Conservative identifiers or multiword names; never fuzzy/short generic links."""
    tokens=re.findall(r'\w+',name); key=normalize(name)
    if len(key)<3 or len(tokens)>12 or not any(c.isalpha() for c in name) or key in ENGLISH_STOP_WORDS: return False
    generic={'method','model','approach','framework','system','network','learning','prediction','data','graph'}
    useful=[t for t in tokens if t.casefold() not in ENGLISH_STOP_WORDS|generic]
    identifier=bool(re.search(r'[a-z][A-Z]|\d',name) or re.fullmatch(r'[A-Z][A-Z0-9()-]{2,}',name))
    return identifier or len(tokens)>=2 and len(useful)>=2 and len(key)>=10


class MentionIndex:
    def __init__(self,nodes,groups,source_types):
        self.mentions={}; self.names={}; self.lookup={n['id']:n for n in nodes}
        sources=[n for n in nodes if n['type'] in source_types]
        # Inverted token postings keep this graph-only index from a quadratic text scan.
        postings=defaultdict(set)
        for n in sources:
            for token in set(re.findall(r'\w+',normalize(n['surface_form']))): postings[token].add(n['id'])
        for group in groups:
            names=sorted({self.lookup[x]['surface_form'] for x in group['node_ids'] if safe_mention_name(self.lookup[x]['surface_form'])})
            evidence=set()
            for name in names:
                tokens=set(re.findall(r'\w+',normalize(name)))
                possible=set.intersection(*(postings[t] for t in tokens)) if tokens else set()
                evidence.update(x for x in possible-set(group['node_ids']) if contains_name(self.lookup[x]['surface_form'],name))
            self.mentions[group['id']]=sorted(evidence); self.names[group['id']]=names

    def assist(self,y,groups,node_ids):
        idx={x:i for i,x in enumerate(node_ids)}; result=y.copy(); transfers=[]
        # Read only ORIGINAL y, so mention chains cannot bootstrap each other.
        for group in groups:
            members=group['node_ids']; own=max((float(y[idx[x]]) for x in members),default=0.)
            evidence=self.mentions[group['id']]
            source=max(evidence,key=lambda x:(y[idx[x]],x),default=None)
            support=float(y[idx[source]]) if source is not None else 0.
            if support>own:
                destination=min(members,key=lambda x:(-y[idx[x]],x))
                result[idx[destination]]=support
                transfers.append(dict(entity_id=group['id'],destination=destination,evidence_node_id=source,own_seed=own,mention_seed=support,added_before_normalization=support-own))
        total=float(result.sum())
        if total: result/=total
        return result,dict(transfers=transfers,normalizer=total,policy='Max support raises one deterministic member per group; no per-duplicate replication; simultaneous update from original y; renormalize globally')


def rank_entities(scores,groups,node_ids):
    idx={x:i for i,x in enumerate(node_ids)}; ranked=[]
    for group in groups:
        score=max(float(scores[idx[x]]) for x in group['node_ids'])
        ranked.append(dict(entity_id=group['id'],node_ids=group['node_ids'],name=group['name'],types=group['types'],article_ids=group['article_ids'],score=score,score_components=dict(diffusion=score)))
    return sorted(ranked,key=lambda r:(-r['score'],r['entity_id']))


def rrf_fuse(a,b,k=60):
    if k<=0: raise ValueError('RRF k must be positive')
    def tied_ranks(ranking):
        result={}; start=0
        while start<len(ranking):
            end=start+1
            while end<len(ranking) and ranking[end]['score']==ranking[start]['score']: end+=1
            average=(start+1+end)/2
            for row in ranking[start:end]: result[row['entity_id']]=average
            start=end
        return result
    rows={r['entity_id']:r for r in a+b}; ra=tied_ranks(a); rb=tied_ranks(b)
    fused=[]
    for ident,row in rows.items():
        ca=1/(k+ra[ident]) if ident in ra else 0.; cb=1/(k+rb[ident]) if ident in rb else 0.
        fused.append(dict(row,score=ca+cb,score_components=dict(rrf_r3=ca,rrf_diffusion=cb),rank_r3=ra.get(ident),rank_diffusion=rb.get(ident)))
    return sorted(fused,key=lambda r:(-r['score'],r['entity_id']))


def propagation_diagnostics(operator,f,nodes,run,top_n=20):
    byid={n['id']:n for n in nodes}; order=sorted(range(len(f)),key=lambda i:(-f[i],operator.ids[i]))
    mass=operator.edge_contributions(f); edge_order=np.argsort(-mass,kind='stable')[:top_n]
    relation_mass=defaultdict(float)
    for e,m in zip(operator.edges,mass): relation_mass[e.relation]+=float(m)
    degree=operator.degree; threshold=float(np.quantile(degree,.9)) if len(degree) else 0.
    rho=float(spearmanr(degree,f).statistic) if np.std(degree)>0 and np.std(f)>0 else None
    total=float(f.sum()); type_mass=run['history'][-1]['score_mass_by_type']; shares={t:v/total if total else 0. for t,v in type_mass.items()}
    return dict(run,operator=operator.metadata,top_propagated_nodes=[dict(node_id=operator.ids[i],type=byid[operator.ids[i]]['type'],surface_form=byid[operator.ids[i]]['surface_form'],score=float(f[i]),degree=float(degree[i])) for i in order[:top_n]],
        degree_score_spearman=rho,high_degree_threshold=threshold,high_degree_score_fraction=float(f[degree>=threshold].sum())/total if total else 0.,
        final_type_fractions=shares,dominated_by_types=[t for t,v in shares.items() if v>.5],
        edge_mass_by_relation=dict(relation_mass),high_arity_threshold=16,high_arity_edge_mass_fraction=float(mass[operator.arity>16].sum()/mass.sum()) if mass.sum() else 0.,
        top_edges=[dict(edge_id=operator.edges[j].id,relation=operator.edges[j].relation,arity=int(operator.arity[j]),weight=float(operator.weights[j]),theta_output_score_mass=float(mass[j])) for j in edge_order],
        diagnostic_extra_incidence_multiplies=2,interpretation='Edge mass decomposes one final Theta application, not the entire resolvent; rank/degree correlation is descriptive, not causation')


def connecting_paths(operator,seeds,target_ids,max_hops=4,limit=3):
    """Approximate positive-weight incidence paths; not directed scientific proofs."""
    if not hasattr(operator,'path_incident'):
        operator.path_incident=defaultdict(list)
        for e in operator.edges:
            if e.weight>0:
                for x in e.multiplicities: operator.path_incident[x].append(e)
    incident=operator.path_incident
    targets=set(target_ids); paths=[]
    for source in seeds:
        if source['y']<=0: continue
        queue=deque([(source['node_id'],[])]); visited={source['node_id']}; seen_edges=set(); found=None
        while queue:
            node,path=queue.popleft()
            if node in targets: found=path; break
            if len(path)>=max_hops: continue
            for e in sorted(incident[node],key=lambda e:(-e.weight/e.arity,e.id)):
                if e.id in seen_edges: continue
                seen_edges.add(e.id)
                for dest in sorted(e.multiplicities):
                    if dest in visited: continue
                    visited.add(dest); queue.append((dest,path+[dict(from_node=node,edge_id=e.id,relation=e.relation,to_node=dest,arity=e.arity,weight=e.weight,
                        theta_transition=float(operator.inv[operator.idx[node]]*e.weight/len(e.multiplicities)*operator.inv[operator.idx[dest]]))]))
        if found is not None:
            paths.append(dict(seed=source,path=found,hops=len(found),interpretation='High-seed, short undirected path; approximate structural support, not unique flow attribution or entailment'))
            if len(paths)>=limit: break
    return paths
