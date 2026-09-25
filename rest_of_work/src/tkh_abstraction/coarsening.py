"""Candidate-restricted greedy minimization with exact local objective deltas."""
from collections import Counter,defaultdict
import heapq
import numpy as np
from sklearn.neighbors import NearestNeighbors
from .temporal import vi_delta

def variant_weights(config,variant,has_previous):
    w=dict(config['weights'])
    if variant=='semantic': w=dict(hyper=0,spectral=0,semantic=1,temporal=0)
    elif variant=='structural': w=dict(hyper=.6,spectral=.4,semantic=0,temporal=0)
    elif variant=='static': w['temporal']=0
    if not has_previous: w['temporal']=0
    total=sum(w.values())
    return {k:v/total for k,v in w.items()}

def agglomerate(graph,spectral,budget,config,variant,previous=None):
    n=len(graph.units)
    if n<=budget: return [[u.id] for u in graph.units],dict(merges=0)
    previous=previous or {}; weights=variant_weights(config,variant,bool(previous))
    units=graph.units; idx={u.id:i for i,u in enumerate(units)}
    sizes={i:u.size for i,u in enumerate(units)}
    sem={i:u.centroid for i,u in enumerate(units)}; spec={i:spectral[i] for i in range(n)}
    groups={i:[u.id] for i,u in enumerate(units)}
    rows={i:Counter(previous[x] for x in u.members if x in previous) for i,u in enumerate(units)}
    total=sum(sum(r.values()) for r in rows.values())
    incidence={i:{} for i in range(n)}; edge_den=sum(e.weight for e in graph.edges) or 1
    factors=[]; support_counts={}
    pairs=set()
    for eidx,e in enumerate(graph.edges):
        factors.append(-2*e.weight/(edge_den*e.arity*(e.arity-1)) if e.arity>1 else 0)
        support_counts[eidx]=len(e.multiplicities)
        if e.internal: continue  # Internal evidence remains in the fixed denominator, with zero merge delta.
        support=sorted(idx[u] for u in e.multiplicities)
        for u,m in e.multiplicities.items(): incidence[idx[u]][eidx]=m
        if weights['hyper'] or weights['spectral']:
            # A support cycle supplies O(I) native candidates, not a clique.
            for a,b in zip(support,support[1:]+support[:1]):
                if a!=b: pairs.add(tuple(sorted((a,b))))
    k=min(config['neighbors']+1,n)
    def add_knn(X):
        near=NearestNeighbors(n_neighbors=k,algorithm='brute',n_jobs=1).fit(X).kneighbors(X,return_distance=False)
        for a,neighbors in enumerate(near):
            for b in neighbors:
                if a!=b: pairs.add(tuple(sorted((a,int(b)))))
    if weights['semantic']: add_knn(np.array(list(sem.values())))
    if weights['spectral']: add_knn(spectral)
    if weights['temporal']:
        old=defaultdict(list)
        for i,r in rows.items():
            for label in r: old[label].append(i)
        for members in old.values():
            for a,b in zip(members,members[1:]): pairs.add((a,b))
    # A deterministic chain guarantees a connected search graph and hard-budget progress.
    pairs.update((a,a+1) for a in range(n-1))
    neigh={i:set() for i in range(n)}
    for a,b in pairs: neigh[a].add(b); neigh[b].add(a)
    base_mass=sum(sizes.values())
    # Normalize SSE by total initial dispersion (a fixed denominator within a level).
    def variance(vectors):
        mean=sum(sizes[i]*vectors[i] for i in range(n))/base_mass
        return max(sum(sizes[i]*float(np.sum((vectors[i]-mean)**2)) for i in range(n)),1e-12)
    sem_scale=variance(sem); spec_scale=variance(spec)
    def terms(a,b):
        factor=sizes[a]*sizes[b]/(sizes[a]+sizes[b])
        ia,ib=incidence[a],incidence[b]
        if len(ia)>len(ib): ia,ib=ib,ia
        h=sum(factors[e]*m*ib.get(e,0) for e,m in ia.items()) if weights['hyper'] else 0.
        s=factor*float(np.sum((spec[a]-spec[b])**2))/spec_scale if weights['spectral'] else 0.
        m=factor*float(np.sum((sem[a]-sem[b])**2))/sem_scale if weights['semantic'] else 0.
        t=vi_delta(rows[a],rows[b],total) if weights['temporal'] else 0.
        return h,s,m,t
    def score(a,b): return sum(w*d for w,d in zip([weights[x] for x in ['hyper','spectral','semantic','temporal']],terms(a,b)))
    heap=[(score(a,b),a,b) for a,b in sorted(pairs)]; heapq.heapify(heap)
    active=set(range(n)); nextid=n; deltas=np.zeros(4); changed=0; pushes=len(heap)
    while len(active)>budget:
        if not heap: raise RuntimeError('candidate graph disconnected unexpectedly')
        value,a,b=heapq.heappop(heap)
        if a not in active or b not in active: continue
        ds=terms(a,b); deltas+=ds; changed+=abs(ds[3])>1e-14
        c=nextid; nextid+=1; mass=sizes[a]+sizes[b]; sizes[c]=mass
        sem[c]=(sizes[a]*sem[a]+sizes[b]*sem[b])/mass
        spec[c]=(sizes[a]*spec[a]+sizes[b]*spec[b])/mass
        groups[c]=groups.pop(a)+groups.pop(b); rows[c]=rows.pop(a)+rows.pop(b)
        ia=incidence.pop(a); ib=incidence.pop(b)
        for e in ia.keys() & ib.keys(): support_counts[e]-=1
        incidence[c]={e:m for e,m in (Counter(ia)+Counter(ib)).items() if support_counts[e]>1}
        nc=(neigh.pop(a)|neigh.pop(b))-{a,b}; active.remove(a); active.remove(b); active.add(c)
        neigh[c]=nc&active
        for x in sorted(neigh[c]):
            neigh[x].discard(a); neigh[x].discard(b); neigh[x].add(c)
            heapq.heappush(heap,(score(x,c),x,c)); pushes+=1
    return [groups[i] for i in sorted(active)],dict(merges=n-len(active),candidate_pairs=len(pairs),heap_pushes=pushes,
        weights=weights,normalization=dict(semantic_total_variance=sem_scale,spectral_total_variance=spec_scale),
        sum_merge_deltas=dict(zip(['hyper','spectral','semantic','temporal'],deltas.tolist())),nonzero_temporal_merges=int(changed))
