import numpy as np

def score(labels,X):
    """Node-weighted mean cosine to centroid, excluding singleton clusters."""
    total=0.; count=0; macro=[]
    for c in np.unique(labels):
        Y=X[labels==c]
        if len(Y)<2: continue
        mu=Y.mean(axis=0); norm=np.linalg.norm(mu)
        v=float(np.mean(Y@mu/norm)) if norm>0 else 0.
        total+=len(Y)*v; count+=len(Y); macro.append(v)
    return (total/count if count else None),float(np.mean(macro)) if macro else None,count

def evaluate(h,nodes,X,seeds=50,seed=42):
    results={}; ids=[n['id'] for n in nodes]; types=np.array([n['type'] for n in nodes])
    for k,level in enumerate(h['levels']):
        lookup={x:i for i,c in enumerate(level) for x in c['member_ids']}; labels=np.array([lookup[x] for x in ids])
        obs,macro,count=score(labels,X)
        if obs is None:
            results[str(k)]=dict(status='not_applicable',reason='all clusters singleton'); continue
        null=[]; typed=[]; rng=np.random.default_rng(seed)
        for _ in range(seeds):
            null.append(score(rng.permutation(labels),X)[0])
            perm=labels.copy()
            for typ in np.unique(types):
                idx=np.flatnonzero(types==typ); perm[idx]=rng.permutation(labels[idx])
            typed.append(score(perm,X)[0])
        def stat(v):
            return dict(mean=float(np.mean(v)),std=float(np.std(v,ddof=1)) if len(v)>1 else None,
                difference=float(obs-np.mean(v)),p_upper=(1+sum(x>=obs for x in v))/(len(v)+1),permutations=len(v))
        sizes=[len(c['member_ids']) for c in level]
        results[str(k)]=dict(status='computed',node_weighted=obs,cluster_weighted=macro,evaluated_nodes=count,
            singleton_nodes=len(nodes)-count,null=stat(null),type_stratified_null=stat(typed),
            cluster_sizes=sizes,largest_fraction=max(sizes)/len(nodes))
    return results
