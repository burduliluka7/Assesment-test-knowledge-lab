import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, LinearOperator

def incidence(graph):
    idx={u.id:i for i,u in enumerate(graph.units)}
    edges=[e for e in graph.edges if not e.internal]
    rows=[]; cols=[]
    for j,e in enumerate(edges):
        for u in e.multiplicities: rows.append(idx[u]); cols.append(j)
    return sparse.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(idx),len(edges))),np.array([e.weight for e in edges])

def laplacian(graph):
    B,w=incidence(graph); de=np.asarray(B.sum(axis=0)).ravel(); dv=np.asarray(B@w).ravel()
    inv=np.zeros_like(dv); inv[dv>0]=1/np.sqrt(dv[dv>0])
    F=sparse.diags(inv)@B@sparse.diags(np.sqrt(w/de) if len(w) else w)
    # Matrix-free sparse incidence products avoid materializing a projected clique.
    n=B.shape[0]
    def mv(x): return x-F@(F.T@x)
    L=LinearOperator((n,n),matvec=mv,matmat=mv,rmatvec=mv,dtype=float)
    rng=np.random.default_rng(0); x=rng.normal(size=n); y=rng.normal(size=n)
    if not np.isclose(x@(L@y),y@(L@x),atol=1e-9): raise ValueError('asymmetric Laplacian')
    return L,B,dv

def coordinates(graph,dimension=8,seed=42):
    L,B,dv=laplacian(graph); n=len(graph.units)
    if n<=2 or B.shape[1]==0: return np.zeros((n,1)),dict(eigenvalues=[],incidence_shape=list(B.shape),isolated=int(sum(dv==0)))
    k=min(dimension,n-1)
    # Largest Theta eigenvalues are smallest L eigenvalues; sparse Lanczos.
    T=LinearOperator((n,n),matvec=lambda x:x-L@x,matmat=lambda x:x-L@x,dtype=float)
    vals,vec=eigsh(T,k=k,which='LA',v0=np.random.default_rng(seed).normal(size=n),tol=1e-7,maxiter=10000)
    order=np.argsort(1-vals); vals=(1-vals)[order]; vec=vec[:,order]
    if min(vals)<-1e-6 or not np.isfinite(vec).all(): raise ValueError('invalid eigensystem')
    vec[dv==0]=0
    return vec,dict(eigenvalues=vals.tolist(),incidence_shape=list(B.shape),isolated=int(sum(dv==0)),residual_max=float(max(np.linalg.norm(L@vec[:,i]-vals[i]*vec[:,i]) for i in range(k))))
