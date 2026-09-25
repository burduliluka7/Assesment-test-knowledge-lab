import numpy as np
from scipy.stats import t

def interval(values):
    x=np.asarray(values,dtype=float); n=len(x)
    if not n: return dict(status='not_applicable',reason='no observations')
    mean=float(x.mean()); sd=float(x.std(ddof=1)) if n>1 else None
    width=float(t.ppf(.975,n-1)*sd/np.sqrt(n)) if n>1 else None
    return dict(status='computed',values=x.tolist(),n=n,mean=mean,std=sd,
        ci95=[mean-width,mean+width] if width is not None else None,
        ci_method='approximate Student t interval over replicate values; unbounded')

def paired_bootstrap(a,b,seed=42,repetitions=2000):
    d=np.asarray(a)-np.asarray(b)
    if not len(d): return dict(status='not_applicable')
    rng=np.random.default_rng(seed)
    means=np.mean(rng.choice(d,size=(repetitions,len(d)),replace=True),axis=1)
    return dict(status='computed',difference=float(d.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),
        method='paired question bootstrap; descriptive, questions are not a random domain sample',n=len(d))
