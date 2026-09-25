from collections import Counter
from sklearn.metrics import adjusted_rand_score
from .statistics import interval
from ..temporal import assignments,normalized_vi

def compare(a,b):
    results={}
    for k,(la,lb) in enumerate(zip(a['levels'],b['levels'])):
        pa,pb=assignments(la),assignments(lb); ids=sorted(set(pa)&set(pb))
        results[str(k)]=dict(status='computed' if len(ids)>1 else 'not_applicable',persistent_nodes=len(ids),
            ari=float(adjusted_rand_score([pa[x] for x in ids],[pb[x] for x in ids])) if len(ids)>1 else None,
            normalized_vi=normalized_vi(pa,pb) if len(ids)>1 else None)
    return results

def summarize_perturbations(records):
    if not records: return {}
    return {k:dict(ari=interval([r['levels'][k]['ari'] for r in records]),
        normalized_vi=interval([r['levels'][k]['normalized_vi'] for r in records])) for k in records[0]['levels']}
