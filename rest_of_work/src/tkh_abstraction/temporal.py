from collections import Counter
import math
import numpy as np
from scipy.optimize import linear_sum_assignment

def f(x): return x*math.log(x) if x else 0.

def vi_delta(a,b,total):
    """Exact normalized VI change from merging contingency rows; new nodes absent."""
    if total<=1: return 0.
    na=sum(a.values()); nb=sum(b.values())
    if len(a)>len(b): a,b=b,a
    cross=sum(f(x+b.get(k,0))-f(x)-f(b.get(k,0)) for k,x in a.items() if k in b)
    return (f(na+nb)-f(na)-f(nb)-2*cross)/(total*math.log(total))

def normalized_vi(p,q):
    common=sorted(set(p)&set(q)); n=len(common)
    if n<=1: return 0.
    a=Counter(p[x] for x in common); b=Counter(q[x] for x in common)
    joint=Counter((p[x],q[x]) for x in common)
    return max(0.,(sum(map(f,a.values()))+sum(map(f,b.values()))-2*sum(map(f,joint.values())))/(n*math.log(n)))

def assignments(level):
    return {x:c['id'] for c in level for x in c['member_ids']}

def track(previous,current,source,target,level,match_threshold=.2,event_threshold=.1):
    events=[]
    def emit(kind,src,dst,scores,**extra):
        events.append(dict(source_snapshot=source,target_snapshot=target,level=level,event_type=kind,
            source_ids=src,target_ids=dst,overlap_scores=scores,**extra))
    if not previous:
        for c in current:
            c['persistent_id']=c['id']; emit('birth',[],[c['id']],[])
        return events
    if all(len(c['member_ids'])==1 for c in previous+current):
        oldmap={c['member_ids'][0]:c for c in previous}
        newmap={c['member_ids'][0]:c for c in current}
        for x,c in newmap.items():
            if x in oldmap:
                c['persistent_id']=oldmap[x]['persistent_id']
                emit('continuation',[oldmap[x]['id']],[c['id']],[dict(jaccard=1.,intersection=1,source_fraction=1.,target_fraction=1.)])
            else:
                c['persistent_id']=c['id']; emit('birth',[],[c['id']],[])
        for x,c in oldmap.items():
            if x not in newmap: emit('death',[c['id']],[],[])
        return events
    old=[set(c['member_ids']) for c in previous]; new=[set(c['member_ids']) for c in current]
    common=set().union(*old)&set().union(*new) if new else set()
    op=[s&common for s in old]; np_=[s&common for s in new]
    jac=np.zeros((len(old),len(new))); links=[]
    for i,a in enumerate(op):
        for j,b in enumerate(np_):
            inter=len(a&b); union=len(a|b)
            jac[i,j]=inter/union if union else 0
            if inter:
                links.append(dict(source_id=previous[i]['id'],target_id=current[j]['id'],intersection=inter,
                    jaccard=float(jac[i,j]),source_fraction=inter/len(a),target_fraction=inter/len(b)))
    matched_new=set(); matched_old=set()
    if jac.size:
        rows,cols=linear_sum_assignment(-jac)
        for i,j in zip(rows,cols):
            if jac[i,j]<match_threshold or jac[i,j]<=0: continue
            matched_new.add(j); matched_old.add(i)
            current[j]['persistent_id']=previous[i]['persistent_id']
            ls=[x for x in links if x['source_id']==previous[i]['id'] and x['target_id']==current[j]['id']]
            emit('growth' if len(new[j])>len(old[i]) else 'continuation',[previous[i]['id']],[current[j]['id']],ls)
    for j,c in enumerate(current):
        if j not in matched_new:
            c['persistent_id']=c['id']
            ls=[x for x in links if x['target_id']==c['id']]
            emit('birth',[],[c['id']],ls,ambiguous=bool(ls),meaning='new identity; overlaps may still indicate split/merge')
    for i,c in enumerate(previous):
        ls=[x for x in links if x['source_id']==c['id']]
        qualified=[x for x in ls if x['source_fraction']>=event_threshold]
        if len(qualified)>1: emit('split',[c['id']],[x['target_id'] for x in qualified],qualified,ambiguous=True)
        if i not in matched_old: emit('death',[c['id']],[],ls,ambiguous=bool(ls),meaning='identity retired; members may continue elsewhere')
    for c in current:
        ls=[x for x in links if x['target_id']==c['id'] and x['target_fraction']>=event_threshold]
        if len(ls)>1: emit('merge',[x['source_id'] for x in ls],[c['id']],ls,ambiguous=True)
    return events
