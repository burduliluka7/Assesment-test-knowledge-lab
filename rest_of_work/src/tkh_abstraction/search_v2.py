"""Answer-key-free typed retrieval. Costs count actual query-vector comparisons."""
from collections import defaultdict
import heapq
import numpy as np
from .retrieval import normalize

def farthest_representatives(X,indices,k):
    if not indices: return []
    mean=normalize(X[indices].mean(axis=0))
    first=max(indices,key=lambda i:(float(X[i]@mean),-i)); selected=[first]
    similarity=X[indices]@X[first]
    while len(selected)<min(k,len(indices)):
        choices=[j for j,i in enumerate(indices) if i not in selected]
        j=min(choices,key=lambda j:(float(similarity[j]),indices[j])); selected.append(indices[j])
        similarity=np.maximum(similarity,X[indices]@X[indices[j]])
    return selected

class SearchIndex:
    def __init__(self,h,nodes,X,allowed_types,prototypes=4):
        self.nodes=nodes; self.X=np.asarray([normalize(row) for row in X]); self.h=h
        self.idx={n['id']:i for i,n in enumerate(nodes)}
        self.eligible={n['id'] for n in nodes if n['type'] in allowed_types}
        self.units={}; self.children=defaultdict(list)
        for level in h['levels']:
            for c in level:
                members=sorted(set(c['member_ids'])&self.eligible)
                if not members: continue
                ii=[self.idx[x] for x in members]; leaf=len(c['member_ids'])==1
                reps=[] if leaf else farthest_representatives(self.X,ii,prototypes)
                self.units[c['id']]=dict(centroid=normalize(self.X[ii].mean(axis=0)),representatives=reps,
                    leaf=leaf,member_id=members[0] if leaf else None,eligible_count=len(ii))
                self.children[c['parent_id']].append(c['id'])
        for children in self.children.values(): children.sort()

    def export(self):
        return {uid:dict(prototype_node_ids=[self.nodes[i]['id'] for i in u['representatives']],
            eligible_count=u['eligible_count'],centroid=u['centroid'].tolist()) for uid,u in self.units.items() if not u['leaf']}

    def search(self,query,budget,method='prototype_best_first',alpha=.5,top_r=2,k=10,beam=3):
        q=normalize(query); trace=[]; seen=set(); scores={}; leaf_scores={}; cost=0; skipped=[]
        prototype=method=='prototype_best_first'
        def score(uid):
            nonlocal cost
            if uid in seen: return scores[uid]
            u=self.units[uid]; charge=1+(len(u['representatives']) if prototype else 0)
            if cost+charge>budget:
                skipped.append(uid); return None
            value=float(u['centroid']@q)
            if prototype and u['representatives']:
                pp=sorted((float(self.X[i]@q) for i in u['representatives']),reverse=True)
                value=alpha*value+(1-alpha)*float(np.mean(pp[:top_r]))
            cost+=charge; seen.add(uid); scores[uid]=value
            trace.append(dict(object_id=uid,comparisons=charge,cumulative_comparisons=cost,score=value))
            if u['leaf']: leaf_scores[u['member_id']]=value
            return value
        roots=self.children[None]
        if method=='legacy_typed_fixed_beam':
            frontier=roots
            while frontier:
                if cost+len(frontier)>budget: break
                ranked=[]
                for uid in sorted(frontier):
                    value=score(uid)
                    if value is not None: ranked.append((value,uid))
                frontier=[c for _,uid in sorted(ranked,key=lambda p:(-p[0],p[1]))[:beam] for c in self.children[uid]]
        else:
            heap=[]
            for uid in roots:
                value=score(uid)
                if value is not None and not self.units[uid]['leaf']: heapq.heappush(heap,(-value,uid))
            while heap and cost<budget:
                _,uid=heapq.heappop(heap)
                for child in self.children[uid]:
                    value=score(child)
                    if value is not None and not self.units[child]['leaf']: heapq.heappush(heap,(-value,child))
        ranked=sorted(leaf_scores,key=lambda x:(-leaf_scores[x],x))
        return dict(returned_ids=ranked[:k],ranked_ids=ranked,leaf_scores=leaf_scores,score_comparisons=cost,
            object_inspections=len(trace),budget=budget,trace=trace,skipped_for_budget=skipped,
            root_layer_complete=all(uid in seen for uid in roots),root_layer_cost=sum(1+(len(self.units[uid]['representatives']) if prototype else 0) for uid in roots),
            output_k=k,method=method)

def flat_search(query,nodes,X,allowed_types,k=10,budget=None,seed=42):
    q=normalize(query); ii=[i for i,n in enumerate(nodes) if n['type'] in allowed_types]
    if budget is not None:
        # Query-independent randomized scan, a secondary equal-work reference only.
        ii=np.random.default_rng(seed).permutation(ii).tolist()[:budget]
    scores={nodes[i]['id']:float(normalize(X[i])@q) for i in ii}
    ranked=sorted(scores,key=lambda x:(-scores[x],x))
    return dict(returned_ids=ranked[:k],ranked_ids=ranked,leaf_scores=scores,score_comparisons=len(ii),
        object_inspections=len(ii),budget=budget,trace=[],output_k=k,
        method='flat_exhaustive' if budget is None else 'flat_query_independent_scan')

def claim_candidates(entity_ids,snap):
    nodes={n['id']:n for n in snap['nodes']}; articles=set(); direct=set(); paths=defaultdict(list)
    for x in entity_ids:
        n=nodes[x]; articles.update(n.get('provenance',{}).get('articles',[]))
        if n['type']=='claim': direct.add(x); paths[x].append(dict(path='retrieved_claim',entity=x))
    for e in snap['hyperedges']:
        shared=set(entity_ids)&set(e['members'])
        if shared:
            aid=e.get('provenance',{}).get('article_id')
            if aid is not None: articles.add(aid)
            for x in e['members']:
                if nodes[x]['type']=='claim':
                    direct.add(x); paths[x].append(dict(path='direct_hyperedge',edge=e['id'],entities=sorted(shared)))
    for x,n in nodes.items():
        if n['type']!='claim': continue
        common=articles&set(n.get('provenance',{}).get('articles',[]))
        if common: paths[x].append(dict(path='entity_article_claim',article_ids=sorted(common)))
    return sorted(paths,key=lambda x:(x not in direct,x)),dict(paths)

def retrieve_claims(query,entities,snap,X,budget=100,k=10,known_scores=None,exhaustive=False):
    idx={n['id']:i for i,n in enumerate(snap['nodes'])}; q=normalize(query)
    if exhaustive:
        ids=sorted(n['id'] for n in snap['nodes'] if n['type']=='claim'); paths={x:[dict(path='flat_claim_corpus')] for x in ids}
    else: ids,paths=claim_candidates(entities,snap)
    known_scores=known_scores or {}; values={}; charges=0; reused=[]; trace=[]
    for x in ids:
        if x in known_scores: values[x]=known_scores[x]; reused.append(x)
        elif exhaustive or charges<budget:
            values[x]=float(normalize(X[idx[x]])@q); charges+=1
            trace.append(dict(node_id=x,cumulative_comparisons=charges))
    ranked=sorted(values,key=lambda x:(-values[x],x)); returned=ranked[:k]
    return dict(returned_ids=returned,ranked_ids=ranked,score_comparisons=charges,trace=trace,
        reused_entity_scores=reused,candidate_count=len(ids),budget=None if exhaustive else budget,
        evidence=[dict(node_id=x,paths=paths[x],surface_form=snap['nodes'][idx[x]]['surface_form'],provenance=snap['nodes'][idx[x]].get('provenance',{})) for x in returned])
