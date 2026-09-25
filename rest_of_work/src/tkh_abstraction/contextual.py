"""Graph-only, relation-aware fields and query-only intent. No evaluation imports."""
from collections import defaultdict
import re,time
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from .target_mapping import normalize,contains_name
from .retrieval import normalize as unit
from .io import digest

FIELDS=('name','claim','claim_article','task','problem','technical','dataset','article')
TYPE_FIELD={'claim':'claim','task':'task','problem':'problem','future_topic':'problem','technique':'technical','component':'technical','dataset':'dataset','metric':'dataset','article':'article'}
STRONG={'presents','claims','addresses','solves','uses_technique','uses_component','evaluated_on'}

def entity_groups(nodes,enabled=True):
    groups={}
    for n in nodes:
        # Claims and people are not scientific named-entity aliases.
        key=normalize(n['surface_form']) if enabled and n['type'] not in {'claim','author'} else n['id']
        groups.setdefault(key,[]).append(n)
    result=[]
    for key,ns in sorted(groups.items()):
        result.append(dict(id='entity:'+digest(sorted(n['id'] for n in ns))[:16],name=min((n['surface_form'] for n in ns),key=lambda s:(len(s),s)),
            node_ids=sorted(n['id'] for n in ns),types=sorted({n['type'] for n in ns}),article_ids=sorted({a for n in ns for a in n.get('provenance',{}).get('articles',[])})))
    return result

def build_representations(snap,cfg):
    nodes={n['id']:n for n in snap['nodes']}; incident=defaultdict(list); articles=defaultdict(list)
    for e in snap['hyperedges']:
        for x in e['members']: incident[x].append(e)
    for n in nodes.values():
        for a in n.get('provenance',{}).get('articles',[]): articles[a].append(n)
    reps={}
    for ident,n in nodes.items():
        fields={f:[] for f in FIELDS}; fields['name']=[dict(node_id=ident,text=n['surface_form'],path='self',priority=0)]
        def add(other,path,priority,field_override=None):
            if other['id']==ident: return
            field=field_override or TYPE_FIELD.get(other['type'])
            if field: fields[field].append(dict(node_id=other['id'],text=other['surface_form'],path=path,priority=priority))
        presented=set()
        for e in sorted(incident[ident],key=lambda e:e['id']):
            if e['relation_type']=='presents': presented.add(e.get('provenance',{}).get('article_id'))
            if e['relation_type'] not in STRONG or len(e['members'])>cfg['context_max_direct_arity']: continue
            # Actual shared strong relation only; no broad cites/extends clique expansion.
            for x in e['members']: add(nodes[x],'direct:'+e['relation_type']+':'+e['id'],0)
        for aid in n.get('provenance',{}).get('articles',[]):
            for other in articles[aid]:
                if other['type']=='article': add(other,'provenance_article:'+str(aid),1)
                elif other['type']=='claim' and contains_name(other['surface_form'],n['surface_form']): add(other,'explicit_name_in_claim:'+str(aid),0)
                elif cfg.get('context_article_expansion',True) and aid in presented and other['type'] in TYPE_FIELD:
                    add(other,'presenting_article_context:'+str(aid),2,'claim_article' if other['type']=='claim' else None)
        used_text=set()
        for f in FIELDS:
            unique={}
            for item in sorted(fields[f],key=lambda x:(x['priority'],x['node_id'])): unique.setdefault(normalize(item['text']),item)
            cap=1 if f=='name' else cfg['context_field_caps'][f]
            fields[f]=[item for key,item in unique.items() if key not in used_text][:cap]
            used_text.update(normalize(item['text']) for item in fields[f])
        reps[ident]=dict(node_id=ident,type=n['type'],cutoff=snap['snapshot'],fields=fields)
    return reps

def analyze_question(question,declared_type=None):
    lower=question.casefold(); data=bool(re.search(r'\b(?:what|which)\s+(?:\w+\s+){0,3}(?:datasets?|benchmarks?)\b',lower[:110]))
    evidence=declared_type=='B' or bool(re.search(r'^how\b|^why\b|\bevidence\b',lower))
    kind='data' if data else 'evidence' if evidence else 'method' if re.search(r'\bmethods?|architectures?|models?\b',lower[:110]) else 'unknown'
    task=re.split(r'\b(?:for|to)\b',question,maxsplit=1,flags=re.I)
    body=task[-1].strip(' ?.') if len(task)>1 else question.strip(' ?.')
    clauses=[p.strip(' ?.,') for p in re.split(r'\b(?:when|where|while|with|without|under|and)\b',body,flags=re.I) if len(p.split())>=3]
    constraints=[m.group(0).strip(' ?.') for m in re.finditer(r'\b(?:when|while|with|without|under)\b[^?;]+',question,re.I)]
    years=re.findall(r'\b(?:19|20)\d{2}\b',question)
    # Views consist exclusively of exact substrings, never answer-conditioned paraphrases.
    views=list(dict.fromkeys([question]+([body] if body!=question else [])+clauses+constraints))[:4]
    return dict(original_question=question,declared_type=declared_type,kind=kind,task=body,constraints=constraints,
        domain_terms=re.findall(r'\b[\w()\-]{4,}\b',body),performance_requirements=[p for p in clauses if re.search(r'\baccur|scal|efficien|preserv|generaliz',p,re.I)],
        temporal_cutoff_text=years[-1] if years else None,plural=bool(re.search(r'\bmethods|datasets|models|architectures\b',lower)),extractive_views=views)

def type_prior(types,intent):
    table={'method':{'method':1.,'technique':.7,'cited_work':.7,'component':.4,'article':.2},
        'data':{'dataset':1.,'method':.6,'cited_work':.6,'task':.4,'metric':.4,'article':.4},
        'evidence':{'claim':1.,'article':.8,'problem':.6,'future_topic':.6,'method':.3}}
    return max((table.get(intent['kind'],{}).get(t,0.) for t in types),default=0.)

class ContextIndex:
    def __init__(self,snap,representations,encoder,cfg,grouped=False):
        self.cfg=cfg; self.groups=entity_groups(snap['nodes'],grouped); self.lookup={g['id']:g for g in self.groups}
        self.representations=representations; texts=sorted({item['text'] for r in representations.values() for field in r['fields'].values() for item in field})
        self.texts=texts; self.text_idx={t:i for i,t in enumerate(texts)}
        self.X=np.asarray([unit(x) for x in encoder.encode(texts,f'context-fields:{snap["snapshot"]}')]); self.field_ids={}; self.rerank_ids={}
        for g in self.groups:
            fields={f:sorted({self.text_idx[item['text']] for x in g['node_ids'] for item in representations[x]['fields'][f]}) for f in FIELDS}
            used=set()
            for f in FIELDS:
                fields[f]=[i for i in fields[f] if normalize(self.texts[i]) not in used]; used.update(normalize(self.texts[i]) for i in fields[f])
            self.field_ids[g['id']]=fields
            self.rerank_ids[g['id']]=sorted({self.text_idx[item['text']] for x in g['node_ids'] for f in ['claim','task','problem'] for item in representations[x]['fields'][f] if item['path'].startswith(('direct:','explicit_name_in_claim:'))})

    def score(self,query_vectors,intent,ids=None,type_enabled=False,surface=False):
        ids=list(self.lookup) if ids is None else ids; Q=np.asarray([unit(v) for v in query_vectors]); needed=sorted({i for x in ids for f,ii in self.field_ids[x].items() if not surface or f=='name' for i in ii})
        sims=self.X[needed]@Q.T; loc={x:i for i,x in enumerate(needed)}; rows=[]
        for ident in ids:
            g=self.lookup[ident]; components={}; per_view=np.zeros(len(Q))
            for field in FIELDS:
                ii=self.field_ids[ident][field]
                if surface and field!='name': components[field]=0.; continue
                if ii:
                    values=sims[[loc[i] for i in ii]]
                    scores=values.max(axis=0) if self.cfg['context_pooling']=='max' else np.sort(values,axis=0)[-self.cfg['context_top_m']:].mean(axis=0)
                else: scores=np.zeros(len(Q))
                weight=1. if surface else self.cfg['context_field_weights'][field]
                components[field]=float(scores.mean())*weight; per_view+=weight*scores
            if self.cfg['context_fusion']=='max': value=float(per_view.max())
            else: value=float(per_view.mean())
            prior=self.cfg['context_type_weight']*type_prior(g['types'],intent) if type_enabled else 0.
            if self.cfg['context_fusion']=='max':
                selected_view=int(per_view.argmax())
                for field in FIELDS:
                    ii=self.field_ids[ident][field]
                    if surface and field!='name' or not ii: components[field]=0.; continue
                    vv=sims[[loc[i] for i in ii],selected_view]
                    pooled=float(vv.max()) if self.cfg['context_pooling']=='max' else float(np.sort(vv)[-self.cfg['context_top_m']:].mean())
                    components[field]=pooled*(1. if surface else self.cfg['context_field_weights'][field])
            components.update(type=prior,temporal=0.); value+=prior
            rows.append(dict(entity_id=ident,node_ids=g['node_ids'],name=g['name'],types=g['types'],article_ids=g['article_ids'],score=value,score_components=components,view_scores=per_view.tolist()))
        rows.sort(key=lambda r:(-r['score'],r['entity_id']))
        return rows,dict(vector_comparisons=len(needed)*len(Q),unique_text_vectors=len(needed),query_views=len(Q),entity_scores=len(ids))

    def search(self,query_vectors,intent,type_enabled=False,surface=False,rerank=False):
        start=time.perf_counter(); ranked,cost=self.score(query_vectors,intent,type_enabled=type_enabled,surface=surface)
        candidate=ranked[:self.cfg['context_candidate_k']]; original_ids=[r['entity_id'] for r in ranked]
        if rerank:
            query_terms=set(re.findall(r'\w+',intent['task'].casefold()))-ENGLISH_STOP_WORDS
            for r in candidate:
                scores=[]
                for i in self.rerank_ids[r['entity_id']]:
                    terms=set(re.findall(r'\w+',self.texts[i].casefold()))-ENGLISH_STOP_WORDS; scores.append(len(query_terms&terms)/max(len(query_terms),1))
                bonus=self.cfg['context_rerank_weight']*(max(scores) if scores else 0.)
                r['score_components']['rerank']=bonus; r['score']+=bonus
            candidate.sort(key=lambda r:(-r['score'],r['entity_id'])); ranked=candidate+ranked[len(candidate):]
        return dict(ranked=ranked,candidate_entity_ids=original_ids,returned=ranked[:100],cost=cost,
            latency_seconds=time.perf_counter()-start,expansions=0,method='flat_surface' if surface else 'flat_context',
            reranker='stopword-filtered query-term overlap in bounded direct/explicit-name claim/task/problem fields' if rerank else None)
