"""Candidate-specific graph evidence after frozen H2 discovery; no answer-key input."""
from collections import defaultdict
from itertools import count
import heapq,math,re,time
import numpy as np
from .contextual import entity_groups
from .diffusion import MentionIndex
from .target_mapping import normalize

FIELD_ORDER=('direct_claims','explicit_mentions','task_problem','technical','dataset','article','path_evidence')
DIRECT_RELATIONS={'claims','addresses','solves','uses_technique','uses_component','evaluated_on','presents'}
TYPE_FIELD={'claim':'direct_claims','task':'task_problem','problem':'task_problem','technique':'technical','component':'technical','dataset':'dataset','metric':'dataset','article':'article'}


class CandidateEvidenceIndex:
    def __init__(self,snap,groups,operator,mention_index,cfg):
        self.snap=snap; self.groups={g['id']:g for g in groups}; self.nodes={n['id']:n for n in snap['nodes']}
        self.op=operator; self.mentions=mention_index; self.cfg=cfg; self.incident=defaultdict(list); self.article_nodes=defaultdict(list)
        for e in snap['hyperedges']:
            for x in e['members']: self.incident[x].append(e)
        for n in snap['nodes']:
            for aid in n.get('provenance',{}).get('articles',[]):
                if n['type']=='article': self.article_nodes[aid].append(n['id'])
        degrees=operator.degree; positive=degrees[degrees>0]
        self.hub_threshold=float(np.quantile(positive,cfg['reranker_hub_quantile'])) if len(positive) else float('inf')
        self.generic={x:self.genericness(x) for x in self.nodes}; self.pools={}
        for group in groups: self.pools[group['id']]=self._pool(group)

    def genericness(self,ident):
        node=self.nodes[ident]; degree=float(self.op.degree[self.op.idx[ident]])
        short=len(re.findall(r'\w+',node['surface_form']))<=self.cfg['reranker_short_label_tokens']
        high=degree>=self.hub_threshold and degree>0
        frequency=len(node.get('provenance',{}).get('articles',[]))
        broad_type=node['type'] in {'task','problem','future_topic','technique'}
        identifier=bool(re.search(r'[a-z][A-Z]|\d',node['surface_form']) or re.fullmatch(r'[A-Z][A-Z0-9()-]{2,}',node['surface_form']))
        generic=high and short and not identifier and (broad_type or frequency>=self.cfg['reranker_cross_article_min'])
        return dict(generic_bridge=generic,weighted_degree=degree,high_degree=high,short_label=short,cross_article_frequency=frequency,
            identifier_signal=identifier,relation_count=len({e['relation_type'] for e in self.incident[ident]}))

    def item(self,ident,kind,priority,path):
        n=self.nodes[ident]
        return dict(node_id=ident,text=n['surface_form'],node_type=n['type'],kind=kind,priority=priority,path=path,
            first_seen_year=n['first_seen_year'],provenance=n.get('provenance',{}),genericness=self.generic[ident])

    def _pool(self,group):
        pools={f:[] for f in FIELD_ORDER}; members=set(group['node_ids']); presented=set()
        for ident in sorted(members):
            for e in sorted(self.incident[ident],key=lambda e:e['id']):
                rel=e['relation_type']
                if rel not in DIRECT_RELATIONS or len(e['members'])>self.cfg['reranker_direct_arity_cap']: continue
                for other in sorted(set(e['members'])-members):
                    n=self.nodes[other]; field=TYPE_FIELD.get(n['type'])
                    if not field: continue
                    if field=='article' and rel!='presents': continue
                    if field=='direct_claims' and rel!='claims': continue
                    if self.generic[other]['generic_bridge']: continue
                    priority=1 if field=='direct_claims' else 3 if field=='task_problem' else 5 if field=='article' else 4
                    item=self.item(other,'direct:'+rel,priority,[e['id']])
                    item['support_kind']='low_arity_direct' if len(e['members'])<=3 else 'comembership_only'
                    item['edge_arity']=len(e['members'])
                    pools[field].append(item)
                    if field=='article': presented.add(other)
        for ident in self.mentions.mentions.get(group['id'],[]):
            if self.nodes[ident]['type']=='claim' and not self.generic[ident]['generic_bridge']:
                pools['explicit_mentions'].append(dict(self.item(ident,'safe_explicit_name',2,[]),support_kind='name_match'))
        # Titles only, never every claim that happens to share a publication.
        if not presented:
            for aid in group['article_ids']:
                for ident in self.article_nodes.get(aid,[]):
                    if ident not in members and not self.generic[ident]['generic_bridge']:
                        pools['article'].append(self.item(ident,'provenance_title_fallback',7,[]))
        return pools

    def support_paths(self,seeds):
        """One query-wide bounded best-support traversal, not one BFS per candidate.

        Reuses native edge memberships and Theta transition factors. Two paths per
        vertex; ranking evidence ignores generic intermediate nodes.
        """
        heap=[]; serial=count(); best=defaultdict(list); cfg=self.cfg
        selected=sorted((s for s in seeds if s['y']>0),key=lambda s:(-s['semantic_score'],s['node_id']))[:cfg['reranker_path_seed_limit']]
        for seed in selected: heapq.heappush(heap,(-float(seed['y']),0,next(serial),seed['node_id'],seed,[]))
        expansions=0
        while heap:
            neg,hops,_,node,seed,path=heapq.heappop(heap)
            if len(best[node])>=cfg['reranker_path_count']: continue
            if any(p['seed_node_id']==seed['node_id'] for p in best[node]): continue
            best[node].append(dict(seed_node_id=seed['node_id'],seed_text=seed['surface_form'],seed_semantic_score=seed['semantic_score'],
                support=-neg,hops=hops,path=path))
            if hops>=cfg['reranker_path_max_hops']: continue
            expansions+=1; visited={seed['node_id']}|{step['to_node'] for step in path}
            for edge in self.incident[node]:
                weight=float(edge.get('weight',1.)); arity=len(edge['members'])
                for dest in sorted(edge['members']):
                    if dest in visited or len(best[dest])>=cfg['reranker_path_count']: continue
                    transition=self.op.inv[self.op.idx[node]]*weight/arity*self.op.inv[self.op.idx[dest]]
                    if transition<=0: continue
                    step=dict(from_node=node,to_node=dest,edge_id=edge['id'],relation=edge['relation_type'],theta_transition=float(transition),
                        from_role='generic_bridge' if self.generic[node]['generic_bridge'] else 'specific_evidence',
                        to_role='generic_bridge' if self.generic[dest]['generic_bridge'] else 'specific_evidence')
                    heapq.heappush(heap,(neg*transition,hops+1,next(serial),dest,seed,path+[step]))
        return dict(best),dict(path_expansions=expansions,seed_nodes=len(selected),max_hops=cfg['reranker_path_max_hops'],
            interpretation='Bounded high-product undirected walks; approximate support, not exact diffusion attribution or entailment')

    def build_candidate_evidence(self,question,candidate,similarities,paths,h2_rank,r3_rank=None):
        group=self.groups[candidate['entity_id']]; members=set(group['node_ids']); pools={f:list(items) for f,items in self.pools[group['id']].items()}
        candidates=[]
        for ident in sorted(members):
            for path in paths.get(ident,[]):
                if path['hops']>0: candidates.append(path)
        selected_paths=[]; seen=set()
        for path in sorted(candidates,key=lambda p:(-p['support'],p['hops'],p['seed_node_id'])):
            key=(path['seed_node_id'],tuple(step['edge_id'] for step in path['path']))
            if key in seen: continue
            seen.add(key)
            annotated=dict(path,path=[dict(step,to_role='target_entity' if step['to_node'] in members else step['to_role']) for step in path['path']])
            annotated['via_generic_bridge']=any(step['to_role']=='generic_bridge' for step in annotated['path'][:-1])
            selected_paths.append(annotated)
            if len(selected_paths)>=self.cfg['reranker_path_count']: break
        # Only the specific source text is eligible: intermediate bridge labels remain diagnostic.
        for path in selected_paths:
            ident=path['seed_node_id']
            if ident not in members and self.nodes[ident]['type'] in {'claim','task','problem'} and not self.generic[ident]['generic_bridge']:
                pools['path_evidence'].append(dict(self.item(ident,'short_path_source',6,path['path']),via_generic_bridge=path['via_generic_bridge'],support_kind='indirect_path_source'))
        fields={}; discarded=[]; used={normalize(self.nodes[x]['surface_form']) for x in members}
        available_counts={}; best_sims={}
        for field in FIELD_ORDER:
            unique={}
            for item in sorted(pools[field],key=lambda x:(x['priority'],x.get('support_kind')=='comembership_only',x['node_id'])):
                key=normalize(item['text'])
                if key not in used: unique.setdefault(key,item)
            ranked=sorted(unique.values(),key=lambda x:(x['priority'],x.get('support_kind')=='comembership_only',-similarities[x['node_id']],x['node_id']))
            available_counts[field]=len(ranked); best_sims[field]=max((float(similarities[x['node_id']]) for x in ranked),default=None)
            cap=self.cfg['reranker_field_caps'][field]; chosen=ranked[:cap]
            fields[field]=[dict(item,similarity=float(similarities[item['node_id']])) for item in chosen]
            used.update(normalize(item['text']) for item in chosen)
            discarded.extend(dict(node_id=item['node_id'],field=field,reason='field_cap') for item in ranked[cap:])
        name_score=max(float(similarities[x]) for x in members)
        bundle=dict(entity_id=group['id'],name=group['name'],types=group['types'],node_ids=group['node_ids'],question=question,cutoff=self.snap['snapshot'],
            h2_rank=h2_rank,h2_score=candidate['score'],r3_rank=r3_rank,fields=fields,name_similarity=name_score,
            available_counts=available_counts,best_similarity_by_field=best_sims,selected_paths=selected_paths,discarded=discarded,
            similarity_ids_consulted=sorted(members|{item['node_id'] for items in pools.values() for item in items}),
            specific_evidence_count=sum(len(fields[f]) for f in FIELD_ORDER if f not in {'article','path_evidence'}))
        # Numeric ranking priors remain metadata, never semantic text for the reranker.
        blocks=[dict(field='name',node_id=None,text='Candidate: '+group['name']+'\nTypes: '+', '.join(group['types']))]
        for f in FIELD_ORDER:
            for item in fields[f]: blocks.append(dict(field=f,node_id=item['node_id'],text=f+' ['+item.get('support_kind',item['kind'])+']: '+item['text']))
        for path in selected_paths:
            # Bridge identity is in selected_paths, not copied into the semantic document.
            rels=' -> '.join(s['relation'] for s in path['path'])
            blocks.append(dict(field='graph_support',node_id=path['seed_node_id'],text='Graph support: '+rels+' -> candidate'))
        bundle['document_blocks']=blocks
        bundle['document']='\n'.join(b['text'] for b in blocks)
        return bundle
