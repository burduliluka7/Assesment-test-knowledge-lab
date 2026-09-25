import time
from .hypergraph import from_snapshot,objective_hyper
from .collapse import collapse
from .spectral import coordinates
from .coarsening import agglomerate
from .temporal import assignments,track
from .baselines import clique_projection

def export_unit(u,snapshot,level):
    return dict(id=u.id,persistent_id=None,snapshot=snapshot,level=level,parent_id=None,
        member_ids=list(u.members),base_node_count=u.size,node_type_distribution=u.types,
        label='',gloss='')

def build(snap,X,cfg,variant='temporal',previous=None):
    start=time.perf_counter(); year=snap['snapshot']; K=len(cfg['budgets'])
    graph=from_snapshot(snap,X,cfg['relation_weights']); native=graph
    if variant=='pairwise': graph=clique_projection(graph)
    levels=[None]*(K+1); coarse_edges={}; diagnostics=[]
    # Snapshot-specific ids avoid confusing temporal correspondence with node identity.
    graph,_=collapse(graph,[[u.id] for u in graph.units],[f'{year}:L{K}:{u.id}' for u in graph.units])
    levels[K]=[export_unit(u,year,K) for u in graph.units]
    for level in reversed(range(K)):
        s,sd=coordinates(graph,cfg['spectral_dim'],cfg['seed']) if variant!='semantic' else (__import__('numpy').zeros((len(graph.units),1)),dict(skipped=True))
        prev=assignments(previous['levels'][level]) if previous else None
        groups,diag=agglomerate(graph,s,cfg['budgets'][level],cfg,variant,prev)
        ids=[f'{year}:L{level}:{i:04d}' for i in range(len(groups))]
        graph,mapping=collapse(graph,groups,ids)
        for c in levels[level+1]: c['parent_id']=mapping[c['id']]
        levels[level]=[export_unit(u,year,level) for u in graph.units]
        coarse_edges[str(level)]=[dict(original_edge_id=e.id,relation_type=e.relation,provenance=e.provenance,
            original_arity=e.arity,original_members=list(e.original_members),multiplicities=e.multiplicities,
            support_arity=len(e.multiplicities),internal=e.internal,weight=e.weight) for e in graph.edges]
        diagnostics.append(dict(level=level,spectral=sd,agglomeration=diag,native_fragmentation=objective_hyper(native,assignments(levels[level]))))
    result=dict(snapshot=year,variant=variant,budgets=cfg['budgets'],levels=levels,coarse_hyperedges=coarse_edges,diagnostics=diagnostics,runtime_seconds=time.perf_counter()-start)
    events=[]
    for level,clusters in enumerate(levels):
        events+=track(previous['levels'][level] if previous else [],clusters,previous['snapshot'] if previous else None,year,level,cfg['match_threshold'],cfg['event_threshold'])
    validate_hierarchy(result,{n['id'] for n in snap['nodes']})
    return result,events

def validate_hierarchy(h,expected):
    for k,level in enumerate(h['levels']):
        if any(not c['member_ids'] for c in level): raise ValueError('partition contains an empty cluster')
        members=[x for c in level for x in c['member_ids']]
        if len(members)!=len(set(members)) or set(members)!=expected: raise ValueError('coverage or overlap invariant failed')
        if len({c['id'] for c in level})!=len(level): raise ValueError('duplicate cluster ids')
        if k<len(h['budgets']) and len(level)>h['budgets'][k]: raise ValueError('hard budget violated')
        if k==0:
            if any(c['parent_id'] is not None for c in level): raise ValueError('root has parent')
        else:
            parents={c['id']:set(c['member_ids']) for c in h['levels'][k-1]}
            for c in level:
                if c['parent_id'] not in parents or not set(c['member_ids'])<=parents[c['parent_id']]: raise ValueError('laminar parent invariant failed')
    if any(len(c['member_ids'])!=1 for c in h['levels'][-1]): raise ValueError('finest level not singleton')
