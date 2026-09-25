from pathlib import Path
from copy import deepcopy
import importlib.metadata
import platform
import time
import numpy as np
from .io import read_json,write_json,locate,digest
from .schema import validate_data
from .snapshots import snapshot,describe
from .semantics import Encoder,node_texts
from .hierarchy import build,validate_hierarchy
from .labeling import label
from .hypergraph import from_snapshot
from .baselines import projection_loss
from .evaluation import coherence,stability,faithfulness,extrinsic

def environment(cfg,data):
    packages=['numpy','scipy','scikit-learn','sentence-transformers','transformers','torch','PyYAML','matplotlib']
    return dict(python=platform.python_version(),platform=platform.platform(),
        dependencies={p:importlib.metadata.version(p) for p in packages},config=cfg,
        config_hash=digest(cfg),data_hash=digest(data),seeds=dict(construction=cfg['seed'],perturbation=cfg['perturbation_seeds']),
        evidence_scope='retrospective corpus publication years; frozen model priors, not historically trained models')

def load(cfg):
    data=read_json(locate(cfg['data_dir'],'tkh_collection10.json')); out=Path(cfg['output'])
    if data.get('meta',{}).get('synthetic') and not any(part.lower()=='synthetic' for part in out.parts):
        raise ValueError('Synthetic runs must use a path containing synthetic')
    quality=validate_data(data); write_json(out/'data_quality.json',quality)
    if quality['status']=='invalid': raise ValueError(f"Data validation failed; see {out/'data_quality.json'}")
    semantics=data['meta']['date_semantics']
    meaning=semantics.get('node_first_seen_year','').lower()
    if 'corpus' not in meaning or not ('earliest' in meaning or 'mention' in meaning):
        raise ValueError('Unrecognized first_seen_year semantics; document and adapt before building')
    return data,out

def run(cfg,command='run-all'):
    started=time.perf_counter(); data,out=load(cfg)
    metadata=environment(cfg,data); write_json(out/'environment.json',metadata)
    snaps={y:snapshot(data,y) for y in cfg['snapshots']}
    if any(not s['nodes'] for s in snaps.values()): raise ValueError('Requested cutoff has no visible corpus nodes; choose a populated year')
    descriptions=[]; previous=None
    for s in snaps.values(): descriptions.append(describe(s,previous)); previous=s
    write_json(out/'descriptive.json',dict(status='computed',snapshots=descriptions,
        historical_origin_range=[min(n['origin_year'] for n in data['nodes'] if n.get('origin_year')),max(n['origin_year'] for n in data['nodes'] if n.get('origin_year'))],
        corpus_first_seen_range=[min(n['first_seen_year'] for n in data['nodes']),max(n['first_seen_year'] for n in data['nodes'])]))
    if command in ['validate-data','describe']: return
    if command=='report':
        from .reporting import generate
        generate(cfg,out); return
    cluster=Encoder(cfg['embedding_model'],cfg['embedding_revision'])
    evaluator=Encoder(cfg['evaluation_model'],cfg['evaluation_revision']); evaluation_error=None
    if cfg['embedding_model']==cfg['evaluation_model']: raise ValueError('Independent coherence model must differ from clustering model')
    if command in ['build','label']:
        evaluation_error='Evaluation not requested'
    else:
        try: evaluator.load()
        except (ImportError,OSError,ValueError,RuntimeError) as e: evaluation_error=str(e)
    matrices={}; evaluations={}; results={}; events_by_variant={}; label_logs={}
    for y,s in snaps.items():
        print(f'Encoding snapshot {y}: {len(s["nodes"])} nodes, {len(s["hyperedges"])} edges',flush=True)
        matrices[y]=cluster.encode(node_texts(s['nodes'],cfg['include_type']),f'real:{y}:clustering')
        if not evaluation_error:
            evaluations[y]=evaluator.encode(node_texts(s['nodes'],False),f'real:{y}:independent-evaluation')
    source_root=Path(__file__).parent
    code_hash=digest({p.relative_to(source_root).as_posix():p.read_text(encoding='utf-8') for p in sorted(source_root.rglob('*.py'))})
    metadata['source_hash']=code_hash
    metadata['input_hashes']={name:digest(read_json(locate(cfg['data_dir'],name))) if name.endswith('.json') else digest(locate(cfg['data_dir'],name).read_text(encoding='utf-8-sig'))
        for name in ['tkh_collection10.json','questions.csv','ground_truth.json','collection10_articles.csv']
        if list(Path(cfg['data_dir']).rglob(name))}
    write_json(out/'environment.json',metadata)
    run_key=digest(dict(config=cfg,data=digest(data),code=code_hash,model=cluster.metadata))
    for variant in cfg['variants']:
        results[variant]={}; events_by_variant[variant]=[]; label_logs[variant]=[]; prev=None
        vdir=out/'variants'/variant; vdir.mkdir(parents=True,exist_ok=True)
        for y,s in snaps.items():
            write_json(vdir/f'label_corpus_{y}.json',[dict(id=n['id'],type=n['type'],surface_form=n['surface_form'],first_seen_year=n['first_seen_year']) for n in s['nodes']])
            path=vdir/f'hierarchy_{y}.json'; lp=vdir/f'label_inputs_{y}.json'; ep=vdir/f'events_{y}.json'
            if path.exists() and lp.exists() and ep.exists() and read_json(path).get('run_key')==run_key:
                h=read_json(path); events=read_json(ep); logs=read_json(lp)
                validate_hierarchy(h,{n['id'] for n in s['nodes']}); print(f'Reusing verified {variant} {y}',flush=True)
            else:
                print(f'Building {variant} {y}',flush=True)
                h,events=build(s,matrices[y],cfg,variant,prev); logs=label(h,s,matrices[y]); h['run_key']=run_key
                write_json(path,h); write_json(lp,logs); write_json(ep,events)
                print(f'Built {variant} {y} in {h["runtime_seconds"]:.1f}s',flush=True)
            results[variant][y]=h; events_by_variant[variant]+=events; label_logs[variant]+=logs; prev=h
        write_json(vdir/'temporal_events.json',events_by_variant[variant])
    primary='temporal' if 'temporal' in results else cfg['variants'][0]
    for y,h in results[primary].items(): write_json(out/f'hierarchy_{y}.json',h)
    write_json(out/'temporal_events.json',events_by_variant[primary])
    write_json(out/'models.json',dict(clustering=cluster.metadata,evaluation=evaluator.metadata))
    if command in ['build','label']: return
    metrics=dict(status='running',data_kind='real' if not data['meta'].get('synthetic') else 'synthetic',
        config_hash=metadata['config_hash'],run_key=run_key,primary_variant=primary,variants={},sensitivity={},
        projection_loss=projection_loss(from_snapshot(snaps[cfg['snapshots'][-1]],matrices[cfg['snapshots'][-1]])))
    for variant in cfg['variants']:
        print(f'Evaluating {variant}',flush=True); vm=dict(status='computed',coherence={},cross_snapshot={},perturbation={},extrinsic={})
        for y,s in snaps.items():
            vm['coherence'][str(y)]=coherence.evaluate(results[variant][y],s['nodes'],evaluations[y],cfg['null_permutations'],cfg['seed']) if not evaluation_error else dict(status='missing_model',reason=evaluation_error)
        years=cfg['snapshots']
        for a,b in zip(years,years[1:]):
            counts={}
            for k in range(len(cfg['budgets'])+1):
                from collections import Counter
                counts[str(k)]=dict(Counter(e['event_type'] for e in events_by_variant[variant] if e['source_snapshot']==a and e['target_snapshot']==b and e['level']==k))
            vm['cross_snapshot'][f'{a}->{b}']=dict(levels=stability.compare(results[variant][a],results[variant][b]),event_counts=counts,
                growth=describe(snaps[b],snaps[a]))
        # Conditional perturbation at the final snapshot: predecessor held fixed, no node dropping.
        y=years[-1]; s=snaps[y]; reference=results[variant][y]
        prev=results[variant][years[-2]] if len(years)>1 else None; records=[]
        for seed in cfg['perturbation_seeds']:
            rng=np.random.default_rng(seed); m=len(s['hyperedges']); count=max(1,int(round(.10*m))) if m else 0
            removed=set(map(int,rng.choice(m,count,replace=False))) if m else set(); pert=dict(s)
            pert['hyperedges']=[e for i,e in enumerate(s['hyperedges']) if i not in removed]
            print(f'Perturbation {variant} {y}, removal seed {seed}',flush=True)
            h,_=build(pert,matrices[y],cfg,variant,prev)
            records.append(dict(seed=seed,removed_edges=[s['hyperedges'][i]['id'] for i in sorted(removed)],fraction=count/m if m else 0.,levels=stability.compare(reference,h)))
        vm['perturbation']=dict(status='computed',snapshot=y,protocol='Uniform removal of round(10% of visible edges), five fixed seeds; predecessor and construction seed fixed',
            records=records,summary=stability.summarize_perturbations(records))
        vm['faithfulness']=faithfulness.evaluate(label_logs[variant],cfg,out/'variants'/variant)
        if not evaluation_error:
            for cutoff in [2025,2026]:
                if cutoff>years[-1]: continue
                ss=snaps.get(cutoff) or snapshot(data,cutoff)
                if cutoff in results[variant]: hh=results[variant][cutoff]
                else:
                    bp=out/'variants'/variant/f'hierarchy_{cutoff}_benchmark.json'
                    if bp.exists() and read_json(bp).get('run_key')==run_key:
                        hh=read_json(bp); validate_hierarchy(hh,{n['id'] for n in ss['nodes']})
                    else:
                        xx=cluster.encode(node_texts(ss['nodes'],cfg['include_type']),f'real:{cutoff}:clustering')
                        predecessors=[t for t in years if t<cutoff]; pp=results[variant][max(predecessors)] if predecessors else None
                        hh,ee=build(ss,xx,cfg,variant,pp); ll=label(hh,ss,xx); hh['run_key']=run_key
                        write_json(bp,hh)
                        write_json(out/'variants'/variant/f'events_{cutoff}_benchmark.json',ee)
                        write_json(out/'variants'/variant/f'label_inputs_{cutoff}_benchmark.json',ll)
                        write_json(out/'variants'/variant/f'label_corpus_{cutoff}.json',[dict(id=n['id'],type=n['type'],surface_form=n['surface_form'],first_seen_year=n['first_seen_year']) for n in ss['nodes']])
                yy=evaluations.get(cutoff)
                if yy is None: yy=evaluator.encode(node_texts(ss['nodes'],False),f'real:{cutoff}:independent-evaluation')
                try:
                    vm['extrinsic'][str(cutoff)]=extrinsic.evaluate(hh,ss,yy,evaluator,cfg,cfg['data_dir'],out/'variants'/variant/f'gt_mapping_{cutoff}.json')
                except FileNotFoundError as e: vm['extrinsic'][str(cutoff)]=dict(status='missing_data',reason=str(e))
        else: vm['extrinsic']=dict(status='missing_model',reason=evaluation_error)
        vm['runtime_seconds']={str(y):h['runtime_seconds'] for y,h in results[variant].items()}
        metrics['variants'][variant]=vm; write_json(out/'metrics.json',metrics)
    if cfg.get('sensitivity') and 'temporal' in results:
        y=cfg['snapshots'][-1]; previous=results['temporal'][cfg['snapshots'][-2]] if len(cfg['snapshots'])>1 else None
        choices={'semantic_low':dict(hyper=.45,spectral=.30,semantic=.10,temporal=.15),
                 'semantic_high':dict(hyper=.12,spectral=.08,semantic=.65,temporal=.15),
                 'temporal_low':dict(hyper=.30,spectral=.20,semantic=.35,temporal=.05),
                 'temporal_high':dict(hyper=.30,spectral=.20,semantic=.35,temporal=.35)}
        for name,w in choices.items():
            print(f'Intrinsic-only sensitivity {name}',flush=True)
            c=deepcopy(cfg); c['weights']=w; h,_=build(snaps[y],matrices[y],c,'temporal',previous)
            metrics['sensitivity'][name]=dict(status='computed',weights=w,snapshot=y,
                protocol='One-step construction sensitivity conditional on default predecessor; no benchmark tuning',
                versus_default=stability.compare(results['temporal'][y],h),
                versus_previous=stability.compare(previous,h) if previous else None,
                coherence=coherence.evaluate(h,snaps[y]['nodes'],evaluations[y],cfg['null_permutations'],cfg['seed']) if not evaluation_error else dict(status='missing_model'))
    metrics['status']='computed'; metrics['wall_seconds']=time.perf_counter()-started; write_json(out/'metrics.json',metrics)
    from .reporting import generate
    generate(cfg,out)
    print(f'Finished. Artifacts: {out}',flush=True)
