"""Explicit failure recovery of completed experiment artifacts; NOT a fresh run.

Recompute summaries/null metrics, replay saved NLI/perturbation results only after
checking their inputs, and execute corrected benchmark scoring. Rankings are
recomputed and checked against the already frozen files. Gate K uses a separate
ordinary full run, never this recovery path.
"""
import sys,time,hashlib
from pathlib import Path
import torch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tkh_abstraction.data import read,write,load_graph,sha
from tkh_abstraction.temporal import transition
from tkh_abstraction import evaluation as ev
from tkh_abstraction.plotting import deliver

out=Path(sys.argv[1]);cfg=read(out/'audit/configuration.json');torch.set_num_threads(cfg['threads'])
graph=load_graph(Path(cfg['data_dir']));snaps=[read(out/f'snapshots/snapshot_{y}.json') for y in cfg['snapshots']]
all_h={v:[read(out/f'hierarchies/{v}/hierarchy_{y}.json') for y in cfg['snapshots']] for v in cfg['variants']}
trans={v:{f"{a['snapshot']}->{b['snapshot']}":transition(a,b) for a,b in zip(hs,hs[1:])} for v,hs in all_h.items()}
base={n['id']:i for i,n in enumerate(graph['nodes'])}
def cached_vectors(texts,name,revision,folder):
    keys=[hashlib.sha256((name+revision+t).encode()).hexdigest() for t in texts]
    return np.array([np.load(Path(cfg['cache_dir'])/folder/(k+'.npy')) for k in keys])
vectors=cached_vectors([f"{n['type']}: {n['surface_form']}" for n in graph['nodes']],cfg['construction_model'],cfg['construction_revision'],'construction')
saved_labels=read(out/'labels/faithfulness.json');pred_sha=sha(out/'retrieval/predictions.json');calls={v:0 for v in cfg['variants']};used=[]
class ReplayedNLI:
    def __init__(self,c):self.rows=[r for r in saved_labels if r['evaluable']]
    def score(self,pairs):
        assert pairs==[(r['held_out_text'],r['gloss']) for r in self.rows]
        return [r['nli'] for r in self.rows]
ev.NLI=ReplayedNLI
def replay_build(snap,x,c,variant,previous):
    if calls[variant]<5:
        seed=cfg['perturbation_seeds'][calls[variant]];calls[variant]+=1;p=out/f'perturbations/{variant}_{seed}.json'
        rng=np.random.default_rng(seed);removed=set(map(int,rng.choice(len(snaps[-1]['hyperedges']),round(cfg['perturbation_fraction']*len(snaps[-1]['hyperedges'])),replace=False)))
        assert snap['hyperedges']==[e for i,e in enumerate(snaps[-1]['hyperedges']) if i not in removed]
    else:
        changed=[k for k in ['alpha','lambda','knn_k'] if c[k]!=cfg[k]];assert len(changed)==1;k=changed[0];p=out/f'sensitivity/{k}_{c[k]}.json'
    used.append(dict(path=str(p),sha256=sha(p)));return read(p)
ev.build=replay_build
start=time.perf_counter();m=ev.execute_evaluation(graph,snaps,read(out/'audit/data_audit.json')['snapshots'],all_h,trans,vectors,base,cfg,out)
assert sha(out/'retrieval/predictions.json')==pred_sha
assert read(out/'labels/faithfulness.json')==saved_labels
m['runtime_seconds']=None
write(out/'metrics.json',m);deliver(m,all_h,cfg,out)
write(out/'audit/recovered_execution.json',dict(status='RECOVERED',ordinary_full_run=False,reason='Null DOI caused evaluation-only source alignment failure after all experiments and predictions completed.',replayed_artifacts=used,label_probabilities_unchanged=True,predictions_unchanged=True,recovery_seconds=time.perf_counter()-start))
print('RECOVERY COMPLETE',out,flush=True)
