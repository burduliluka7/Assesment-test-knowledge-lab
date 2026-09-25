"""Compare regenerated partitions with the delivered real-data hierarchies."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tkh_abstraction.io import read_json,write_json
from tkh_abstraction.semantics import Encoder,node_texts
from tkh_abstraction.snapshots import snapshot
from tkh_abstraction.hierarchy import build
import numpy as np

out=Path('artifacts/full'); cfg=read_json(out/'environment.json')['config']
data=read_json(Path(cfg['data_dir'])/'tkh_collection10.json')
enc=Encoder(cfg['embedding_model'],cfg['embedding_revision']); rows=[]
snaps={y:snapshot(data,y) for y in cfg['snapshots']}
Xs={y:enc.encode(node_texts(s['nodes'],cfg['include_type']),f'real:{y}:clustering') for y,s in snaps.items()}
y=cfg['snapshots'][0]
fresh=enc.encode(node_texts(snaps[y]['nodes'],cfg['include_type']),'fresh-reproduction-check')
assert fresh.dtype==Xs[y].dtype==np.float32
assert np.array_equal(fresh,Xs[y]),'Fresh and cached real model embeddings differ'
def partition(h): return [[(c['id'],c['parent_id'],c['persistent_id'],c['member_ids']) for c in level] for level in h['levels']]
for v in cfg['variants']:
    previous=None
    for y,s in snaps.items():
        old=read_json(out/'variants'/v/f'hierarchy_{y}.json')
        h,_=build(s,Xs[y],cfg,v,previous)
        assert partition(h)==partition(old),f'Partition mismatch: {v} {y}'
        rows.append(dict(variant=v,snapshot=y,identical=True)); previous=h
        print('Identical',v,y,flush=True)
write_json(out/'reproduction_check.json',dict(status='passed',fresh_cached_model_vectors_identical=True,hierarchies=rows))
