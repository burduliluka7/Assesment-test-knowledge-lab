"""Re-evaluate benchmark after a mapping correction without repeating graph experiments.

This narrowly scoped maintenance script was used after preserving model-name
hyphens/plus signs. Normal clean reproduction uses run-all with the corrected rule.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tkh_abstraction.io import read_json,write_json,digest
from tkh_abstraction.semantics import Encoder,node_texts
from tkh_abstraction.snapshots import snapshot
from tkh_abstraction.evaluation.extrinsic import evaluate
from tkh_abstraction.reporting import generate

out=Path(sys.argv[1] if len(sys.argv)>1 else 'artifacts/full')
env=read_json(out/'environment.json'); cfg=env['config']; m=read_json(out/'metrics.json')
if m['status']!='computed': raise RuntimeError('Wait for the complete graph experiment before refreshing retrieval')
data=read_json(Path(cfg['data_dir'])/'tkh_collection10.json')
enc=Encoder(cfg['evaluation_model'],cfg['evaluation_revision'])
for variant in cfg['variants']:
    for cutoff in [2025,2026]:
        vd=out/'variants'/variant; hp=vd/f'hierarchy_{cutoff}.json'
        if not hp.exists(): hp=vd/f'hierarchy_{cutoff}_benchmark.json'
        if not hp.exists(): continue
        s=snapshot(data,cutoff); h=read_json(hp)
        X=enc.encode(node_texts(s['nodes'],False),f'real:{cutoff}:independent-evaluation')
        m['variants'][variant]['extrinsic'][str(cutoff)]=evaluate(h,s,X,enc,cfg,cfg['data_dir'],vd/f'gt_mapping_{cutoff}.json')
    print('Refreshed retrieval',variant,flush=True)
root=Path('src/tkh_abstraction')
m['retrieval_revision']=dict(reason='Preserve hyphens and plus signs in target names; avoid collapsing distinct model variants. Correctness fix after exploratory mapping audit, without changing construction or tuning on retrieval.',
    source_hash=digest({p.relative_to(root).as_posix():p.read_text(encoding='utf-8') for p in sorted(root.rglob('*.py'))}))
write_json(out/'metrics.json',m); generate(cfg,out)
